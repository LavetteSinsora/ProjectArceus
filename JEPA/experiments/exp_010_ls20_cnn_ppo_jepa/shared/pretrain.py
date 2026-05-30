"""Random-policy data collection + offline JEPA pretraining (exp_010_2).

`collect_random(...)` runs a uniform-random agent in the real LS20 env and
stores `(obs_t, a_t, obs_{t+1})` transitions (episode-ending steps skipped, so
`obs_{t+1}` is always a real env transition, never a reset). It reports the
number of **environment steps** consumed — the quantity the spec asks us to
report for the random-policy variant.

`pretrain_jepa(...)` trains encoder + forward predictor + IDM on that buffer
until the held-out JEPA loss plateaus, then writes `encoder_final.pt` (plus the
full module bundle) which the PPO phase loads as its encoder warm start.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .device import get_device
from .model import ActorCritic, ActionConditionedPredictor, InverseDynamicsModel
from .ls20_vec_env import VecLS20Env
from .jepa import jepa_losses_on_batch
from .metrics import MetricsWriter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def collect_random(cfg, n_transitions: int, out_path: Path, smoke: bool = False) -> dict:
    """Collect ~n_transitions random (s, a, s') tuples. Returns meta dict."""
    if smoke:
        n_transitions = 256
    np.random.seed(cfg.seed)
    envs = VecLS20Env(cfg.env_name, n_envs=cfg.n_envs,
                      max_episode_steps=cfg.max_episode_steps, seed=cfg.seed)
    F = envs.FRAME
    obs_buf = np.zeros((n_transitions, F, F), dtype=np.uint8)
    nxt_buf = np.zeros((n_transitions, F, F), dtype=np.uint8)
    act_buf = np.zeros((n_transitions,), dtype=np.int64)

    filled = 0
    env_steps = 0
    obs = envs.current_obs()
    t0 = time.time()
    while filled < n_transitions:
        actions = np.random.randint(0, envs.n_actions, size=envs.n_envs)
        next_obs, _, dones, _ = envs.step(actions)
        env_steps += envs.n_envs
        for i in range(envs.n_envs):
            if dones[i]:
                continue  # s' is a reset frame, not a real transition
            if filled < n_transitions:
                obs_buf[filled] = obs[i]
                nxt_buf[filled] = next_obs[i]
                act_buf[filled] = actions[i]
                filled += 1
        obs = next_obs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, obs=obs_buf[:filled], next_obs=nxt_buf[:filled],
                        actions=act_buf[:filled])
    meta = {
        "n_transitions": int(filled),
        "env_steps_used": int(env_steps),
        "collect_seconds": round(time.time() - t0, 2),
        "env_name": cfg.env_name,
        "max_episode_steps": cfg.max_episode_steps,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[exp010_2/collect] {filled} transitions from {env_steps} env steps "
          f"-> {out_path.name}")
    return meta


def pretrain_jepa(cfg, buffer_path: Path, smoke: bool = False) -> dict:
    """Train encoder+predictor+IDM on the random buffer until plateau.

    Plateau rule: stop when the held-out JEPA loss fails to improve by more than
    `min_delta` (relative) for `patience` consecutive epochs, or at max_epochs.
    """
    device = get_device()
    torch.manual_seed(cfg.seed)
    data = np.load(buffer_path)
    obs = torch.from_numpy(data["obs"])
    nxt = torch.from_numpy(data["next_obs"])
    act = torch.from_numpy(data["actions"])
    n = obs.shape[0]

    # 90/10 train/val split.
    perm = np.random.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    predictor = ActionConditionedPredictor(cfg.trunk_dim, cfg.n_actions,
                                           cfg.action_emb_dim).to(device)
    idm = InverseDynamicsModel(cfg.trunk_dim, cfg.n_actions).to(device)
    params = (list(model.encoder.parameters()) + list(predictor.parameters())
              + list(idm.parameters()))
    opt = torch.optim.Adam(params, lr=cfg.learning_rate)

    exp_dir = _repo_root() / cfg.exp_dir
    run_name = f"jepa_pretrain_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = exp_dir / "runs" / run_name
    writer = MetricsWriter(run_dir)

    max_epochs = 3 if smoke else getattr(cfg, "pretrain_max_epochs", 40)
    patience = 1 if smoke else getattr(cfg, "pretrain_patience", 4)
    min_delta = getattr(cfg, "pretrain_min_delta", 0.002)
    bs = 256

    def run_epoch(idxs, train: bool):
        if train:
            model.train(); predictor.train(); idm.train()
        else:
            model.eval(); predictor.eval(); idm.eval()
        order = np.random.permutation(idxs) if train else idxs
        jl = il = ia = 0.0
        steps = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for s in range(0, len(order), bs):
                sel = order[s:s + bs]
                o = obs[sel].to(device); no = nxt[sel].to(device); a = act[sel].to(device)
                l_jepa, l_idm, acc = jepa_losses_on_batch(model, predictor, idm, o, no, a,
                                                          cfg.idm_coef)
                if train:
                    loss = cfg.jepa_coef * l_jepa + cfg.idm_coef * l_idm
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step()
                jl += l_jepa.item(); il += l_idm.item(); ia += acc; steps += 1
        steps = max(1, steps)
        return jl / steps, il / steps, ia / steps

    best_val = float("inf")
    bad = 0
    env_steps_meta = json.loads((buffer_path.with_suffix(".meta.json")).read_text()) \
        if buffer_path.with_suffix(".meta.json").exists() else {}
    for epoch in range(1, max_epochs + 1):
        tr_j, tr_i, tr_a = run_epoch(tr_idx, True)
        va_j, va_i, va_a = run_epoch(val_idx, False)
        writer.write({
            "step": epoch, "update": epoch,
            "train_jepa_loss": tr_j, "train_idm_loss": tr_i, "train_idm_acc": tr_a,
            "val_jepa_loss": va_j, "val_idm_loss": va_i, "val_idm_acc": va_a,
        })
        print(f"[exp010_2/pretrain] epoch {epoch} val_jepa={va_j:.4f} val_idm_acc={va_a:.3f}")
        if va_j < best_val * (1 - min_delta):
            best_val = va_j; bad = 0
            _save_encoder(cfg, model, predictor, idm, exp_dir, env_steps_meta, best_val, epoch)
        else:
            bad += 1
            if bad >= patience:
                print(f"[exp010_2/pretrain] plateaued at epoch {epoch} (best val_jepa={best_val:.4f})")
                break
    writer.close()
    return {"best_val_jepa": best_val, "epochs_run": epoch,
            "data_env_steps": env_steps_meta.get("env_steps_used")}


def _save_encoder(cfg, model, predictor, idm, exp_dir, data_meta, best_val, epoch):
    out = exp_dir / "jepa_pretrained"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": model.encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "idm": idm.state_dict(),
        "best_val_jepa": best_val,
        "epoch": epoch,
        "data_meta": data_meta,
        "config_frame": {"n_colors": cfg.n_colors, "frame_size": cfg.frame_size,
                         "trunk_dim": cfg.trunk_dim, "n_actions": cfg.n_actions},
    }, out / "encoder_final.pt")
