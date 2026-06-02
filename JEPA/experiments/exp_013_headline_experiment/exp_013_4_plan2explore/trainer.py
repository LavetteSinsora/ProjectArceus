"""exp_013_4 — ensemble DISAGREEMENT training loop. See disagreement.py / config.py.

Per update:
  collect rollout → φ(s), φ(s') via the FROZEN encoder → reward = disagreement(φ(s), a)
  (variance across the ensemble) → normalise (warm-up + raw-clip + EMA-std, no centring)
  → non-episodic single-head GAE → PPO → train the ensemble (each member regresses the
  frozen φ(s')) → stop on first extrinsic reward.

No ICM / no φ-freeze: φ is a fixed random encoder, stationary from t=0. Reuses the
audited exp_013_1 helpers (normaliser, GAE, checkpoint).
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.metrics import MetricsWriter, mean_feature_cosine
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter

from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.trainer import (
    _EMAStd, _gae_nonepisodic, _gae_episodic,
)
from .disagreement import FrozenPhi, ForwardEnsemble


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _save_ckpt(run_dir: Path, model, ens, frozen, cfg, global_step):
    ck = run_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    path = ck / f"step_{global_step:08d}.pt"
    torch.save({"step": int(global_step), "config": dataclasses.asdict(cfg),
                "model": model.state_dict(), "ensemble": ens.state_dict(),
                "frozen_phi": frozen.state_dict()}, path)
    return path


@torch.no_grad()
def _phi_batch(frozen: FrozenPhi, frames: torch.Tensor, device, dim, chunk=512) -> torch.Tensor:
    """(M,F,F) uint8 → (M, dim) φ on CPU (frozen encoder)."""
    M = frames.shape[0]
    out = torch.empty(M, dim, dtype=torch.float32)
    for s in range(0, M, chunk):
        out[s:s + chunk] = frozen(frames[s:s + chunk].to(device)).to("cpu")
    return out


@torch.no_grad()
def _disagreement_reward(ens: ForwardEnsemble, phi_s: torch.Tensor, actions: torch.Tensor,
                         dones: torch.Tensor, device, chunk=2048) -> np.ndarray:
    """phi_s (T,N,dim), actions (T,N) → (T,N) disagreement, done-steps zeroed."""
    T, N, D = phi_s.shape
    flat_phi = phi_s.reshape(-1, D)
    flat_a = actions.reshape(-1)
    M = flat_phi.shape[0]
    out = torch.empty(M, dtype=torch.float32)
    for s in range(0, M, chunk):
        out[s:s + chunk] = ens.disagreement(flat_phi[s:s + chunk].to(device),
                                             flat_a[s:s + chunk].to(device)).to("cpu")
    out = out.reshape(T, N) * (~dones).float()
    return out.numpy()


def _ensemble_update(ens: ForwardEnsemble, opt, phi_s, phi_sn, actions, dones, cfg, device) -> float:
    """Train each member to regress φ(s') from (φ(s),a) on non-done transitions."""
    T, N, D = phi_s.shape
    valid = (~dones).reshape(-1)
    ps = phi_s.reshape(-1, D)[valid]
    pn = phi_sn.reshape(-1, D)[valid]
    a = actions.reshape(-1)[valid]
    n = ps.shape[0]
    if n == 0:
        return float("nan")
    mb = max(1, n // cfg.minibatches)
    idx = np.arange(n)
    tot = 0.0
    steps = 0
    for _ in range(cfg.ensemble_epochs):
        np.random.shuffle(idx)
        for s in range(0, n, mb):
            sel = idx[s:s + mb]
            loss = ens.loss(ps[sel].to(device), a[sel].to(device), pn[sel].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ens.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item()
            steps += 1
    return tot / max(1, steps)


def train(cfg, smoke: bool = False) -> dict:
    if smoke:
        cfg = cfg.smoke()
    device = get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    exp_dir = _repo_root() / cfg.exp_dir
    run_name = f"{cfg.exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = exp_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = MetricsWriter(run_dir)

    envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=cfg.seed,
                           level_index=cfg.level_index)
    if cfg.n_actions is None:
        cfg.n_actions = envs.n_actions
    assert cfg.n_actions == envs.n_actions
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    print(f"[exp013_4] {cfg.exp_name}  device={device}  n_actions={cfg.n_actions}  "
          f"K={cfg.n_ensemble}  cap={cfg.max_env_steps}")

    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    frozen = FrozenPhi(n_colors=cfg.n_colors, frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    ens = ForwardEnsemble(k=cfg.n_ensemble, dim=cfg.trunk_dim, n_actions=cfg.n_actions,
                          hidden=cfg.ensemble_hidden).to(device)

    ppo_opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    ens_opt = torch.optim.Adam(ens.parameters(), lr=cfg.ensemble_lr)
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps, c_value=cfg.c_value,
                        c_entropy=cfg.c_entropy, grad_clip=cfg.grad_clip, epochs=cfg.epochs,
                        minibatches=cfg.minibatches)

    rff = RewardForwardFilter(cfg.gamma)
    int_ret_std = _EMAStd(cfg.int_norm_decay)
    raw_mean_ema: float | None = None

    global_step = 0
    first_reward_step: int | None = None
    t_start = time.time()
    stop_now = False

    for update in range(1, cfg.total_updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)
        T, N = rollout.actions.shape
        Fz = rollout.frame
        D = cfg.trunk_dim

        # FROZEN φ for s and s'; cache for the ensemble update.
        phi_s = _phi_batch(frozen, rollout.obs.reshape(-1, Fz, Fz), device, D).reshape(T, N, D)
        phi_sn = _phi_batch(frozen, rollout.next_obs.reshape(-1, Fz, Fz), device, D).reshape(T, N, D)

        raw_i = _disagreement_reward(ens, phi_s, rollout.actions, rollout.dones, device)  # (T,N)
        raw_mean_pre = float(raw_i.mean())
        raw_max_pre = float(raw_i.max())

        warming = update <= cfg.norm_warmup_updates
        if warming:
            norm_i = np.zeros_like(raw_i)
        else:
            if cfg.reward_clip_k is not None:
                raw_mean_ema = (raw_mean_pre if raw_mean_ema is None
                                else 0.99 * raw_mean_ema + 0.01 * raw_mean_pre)
                if raw_mean_ema > 0:
                    raw_i = np.minimum(raw_i, cfg.reward_clip_k * raw_mean_ema)
            rems = np.stack([rff.update(raw_i[t]) for t in range(T)])
            int_ret_std.update(rems)
            norm_i = raw_i / (int_ret_std.std + cfg.int_norm_eps)

        extrinsic = rollout.rewards.clone()                  # +1 = stop signal only
        rollout.rewards = torch.from_numpy(norm_i.astype(np.float32))
        if cfg.intrinsic_episodic:
            _gae_episodic(rollout, cfg.gamma, cfg.gae_lambda)
        else:
            _gae_nonepisodic(rollout, cfg.gamma, cfg.gae_lambda)
        ustats = ppo_update(model, ppo_opt, rollout, ppo_cfg, device)

        ens_loss = _ensemble_update(ens, ens_opt, phi_s, phi_sn, rollout.actions, rollout.dones, cfg, device)
        global_step += cfg.rollout_steps * cfg.n_envs

        if first_reward_step is None and bool((extrinsic > 0).any().item()):
            t_idx = int((extrinsic > 0).float().sum(dim=1).nonzero()[0].item())
            first_reward_step = (global_step - cfg.rollout_steps * cfg.n_envs) + (t_idx + 1) * cfg.n_envs
            print(f"[exp013_4] *** FIRST REWARD at ~{first_reward_step} env steps (update {update}) ***")
            if cfg.stop_on_first_reward:
                stop_now = True

        done_eps = envs.drain_completed_episodes()
        train_succ = float(np.mean([e.success for e in done_eps])) if done_eps else float("nan")

        record = {
            "step": global_step, "update": update,
            "policy_loss": ustats.policy_loss, "value_loss": ustats.value_loss,
            "policy_entropy": ustats.entropy, "approx_kl": ustats.approx_kl,
            "clipfrac": ustats.clipfrac, "grad_norm_total": ustats.grad_norm_total,
            "disagreement_raw_mean": raw_mean_pre, "disagreement_raw_max": raw_max_pre,
            "intrinsic_reward_norm_mean": float(norm_i.mean()), "intrinsic_return_std": int_ret_std.std,
            "v_int_mean": float(rollout.values.mean()), "ret_int_mean": float(rollout.returns.mean()),
            "ensemble_loss": ens_loss, "norm_warming": bool(warming),
            "mean_feature_cosine": mean_feature_cosine(rollout.features, rollout.ep_starts),
            "train_success_rate": train_succ, "train_episodes": len(done_eps),
            "env_steps_to_first_reward": first_reward_step,
            "sps": global_step / max(1e-6, time.time() - t_start),
        }
        if cfg.log_every > 0 and (update % cfg.log_every == 0 or stop_now):
            writer.write(record)
        if update % 25 == 0 or stop_now:
            print(f"[exp013_4] u{update}/{cfg.total_updates} step={global_step} frr={first_reward_step} "
                  f"disag_raw={raw_mean_pre:.4g} r_norm={record['intrinsic_reward_norm_mean']:.4g} "
                  f"ens_loss={ens_loss:.4g} ent={ustats.entropy:.3f}")

        if cfg.save_every > 0 and update % cfg.save_every == 0:
            _save_ckpt(run_dir, model, ens, frozen, cfg, global_step)
        if stop_now:
            break

    writer.close()
    _save_ckpt(run_dir, model, ens, frozen, cfg, global_step)
    solved = first_reward_step is not None
    result = {
        "exp_name": cfg.exp_name, "method": "disagreement", "game": cfg.game,
        "level_index": cfg.level_index, "seed": cfg.seed, "n_ensemble": cfg.n_ensemble,
        "env_steps_to_first_reward": first_reward_step, "solved": solved, "censored": not solved,
        "total_env_steps": global_step, "max_env_steps": cfg.max_env_steps,
        "wall_seconds": time.time() - t_start,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[exp013_4] DONE {cfg.exp_name}: "
          f"{'first reward @ ' + str(first_reward_step) if solved else 'CENSORED'} env steps; total={global_step}")
    return result
