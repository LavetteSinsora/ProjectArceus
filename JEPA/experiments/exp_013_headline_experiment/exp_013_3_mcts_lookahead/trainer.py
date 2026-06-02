"""exp_013_5 — proposal D, B1 lookahead-softmax controller. See lookahead.py / config.py.

Per update:
  collect rollout via the LOOKAHEAD-SOFTMAX policy (model used for the action decision)
  → real novelty at the REAL φ(s') → normalise → non-episodic GAE returns
  → train V_int (regress to returns), ICM (φ+inverse+forward; φ frozen after warm-up),
    RND predictor (+leak) → stop on first extrinsic reward.
No policy gradient (the policy is the lookahead, recomputed each step) → no phantom-advantage.
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
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.metrics import MetricsWriter, mean_feature_cosine
from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule, icm_update_from_rollout
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter

from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.trainer import (
    _EMAStd, _phi_and_novelty, _rnd_update, _gae_nonepisodic, _gae_episodic,
    _collect_holdout, _eval_holdout_inv_acc,
)
from .lookahead import ValueMLP, collect_lookahead_rollout, value_update


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _save_ckpt(run_dir: Path, value, icm, rndphi, cfg, global_step):
    ck = run_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    path = ck / f"step_{global_step:08d}.pt"
    torch.save({"step": int(global_step), "config": dataclasses.asdict(cfg),
                "value": value.state_dict(), "icm": icm.state_dict(),
                "rnd_target": rndphi.target.state_dict(),
                "rnd_predictor": rndphi.predictor.state_dict()}, path)
    return path


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
    print(f"[exp013_5] {cfg.exp_name}  device={device}  n_actions={cfg.n_actions}  "
          f"τ={cfg.tau}  cap={cfg.max_env_steps}")

    icm = ICMModule(n_actions=cfg.n_actions, n_colors=cfg.n_colors, frame_size=cfg.frame_size,
                    trunk_dim=cfg.trunk_dim, hidden=cfg.icm_hidden).to(device)
    rndphi = RNDPhi(dim=cfg.trunk_dim, hidden=cfg.rnd_hidden, out=cfg.rnd_feature_dim,
                    leak=cfg.leak).to(device)
    value = ValueMLP(dim=cfg.trunk_dim, hidden=cfg.value_hidden).to(device)

    icm_opt = torch.optim.Adam(icm.parameters(), lr=cfg.icm_lr)
    rnd_opt = torch.optim.Adam(rndphi.predictor.parameters(), lr=cfg.rnd_lr)
    value_opt = torch.optim.Adam(value.parameters(), lr=cfg.value_lr)

    rff = RewardForwardFilter(cfg.gamma)
    int_ret_std = _EMAStd(cfg.int_norm_decay)
    holdout = _collect_holdout(cfg.game, cfg.level_index, cfg.seed, cfg.holdout_size, device)

    phi_frozen = False
    inv_streak = 0
    freeze_step: int | None = None
    last_inv_acc = float("nan")
    last_holdout_inv = float("nan")
    raw_mean_ema: float | None = None
    global_step = 0
    first_reward_step: int | None = None
    t_start = time.time()
    stop_now = False

    for update in range(1, cfg.total_updates + 1):
        # ACT via lookahead-softmax (model used only for the decision); REAL transitions recorded.
        rollout = collect_lookahead_rollout(envs, icm, rndphi, value, device,
                                            cfg.rollout_steps, cfg.n_actions, cfg.gamma, cfg.tau)

        # REAL novelty at φ(s') → the intrinsic reward (this is what V_int learns).
        phi_cached, nov = _phi_and_novelty(icm, rndphi, rollout, device)
        raw_i = nov.numpy()
        T, N = raw_i.shape
        raw_mean_pre = float(raw_i.mean())

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

        # LEARN FROM REALITY: V_int regressed to the real returns; model error never enters here.
        v_loss = value_update(value, value_opt, rollout, icm, cfg, device)

        # ICM update ALWAYS (trains φ pre-freeze; only inverse/forward heads post-freeze, so the
        # forward model the lookahead uses keeps improving). φ frozen at the held-out gate.
        icm_stats = icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)
        last_inv_acc = icm_stats["inverse_acc"]
        if not phi_frozen:
            last_holdout_inv = _eval_holdout_inv_acc(icm, holdout, device)
            trig = last_holdout_inv if cfg.freeze_metric == "holdout" else last_inv_acc
            inv_streak = inv_streak + 1 if trig >= cfg.phi_freeze_inverse_acc else 0
            if inv_streak >= cfg.phi_freeze_patience or update >= cfg.phi_freeze_max_updates:
                for p in icm.phi.parameters():
                    p.requires_grad_(False)
                icm.phi.eval()
                phi_frozen = True
                freeze_step = global_step + cfg.rollout_steps * cfg.n_envs
                print(f"[exp013_5] φ FROZEN @u{update} holdout_inv={last_holdout_inv:.3f}")

        rnd_loss = _rnd_update(rndphi, rnd_opt, phi_cached, rollout.dones, cfg, device)
        global_step += cfg.rollout_steps * cfg.n_envs

        if first_reward_step is None and bool((extrinsic > 0).any().item()):
            t_idx = int((extrinsic > 0).float().sum(dim=1).nonzero()[0].item())
            first_reward_step = (global_step - cfg.rollout_steps * cfg.n_envs) + (t_idx + 1) * cfg.n_envs
            print(f"[exp013_5] *** FIRST REWARD at ~{first_reward_step} env steps (update {update}) ***")
            if cfg.stop_on_first_reward:
                stop_now = True

        done_eps = envs.drain_completed_episodes()
        train_succ = float(np.mean([e.success for e in done_eps])) if done_eps else float("nan")

        record = {
            "step": global_step, "update": update,
            "policy_entropy": float(getattr(rollout, "policy_entropy_mean", float("nan"))),
            "value_loss": v_loss,
            "novelty_raw_mean": raw_mean_pre,
            "intrinsic_reward_norm_mean": float(norm_i.mean()),
            "intrinsic_return_std": int_ret_std.std,
            "v_int_mean": float(rollout.values.mean()), "ret_int_mean": float(rollout.returns.mean()),
            "rnd_predictor_loss": rnd_loss,
            "inverse_acc": last_inv_acc, "holdout_inv_acc": last_holdout_inv,
            "phi_frozen": bool(phi_frozen), "freeze_step": freeze_step, "norm_warming": bool(warming),
            "mean_feature_cosine": mean_feature_cosine(rollout.features, rollout.ep_starts),
            "train_success_rate": train_succ, "train_episodes": len(done_eps),
            "env_steps_to_first_reward": first_reward_step,
            "sps": global_step / max(1e-6, time.time() - t_start),
            **icm_stats,
        }
        if cfg.log_every > 0 and (update % cfg.log_every == 0 or stop_now):
            writer.write(record)
        if update % 25 == 0 or stop_now:
            print(f"[exp013_5] u{update}/{cfg.total_updates} step={global_step} frr={first_reward_step} "
                  f"nov={raw_mean_pre:.4g} r_norm={record['intrinsic_reward_norm_mean']:.4g} "
                  f"v_loss={v_loss:.4g} π_ent={record['policy_entropy']:.3f} "
                  f"inv(hold)={last_holdout_inv:.2f} frozen={phi_frozen}")

        if cfg.save_every > 0 and update % cfg.save_every == 0:
            _save_ckpt(run_dir, value, icm, rndphi, cfg, global_step)
        if stop_now:
            break

    writer.close()
    _save_ckpt(run_dir, value, icm, rndphi, cfg, global_step)
    solved = first_reward_step is not None
    result = {
        "exp_name": cfg.exp_name, "method": "lookahead_mcts", "game": cfg.game,
        "level_index": cfg.level_index, "seed": cfg.seed, "tau": cfg.tau,
        "env_steps_to_first_reward": first_reward_step, "solved": solved, "censored": not solved,
        "total_env_steps": global_step, "max_env_steps": cfg.max_env_steps,
        "phi_freeze_step": freeze_step, "holdout_inv_acc_final": last_holdout_inv,
        "wall_seconds": time.time() - t_start,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[exp013_5] DONE {cfg.exp_name}: "
          f"{'first reward @ ' + str(first_reward_step) if solved else 'CENSORED'} env steps; total={global_step}")
    return result
