"""exp_013_2 — ADDITIVE RND+ICM training loop. See config.py / SYSTEM_CARD.

reward_t = w_icm · norm(ICM_forward_error_t) + (1 − w_icm) · norm(RND_on_φ_error_t)

Both raw signals are computed BEFORE the ICM/RND updates (on the current models),
each passed through its OWN normaliser (raw-clip → RewardForwardFilter → EMA std,
no centring) so they sit on a common ~unit scale, then mixed by w_icm. Intrinsic-
only (the env +1 is the stop signal, never a reward).

Reuses the audited exp_013_1 helpers. ONE deliberate difference from exp_013_1:
the ICM update keeps running AFTER φ-freeze (φ is frozen for RND's stationary
ruler, but the forward/inverse HEADS keep training — Adam skips frozen-φ params —
so the ICM-forward-error reward stays a meaningful, decaying curiosity signal).
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
from JEPA.experiments.exp_011_ls20_icm.shared.icm import (
    ICMModule, icm_update_from_rollout, intrinsic_raw_error,
)
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter

from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.rnd_phi import RNDPhi
from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.trainer import (
    _EMAStd, _phi_and_novelty, _rnd_update, _gae_nonepisodic, _gae_episodic,
    _collect_holdout, _eval_holdout_inv_acc, _save_ckpt,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


class _SignalNorm:
    """Per-signal normaliser: raw-clip (k× running mean) → RewardForwardFilter →
    EMA std of returns; divide (no centring). One per intrinsic signal so the two
    are on a common ~unit scale before the convex combination."""

    def __init__(self, gamma, decay, clip_k, eps):
        self.rff = RewardForwardFilter(gamma)
        self.ema = _EMAStd(decay)
        self.clip_k = clip_k
        self.eps = eps
        self.raw_mean_ema: float | None = None

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        if self.clip_k is not None:
            m = float(raw.mean())
            self.raw_mean_ema = m if self.raw_mean_ema is None else 0.99 * self.raw_mean_ema + 0.01 * m
            if self.raw_mean_ema > 0:
                raw = np.minimum(raw, self.clip_k * self.raw_mean_ema)
        T = raw.shape[0]
        rems = np.stack([self.rff.update(raw[t]) for t in range(T)])
        self.ema.update(rems)
        return raw / (self.ema.std + self.eps)


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

    print(f"[exp013_2] {cfg.exp_name}  device={device}  n_actions={cfg.n_actions}  "
          f"cap={cfg.max_env_steps}  w_icm={cfg.w_icm}  leak={cfg.leak}")

    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    icm = ICMModule(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                    frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim,
                    hidden=cfg.icm_hidden).to(device)
    rndphi = RNDPhi(dim=cfg.trunk_dim, hidden=cfg.rnd_hidden, out=cfg.rnd_feature_dim,
                    leak=cfg.leak).to(device)

    ppo_opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    icm_opt = torch.optim.Adam(icm.parameters(), lr=cfg.icm_lr)
    rnd_opt = torch.optim.Adam(rndphi.predictor.parameters(), lr=cfg.rnd_lr)
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps,
                        c_value=cfg.c_value, c_entropy=cfg.c_entropy,
                        grad_clip=cfg.grad_clip, epochs=cfg.epochs, minibatches=cfg.minibatches)

    # TWO independent normalisers — one per intrinsic signal (common scale → clean w).
    norm_icm = _SignalNorm(cfg.gamma, cfg.int_norm_decay, cfg.reward_clip_k, cfg.int_norm_eps)
    norm_rnd = _SignalNorm(cfg.gamma, cfg.int_norm_decay, cfg.reward_clip_k, cfg.int_norm_eps)

    holdout = _collect_holdout(cfg.game, cfg.level_index, cfg.seed, cfg.holdout_size, device)
    print(f"[exp013_2]   freeze_metric={cfg.freeze_metric}  holdout={holdout[0].shape[0]}  "
          f"reward_clip_k={cfg.reward_clip_k}")

    phi_frozen = False
    inv_streak = 0
    freeze_step: int | None = None
    last_inv_acc = float("nan")
    last_holdout_inv = float("nan")
    global_step = 0
    first_reward_step: int | None = None
    t_start = time.time()
    stop_now = False
    w = cfg.w_icm

    for update in range(1, cfg.total_updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)

        # Both raw signals on the CURRENT models (before the ICM/RND updates).
        phi_cached, rnd_nov = _phi_and_novelty(icm, rndphi, rollout, device)   # RND-on-φ (T,N)
        rnd_raw = rnd_nov.numpy()
        icm_raw_t, _m = intrinsic_raw_error(icm, rollout, device)              # ICM fwd err (T,N), done-zeroed
        icm_raw = icm_raw_t.numpy()
        T, N = rnd_raw.shape

        warming = update <= cfg.norm_warmup_updates
        if warming:
            r = np.zeros_like(rnd_raw)
            n_icm = n_rnd = r
        else:
            n_icm = norm_icm(icm_raw)          # each ~unit scale
            n_rnd = norm_rnd(rnd_raw)
            r = w * n_icm + (1.0 - w) * n_rnd

        extrinsic = rollout.rewards.clone()                  # +1 = stop signal only
        rollout.rewards = torch.from_numpy(r.astype(np.float32))
        if cfg.intrinsic_episodic:
            _gae_episodic(rollout, cfg.gamma, cfg.gae_lambda)
        else:
            _gae_nonepisodic(rollout, cfg.gamma, cfg.gae_lambda)
        ustats = ppo_update(model, ppo_opt, rollout, ppo_cfg, device)

        # ICM update ALWAYS (trains φ pre-freeze; only the heads post-freeze, since
        # frozen-φ params get no grad) → the ICM-forward reward stays a live signal.
        icm_stats = icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)
        last_inv_acc = icm_stats["inverse_acc"]
        if not phi_frozen:
            last_holdout_inv = _eval_holdout_inv_acc(icm, holdout, device)
            trig_inv = last_holdout_inv if cfg.freeze_metric == "holdout" else last_inv_acc
            inv_streak = inv_streak + 1 if trig_inv >= cfg.phi_freeze_inverse_acc else 0
            hit_thresh = inv_streak >= cfg.phi_freeze_patience
            hit_fallback = update >= cfg.phi_freeze_max_updates
            if hit_thresh or hit_fallback:
                for p in icm.phi.parameters():
                    p.requires_grad_(False)
                icm.phi.eval()
                phi_frozen = True
                freeze_step = global_step + cfg.rollout_steps * cfg.n_envs
                reason = f"{cfg.freeze_metric} inv_acc plateau" if hit_thresh else "MAX-UPDATES FALLBACK"
                print(f"[exp013_2] φ FROZEN ({reason}) @u{update} "
                      f"holdout_inv={last_holdout_inv:.3f} onpolicy_inv={last_inv_acc:.3f}")
                if hit_fallback and not hit_thresh and last_holdout_inv < cfg.phi_freeze_inverse_acc:
                    print(f"[exp013_2] WARNING: φ frozen by FALLBACK with held-out inv_acc="
                          f"{last_holdout_inv:.3f} < {cfg.phi_freeze_inverse_acc} — φ NOT controllable; "
                          f"RND ruler near-random (SYSTEM_CARD §9).")

        rnd_loss = _rnd_update(rndphi, rnd_opt, phi_cached, rollout.dones, cfg, device)
        global_step += cfg.rollout_steps * cfg.n_envs

        if first_reward_step is None and bool((extrinsic > 0).any().item()):
            t_idx = int((extrinsic > 0).float().sum(dim=1).nonzero()[0].item())
            first_reward_step = (global_step - cfg.rollout_steps * cfg.n_envs) + (t_idx + 1) * cfg.n_envs
            print(f"[exp013_2] *** FIRST REWARD at ~{first_reward_step} env steps (update {update}) ***")
            if cfg.stop_on_first_reward:
                stop_now = True

        done_eps = envs.drain_completed_episodes()
        train_succ = float(np.mean([e.success for e in done_eps])) if done_eps else float("nan")

        record = {
            "step": global_step, "update": update,
            "policy_loss": ustats.policy_loss, "value_loss": ustats.value_loss,
            "policy_entropy": ustats.entropy, "approx_kl": ustats.approx_kl,
            "clipfrac": ustats.clipfrac, "grad_norm_total": ustats.grad_norm_total,
            "icm_raw_mean": float(icm_raw.mean()), "rnd_raw_mean": float(rnd_raw.mean()),
            "icm_norm_mean": float(np.mean(n_icm)), "rnd_norm_mean": float(np.mean(n_rnd)),
            "reward_mean": float(r.mean()), "w_icm": w,
            "icm_ret_std": norm_icm.ema.std, "rnd_ret_std": norm_rnd.ema.std,
            "v_int_mean": float(rollout.values.mean()), "ret_int_mean": float(rollout.returns.mean()),
            "rnd_predictor_loss": rnd_loss,
            "inverse_acc": last_inv_acc, "holdout_inv_acc": last_holdout_inv,
            "phi_frozen": bool(phi_frozen), "freeze_step": freeze_step, "norm_warming": bool(warming),
            "mean_feature_cosine": mean_feature_cosine(rollout.features, rollout.ep_starts),
            "train_success_rate": train_succ, "train_episodes": len(done_eps),
            "env_steps_to_first_reward": first_reward_step,
            "sps": global_step / max(1e-6, time.time() - t_start),
        }
        if cfg.log_every > 0 and (update % cfg.log_every == 0 or stop_now):
            writer.write(record)
        if update % 25 == 0 or stop_now:
            print(f"[exp013_2] u{update}/{cfg.total_updates} step={global_step} frr={first_reward_step} "
                  f"icm_n={record['icm_norm_mean']:.3g} rnd_n={record['rnd_norm_mean']:.3g} "
                  f"inv(onpol/hold)={last_inv_acc:.2f}/{last_holdout_inv:.2f} "
                  f"frozen={phi_frozen} ent={ustats.entropy:.3f}")

        if cfg.save_every > 0 and update % cfg.save_every == 0:
            _save_ckpt(run_dir, model, icm, rndphi, cfg, global_step, phi_frozen)
        if stop_now:
            break

    writer.close()
    _save_ckpt(run_dir, model, icm, rndphi, cfg, global_step, phi_frozen)
    solved = first_reward_step is not None
    result = {
        "exp_name": cfg.exp_name, "method": "rnd_icm_additive", "game": cfg.game,
        "level_index": cfg.level_index, "seed": cfg.seed, "w_icm": w,
        "env_steps_to_first_reward": first_reward_step, "solved": solved, "censored": not solved,
        "total_env_steps": global_step, "max_env_steps": cfg.max_env_steps,
        "phi_freeze_step": freeze_step, "freeze_metric": cfg.freeze_metric,
        "holdout_inv_acc_final": last_holdout_inv, "onpolicy_inv_acc_final": last_inv_acc,
        "wall_seconds": time.time() - t_start,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[exp013_2] DONE {cfg.exp_name}: "
          f"{'first reward @ ' + str(first_reward_step) if solved else 'CENSORED'} "
          f"env steps; total={global_step}")
    return result
