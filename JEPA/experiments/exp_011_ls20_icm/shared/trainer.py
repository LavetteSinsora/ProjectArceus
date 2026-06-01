"""ICM + PPO training loop for exp_011_0.

This is the exp_010 `trainer.train` loop with exactly one change: between
collecting a rollout and computing GAE, we add the ICM intrinsic reward to the
(terminal-only) extrinsic reward, then train the ICM on the same rollout with
its own optimiser. PPO itself — model, optimiser, hyperparameters — is the
exp_010 recipe, untouched.

Two optimisers (SYSTEM_CARD §5):
    ppo_opt  = Adam(actor-critic params,   lr=cfg.learning_rate = 3e-4)
    icm_opt  = Adam(ICM phi+inv+fwd params, lr=cfg.icm_lr        = 1e-3)

The intrinsic-reward scale eta is auto-calibrated once on the first rollout so
the mean per-step intrinsic reward ~= cfg.intrinsic_target, then frozen and
logged (SYSTEM_CARD §4.5). Set cfg.eta to a float to skip calibration.

Artifacts are the exp_010 dashboard layout:
    <exp_dir>/checkpoints/step_<env_step>.pt
    <exp_dir>/runs/<run_name>/metrics.jsonl  +  config.json
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Everything except the ICM is imported unchanged from exp_010.
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
from .ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.evaluator import evaluate
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.metrics import (
    MetricsWriter, mean_feature_cosine, feature_health,
)

from .icm import ICMModule, intrinsic_raw_error, icm_update_from_rollout


def _repo_root() -> Path:
    # shared -> exp_011_.. -> experiments -> JEPA -> Code Repo
    return Path(__file__).resolve().parents[4]


def save_checkpoint(cfg, model, icm, eta, global_step, update, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{global_step:08d}.pt"
    torch.save({
        "step": int(global_step),
        "update": int(update),
        "config": dataclasses.asdict(cfg),
        "model": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "icm": icm.state_dict(),
        "eta": float(eta) if eta is not None else None,
    }, path)
    return path


def train(cfg, smoke: bool = False):
    if smoke:
        cfg = cfg.smoke()
    device = get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    exp_dir = _repo_root() / cfg.exp_dir
    run_name = f"{cfg.exp_name}_seed{cfg.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = exp_dir / "runs" / run_name
    # Per-run checkpoint dir so parallel multi-seed runs never overwrite each
    # other (the headline metric needs several seeds — SYSTEM_CARD §10).
    ckpt_dir = exp_dir / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    writer = MetricsWriter(run_dir)

    print(f"[exp011] device={device}  run={run_name}")
    print(f"[exp011] env={cfg.env_name} level_index={cfg.level_index}  "
          f"ICM beta={cfg.beta} icm_lr={cfg.icm_lr} "
          f"eta={'auto' if cfg.eta is None else cfg.eta}  "
          f"updates={cfg.total_updates}  steps/update={cfg.rollout_steps*cfg.n_envs}")

    envs = VecLS20EnvLevel(cfg.env_name, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=cfg.seed,
                           level_index=cfg.level_index)
    eval_envs = VecLS20EnvLevel(cfg.env_name, n_envs=cfg.n_envs,
                                max_episode_steps=cfg.max_episode_steps, seed=cfg.seed + 777,
                                level_index=cfg.level_index)

    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    icm = ICMModule(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                    frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim,
                    hidden=cfg.icm_hidden).to(device)

    ppo_opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    icm_opt = torch.optim.Adam(icm.parameters(), lr=cfg.icm_lr)
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps,
                        c_value=cfg.c_value, c_entropy=cfg.c_entropy,
                        grad_clip=cfg.grad_clip, epochs=cfg.epochs,
                        minibatches=cfg.minibatches)

    eta = cfg.eta                      # None until calibrated
    first_reward_step = None           # headline metric (SYSTEM_CARD §6)
    best_success = -1.0                # plateau early-stop bookkeeping
    evals_without_improve = 0
    eval_count = 0
    global_step = 0
    t_start = time.time()
    for update in range(1, cfg.total_updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)
        global_step += cfg.rollout_steps * cfg.n_envs

        # ── intrinsic reward (added to the extrinsic terminal reward) ────────
        extrinsic = rollout.rewards.clone()                 # keep the pure r^e
        raw, mean_raw = intrinsic_raw_error(icm, rollout, device)
        if eta is None:                                     # one-time calibration
            eta = 2.0 * cfg.intrinsic_target / (mean_raw + 1e-8)
            print(f"[exp011] calibrated eta={eta:.4g} "
                  f"(mean forward error={mean_raw:.4g}, target r^i={cfg.intrinsic_target})")
        intrinsic = 0.5 * eta * raw
        rollout.rewards = extrinsic + intrinsic             # r = r^e + r^i

        compute_gae(rollout, cfg.gamma, cfg.gae_lambda)

        # ── PPO (untouched) then ICM (own optimiser) ─────────────────────────
        ustats = ppo_update(model, ppo_opt, rollout, ppo_cfg, device)
        icm_stats = icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)

        # ── headline: env steps to the FIRST extrinsic reward ────────────────
        if first_reward_step is None and bool((extrinsic > 0).any().item()):
            # locate the step index within this rollout for a precise count
            t_idx = int((extrinsic > 0).float().sum(dim=1).nonzero()[0].item())
            first_reward_step = (global_step - cfg.rollout_steps * cfg.n_envs
                                 + (t_idx + 1) * cfg.n_envs)
            print(f"[exp011] *** FIRST EXTRINSIC REWARD at ~{first_reward_step} env steps "
                  f"(update {update}) ***")

        done_eps = envs.drain_completed_episodes()
        train_succ = (float(np.mean([e.success for e in done_eps]))
                      if done_eps else float("nan"))

        record = {
            "step": global_step,
            "update": update,
            "policy_loss": ustats.policy_loss,
            "value_loss": ustats.value_loss,
            "policy_entropy": ustats.entropy,
            "approx_kl": ustats.approx_kl,
            "clipfrac": ustats.clipfrac,
            "grad_norm_total": ustats.grad_norm_total,
            "mean_feature_cosine": mean_feature_cosine(rollout.features, rollout.ep_starts),
            "train_success_rate": train_succ,
            "train_episodes": len(done_eps),
            "sps": global_step / max(1e-6, time.time() - t_start),
            # ICM-specific (SYSTEM_CARD §6)
            "eta": float(eta),
            "intrinsic_reward_mean": float(intrinsic.mean().item()),
            "intrinsic_reward_std": float(intrinsic.std().item()),
            "extrinsic_reward_sum": float(extrinsic.sum().item()),
            "first_reward_step": first_reward_step,
            **icm_stats,
        }

        is_eval = (update % cfg.eval_every == 0 or update == cfg.total_updates)
        if is_eval:
            record.update(evaluate(model, eval_envs, device, cfg.eval_episodes))
            record.update(feature_health(rollout.features))

        if update % cfg.log_every == 0:
            writer.write(record)

        # ── plateau early stop on eval success rate ──────────────────────────
        stop_now = False
        if is_eval:
            eval_count += 1
            esr = record.get("success_rate", float("nan"))
            esr_val = esr if (esr == esr) else 0.0     # NaN (no successes) -> 0
            improved = esr_val > best_success + cfg.early_stop_min_delta
            if improved:
                best_success = esr_val
                evals_without_improve = 0
            else:
                evals_without_improve += 1
            print(f"[exp011] update {update}/{cfg.total_updates} step={global_step} "
                  f"eval_success_rate={esr_val:.3f} best={max(best_success,0.0):.3f} "
                  f"stale={evals_without_improve}/{cfg.early_stop_patience} "
                  f"r^i_mean={record['intrinsic_reward_mean']:.4g} "
                  f"inv_acc={icm_stats['inverse_acc']:.3f} "
                  f"first_reward_step={first_reward_step}")
            if (cfg.early_stop_enabled and eval_count >= cfg.early_stop_warmup_evals
                    and evals_without_improve >= cfg.early_stop_patience):
                stop_now = True

        if update % cfg.save_every == 0 or update == cfg.total_updates or stop_now:
            p = save_checkpoint(cfg, model, icm, eta, global_step, update, ckpt_dir)
            print(f"[exp011]   saved {p.name}")

        if stop_now:
            print(f"[exp011] EARLY STOP: eval success_rate plateaued at "
                  f"{max(best_success,0.0):.3f} ({cfg.early_stop_patience} evals "
                  f"without > {cfg.early_stop_min_delta} improvement) at step={global_step}")
            break

    writer.close()
    print(f"[exp011] done. first_reward_step={first_reward_step}. checkpoints in {ckpt_dir}")
    return ckpt_dir
