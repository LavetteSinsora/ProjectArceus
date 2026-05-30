"""Main training loop. Called by per-variant train.py entry points."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config_base import Config
from .device import pick_device
from .model import ActorCritic
from .vec_env import VecMiniEnv
from .rewards import make_shaping_fn
from .rollout import collect_rollout, compute_gae
from .ppo import PPOConfig, ppo_update, grad_norm_decomp
from .metrics import mean_feature_cosine, run_eval_episodes, summarise_completed


def _make_run_dir(cfg: Config) -> Path:
    base = Path(cfg.runs_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run = base / f"{cfg.exp_name}_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    return run


def _log_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def train(cfg: Config, max_updates: int | None = None) -> Path:
    """Run the full training loop. Returns the run directory.

    `max_updates`, if provided, overrides cfg.total_updates — handy for
    smoke tests and short validations.
    """
    device = pick_device()
    print(f"[exp_007] device={device}  exp={cfg.exp_name}  mode={cfg.reward_mode}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = _make_run_dir(cfg)
    print(f"[exp_007] run_dir={run_dir}")

    # Persist config so eval/inspect can read it back.
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Build env, model, optimizer.
    envs = VecMiniEnv(cfg.level_path, n_envs=cfg.n_envs, seed=cfg.seed)
    envs.reset_all()

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    shape_fn = make_shaping_fn(
        cfg.reward_mode,
        wall_penalty=cfg.wall_penalty,
        match_bonus=cfg.match_bonus,
        unmatch_penalty=cfg.unmatch_penalty,
    )
    ppo_cfg = PPOConfig(
        clip_eps=cfg.clip_eps,
        vf_clip_eps=cfg.vf_clip_eps,
        c_value=cfg.c_value,
        c_entropy=cfg.c_entropy,
        grad_clip=cfg.grad_clip,
        epochs=cfg.epochs,
        minibatches=cfg.minibatches,
    )

    log_path = run_dir / "metrics.jsonl"

    total_updates = max_updates if max_updates is not None else cfg.total_updates
    print(f"[exp_007] total_updates={total_updates}  rollout_steps={cfg.rollout_steps}  n_envs={cfg.n_envs}")

    env_step = 0
    t0 = time.time()

    for update in range(1, total_updates + 1):
        # --- Collect rollout (with reward shaping applied inline) ---
        rollout = collect_rollout(envs, model, device=device,
                                   T=cfg.rollout_steps, shape_fn=shape_fn)
        env_step += cfg.rollout_steps * cfg.n_envs

        # --- Drain completed episodes (for train-side success rate) ---
        completed = envs.drain_completed_episodes()
        train_summary = summarise_completed(completed)

        # --- GAE ---
        rollout = compute_gae(rollout, gamma=cfg.gamma, lam=cfg.gae_lambda)

        # --- Cheap per-update metric: feature cosine ---
        feat_cos = mean_feature_cosine(rollout)

        # --- PPO update ---
        stats = ppo_update(model, optimizer, rollout, ppo_cfg, device=device)

        # --- Build per-update log record ---
        rec = {
            "update": update,
            "env_step": env_step,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy": stats.entropy,
            "approx_kl": stats.approx_kl,
            "clipfrac": stats.clipfrac,
            "grad_norm_total": stats.grad_norm_total,
            "mean_feature_cosine": feat_cos,
            "wall_clock_s": time.time() - t0,
        }
        rec.update(train_summary)

        # --- Periodic gradient decomposition ---
        if update % cfg.grad_decomp_every == 0:
            try:
                # collect a small fresh rollout JUST for decomp to avoid
                # disturbing graph state from the update we just did.
                tmp = collect_rollout(envs, model, device=device, T=32,
                                        shape_fn=shape_fn)
                tmp = compute_gae(tmp, gamma=cfg.gamma, lam=cfg.gae_lambda)
                gn = grad_norm_decomp(model, tmp, ppo_cfg, device=device, n_samples=128)
                rec.update(gn)
            except Exception as e:
                rec["grad_decomp_error"] = repr(e)

        # --- Periodic eval ---
        if update % cfg.eval_every == 0 or update == 1:
            eval_metrics = run_eval_episodes(cfg.level_path, model, device=device,
                                              n_episodes=cfg.eval_episodes)
            rec.update(eval_metrics)
            print(f"[exp_007] update={update:5d} env_step={env_step:9d} "
                  f"eval_success={eval_metrics['eval_success_rate']:.2f} "
                  f"entropy={stats.entropy:.3f} feat_cos={feat_cos:.3f}")

        _log_jsonl(log_path, rec)

        # --- Checkpoint ---
        if update % cfg.save_every == 0 or update == total_updates:
            ckpt_path = run_dir / "checkpoints" / f"update_{update:06d}.pt"
            torch.save({
                "update": update,
                "env_step": env_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
            }, ckpt_path)

    # Final checkpoint alias.
    final_path = run_dir / "checkpoints" / "final.pt"
    torch.save({
        "update": total_updates,
        "env_step": env_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
    }, final_path)

    print(f"[exp_007] done. run_dir={run_dir}")
    return run_dir
