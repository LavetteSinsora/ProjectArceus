"""Frozen-encoder PPO. The CNN encoder is initialised from a pretrained JEPA
checkpoint, frozen (requires_grad=False, eval mode), and PPO trains only the
policy and value heads.

Usage:
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_ppo --env 1rot
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_ppo --env 2rot --smoke

Loads the most recent encoder_final.pt from jepa_runs/<env_tag>_*/, unless
--encoder_ckpt is passed. The loop otherwise mirrors shared.trainer.train()
(same hyperparameters, same logging) but the optimizer only sees the
policy+value head parameters.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.device import pick_device
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.model import ActorCritic
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.vec_env import VecMiniEnv
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.rewards import make_shaping_fn
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.rollout import (
    collect_rollout, compute_gae,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.ppo import PPOConfig, ppo_update
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.metrics import (
    mean_feature_cosine, run_eval_episodes, summarise_completed,
)

from .config import (
    ENV_TAG_TO_LEVEL,
    FrozenPPOConfig,
    JEPA_RUNS_DIR,
    PPO_RUNS_DIR,
    level_path_for,
)


def _latest_jepa_ckpt(env_tag: str) -> Path:
    candidates = sorted(JEPA_RUNS_DIR.glob(f"{env_tag}_*/encoder_final.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no JEPA encoder found at {JEPA_RUNS_DIR}/{env_tag}_*/encoder_final.pt."
            f"  Run train_jepa.py --env {env_tag} first."
        )
    return candidates[-1]


def _make_run_dir(cfg: FrozenPPOConfig) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run = PPO_RUNS_DIR / f"{cfg.exp_name}__{cfg.env_tag}_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    return run


def _log_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _freeze_encoder(model: ActorCritic) -> None:
    """Set encoder to eval() and disable grads on every encoder param.

    `ActorCritic` here has no BatchNorm or Dropout in the encoder, but
    .eval() is still set for hygiene (matches what would be expected
    by anyone reading the code).
    """
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)


def train_frozen_ppo(cfg: FrozenPPOConfig, encoder_ckpt: Path,
                     max_updates: int | None = None) -> Path:
    device = pick_device()
    print(f"[ppo-frozen] device={device}  env={cfg.env_tag}  "
          f"encoder_ckpt={encoder_ckpt}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.level_path = level_path_for(cfg.env_tag)
    cfg.encoder_ckpt = str(encoder_ckpt)

    run_dir = _make_run_dir(cfg)
    (run_dir / "config.json").write_text(json.dumps({
        **asdict(cfg),
        "role": "frozen_jepa_ppo",
        "level_path": cfg.level_path,
    }, indent=2))
    log_path = run_dir / "metrics.jsonl"
    print(f"[ppo-frozen] run_dir={run_dir}")

    envs = VecMiniEnv(cfg.level_path, n_envs=cfg.n_envs, seed=cfg.seed)
    envs.reset_all()

    model = ActorCritic().to(device)
    # Load encoder weights from JEPA checkpoint.
    ckpt = torch.load(encoder_ckpt, map_location=device, weights_only=False)
    enc_sd = ckpt["encoder_state_dict"]
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"encoder state-dict mismatch: missing={missing} unexpected={unexpected}"
        )
    _freeze_encoder(model)

    # Sanity check: encoder params should NOT be in the optimizer.
    pp_params = (list(model.policy_head.parameters())
                 + list(model.value_head.parameters()))
    optimizer = torch.optim.Adam(pp_params, lr=cfg.learning_rate)

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

    total_updates = max_updates if max_updates is not None else cfg.total_updates
    print(f"[ppo-frozen] total_updates={total_updates}  rollout_steps={cfg.rollout_steps}  "
          f"n_envs={cfg.n_envs}")

    env_step = 0
    t0 = time.time()
    enc_param_signature = _param_signature(model.encoder)

    for update in range(1, total_updates + 1):
        rollout = collect_rollout(envs, model, device=device,
                                   T=cfg.rollout_steps, shape_fn=shape_fn)
        env_step += cfg.rollout_steps * cfg.n_envs

        completed = envs.drain_completed_episodes()
        train_summary = summarise_completed(completed)

        rollout = compute_gae(rollout, gamma=cfg.gamma, lam=cfg.gae_lambda)
        feat_cos = mean_feature_cosine(rollout)

        stats = ppo_update(model, optimizer, rollout, ppo_cfg, device=device)

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

        if update % cfg.eval_every == 0 or update == 1:
            eval_metrics = run_eval_episodes(cfg.level_path, model, device=device,
                                              n_episodes=cfg.eval_episodes)
            rec.update(eval_metrics)
            print(f"[ppo-frozen] update={update:5d} env_step={env_step:9d} "
                  f"eval_success={eval_metrics['eval_success_rate']:.2f} "
                  f"entropy={stats.entropy:.3f} feat_cos={feat_cos:.3f}")

        _log_jsonl(log_path, rec)

        if update % cfg.save_every == 0 or update == total_updates:
            ckpt_path = run_dir / "checkpoints" / f"update_{update:06d}.pt"
            torch.save({
                "update": update,
                "env_step": env_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
            }, ckpt_path)

    final_path = run_dir / "checkpoints" / "final.pt"
    torch.save({
        "update": total_updates,
        "env_step": env_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
    }, final_path)

    # Encoder must not have drifted.
    if _param_signature(model.encoder) != enc_param_signature:
        raise RuntimeError("encoder parameters drifted during training — freeze leak!")

    print(f"[ppo-frozen] done. run_dir={run_dir}")
    return run_dir


def _param_signature(module: torch.nn.Module) -> tuple[float, ...]:
    """Quick fingerprint of module parameters; used to assert freeze."""
    return tuple(float(p.detach().abs().sum().item()) for p in module.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=sorted(ENV_TAG_TO_LEVEL), required=True)
    p.add_argument("--encoder_ckpt", type=str, default=None,
                   help="path to encoder_final.pt; default = latest matching --env")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="5 updates only")
    p.add_argument("--short", action="store_true", help="50 updates")
    p.add_argument("--updates", type=int, default=None)
    args = p.parse_args()

    cfg = FrozenPPOConfig(env_tag=args.env, seed=args.seed)
    ckpt = Path(args.encoder_ckpt) if args.encoder_ckpt else _latest_jepa_ckpt(args.env)

    if args.smoke:
        max_updates = 5
    elif args.short:
        max_updates = 50
    else:
        max_updates = args.updates

    train_frozen_ppo(cfg, ckpt, max_updates=max_updates)


if __name__ == "__main__":
    main()
