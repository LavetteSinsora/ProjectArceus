"""Transfer PPO: train a fresh policy on a NEW env on top of an encoder that
was learned on simple_1_rotation.

One parametrised driver for the whole 008_3 matrix:

    --source {jepa, ppo_early, ppo_final, scratch}
    --freeze / --no-freeze
    --env {hard1, hard2}

Frozen treatment mirrors exp_008_2 (optimiser over policy+value heads only,
freeze-leak assert); unfrozen mirrors exp_008_4 (optimiser over all params,
encoder-moved assert). `scratch` is always unfrozen (a frozen random encoder
is meaningless) and is the zero-transfer control.

Usage:
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source jepa --freeze --env hard1 --updates 488
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source scratch --no-freeze --env hard2 --smoke
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
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.diagnostics import (
    all_diagnostics,
)

from . import encoders
from .config import (
    ENV_TAG_TO_LEVEL,
    PPO_RUNS_DIR,
    SOURCE_TAGS,
    TransferPPOConfig,
    freeze_tag,
    level_path_for,
)


def _make_run_dir(cfg: TransferPPOConfig) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = (f"{cfg.exp_name}__{cfg.source}_{freeze_tag(cfg.freeze)}__"
            f"{cfg.env_tag}_s{cfg.seed}_{ts}")
    run = PPO_RUNS_DIR / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    return run


def _log_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _param_signature(module: torch.nn.Module) -> tuple[float, ...]:
    return tuple(float(p.detach().abs().sum().item()) for p in module.parameters())


def _save_ckpt(path: Path, model, optimizer, update, env_step, cfg) -> None:
    torch.save({
        "update": update,
        "env_step": env_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
    }, path)


def train_transfer_ppo(cfg: TransferPPOConfig,
                       override_ckpt: str | None = None,
                       max_updates: int | None = None) -> Path:
    if cfg.source == "scratch" and cfg.freeze:
        raise ValueError("scratch + frozen is not a valid condition "
                         "(a frozen random encoder is meaningless); use --no-freeze.")

    device = pick_device()
    print(f"[transfer] device={device}  source={cfg.source}  "
          f"freeze={cfg.freeze}  env={cfg.env_tag}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.level_path = level_path_for(cfg.env_tag)

    # ── resolve + load the transferred encoder ───────────────────────────
    enc_sd, src_path = encoders.resolve_encoder(cfg.source, override_ckpt,
                                                 map_location=device)
    model = ActorCritic().to(device)
    if enc_sd is not None:
        model.encoder.load_state_dict(enc_sd, strict=True)
        cfg.encoder_ckpt = str(src_path)
        print(f"[transfer] loaded encoder from {src_path}")
    else:
        cfg.encoder_ckpt = ""
        print("[transfer] scratch — random-init encoder (no load)")

    # ── freeze treatment ─────────────────────────────────────────────────
    if cfg.freeze:
        model.encoder.eval()
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        trainable = (list(model.policy_head.parameters())
                     + list(model.value_head.parameters()))
    else:
        for p in model.encoder.parameters():
            p.requires_grad_(True)
        trainable = list(model.parameters())
    optimizer = torch.optim.Adam(trainable, lr=cfg.learning_rate)

    run_dir = _make_run_dir(cfg)
    (run_dir / "config.json").write_text(json.dumps({
        **asdict(cfg),
        "role": f"{cfg.source}_{freeze_tag(cfg.freeze)}",
        "level_path": cfg.level_path,
        "encoder_ckpt": cfg.encoder_ckpt,
    }, indent=2))
    log_path = run_dir / "metrics.jsonl"
    print(f"[transfer] run_dir={run_dir}")

    envs = VecMiniEnv(cfg.level_path, n_envs=cfg.n_envs, seed=cfg.seed)
    envs.reset_all()

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
    print(f"[transfer] total_updates={total_updates}  rollout_steps={cfg.rollout_steps}  "
          f"n_envs={cfg.n_envs}")

    env_step = 0
    t0 = time.time()
    enc_sig_start = _param_signature(model.encoder)
    log_diag = not cfg.freeze  # collapse diagnostics only matter when encoder moves

    consec_solved = 0          # consecutive evals at/above the solved threshold
    stop_reason = "budget"
    last_update = total_updates

    for update in range(1, total_updates + 1):
        rollout = collect_rollout(envs, model, device=device,
                                  T=cfg.rollout_steps, shape_fn=shape_fn)
        env_step += cfg.rollout_steps * cfg.n_envs

        completed = envs.drain_completed_episodes()
        train_summary = summarise_completed(completed)

        rollout = compute_gae(rollout, gamma=cfg.gamma, lam=cfg.gae_lambda)
        feat_cos = mean_feature_cosine(rollout)

        diag = {}
        if log_diag:
            feat_flat = rollout.features.reshape(-1, rollout.features.shape[-1])
            diag = all_diagnostics(feat_flat)

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
        rec.update(diag)
        rec.update(train_summary)

        triggered_early_stop = False
        if update % cfg.eval_every == 0 or update == 1:
            eval_metrics = run_eval_episodes(cfg.level_path, model, device=device,
                                             n_episodes=cfg.eval_episodes)
            rec.update(eval_metrics)
            print(f"[transfer] update={update:5d} env_step={env_step:9d} "
                  f"eval={eval_metrics['eval_success_rate']:.2f} "
                  f"H={stats.entropy:.3f} cos={feat_cos:.3f}"
                  + (f" rank={diag['feat_effective_rank']:.1f}" if diag else ""))

            # ── early stop: solved & saturated ───────────────────────────
            if eval_metrics["eval_success_rate"] >= cfg.early_stop_threshold:
                consec_solved += 1
            else:
                consec_solved = 0
            if cfg.early_stop and consec_solved >= cfg.early_stop_patience:
                triggered_early_stop = True
                stop_reason = "saturated"
                rec["early_stopped"] = True

        _log_jsonl(log_path, rec)

        if update % cfg.save_every == 0 or triggered_early_stop:
            _save_ckpt(run_dir / "checkpoints" / f"update_{update:06d}.pt",
                       model, optimizer, update, env_step, cfg)

        if triggered_early_stop:
            last_update = update
            print(f"[transfer] EARLY STOP at update={update} "
                  f"(eval_success_rate>={cfg.early_stop_threshold} for "
                  f"{cfg.early_stop_patience} consecutive evals)")
            break
    else:
        last_update = total_updates

    _save_ckpt(run_dir / "checkpoints" / "final.pt",
               model, optimizer, last_update, env_step, cfg)
    (run_dir / "stop.json").write_text(json.dumps({
        "stop_reason": stop_reason,
        "final_update": last_update,
        "final_env_step": env_step,
        "total_budget_updates": total_updates,
    }, indent=2))

    # ── freeze / unfreeze integrity assert ───────────────────────────────
    enc_sig_end = _param_signature(model.encoder)
    if cfg.freeze:
        if enc_sig_end != enc_sig_start:
            raise RuntimeError("frozen encoder drifted during training — freeze leak!")
    else:
        if enc_sig_end == enc_sig_start:
            raise RuntimeError("unfrozen encoder never changed — encoder was not trained!")

    print(f"[transfer] done. run_dir={run_dir}")
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=SOURCE_TAGS, required=True)
    p.add_argument("--env", choices=sorted(ENV_TAG_TO_LEVEL), required=True)
    p.add_argument("--freeze", action=argparse.BooleanOptionalAction, default=None,
                   help="freeze the encoder (--freeze) or fine-tune it (--no-freeze)")
    p.add_argument("--encoder_ckpt", type=str, default=None,
                   help="override the auto-resolved encoder checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="5 updates only")
    p.add_argument("--short", action="store_true", help="50 updates")
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=None,
                   help="eval cadence in updates; overrides the config default")
    p.add_argument("--early_stop", action=argparse.BooleanOptionalAction, default=None,
                   help="stop once eval success saturates at the solved level")
    p.add_argument("--early_stop_threshold", type=float, default=None,
                   help="success rate that counts as 'solved' (default 0.99)")
    p.add_argument("--early_stop_patience", type=int, default=None,
                   help="consecutive solved evals required to stop (default 5)")
    args = p.parse_args()

    if args.freeze is None:
        # Default: scratch is unfrozen; everything else frozen unless told otherwise.
        freeze = args.source != "scratch"
    else:
        freeze = args.freeze

    cfg = TransferPPOConfig(env_tag=args.env, source=args.source,
                            freeze=freeze, seed=args.seed)
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.early_stop is not None:
        cfg.early_stop = args.early_stop
    if args.early_stop_threshold is not None:
        cfg.early_stop_threshold = args.early_stop_threshold
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience

    if args.smoke:
        max_updates = 5
    elif args.short:
        max_updates = 50
    else:
        max_updates = args.updates

    train_transfer_ppo(cfg, override_ckpt=args.encoder_ckpt, max_updates=max_updates)


if __name__ == "__main__":
    main()
