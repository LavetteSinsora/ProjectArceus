"""Joint-training PPO+CNN baseline (the exp_007_0_naive recipe), retargeted
at the 2-rot mini-env. The 1-rot baseline is reused from the existing
exp_007_0_naive runs and is **not** re-trained here.

Usage:
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_baseline --env 2rot
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_baseline --env 2rot --smoke

Everything but `level_path` matches exp_007_0_naive: same model, same loss,
same hyperparameters, same logging. Only difference is where the run lands
on disk and the env it trains in.
"""

from __future__ import annotations

import argparse

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.trainer import train as shared_train

from .config import (
    ENV_TAG_TO_LEVEL,
    JointPPOBaselineConfig,
    PPO_RUNS_DIR,
    level_path_for,
)


def train_joint_baseline(env_tag: str, seed: int = 0,
                          max_updates: int | None = None):
    cfg = JointPPOBaselineConfig(env_tag=env_tag, seed=seed)
    cfg.level_path = level_path_for(env_tag)
    cfg.exp_name = f"{cfg.exp_name}__{env_tag}"
    cfg.runs_dir = str(PPO_RUNS_DIR)
    print(f"[ppo-joint] env={env_tag}  level={cfg.level_path}  runs_dir={cfg.runs_dir}")
    return shared_train(cfg, max_updates=max_updates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=sorted(ENV_TAG_TO_LEVEL), required=True,
                   help="env tag (typically 2rot — 1rot baseline reuses exp_007_0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="5 updates")
    p.add_argument("--short", action="store_true", help="50 updates")
    p.add_argument("--updates", type=int, default=None)
    args = p.parse_args()

    if args.smoke:
        max_updates = 5
    elif args.short:
        max_updates = 50
    else:
        max_updates = args.updates

    train_joint_baseline(args.env, seed=args.seed, max_updates=max_updates)


if __name__ == "__main__":
    main()
