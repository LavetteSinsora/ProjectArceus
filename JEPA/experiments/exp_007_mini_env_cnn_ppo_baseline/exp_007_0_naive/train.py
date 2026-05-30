"""Entry point for exp_007_0_naive (terminal reward only).

Usage:
    uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_0_naive.train
    uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_0_naive.train --smoke
"""

from __future__ import annotations

import argparse

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_0_naive.config import Config
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.trainer import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="run 5 updates only")
    p.add_argument("--short", action="store_true", help="run 50 updates only")
    p.add_argument("--updates", type=int, default=None, help="override total updates")
    p.add_argument("--save-every", type=int, default=None, help="override checkpoint cadence (in updates)")
    args = p.parse_args()

    cfg = Config()
    if args.save_every is not None:
        cfg.save_every = args.save_every
    if args.smoke:
        max_updates = 5
    elif args.short:
        max_updates = 50
    else:
        max_updates = args.updates
    train(cfg, max_updates=max_updates)


if __name__ == "__main__":
    main()
