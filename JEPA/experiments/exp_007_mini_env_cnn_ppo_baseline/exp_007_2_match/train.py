"""Entry point for exp_007_2_match (terminal + wall + rotation-match shaping).

Usage:
    uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_2_match.train
"""

from __future__ import annotations

import argparse

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_2_match.config import Config
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.trainer import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--short", action="store_true")
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None, help="override checkpoint cadence (in updates)")
    args = p.parse_args()

    cfg = Config()
    if args.save_every is not None:
        cfg.save_every = args.save_every
    max_updates = 5 if args.smoke else (50 if args.short else args.updates)
    train(cfg, max_updates=max_updates)


if __name__ == "__main__":
    main()
