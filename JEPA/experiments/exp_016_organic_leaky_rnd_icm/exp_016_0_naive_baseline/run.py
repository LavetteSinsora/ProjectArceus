"""exp_016_0 naive leaky-RND-on-IDM baseline — single run CLI.

    uv run python -m JEPA.experiments.exp_016_organic_leaky_rnd_icm.\
exp_016_0_naive_baseline.run --game ls20 --level 0 --seed 0

    # plumbing smoke test (a few tiny updates, < 1 min):
    uv run python -m JEPA.experiments.exp_016_organic_leaky_rnd_icm.\
exp_016_0_naive_baseline.run --smoke
"""
from __future__ import annotations

import argparse

from .config import Config, GAME_N_ACTIONS
from .trainer import train


def main():
    p = argparse.ArgumentParser(description="exp_016_0 naive leaky-RND on IDM features")
    p.add_argument("--game", choices=list(GAME_N_ACTIONS), default="ls20")
    p.add_argument("--level", type=int, default=0, help="0-indexed (0 = Level 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-env-steps", type=int, default=None)
    p.add_argument("--leak", type=float, default=None, help="μ predictor shrink-to-init")
    p.add_argument("--no-reward-zscore", action="store_true")
    p.add_argument("--reward-center", action="store_true", help="also subtract running mean from reward")
    p.add_argument("--no-baseline", action="store_true", help="ablation: drop the batch-mean return baseline")
    p.add_argument("--value-head", action="store_true", help="state-dependent baseline V(s): advantage = return − V(s)")
    p.add_argument("--return-scale", action="store_true", help="divide advantage by batch std")
    p.add_argument("--idm-layernorm", action="store_true",
                   help="ablation: LayerNorm encoder features before inverse head + RND")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cfg = Config(game=args.game, level_index=args.level, seed=args.seed)
    if args.max_env_steps is not None:
        cfg.max_env_steps = args.max_env_steps
    if args.leak is not None:
        cfg.leak = args.leak
    if args.no_reward_zscore:
        cfg.reward_zscore = False
    if args.reward_center:
        cfg.reward_center = True
    if args.no_baseline:
        cfg.use_baseline = False
    if args.value_head:
        cfg.use_value_head = True
    if args.return_scale:
        cfg.return_scale_by_std = True
    if args.idm_layernorm:
        cfg.idm_layernorm = True

    print(train(cfg, smoke=args.smoke))


if __name__ == "__main__":
    main()
