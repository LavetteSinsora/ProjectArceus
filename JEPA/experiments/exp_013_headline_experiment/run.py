"""exp_013 single-run CLI.

One process = one (method × game × level × seed). Launch many in parallel (one
per seed / config) for the multi-seed averaging the calibration needs.

Examples
--------
    # smoke test (a few updates, < 1 min):
    uv run python -m JEPA.experiments.exp_013_headline_experiment.run \
        --method rnd --game ls20 --level 0 --seed 0 --smoke

    # a real run, stop on first reward, cap at 3M env steps:
    uv run python -m JEPA.experiments.exp_013_headline_experiment.run \
        --method icm --game ls20 --level 0 --seed 1 --max-env-steps 3000000
"""

from __future__ import annotations

import argparse

from JEPA.experiments.exp_013_headline_experiment.shared.config import Config, GAME_N_ACTIONS
from JEPA.experiments.exp_013_headline_experiment.shared.trainer import train


def build_cfg(args) -> Config:
    cfg = Config(method=args.method, game=args.game, level_index=args.level, seed=args.seed)
    if args.max_env_steps is not None:
        cfg.max_env_steps = args.max_env_steps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.max_episode_steps is not None:
        cfg.max_episode_steps = args.max_episode_steps
    if args.no_stop_on_first_reward:
        cfg.stop_on_first_reward = False
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    return cfg


def main():
    p = argparse.ArgumentParser(description="exp_013 sparse-reward exploration run")
    p.add_argument("--method", choices=["icm", "rnd"], required=True)
    p.add_argument("--game", choices=list(GAME_N_ACTIONS), default="ls20")
    p.add_argument("--level", type=int, default=0, help="0-indexed level to drop into")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-env-steps", type=int, default=None, help="hard cap / censoring point")
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None, help="updates between evals (0=off)")
    p.add_argument("--no-stop-on-first-reward", action="store_true",
                   help="run to the cap even after the first reward")
    p.add_argument("--smoke", action="store_true", help="tiny plumbing test")
    args = p.parse_args()

    cfg = build_cfg(args)
    result = train(cfg, smoke=args.smoke)
    print(result)


if __name__ == "__main__":
    main()
