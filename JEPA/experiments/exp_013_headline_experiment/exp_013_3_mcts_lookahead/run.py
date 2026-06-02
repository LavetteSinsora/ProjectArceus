"""exp_013_5 lookahead-softmax (proposal D) single-run CLI.

    uv run python -m JEPA.experiments.exp_013_headline_experiment.exp_013_3_mcts_lookahead.run \
        --game ls20 --level 0 --seed 0 --max-env-steps 250000

    uv run python -m JEPA.experiments.exp_013_headline_experiment.exp_013_3_mcts_lookahead.run \
        --game ls20 --level 0 --seed 0 --smoke
"""

from __future__ import annotations

import argparse

from .config import Config, GAME_N_ACTIONS
from .trainer import train


def main():
    p = argparse.ArgumentParser(description="exp_013_5 lookahead-softmax (MCTS-organized) run")
    p.add_argument("--game", choices=list(GAME_N_ACTIONS), default="ls20")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau", type=float, default=None, help="softmax temperature")
    p.add_argument("--max-env-steps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--no-stop-on-first-reward", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cfg = Config(game=args.game, level_index=args.level, seed=args.seed)
    if args.tau is not None:
        cfg.tau = args.tau
    if args.max_env_steps is not None:
        cfg.max_env_steps = args.max_env_steps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.no_stop_on_first_reward:
        cfg.stop_on_first_reward = False

    result = train(cfg, smoke=args.smoke)
    print(result)


if __name__ == "__main__":
    main()
