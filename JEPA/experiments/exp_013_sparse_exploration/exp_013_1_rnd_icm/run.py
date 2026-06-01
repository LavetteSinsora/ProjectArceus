"""exp_013_1 RND+ICM ("OCC") single-run CLI.

    uv run python -m JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.run \
        --game ls20 --level 0 --seed 0 --max-env-steps 250000

    # smoke (a few updates, < 1 min):
    uv run python -m JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.run \
        --game ls20 --level 0 --seed 0 --smoke
"""

from __future__ import annotations

import argparse

from .config import Config, GAME_N_ACTIONS
from .trainer import train


def main():
    p = argparse.ArgumentParser(description="exp_013_1 RND+ICM (OCC) run")
    p.add_argument("--game", choices=list(GAME_N_ACTIONS), default="ls20")
    p.add_argument("--level", type=int, default=0, help="0-indexed level to drop into")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-env-steps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--leak", type=float, default=None, help="predictor shrink-to-init rate μ")
    p.add_argument("--phi-mode", choices=["icm", "frozen"], default=None,
                   help="RND ruler: ICM learned features (icm) or a fixed random encoder (frozen)")
    p.add_argument("--init-phi-ckpt", type=str, default=None,
                   help="init φ from a saved run's checkpoint (cross-level transfer)")
    p.add_argument("--no-stop-on-first-reward", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cfg = Config(game=args.game, level_index=args.level, seed=args.seed)
    if args.max_env_steps is not None:
        cfg.max_env_steps = args.max_env_steps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.leak is not None:
        cfg.leak = args.leak
    if args.phi_mode is not None:
        cfg.phi_mode = args.phi_mode
    if args.init_phi_ckpt is not None:
        cfg.init_phi_ckpt = args.init_phi_ckpt
    if args.no_stop_on_first_reward:
        cfg.stop_on_first_reward = False

    result = train(cfg, smoke=args.smoke)
    print(result)


if __name__ == "__main__":
    main()
