"""Train exp_011_2 (ICM + PPO on the HARDER LS20 Level 2).

    uv run python -m JEPA.experiments.exp_011_ls20_icm.exp_011_2_icm_ls20_l2.train --seed 0
    uv run python -m JEPA.experiments.exp_011_ls20_icm.exp_011_2_icm_ls20_l2.train --smoke
"""

import argparse
import dataclasses

from JEPA.experiments.exp_011_ls20_icm.shared.trainer import train
from .config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eta", type=float, default=None,
                    help="fixed intrinsic scale; omit to auto-calibrate")
    ap.add_argument("--total-env-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.eta is not None:
        overrides["eta"] = args.eta
    if args.total_env_steps is not None:
        overrides["total_env_steps"] = args.total_env_steps
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
