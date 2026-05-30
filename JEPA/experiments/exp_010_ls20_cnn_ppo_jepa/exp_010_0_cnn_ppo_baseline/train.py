"""Train exp_010_0 (CNN + PPO baseline on real LS20).

    uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_0_cnn_ppo_baseline.train
    uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_0_cnn_ppo_baseline.train --smoke
"""

import argparse

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.trainer import train
from .config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train(Config(), smoke=args.smoke)


if __name__ == "__main__":
    main()
