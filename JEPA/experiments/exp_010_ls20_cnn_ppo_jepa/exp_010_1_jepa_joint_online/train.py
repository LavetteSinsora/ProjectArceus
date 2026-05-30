"""Train exp_010_1 (joint online JEPA + PPO on real LS20).

    uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_1_jepa_joint_online.train
    uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_1_jepa_joint_online.train --smoke
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
