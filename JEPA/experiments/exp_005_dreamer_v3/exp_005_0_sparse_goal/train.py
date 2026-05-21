"""Run sub-experiment A: canonical DV3 + sparse goal reward.

Usage:
    uv run python -m JEPA.experiments.exp_005_dreamer_v3.exp_005_0_sparse_goal.train
"""

from JEPA.experiments.exp_005_dreamer_v3.exp_005_0_sparse_goal.config import Config
from JEPA.experiments.exp_005_dreamer_v3.exp_005_0_sparse_goal.reward_shaping import make_reward_fn
from JEPA.experiments.exp_005_dreamer_v3.shared.trainer import train


def main():
    cfg = Config()
    reward_fn = make_reward_fn(cfg)
    train(cfg, reward_fn)


if __name__ == "__main__":
    main()
