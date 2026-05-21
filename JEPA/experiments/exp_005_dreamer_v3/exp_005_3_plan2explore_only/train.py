"""Run sub-experiment D: pure P2E, zero extrinsic."""

from JEPA.experiments.exp_005_dreamer_v3.exp_005_3_plan2explore_only.config import Config
from JEPA.experiments.exp_005_dreamer_v3.exp_005_3_plan2explore_only.reward_shaping import make_reward_fn
from JEPA.experiments.exp_005_dreamer_v3.shared.trainer import train


def main():
    cfg = Config()
    train(cfg, make_reward_fn(cfg))


if __name__ == "__main__":
    main()
