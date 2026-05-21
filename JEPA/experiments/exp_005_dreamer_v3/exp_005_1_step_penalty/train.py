"""Run sub-experiment B: DV3 + sparse + step penalty."""

from JEPA.experiments.exp_005_dreamer_v3.exp_005_1_step_penalty.config import Config
from JEPA.experiments.exp_005_dreamer_v3.exp_005_1_step_penalty.reward_shaping import make_reward_fn
from JEPA.experiments.exp_005_dreamer_v3.shared.trainer import train


def main():
    cfg = Config()
    train(cfg, make_reward_fn(cfg))


if __name__ == "__main__":
    main()
