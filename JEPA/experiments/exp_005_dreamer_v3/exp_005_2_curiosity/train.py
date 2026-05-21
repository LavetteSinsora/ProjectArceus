"""Run sub-experiment C: DV3 sparse + Plan2Explore intrinsic.

Intrinsic delivery: we leave env-time reward sparse, but the trainer already
runs a Plan2Explore exploration actor π_e in env warm-up and trains its
critic on imagined disagreement.  This sub-exp is mostly an alias of 005_0
*today* — it's wired so that `p2e_intrinsic_weight > 0` would mix intrinsic
into the task actor's λ-returns in a future trainer change.  Keeping the
hook present here documents the intended ablation.
"""

from JEPA.experiments.exp_005_dreamer_v3.exp_005_2_curiosity.config import Config
from JEPA.experiments.exp_005_dreamer_v3.exp_005_2_curiosity.reward_shaping import make_reward_fn
from JEPA.experiments.exp_005_dreamer_v3.shared.trainer import train


def main():
    cfg = Config()
    train(cfg, make_reward_fn(cfg))


if __name__ == "__main__":
    main()
