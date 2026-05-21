"""Sub-exp C: DV3 sparse extrinsic + P2E intrinsic added to TASK reward.

The task actor is trained on (r_ext + α·r_int) — gives the task actor a dense
gradient toward novel states even before it has seen the goal.  The P2E
exploration actor remains as the warm-start policy in the env.
"""

from dataclasses import dataclass

from JEPA.experiments.exp_005_dreamer_v3.shared.config_base import ConfigBase


@dataclass(frozen=True)
class Config(ConfigBase):
    reward_mode: str = "curiosity"
    # NOTE: We mix the intrinsic disagreement into the env-time reward stored
    # in the buffer, so the task actor's extrinsic-reward head sees it.  α is
    # small because ensemble variance grows unbounded early in training.
    p2e_intrinsic_weight: float = 0.1
    run_name: str = "exp_005_2_curiosity"
