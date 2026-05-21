"""Sub-exp A: canonical Dreamer V3 on LS20 Level 1.

Reward: 1.0 on level_completed, 0.0 otherwise.  No shaping.
Exploration: P2E actor for the first 100K env steps, then task actor.
"""

from dataclasses import dataclass

from JEPA.experiments.exp_005_dreamer_v3.shared.config_base import ConfigBase


@dataclass(frozen=True)
class Config(ConfigBase):
    reward_mode: str = "sparse"
    p2e_intrinsic_weight: float = 0.0
    run_name: str = "exp_005_0_sparse_goal"
