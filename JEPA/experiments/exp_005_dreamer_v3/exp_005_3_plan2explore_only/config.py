"""Sub-exp D: pure Plan2Explore, no extrinsic reward.

The env-time reward is always 0; the P2E exploration actor π_e is used for
the *entire* run (no handoff to the task actor).  If even this never visits
the goal, the bottleneck on LS20 L1 is exploration capacity, not reward.
"""

from dataclasses import dataclass

from JEPA.experiments.exp_005_dreamer_v3.shared.config_base import ConfigBase


@dataclass(frozen=True)
class Config(ConfigBase):
    reward_mode: str = "p2e_only"
    # π_e acts forever (set acting_steps > max_env_steps)
    p2e_acting_steps: int = 10_000_000
    run_name: str = "exp_005_3_plan2explore_only"
