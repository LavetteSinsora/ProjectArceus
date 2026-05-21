"""Sub-exp B: DV3 + sparse goal + −0.01/step penalty.

Tests whether DV3's percentile return scaling tolerates a small dense
penalty without destabilising the critic.
"""

from dataclasses import dataclass

from JEPA.experiments.exp_005_dreamer_v3.shared.config_base import ConfigBase


@dataclass(frozen=True)
class Config(ConfigBase):
    reward_mode: str = "step_penalty"
    step_penalty: float = 0.01
    run_name: str = "exp_005_1_step_penalty"
