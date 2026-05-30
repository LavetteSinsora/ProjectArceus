from dataclasses import dataclass

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    exp_name: str = "exp_007_0_naive_gaefix"   # post GAE off-by-one fix; see shared/rollout.py
    reward_mode: str = "terminal_only"
