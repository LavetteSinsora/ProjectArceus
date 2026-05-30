from dataclasses import dataclass

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    exp_name: str = "exp_010_0_cnn_ppo"
    exp_dir: str = "JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_0_cnn_ppo_baseline"
    jepa_mode: str = "none"
    # Terminal-only reward on real LS20 is genuinely sparse, so give it room.
    total_env_steps: int = 3_000_000
