from dataclasses import dataclass

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    exp_name: str = "exp_010_1_jepa_joint_online"
    exp_dir: str = "JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_1_jepa_joint_online"
    jepa_mode: str = "online"
    # Encoder is shaped by PPO *and* JEPA every update.
    jepa_coef: float = 1.0
    idm_coef: float = 1.0
    jepa_epochs: int = 1
    total_env_steps: int = 3_000_000
