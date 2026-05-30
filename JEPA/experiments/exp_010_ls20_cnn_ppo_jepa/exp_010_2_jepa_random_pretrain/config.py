from dataclasses import dataclass

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.config_base import Config as _Base

_EXP_DIR = "JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain"


@dataclass
class Config(_Base):
    exp_name: str = "exp_010_2_jepa_random_pretrain"
    exp_dir: str = _EXP_DIR

    # PPO phase: no online JEPA — the encoder is only *initialised* from the
    # pretrained weights, then trained by PPO (unfrozen).
    jepa_mode: str = "none"
    freeze_encoder: bool = False
    # Resolved at runtime by train_ppo.py if left None (latest encoder_final.pt).
    init_encoder_ckpt: str | None = None
    total_env_steps: int = 3_000_000

    # Random-data collection.
    n_random_transitions: int = 500_000

    # JEPA pretraining plateau controls (read via getattr in shared.pretrain).
    pretrain_max_epochs: int = 40
    pretrain_patience: int = 4
    pretrain_min_delta: float = 0.002
