"""Config for exp_008_4: pretrained JEPA + unfrozen PPO.

Reuses the env-tag map and PPO base config from exp_008_2. Only the
warm-start treatment differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.config_base import (
    Config as _PPOBase,
)
from JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.config import (
    ENV_TAG_TO_LEVEL,
    level_path_for,
)


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
PPO_RUNS_DIR = EXP_DIR / "ppo_runs"

# Source of the pretrained JEPA encoders we'll warm-start from.
EXP_008_2_DIR = EXP_DIR.parent / "exp_008_2_frozen_jepa_ppo"
EXP_008_2_JEPA_RUNS = EXP_008_2_DIR / "jepa_runs"


@dataclass
class UnfrozenPPOConfig(_PPOBase):
    """PPO with encoder initialised from a JEPA checkpoint but *not* frozen."""
    exp_name: str = "exp_008_4_pretrained_jepa_unfrozen_ppo"
    env_tag: str = "1rot"
    encoder_ckpt: str = ""            # absolute path, filled by CLI / loader
    save_every: int = 100
    total_env_steps: int = 1_000_000  # actual budget capped by --updates flag
