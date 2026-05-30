"""Shared configuration for exp_008_2.

Two env tags, mapped to MiniLS20 level configs:

    1rot  → mini_env/configs/level_01/simple_1_rotation.json
    2rot  → mini_env/configs/level_01/simple_2_rotation.json

The exp_007_0_naive recipe (CNN + PPO, joint) is reused for the 1-rot baseline
without re-running and is replicated for the new 2-rot baseline by overriding
`level_path` only — see train_baseline.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.config_base import (
    Config as _PPOBase,
)


# ───────────────────────────────────────────────────────────────────────────
# Env tag <-> level path
# ───────────────────────────────────────────────────────────────────────────

ENV_TAG_TO_LEVEL: dict[str, str] = {
    "1rot": "mini_env/configs/level_01/simple_1_rotation.json",
    "2rot": "mini_env/configs/level_01/simple_2_rotation.json",
}


def level_path_for(env_tag: str) -> str:
    if env_tag not in ENV_TAG_TO_LEVEL:
        raise ValueError(
            f"unknown env_tag {env_tag!r}; expected one of {sorted(ENV_TAG_TO_LEVEL)}"
        )
    return ENV_TAG_TO_LEVEL[env_tag]


# ───────────────────────────────────────────────────────────────────────────
# Directory layout
# ───────────────────────────────────────────────────────────────────────────

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
DATA_DIR = EXP_DIR / "data"
JEPA_RUNS_DIR = EXP_DIR / "jepa_runs"
PPO_RUNS_DIR = EXP_DIR / "ppo_runs"


def env_results_dir(env_tag: str) -> Path:
    return RESULTS_DIR / env_tag


# ───────────────────────────────────────────────────────────────────────────
# Sub-configs
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class CollectConfig:
    """Uniform-random data collection."""
    env_tag: str = "1rot"
    n_envs: int = 8
    n_transitions: int = 200_000   # valid (non-done-crossing) transitions
    seed: int = 0


@dataclass
class JEPATrainConfig:
    """Offline JEPA (encoder + forward predictor + IDM) on the random buffer."""
    env_tag: str = "1rot"
    batch_size: int = 256
    learning_rate: float = 3e-4
    n_epochs: int = 10
    val_split: float = 0.05
    d_action: int = 32
    predictor_hidden: int = 256
    idm_hidden: int = 256
    idm_loss_weight: float = 1.0
    grad_clip: float = 0.5
    seed: int = 0
    log_every: int = 50            # batches
    save_every_epochs: int = 1


@dataclass
class FrozenPPOConfig(_PPOBase):
    """PPO with a *frozen* encoder loaded from a JEPA-trained checkpoint."""
    exp_name: str = "exp_008_2_frozen_jepa_ppo"
    env_tag: str = "1rot"
    encoder_ckpt: str = ""          # absolute path, filled by CLI/loader
    save_every: int = 200
    # PPO budget matches exp_007_0_naive (the joint baseline) so curves line up.
    total_env_steps: int = 1_000_000


@dataclass
class JointPPOBaselineConfig(_PPOBase):
    """exp_007_0_naive recipe, retargeted at one of our env tags."""
    exp_name: str = "exp_008_2_joint_cnn_ppo"
    env_tag: str = "2rot"
    save_every: int = 200
    total_env_steps: int = 1_000_000
    reward_mode: str = "terminal_only"
