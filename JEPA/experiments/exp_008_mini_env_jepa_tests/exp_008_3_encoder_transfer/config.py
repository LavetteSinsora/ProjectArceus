"""Config for exp_008_3: encoder transfer (JEPA vs PPO encoder on a new env).

Two *target* env tags, mapped to MiniLS20 level configs the encoders were
NEVER trained on:

    hard1 → mini_env/configs/level_01/hard_1_rotation.json
    hard2 → mini_env/configs/level_01/hard_2_rotation.json

Encoder sources (all trained on simple_1_rotation, reused as-is):

    jepa       → exp_008_2 jepa_runs/1rot_*/encoder_final.pt   (bare encoder_state_dict)
    ppo_early  → exp_007_0_naive_gaefix first-solve checkpoint (ActorCritic model_state_dict)
    ppo_final  → exp_007_0_naive_gaefix final.pt               (ActorCritic model_state_dict)
    scratch    → random init (zero-transfer control, always unfrozen)

See SYSTEM_CARD.md for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.config_base import (
    Config as _PPOBase,
)


# ───────────────────────────────────────────────────────────────────────────
# Target env tag <-> level path
# ───────────────────────────────────────────────────────────────────────────

ENV_TAG_TO_LEVEL: dict[str, str] = {
    "hard1": "mini_env/configs/level_01/hard_1_rotation.json",
    "hard2": "mini_env/configs/level_01/hard_2_rotation.json",
}


def level_path_for(env_tag: str) -> str:
    if env_tag not in ENV_TAG_TO_LEVEL:
        raise ValueError(
            f"unknown env_tag {env_tag!r}; expected one of {sorted(ENV_TAG_TO_LEVEL)}"
        )
    return ENV_TAG_TO_LEVEL[env_tag]


# ───────────────────────────────────────────────────────────────────────────
# Encoder sources
# ───────────────────────────────────────────────────────────────────────────

SOURCE_TAGS: tuple[str, ...] = ("jepa", "ppo_early", "ppo_final", "scratch")


def freeze_tag(freeze: bool) -> str:
    return "frozen" if freeze else "unfrozen"


# ───────────────────────────────────────────────────────────────────────────
# Directory layout
# ───────────────────────────────────────────────────────────────────────────

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
PPO_RUNS_DIR = EXP_DIR / "ppo_runs"

# JEPA encoder source: reuse 008_2's offline 1rot encoder (trained on simple_1_rotation).
EXP_008_2_DIR = EXP_DIR.parent / "exp_008_2_frozen_jepa_ppo"
EXP_008_2_JEPA_RUNS = EXP_008_2_DIR / "jepa_runs"
JEPA_SOURCE_GLOB = "1rot_*/encoder_final.pt"

# PPO encoder source: the GAE-fixed naive run is the only exp_007_0 run that
# actually solves simple_1_rotation (100% from update 50, 50-update cadence).
EXP_007_RUNS = Path("JEPA/experiments/exp_007_mini_env_cnn_ppo_baseline/runs")
PPO_SOURCE_RUN_GLOB = "exp_007_0_naive_gaefix_*"


def env_results_dir(env_tag: str) -> Path:
    return RESULTS_DIR / env_tag


# ───────────────────────────────────────────────────────────────────────────
# Run config
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class TransferPPOConfig(_PPOBase):
    """PPO on a target env with an encoder transferred from simple_1_rotation."""
    exp_name: str = "008_3_transfer"
    env_tag: str = "hard1"
    source: str = "jepa"          # one of SOURCE_TAGS
    freeze: bool = True           # frozen encoder if True, else fine-tuned by PPO
    encoder_ckpt: str = ""        # absolute path of the transferred encoder, filled by loader
    save_every: int = 100
    # Finer eval cadence than the exp_007 default (50): the fast learners solve
    # within the first 50 updates, so a 50-update eval censors them into a tie.
    # eval_every=10 → eval at env_step 10.2K, 20.5K, … resolving sub-50K solves.
    eval_every: int = 10
    # Early stop: once eval_success_rate stays >= threshold for `patience`
    # consecutive evals, the run has solved & saturated — stop and free the GPU
    # for the next cell. Deliberately only triggers at the SOLVED level; low
    # plateaus run the full budget (a frozen encoder can climb late, so a
    # low-level flat stretch is not proof it won't improve).
    early_stop: bool = True
    early_stop_threshold: float = 0.99
    early_stop_patience: int = 5
    # Budget matches 008_2 / 008_4 (capped by --updates at the CLI, default 488 ≈ 500K).
    total_env_steps: int = 1_000_000
