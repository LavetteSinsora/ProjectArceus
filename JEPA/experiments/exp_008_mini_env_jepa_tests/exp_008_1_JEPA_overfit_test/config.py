"""Config for exp_008_1_JEPA_overfit_test (offline analysis, no training)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # Where the 7_4 checkpoints live.
    ckpt_sweep_dir: str = (
        "JEPA/experiments/exp_007_mini_env_cnn_ppo_baseline/runs/"
        "exp_007_4_jepa_sg_idm_novfclip_20260525_161738/checkpoints"
    )
    level_path: str = "mini_env/configs/level_01/simple_1_rotation.json"

    # Sample sizes per source.
    n_transitions_per_source: int = 50_000

    # Rollout collection.
    n_envs: int = 8
    seed_trained: int = 0
    seed_random: int = 1

    # Scoring.
    score_batch_size: int = 1024

    # Where to write CSV / JSON outputs.
    output_dir: str = (
        "JEPA/experiments/exp_008_mini_env_jepa_tests/"
        "exp_008_1_JEPA_overfit_test/results"
    )
