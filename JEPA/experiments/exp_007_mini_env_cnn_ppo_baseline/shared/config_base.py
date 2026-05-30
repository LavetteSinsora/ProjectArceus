"""Shared hyperparameters. Per-variant configs override `reward_mode` and
optionally other fields."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # Experiment identity (variants override)
    exp_name: str = "exp_007_base"
    reward_mode: str = "terminal_only"   # one of REWARD_MODES in rewards.py

    # Env
    level_path: str = "mini_env/configs/level_01/simple_1_rotation.json"
    n_envs: int = 8
    seed: int = 0

    # Rollout
    rollout_steps: int = 128

    # Total training budget
    total_env_steps: int = 1_000_000

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_clip_eps: float = 0.2          # PPO value-function clip range; None disables
    c_value: float = 0.5
    c_entropy: float = 0.01
    grad_clip: float = 0.5
    epochs: int = 2
    minibatches: int = 4
    learning_rate: float = 3e-4

    # Reward shaping (only used when reward_mode != "terminal_only")
    wall_penalty: float = -0.05
    match_bonus: float = 0.1
    unmatch_penalty: float = -0.1

    # Logging / eval
    log_every: int = 1                # log per-update cheap metrics every N updates
    eval_every: int = 50              # full eval rollout every N updates
    eval_episodes: int = 32
    grad_decomp_every: int = 10
    save_every: int = 200

    # IO
    runs_dir: str = "JEPA/experiments/exp_007_mini_env_cnn_ppo_baseline/runs"

    @property
    def total_updates(self) -> int:
        per_update = self.rollout_steps * self.n_envs
        return self.total_env_steps // per_update
