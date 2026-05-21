"""Hyperparameter configuration for the claude_automate RL framework.

Every field is game-agnostic. Nothing here encodes LS20-specific knowledge:
`game_id` simply selects which ARC-AGI bundle to load, and all reward weights
are generic (completion / step / stuck / novelty).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Environment ──────────────────────────────────────────────────────────
    game_id: str = "ls20-9607627b"
    level_index: int = 0                  # 0-indexed level each episode starts on
    max_episode_steps: int = 200          # hard cap; real episodes end ~130

    # ── Frame representation ─────────────────────────────────────────────────
    n_colors: int = 16                    # ARC palette size → one-hot channels
    frame_size: int = 64

    # ── Reward weights (all generalizable — see README) ──────────────────────
    w_complete: float = 20.0              # terminal bonus on level_completed
    w_step: float = 0.01                  # per-action time penalty
    w_stuck: float = 0.05                 # penalty when masked frame unchanged
    w_novel: float = 0.3                  # GLOBAL count-based novelty scale
    w_novel_episodic: float = 0.05        # EPISODIC novelty scale (resets/episode)
    novelty_clip: float = 1.0             # cap on a single-step novelty term

    # ── Exploration counter ──────────────────────────────────────────────────
    count_mode: str = "exact"             # "exact" (discrete obs) | "simhash"
    hash_bits: int = 28                   # k-bit SimHash code length
    hash_seed: int = 0

    # ── PPO ──────────────────────────────────────────────────────────────────
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.04
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    ppo_epochs: int = 4
    minibatch_size: int = 256
    rollout_episodes: int = 8             # full episodes collected per update

    # ── Network ──────────────────────────────────────────────────────────────
    hidden_dim: int = 512

    # ── Training loop ────────────────────────────────────────────────────────
    total_env_steps: int = 600_000
    eval_every_updates: int = 20
    eval_episodes: int = 20
    checkpoint_every_updates: int = 50
    seed: int = 0
    device: str = "auto"                  # auto | cpu | cuda | mps

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
