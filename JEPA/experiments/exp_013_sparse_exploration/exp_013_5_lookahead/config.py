"""Config for exp_013_5 — proposal D, the B1 lookahead-softmax controller. See lookahead.py.

Actor-free: a value head V_int(φ) + ICM (φ, inverse, forward) + RND-on-φ (leaky). Acting is
`softmax(standardize(nov(φ̂'_a)+γV(φ̂'_a))/τ)`. Intrinsic-only; stop-on-first-reward.
No PPO policy/clip/entropy knobs — instead a value learner + the exploration temperature τ.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

GAME_N_ACTIONS = {"ls20": 4, "tu93": 4, "re86": 5, "g50t": 5}


@dataclass
class Config:
    game: str = "ls20"
    level_index: int = 0
    seed: int = 0
    exp_dir: str = "JEPA/experiments/exp_013_sparse_exploration/exp_013_5_lookahead"

    # Env
    n_envs: int = 16
    max_episode_steps: int = 200
    n_actions: int | None = None
    n_colors: int = 16
    frame_size: int = 64
    trunk_dim: int = 256

    # Rollout / budget / stop
    rollout_steps: int = 128
    max_env_steps: int = 250_000
    stop_on_first_reward: bool = True

    # Lookahead control
    tau: float = 0.25                # softmax temperature over standardised per-state Q (0.25 → max-prob~0.85, entropy~0.36; τ=1.0 was ~uniform/random-walk)
    gamma: float = 0.95              # intrinsic horizon (value bootstrap in Q + GAE)
    gae_lambda: float = 0.95
    intrinsic_episodic: bool = False

    # Value learner V_int(φ)
    value_hidden: int = 256
    value_lr: float = 3e-4
    value_epochs: int = 4
    vf_clip_eps: float | None = 0.2
    grad_clip: float = 0.5
    minibatches: int = 4

    # ICM (φ + inverse + forward; forward = the lookahead model). Trained, frozen after warm-up.
    beta: float = 0.2
    icm_lr: float = 1e-3
    icm_hidden: int = 256
    icm_epochs: int = 1
    phi_freeze_inverse_acc: float = 0.70  # lowered 0.90→0.70 (holdout maxes ~0.76; adaptive freeze)
    phi_freeze_patience: int = 3
    phi_freeze_max_updates: int = 100
    freeze_metric: str = "holdout"
    holdout_size: int = 2000

    # RND-on-φ novelty + leak
    rnd_feature_dim: int = 256
    rnd_hidden: int = 256
    rnd_lr: float = 1e-4
    rnd_epochs: int = 1
    leak: float = 0.01

    # Intrinsic-reward normalisation
    int_norm_decay: float = 0.99
    norm_warmup_updates: int = 2
    int_norm_eps: float = 1e-8
    reward_clip_k: float | None = 5.0

    # Logging / cadence (0 disables; final ckpt always saved)
    log_every: int = 1
    eval_every: int = 0
    save_every: int = 0

    @property
    def exp_name(self) -> str:
        return f"exp013_5_lookahead_{self.game}_L{self.level_index + 1}_seed{self.seed}"

    @property
    def total_updates(self) -> int:
        return self.max_env_steps // (self.rollout_steps * self.n_envs)

    def smoke(self) -> "Config":
        return dataclasses.replace(
            self, n_envs=2, rollout_steps=16, max_episode_steps=40,
            max_env_steps=16 * 2 * 8, minibatches=2, value_epochs=1,
            norm_warmup_updates=1, phi_freeze_max_updates=2, phi_freeze_patience=1,
            stop_on_first_reward=False, save_every=0, holdout_size=64,
        )
