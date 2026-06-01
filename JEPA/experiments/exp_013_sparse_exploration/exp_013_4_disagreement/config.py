"""Config for exp_013_4 — ensemble DISAGREEMENT (Plan2Explore-style). See disagreement.py.

Reward = normalised ensemble-disagreement (variance across K forward models predicting
φ(s') from (φ(s),a)), φ = a FROZEN RANDOM encoder. Intrinsic-only (env +1 = stop signal).
No ICM, no φ-freeze logic (φ is fixed from the start). All the exp_013 stop-rule /
normalisation / non-episodic-GAE / entropy conventions carry over.
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
    exp_dir: str = "JEPA/experiments/exp_013_sparse_exploration/exp_013_4_disagreement"

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

    # PPO (single intrinsic value head, non-episodic by default)
    gamma: float = 0.95              # 0.99→0.95: curb non-episodic return inflation (value-lag fix)
    gae_lambda: float = 0.95
    intrinsic_episodic: bool = False
    clip_eps: float = 0.2
    vf_clip_eps: float = 0.2
    c_value: float = 1.0             # 0.5→1.0: faster value tracking (reduce the lag)
    c_entropy: float = 0.05            # raised (entropy-collapse fix); see probes/
    grad_clip: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4

    # Disagreement ensemble
    n_ensemble: int = 5
    ensemble_hidden: int = 256
    ensemble_lr: float = 1e-3
    ensemble_epochs: int = 1

    # Intrinsic-reward normalisation (warm-up + EMA std of returns, no centring + clip)
    int_norm_decay: float = 0.99
    norm_warmup_updates: int = 2
    int_norm_eps: float = 1e-8
    reward_clip_k: float | None = 5.0

    # Logging / eval / checkpoint cadence (updates; 0 disables, final always saved)
    log_every: int = 1
    eval_every: int = 0
    eval_episodes: int = 16
    save_every: int = 0

    @property
    def exp_name(self) -> str:
        return f"exp013_4_disagree_{self.game}_L{self.level_index + 1}_seed{self.seed}"

    @property
    def total_updates(self) -> int:
        return self.max_env_steps // (self.rollout_steps * self.n_envs)

    def smoke(self) -> "Config":
        return dataclasses.replace(
            self, n_envs=2, rollout_steps=16, max_episode_steps=40,
            max_env_steps=16 * 2 * 8, minibatches=2, epochs=1, n_ensemble=3,
            norm_warmup_updates=1, stop_on_first_reward=False, eval_every=0, save_every=0,
        )
