"""Shared config for exp_010. Per-variant configs subclass and override.

`exp_dir` is the directory of the *sub-experiment* (e.g.
.../exp_010_0_cnn_ppo_baseline). Checkpoints are written flat to
`<exp_dir>/checkpoints/step_*.pt` and metrics to
`<exp_dir>/runs/<run>/metrics.jsonl`, which is exactly the layout the main
JEPA dashboard (port 8787) reads via /api/checkpoints and /api/training/metrics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # Identity
    exp_name: str = "exp_010_base"
    exp_dir: str = "JEPA/experiments/exp_010_ls20_cnn_ppo_jepa"

    # Env (real LS20 via JEPA.shared.env_wrapper)
    env_name: str = "ls20"
    n_envs: int = 8
    max_episode_steps: int = 200
    seed: int = 0

    # Model
    n_actions: int = 4
    n_colors: int = 16
    frame_size: int = 64
    trunk_dim: int = 256

    # Rollout / budget
    rollout_steps: int = 128            # 128 * 8 = 1024 transitions / update
    total_env_steps: int = 2_000_000

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_clip_eps: float = 0.2
    c_value: float = 0.5
    c_entropy: float = 0.01
    grad_clip: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4

    # JEPA (exp_010_1 online / exp_010_2 pretrain). Ignored when jepa_mode="none".
    jepa_mode: str = "none"             # "none" | "online"
    jepa_coef: float = 1.0
    idm_coef: float = 1.0
    jepa_epochs: int = 1                # JEPA passes over each rollout (online)
    action_emb_dim: int = 32

    # Encoder warm-start (exp_010_2 PPO phase). Path to an encoder state-dict.
    init_encoder_ckpt: str | None = None
    freeze_encoder: bool = False

    # Logging / eval / checkpoint cadence (in updates)
    log_every: int = 1
    eval_every: int = 50
    eval_episodes: int = 32
    save_every: int = 100

    # Early stopping (PPO): stop once eval success_rate stays at/above
    # `early_stop_success_rate` for `early_stop_patience` consecutive evals.
    # Set early_stop_enabled=False to always run the full budget.
    early_stop_enabled: bool = True
    early_stop_success_rate: float = 0.99
    early_stop_patience: int = 3

    @property
    def total_updates(self) -> int:
        return self.total_env_steps // (self.rollout_steps * self.n_envs)

    def smoke(self) -> "Config":
        """Return a tiny variant for plumbing tests (a few updates)."""
        import dataclasses
        return dataclasses.replace(
            self,
            n_envs=2, rollout_steps=16, max_episode_steps=40,
            total_env_steps=16 * 2 * 3, eval_every=2, eval_episodes=2,
            save_every=2, minibatches=2, epochs=1, jepa_epochs=1,
        )
