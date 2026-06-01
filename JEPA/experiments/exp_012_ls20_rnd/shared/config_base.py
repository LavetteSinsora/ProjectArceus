"""Shared config for exp_012. Per-variant configs subclass and override.

Inherits exp_010's PPO recipe and adds the RND dual-stream knobs (two
discounts, ext/int advantage coefficients, RND network sizing, intrinsic-reward
normalisation). `exp_dir` is the sub-experiment directory; checkpoints are
written flat to `<exp_dir>/checkpoints/step_*.pt` and metrics to
`<exp_dir>/runs/<run>/metrics.jsonl` — the layout the main JEPA dashboard reads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # Identity
    exp_name: str = "exp_012_base"
    exp_dir: str = "JEPA/experiments/exp_012_ls20_intrinsic_exploration"

    # Env (real LS20 via JEPA.shared.env_wrapper)
    env_name: str = "ls20"
    n_envs: int = 16                    # more parallel envs than exp_010 (8): bigger
                                        # GPU batch + more diverse RND exploration
    max_episode_steps: int = 200
    stop_levels: int = 1                # clear this many LS20 levels per episode
                                        # (1 = the L1 task; 2 = clear L1 then L2 ...)
    seed: int = 0

    # Model
    n_actions: int = 4
    n_colors: int = 16
    frame_size: int = 64
    trunk_dim: int = 256

    # Rollout / budget
    rollout_steps: int = 128            # 128 * 16 = 2048 transitions / update
    total_env_steps: int = 3_000_000

    # PPO (exp_010 recipe)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_clip_eps: float = 0.2
    c_value: float = 0.5
    c_entropy: float = 0.01
    grad_clip: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4

    # RND dual-stream (faithful to Burda et al. 2018)
    gamma_ext: float = 0.999            # extrinsic, EPISODIC
    gamma_int: float = 0.99             # intrinsic, NON-EPISODIC
    ext_coef: float = 2.0               # A = ext_coef * A_E + int_coef * A_I
    int_coef: float = 1.0
    rnd_feature_dim: int = 256          # target/predictor output dim (paper: 512)
    rnd_predictor_hidden: int = 256     # extra FC width in the predictor head
    rnd_loss_coef: float = 1.0          # predictor distillation loss weight
    predictor_update_proportion: float = 1.0   # 1.0 at <=32 envs (paper)
    int_norm_eps: float = 1e-8          # floor on the intrinsic-return std

    # Warm-start: path to a checkpoint (.pt) to load model + RND target + predictor
    # from (e.g. an L1 run, to continue into L2). None = train from scratch.
    init_ckpt: str | None = None

    # Logging / eval / checkpoint cadence (in updates)
    log_every: int = 1
    eval_every: int = 25
    eval_episodes: int = 32
    save_every: int = 100

    # Early stopping (PPO): stop once eval success_rate stays at/above
    # `early_stop_success_rate` for `early_stop_patience` consecutive evals.
    early_stop_enabled: bool = True
    early_stop_success_rate: float = 0.99
    early_stop_patience: int = 3

    @property
    def total_updates(self) -> int:
        return self.total_env_steps // (self.rollout_steps * self.n_envs)

    def smoke(self) -> "Config":
        """Tiny variant for plumbing tests (a few updates, < 1 min)."""
        import dataclasses
        return dataclasses.replace(
            self,
            n_envs=2, rollout_steps=16, max_episode_steps=40,
            total_env_steps=16 * 2 * 3, eval_every=2, eval_episodes=2,
            save_every=2, minibatches=2, epochs=1,
        )
