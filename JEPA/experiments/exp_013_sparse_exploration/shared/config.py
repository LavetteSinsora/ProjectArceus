"""Unified config for exp_013 — the sparse-reward exploration calibration.

One Config drives every (method × game × level × seed) run. The only knobs that
change between the two intrinsic baselines are `method` ("icm" | "rnd") and the
method-specific blocks below; everything else (the dual-stream PPO backbone, the
intrinsic-return normaliser, the stop rule) is shared, so ICM-vs-RND is a
controlled comparison where only the bonus differs.

Goal metric: `env_steps_to_first_reward` (total env steps, summed across actors,
to the first positive extrinsic reward). We **stop on the first reward** and cap
at `max_env_steps` for the censored "never solved" case.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# Per-game action counts (the wrapper is authoritative; this is for sanity only).
GAME_N_ACTIONS = {"ls20": 4, "tu93": 4, "re86": 5, "g50t": 5}


@dataclass
class Config:
    # ── Identity ────────────────────────────────────────────────────────────
    method: str = "rnd"                 # "icm" | "rnd"
    game: str = "ls20"                  # ls20 | tu93 | re86 | g50t
    level_index: int = 0                # 0-indexed; drop the agent INTO this level
    seed: int = 0
    exp_dir: str = "JEPA/experiments/exp_013_sparse_exploration"

    # ── Env ─────────────────────────────────────────────────────────────────
    n_envs: int = 16
    max_episode_steps: int = 200
    n_actions: int | None = None        # filled from the env wrapper at build time
    n_colors: int = 16
    frame_size: int = 64
    trunk_dim: int = 256

    # ── Rollout / budget / stop rule ─────────────────────────────────────────
    rollout_steps: int = 128            # 128 * 16 = 2048 transitions / update
    max_env_steps: int = 3_000_000      # hard cap (censoring point) per run
    stop_on_first_reward: bool = True   # the exp_013 stop rule

    # ── PPO (exp_010/exp_012 recipe) ─────────────────────────────────────────
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_clip_eps: float = 0.2
    c_value: float = 0.5
    c_entropy: float = 0.01
    grad_clip: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4

    # ── Dual-stream (faithful to RND; shared by both methods) ────────────────
    gamma_ext: float = 0.999            # extrinsic, EPISODIC
    gamma_int: float = 0.99             # intrinsic, NON-EPISODIC
    ext_coef: float = 2.0               # A = ext_coef * A_E + int_coef * A_I
    int_coef: float = 1.0
    int_norm_eps: float = 1e-8          # floor on the intrinsic-return std

    # ── Intrinsic-reward normalisation robustness (exp_013 fix) ──────────────
    # ICM's untrained-model startup error is ~700× its converged value; a
    # cumulative RMS bakes that transient in and crushes the bonus for ~43k
    # steps (see SMOKE_TEST_FINDINGS.md). The principled fix:
    #   (1) WARM-UP — give the policy ZERO intrinsic reward for the first
    #       `norm_warmup_updates` updates while the bonus net still trains (a
    #       meaningless prediction error is not "surprise", so don't reward it);
    #   (2) scale by an EMA std of the intrinsic RETURNS (decay `int_norm_decay`)
    #       that tracks the CURRENT novelty scale, not a frozen cumulative one;
    #   (3) never subtract the mean (keep the bonus ≥ 0, RND-faithful);
    #   (4) optionally CLIP the normalised bonus as a value-target safety rail.
    # METHOD-SPECIFIC normalisation (left None -> per-method default in __post_init__):
    #   * int_norm_mode: how the intrinsic-return std is tracked.
    #       "cumulative" = RND-paper-faithful running RMS (TESTED-good for RND; its
    #                      novelty is small + stationary, so the cumulative std is fine).
    #       "ema"        = decaying std that tracks the CURRENT scale (needed for ICM,
    #                      whose untrained-model error has a ~700× startup transient
    #                      that a cumulative RMS bakes in forever -> bonus collapse).
    #   * norm_warmup_updates: updates of ZERO intrinsic reward while the bonus net
    #     trains (ICM=3 to skip its transient; RND=0 — its target is informative at t=0).
    # RND therefore runs byte-for-byte its tested config (cumulative RMS, no warm-up).
    int_norm_mode: str | None = None       # "cumulative" | "ema"
    norm_warmup_updates: int | None = None
    int_norm_decay: float = 0.99           # used only when int_norm_mode == "ema"
    int_reward_clip: float | None = None   # cap on the normalised bonus (None = off)

    # ── RND-specific (Burda et al. 2018) ─────────────────────────────────────
    rnd_feature_dim: int = 256
    rnd_predictor_hidden: int = 256
    rnd_loss_coef: float = 1.0
    rnd_lr: float = 1e-4                # predictor's own optimiser
    predictor_update_proportion: float = 1.0

    # ── ICM-specific (Pathak 2017, NORMALIZED variant — no frozen η) ─────────
    # The raw forward error is fed through the SAME running-return-std normaliser
    # as RND (the 2018 large-scale-study fix), so curiosity does not collapse via
    # a frozen η. This is a deliberate, documented deviation from vanilla 2017.
    icm_beta: float = 0.2               # (1-β)L_inverse + β L_forward
    icm_hidden: int = 256
    icm_lr: float = 1e-3
    icm_epochs: int = 1

    # ── Logging / eval / checkpoint cadence (in updates; 0 disables) ─────────
    log_every: int = 1
    eval_every: int = 0                 # eval is NOT the metric; off by default
    eval_episodes: int = 16
    save_every: int = 0                 # checkpoints not needed for the metric

    def __post_init__(self):
        if self.int_norm_mode is None:
            self.int_norm_mode = "ema" if self.method == "icm" else "cumulative"
        if self.norm_warmup_updates is None:
            self.norm_warmup_updates = 3 if self.method == "icm" else 0

    @property
    def exp_name(self) -> str:
        return f"exp013_{self.method}_{self.game}_L{self.level_index + 1}_seed{self.seed}"

    @property
    def total_updates(self) -> int:
        return self.max_env_steps // (self.rollout_steps * self.n_envs)

    def smoke(self) -> "Config":
        """Tiny variant for plumbing tests (a handful of updates, < 1 min)."""
        return dataclasses.replace(
            self,
            n_envs=2, rollout_steps=16, max_episode_steps=40,
            max_env_steps=16 * 2 * 6, minibatches=2, epochs=1,
            stop_on_first_reward=False, eval_every=0, save_every=0,
        )
