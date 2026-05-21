"""
Exp-003-3 — State-only curiosity reward (action predictor retained for JEPA).

Minimal-diff fork of exp-003-3's parent: the action-CE term is dropped from
the per-step intrinsic reward (reward_w_action = 0.0) while the action
predictor remains in the JEPA loss for anti-collapse. State-prediction error
is restored to magnitude 1.0 (reward_w_state = 1.0) to keep per-step reward
scale comparable to the parent's average. See system_card.md §2–3.
"""

from dataclasses import dataclass

from JEPA.experiments.exp_003_0_normalized_latent_jepa.config import Config as _Base003


@dataclass(frozen=True)
class Config(_Base003):
    # ── Action predictor (NEW) ────────────────────────────────────────────────
    action_predictor_hidden: int = 512
    action_predictor_lr: float = 1e-4
    action_predictor_wd: float = 0.01

    # ── JEPA loss weighting ───────────────────────────────────────────────────
    lambda_state:  float = 0.5
    lambda_action: float = 0.5

    # ── Reward weighting + cap ────────────────────────────────────────────────
    # Action-CE term dropped from reward (system_card.md §3 — noisy-TV trap);
    # state term magnitude restored to 1.0 to match parent's per-step scale.
    reward_w_state:  float = 1.0
    reward_w_action: float = 0.0
    reward_clamp:    float = 50.0

    # ── State-predictor optimiser (renamed for clarity; mirrors predictor_lr) ─
    state_predictor_lr: float = 1e-4
    state_predictor_wd: float = 0.01

    # ── Metric cadences ───────────────────────────────────────────────────────
    eval_freq:         int = 5_000   # full eval pass every N env steps
    n_eval_episodes:   int = 5
    grad_decomp_freq:  int = 25      # per-source × per-sub-block gnorm every Nth JEPA update
    uwr_freq:          int = 25      # update-to-weight ratio cadence
    capture_attn_freq: int = 25      # eval-mode attention probe every Nth JEPA update

    # ── Weights & Biases ──────────────────────────────────────────────────────
    # Off by default; enable with --wandb on the CLI.
    wandb_project: str = "ProjectArceus"
    wandb_entity:  str | None = None       # None → user's default entity
    wandb_mode:    str = "online"          # "online" | "offline" | "disabled"
