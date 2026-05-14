"""
Exp-003-2 — Action Predictor (no EMA).

Anti-collapse: replace the EMA target encoder of exp_003_1 with an action
predictor that recovers a_t from (h_t, h_{t+1}). Encoder gets gradient from
three paths per step: L_state via h_t, L_action via h_t, L_action via h_{t+1}.

Replay buffer now stores raw next_frame (uint8); both h_t and h_{t+1} are
re-encoded with the live encoder at every JEPA step. Sampling is uniform.

See system_card.md §8 — every number here mirrors the hyperparameter table.
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
    reward_w_state:  float = 0.5
    reward_w_action: float = 0.5
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
