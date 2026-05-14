"""
Exp-004-0 — LS20 + TU93 shared-encoder multi-env JEPA.

Merges the three exp_003_X isolates:
  - exp_003_2: dual predictors (state + action), no EMA target encoder
  - exp_003_3: state-only intrinsic reward (action term retained in JEPA loss only)
  - exp_003_4: Perceiver Resampler is cross-attention only (no SA among latents)

New axis: joint training on two 4-action games (LS20 + TU93). Shared encoder,
shared predictors, per-env action embeddings, per-env policies, per-env buffers,
balanced JEPA sampling.

See system_card.md for the full description.
"""

from dataclasses import dataclass, field
from typing import Tuple

from JEPA.experiments.exp_003_2_action_pred_no_ema.config import Config as _Base032


@dataclass(frozen=True)
class Config(_Base032):
    # ── Multi-env axis ────────────────────────────────────────────────────────
    # The two 4-action games we jointly train on. The wrapper for each game is
    # selected via the short prefix ('ls20' / 'tu93') by JEPA.shared.env_wrapper.make_env.
    game_ids: Tuple[str, str] = ("ls20-9607627b", "tu93-0768757b")
    env_names: Tuple[str, str] = ("ls20", "tu93")

    # ── Per-env replay buffer capacity ────────────────────────────────────────
    # Total memory matches a single 50K NextFrameLatentBuffer used in exp_003_X.
    buffer_size_per_env: int = 25_000

    # ── Reward weighting (state-only, inherited from exp_003_3) ───────────────
    # NOTE: the JEPA loss still uses 0.5 / 0.5 — the action predictor remains in
    # the loss for its anti-collapse role. Only the *reward* drops the action term.
    reward_w_state:  float = 1.0
    reward_w_action: float = 0.0
    # reward_clamp stays at 50.0 (inherited from _Base032)

    # ── Resampler self-attention among latents ────────────────────────────────
    # Disabled — inherited from exp_003_4. See models/encoder.py for the implementation.
    perceiver_self_attn_among_latents: bool = False

    # ── Dying-step exclusion ──────────────────────────────────────────────────
    # When True, the rollout step that triggered life_end is excluded from
    # the replay buffer (already), the policy buffer, AND all rollout health
    # metrics. See system_card.md §3.3.
    exclude_dying_step: bool = True
