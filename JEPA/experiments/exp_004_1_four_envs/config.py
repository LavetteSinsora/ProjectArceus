"""
Exp-004-1 — LS20 + TU93 + RE86 + G50T shared-encoder multi-env JEPA.

Extends exp_004_0_ls20_tu93 from two 4-action games to all four ARC-AGI-3 envs
registered in JEPA.shared.env_wrapper. The new axis is heterogeneous action
spaces: LS20 and TU93 are 4-action; RE86 and G50T are 5-action. We use a single
5-way action predictor head + per-env Embedding(5, 32) action embeddings +
per-env PolicyNetwork(out=5), with the wrappers' `available_actions` masking
preventing 4-action envs from ever sampling index 4.

See system_card.md for the full description.
"""

from dataclasses import dataclass
from typing import Tuple

from JEPA.experiments.exp_003_2_action_pred_no_ema.config import Config as _Base032


@dataclass(frozen=True)
class Config(_Base032):
    # ── Multi-env axis (all four 64×64 ARC-AGI-3 games) ───────────────────────
    # The wrapper for each game is selected via the short prefix
    # ('ls20' / 'tu93' / 're86' / 'g50t') by JEPA.shared.env_wrapper.make_env.
    game_ids: Tuple[str, str, str, str] = (
        "ls20-9607627b",
        "tu93-0768757b",
        "re86-8af5384d",
        "g50t-5849a774",
    )
    env_names: Tuple[str, str, str, str] = ("ls20", "tu93", "re86", "g50t")

    # ── Action space (max over all envs) ──────────────────────────────────────
    # Shared 5-way action predictor head; per-env Embedding(5, 32); per-env
    # policy out=5. 4-action envs mask action index 4 via `available_actions`.
    n_actions: int = 5

    # ── Per-env replay buffer capacity ────────────────────────────────────────
    # 4 × 15K ≈ 60K total, vs exp_004_0's 2 × 25K = 50K total. Per-env recency
    # horizons: ~115 LS20 full lives, ~300 TU93 episodes, similar for re86/g50t.
    buffer_size_per_env: int = 15_000

    # ── Reward weighting (state-only, inherited from exp_003_3 via exp_004_0) ─
    # JEPA loss still uses 0.5 / 0.5 — the action predictor stays in the loss
    # for its anti-collapse role. Only the *reward* drops the action term.
    reward_w_state:  float = 1.0
    reward_w_action: float = 0.0
    # reward_clamp inherited from _Base032 (50.0)

    # ── Resampler self-attention among latents ────────────────────────────────
    # Disabled — inherited from exp_003_4 via exp_004_0. See exp_003_4's
    # models/encoder.py for the implementation.
    perceiver_self_attn_among_latents: bool = False

    # ── Dying-step exclusion ──────────────────────────────────────────────────
    # When True, the rollout step that triggered life_end is excluded from the
    # replay buffer (already), the policy buffer, AND all rollout health
    # metrics. See exp_004_0/system_card.md §3.3 (inherited).
    exclude_dying_step: bool = True

    # ── Warm-up ───────────────────────────────────────────────────────────────
    # 4000 global env steps. Under equal round-robin sampling across 4 envs,
    # this is ~1000 random-exploration steps per env before its policy turns on.
    warmup_steps: int = 4_000
