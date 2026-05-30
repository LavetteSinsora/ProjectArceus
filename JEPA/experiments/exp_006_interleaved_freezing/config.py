"""
Exp-006-0 — Interleaved freezing schedule on top of exp_003_3.

Minimal-diff fork: architecture, reward, JEPA loss, optimisers, buffers, and
policy are inherited unchanged. The only addition is the FreezeScheduler
config — see system_card.md §2.1 and §3 for rationale.
"""

from dataclasses import dataclass

from JEPA.experiments.exp_003_3_state_only_reward.config import Config as _Base033


@dataclass(frozen=True)
class Config(_Base033):
    # ── Interleaved freeze schedule (NEW) ──────────────────────────────────────
    freeze_enabled: bool = True
    freeze_phase_len: int = 500              # JEPA updates per freeze phase
    freeze_initial_mode: str = "interleaved" # "interleaved" | "joint" — what the run starts in
    freeze_initial_phase: str = "encoder_frozen"  # which freeze phase fires first when interleaved
    freeze_threshold_window: int = 200       # JEPA updates of running mean used for thresholds
    freeze_threshold_min_phases: int = 4     # ≥ N complete phases before INTERLEAVED→JOINT can fire

    # Exit criterion (INTERLEAVED → JOINT): both must hold simultaneously
    freeze_l_action_exit: float = 0.05       # PLACEHOLDER — tune via calibration run (§6.3)
    freeze_cossim_exit:   float = 0.30       # PLACEHOLDER — tune via calibration run (§6.3)

    # Re-entry criterion (JOINT → INTERLEAVED): collapse detected
    freeze_cossim_reentry: float = 0.70      # PLACEHOLDER — tune via calibration run (§6.3)
    freeze_reentry_cooldown: int = 1000      # JEPA updates after exit before re-entry can fire
