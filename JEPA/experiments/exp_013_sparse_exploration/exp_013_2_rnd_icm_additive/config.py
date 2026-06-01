"""Config for exp_013_2 — ADDITIVE RND+ICM. See SYSTEM_CARD (and exp_013_1).

Reward = w_icm · norm(ICM_forward_error) + (1 − w_icm) · norm(RND_on_φ_error),
each signal normalised INDEPENDENTLY to ~unit scale BEFORE combining (so w is a
true mixing knob, not entangled with the ~100× raw-scale gap). Intrinsic-only:
the extrinsic +1 is the stop signal, never a reward. All exp_013_1 techniques
carry over (φ-space, held-out freeze, leak, entropy=0.05, non-episodic GAE,
reward clip). Default cell = ls20 L2 (level_index=1), an E=∞ (random-unreachable)
cell — the fair arena for directed exploration.

Distinct from exp_013_1 (which COMPOSES: RND inside φ-space, one reward). Here both
ICM-forward-error AND RND-on-φ are SEPARATE rewards, summed.
"""

from __future__ import annotations

from dataclasses import dataclass

from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.config import (
    Config as _Base, GAME_N_ACTIONS,  # noqa: F401  (re-exported for run.py)
)


@dataclass
class Config(_Base):
    exp_dir: str = "JEPA/experiments/exp_013_sparse_exploration/exp_013_2_rnd_icm_additive"
    level_index: int = 1                 # ls20 L2 (random E = ∞)
    max_env_steps: int = 500_000         # stretch budget for a hard cell
    w_icm: float = 0.5                   # r = w_icm·norm(ICM) + (1−w_icm)·norm(RND)

    @property
    def exp_name(self) -> str:
        return f"exp013_2_additive_{self.game}_L{self.level_index + 1}_seed{self.seed}"
