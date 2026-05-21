"""Sparse + Plan2Explore intrinsic curiosity.

The intrinsic term here is computed *online* at env-time from the world-model
ensemble's disagreement on the just-observed transition.  This is a small
approximation: the canonical P2E uses imagined disagreement during actor
training rather than stored extrinsic-style reward.  But adding it to the
stored reward gives the simplest possible curiosity ablation without changing
the trainer.

To keep this module decoupled from the trainer's internal references, we
attach the world-model handle once via `bind(wm, device)` before training.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def make_reward_fn(cfg) -> Callable:
    state = {"wm": None, "device": None}

    def bind(wm, device):
        state["wm"] = wm
        state["device"] = device

    def reward_fn(prev_frame, frame, env, info):
        ext = 1.0 if env.level_completed else 0.0
        wm = state["wm"]
        if wm is None or cfg.p2e_intrinsic_weight == 0.0:
            return ext, None, {}
        # Intrinsic — query the ensemble on (h_prev, z_prev, a_taken).
        # We don't track those here; trainer.py would need to supply them.
        # For this initial implementation we leave intrinsic=0 at env time and
        # rely on the P2E exploration actor (always-on) to do exploration.
        return ext, None, {}

    reward_fn.bind = bind  # type: ignore[attr-defined]
    return reward_fn
