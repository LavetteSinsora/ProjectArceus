"""Per-step reward shaping. The mini env returns only a binary terminal
signal — everything else is layered on here."""

from __future__ import annotations

from typing import Callable

import numpy as np


REWARD_MODES = ("terminal_only", "wall", "wall+match")


def make_shaping_fn(
    mode: str,
    wall_penalty: float = -0.05,
    match_bonus: float = 0.1,
    unmatch_penalty: float = -0.1,
) -> Callable[[np.ndarray, list[dict]], np.ndarray]:
    """Return f(raw_rewards, infos) -> shaped_rewards. Mutates a copy."""

    if mode not in REWARD_MODES:
        raise ValueError(f"unknown reward mode {mode!r}; valid: {REWARD_MODES}")

    def shape(raw: np.ndarray, infos: list[dict]) -> np.ndarray:
        r = raw.astype(np.float32, copy=True)
        if mode == "terminal_only":
            return r

        for i, info in enumerate(infos):
            if info["wall_hit"]:
                r[i] += wall_penalty

            if mode == "wall+match":
                pre = info["prev_rot"]
                post = info["post_rot"]
                goal = info["goal_rot"]
                was_match = (pre == goal)
                is_match = (post == goal)
                if not was_match and is_match:
                    r[i] += match_bonus
                elif was_match and not is_match:
                    r[i] += unmatch_penalty
        return r

    return shape
