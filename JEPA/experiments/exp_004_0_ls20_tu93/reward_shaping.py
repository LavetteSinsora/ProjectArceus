"""
Per-environment life-end detection for exp_004_0.

LS20: 3 lives, energy-bounded. Life counter at row 61 cols 55-63.
      Reuses the count_lives / is_end_of_life_ls20 logic from prior experiments.

TU93: No intra-game lives. One game = one episode. Termination via
      `state in (WIN, GAME_OVER)` or `levels_completed >= 1`.
      Empirically: 50 random-action steps → GAME_OVER (see probe verdict in
      probe_runs/<timestamp>/verdict.md and system_card.md §7).

Dispatch via is_end_of_life(env_name, ...).
"""

from __future__ import annotations

import numpy as np

# ── LS20 (re-exported from prior experiments) ─────────────────────────────────

# Energy bar: rows 61-62, cols 13-54 (42 pixels wide), color 11 = remaining
_ENERGY_ROWS = (61, 62)
_ENERGY_COLS = slice(13, 55)
_ENERGY_COLOR = 11

# Life counter: row 61, cols 55-63
_LIFE_ROW = 61
_LIFE_COLS = slice(55, 64)
_LIFE_COLOR = 8
_LIFE_OFFSETS = [(1, 2), (4, 5), (7, 8)]


def count_energy_ls20(frame: np.ndarray) -> int:
    """Count remaining LS20 energy steps (0-42)."""
    return int((frame[_ENERGY_ROWS[0], _ENERGY_COLS] == _ENERGY_COLOR).sum())


def count_lives_ls20(frame: np.ndarray) -> int:
    """Count remaining LS20 lives (0-3)."""
    life_pixels = frame[_LIFE_ROW, _LIFE_COLS]
    count = 0
    for a, b in _LIFE_OFFSETS:
        if life_pixels[a] == _LIFE_COLOR and life_pixels[b] == _LIFE_COLOR:
            count += 1
    return count


def is_end_of_life_ls20(frame: np.ndarray, next_frame: np.ndarray,
                        is_terminal: bool) -> bool:
    """End of one LS20 life: GAME_OVER OR life counter decremented."""
    if is_terminal:
        return True
    return count_lives_ls20(next_frame) < count_lives_ls20(frame)


# ── TU93 (no intra-game lives; one game = one episode) ───────────────────────

def is_end_of_life_tu93(frame: np.ndarray, next_frame: np.ndarray,
                        is_terminal: bool) -> bool:
    """End of one TU93 life is the same as game end (probe verdict, §7 of system card)."""
    return is_terminal


# ── Dispatcher ───────────────────────────────────────────────────────────────

def is_end_of_life(env_name: str, frame: np.ndarray, next_frame: np.ndarray,
                   is_terminal: bool) -> bool:
    if env_name == "ls20":
        return is_end_of_life_ls20(frame, next_frame, is_terminal)
    if env_name == "tu93":
        return is_end_of_life_tu93(frame, next_frame, is_terminal)
    raise ValueError(f"unknown env_name: {env_name!r}")
