"""
Reward shaping and single-life detection for exp_002.

End-of-life detection (empirically confirmed via environment probe):
  - Game has 3 lives; each life = 42 energy steps (wall hits)
  - Energy bar: rows 61-62, cols 13-54, color 11 = remaining steps
  - Life counter: row 61, cols 55-63: groups of (5, 8, 8) per alive life
  - End-of-life signal: count_lives(next_frame) < count_lives(frame)
  - GAME_OVER (is_terminal=True) = all lives lost = also end-of-last-life

Training exclusion:
  The dying transition (s_dying → s_next_life_start or s_game_over) is excluded
  from the replay buffer. Train only on s_0 → ... → s_{T-2} → s_{T-1}.

Reward:
  Intrinsic curiosity reward = flow-matching prediction error (computed in train.py).
  No explicit shaping by default; this module handles detection only.
"""

import numpy as np

# ── Frame regions (confirmed by environment probe) ────────────────────────────

# Energy bar: rows 61-62, cols 13-54 (42 pixels wide), color 11 = remaining
_ENERGY_ROWS = (61, 62)
_ENERGY_COLS = slice(13, 55)    # 42 pixels
_ENERGY_COLOR = 11

# Life counter: row 61, cols 55-63
# Pattern: [5, 8, 8,  5, 8, 8,  5, 8, 8] = 3 lives (color 8 = alive)
_LIFE_ROW = 61
_LIFE_COLS = slice(55, 64)      # 9 pixels
_LIFE_COLOR = 8
# Each life occupies 3 pixels; the live indicator is at offsets 1 and 2 within each group
_LIFE_OFFSETS = [(1, 2), (4, 5), (7, 8)]   # indices within the 9-pixel slice

# Step counter display (changes every step — mask when comparing frames)
STEP_COUNTER_ROWS = slice(61, 63)


def count_energy(frame: np.ndarray) -> int:
    """
    Count remaining energy steps (0–42).
    frame: (64, 64) uint8
    """
    return int((frame[_ENERGY_ROWS[0], _ENERGY_COLS] == _ENERGY_COLOR).sum())


def count_lives(frame: np.ndarray) -> int:
    """
    Count remaining lives (0–3).
    frame: (64, 64) uint8
    """
    life_pixels = frame[_LIFE_ROW, _LIFE_COLS]
    count = 0
    for a, b in _LIFE_OFFSETS:
        if life_pixels[a] == _LIFE_COLOR and life_pixels[b] == _LIFE_COLOR:
            count += 1
    return count


def is_end_of_life(frame: np.ndarray, next_frame: np.ndarray, is_terminal: bool) -> bool:
    """
    Detect whether this transition corresponds to a life being lost.

    Returns True if:
      (a) is_terminal=True  (GAME_OVER, all lives lost)
      (b) count_lives(next_frame) < count_lives(frame)
              (intermediate life loss: life counter decreased)

    frame:      (64, 64) uint8 — current frame (before action)
    next_frame: (64, 64) uint8 — next frame (after action)
    is_terminal: bool from env.step()
    """
    if is_terminal:
        return True
    return count_lives(next_frame) < count_lives(frame)


def frames_changed(frame: np.ndarray, next_frame: np.ndarray) -> bool:
    """
    True if any game-relevant pixels changed (masking the step counter rows).
    Used as the base signal for intrinsic curiosity.
    """
    f1 = frame.copy()
    f2 = next_frame.copy()
    f1[STEP_COUNTER_ROWS, :] = 0
    f2[STEP_COUNTER_ROWS, :] = 0
    return not np.array_equal(f1, f2)
