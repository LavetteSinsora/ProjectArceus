"""
Player position tracking and progress reward shaping for LS20 Level 1.

Level 1 goal: navigate from starting position (~row 45) to the rotation
modifier near row 10.  The binary ±1 frame-change reward teaches wall
avoidance but gives no directional signal.  This module adds a small
bonus reward each time the player moves toward the goal (Y decreases).

How player tracking works
--------------------------
The player sprite uses StartColor=9.  When the player moves, pixels at its
OLD position change FROM 9 to background, and pixels at its NEW position
change FROM background TO 9.  The goal area also contains color-9 pixels
(GoalColor=9) but those are STATIONARY, so they never appear in the
"changed AND now color 9" set.  This lets us isolate the player cleanly.

Reward structure
----------------
  base       : +1 if player actually moved, −1 if wall hit
  progress   : +PROGRESS_SCALE * (pixels_moved_up / GOAL_DIST) if moved toward goal
  completion : +COMPLETION_BONUS on level finish
"""

import numpy as np

_STEP_ROW = slice(61, 63)   # rows 61-62: step counter display (always changes)
PLAYER_START_Y  = 45.0      # approximate starting row of player sprite
GOAL_Y          = 10.0      # approximate target row (rotation modifier)
GOAL_DIST       = PLAYER_START_Y - GOAL_Y   # 35 pixels total vertical travel
PROGRESS_SCALE  = 5.0       # reward per full goal-distance traversed upward
COMPLETION_BONUS = 50.0


def track_player_y(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float | None:
    """
    Estimate player Y (row) from frame difference.

    Returns the mean row of pixels that just became color 9 (i.e. the
    player's new position after a successful move), or None if no
    color-9 pixels changed (wall hit — player didn't move).

    prev_frame, curr_frame: (64, 64) uint8 arrays.
    """
    prev_m = prev_frame.copy(); prev_m[_STEP_ROW] = 0
    curr_m = curr_frame.copy(); curr_m[_STEP_ROW] = 0

    # Pixels that changed AND are now color 9 = player's new location
    new_color9 = (prev_m != curr_m) & (curr_m == 9)
    if new_color9.any():
        return float(np.where(new_color9)[0].mean())
    return None


def progress_bonus(prev_y: float, curr_y: float) -> float:
    """
    Reward for upward progress (decreasing Y = closer to goal).

    Only positive: moving toward the goal is rewarded; moving away is not
    penalised (the maze may require temporary detours sideways/downward).

    Scale: PROGRESS_SCALE points for traversing the full GOAL_DIST in one go.
    With GOAL_DIST=35 pixels and each move ~5 pixels, one upward step gives
    PROGRESS_SCALE * (5/35) ≈ 0.7 bonus on top of the base +1 reward.
    """
    delta = prev_y - curr_y      # positive = moved up (toward goal)
    if delta <= 0:
        return 0.0               # no bonus for lateral or downward moves
    return PROGRESS_SCALE * (delta / GOAL_DIST)


def compute_reward(
    frame_np: np.ndarray,
    next_np: np.ndarray,
    prev_player_y: float,
    is_terminal: bool,
    level_completed: bool,
) -> tuple[float, float]:
    """
    Full reward for one step.

    Returns (reward, new_player_y) where new_player_y is the estimated
    player Y for the next call (unchanged if the player didn't move).

    reward breakdown:
      base reward   : +1 move / −1 wall hit  (masked, excluding step counter)
      progress bonus: for upward movement toward goal
      completion    : flat bonus when level finishes
    """
    _STEP = slice(61, 63)
    f_m  = frame_np.copy(); f_m[_STEP]  = 0
    nf_m = next_np.copy();  nf_m[_STEP] = 0
    frame_changed = not np.array_equal(f_m, nf_m)

    base = 1.0 if frame_changed else -1.0

    curr_player_y = track_player_y(frame_np, next_np)
    if curr_player_y is not None:
        bonus = progress_bonus(prev_player_y, curr_player_y)
        new_player_y = curr_player_y
    else:
        bonus = 0.0
        new_player_y = prev_player_y   # no movement → Y unchanged

    completion = COMPLETION_BONUS if (is_terminal and level_completed) else 0.0

    return base + bonus + completion, new_player_y
