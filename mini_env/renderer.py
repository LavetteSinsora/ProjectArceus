"""Pure-numpy renderer for the 32x32 mini env, styled like LS20.

LS20 visual conventions (matched here):
    bg          light grey  (walkable empty)
    wall        near-black  (impassable)
    player      blue body with a FIXED orange top half (never rotates) —
                mirrors LS20's `sfqyzhzkij` 5x5 player sprite
    goal        blue body with a 3-pixel ORANGE L-mark rotated by
                `goal_rotation` — mirrors LS20's `kvynsvxbpi` shape at the
                goal position, colored per `GoalColor`, rotated per `GoalRotation`
    cross       white "+"
    match cue   bright-lime 1-px ring outside the goal tile
    UI strip    bottom 4 px, LS20 layout:
                    cols 0..3    pattern-preview tile (blue L-mark rotated
                                 by player_rotation on dark bg)
                    cols 4..31   yellow energy bar shrinking as
                                 step_counter decrements

Returned frame is (32, 32) uint8 — colour indices in the same 16-colour
palette LS20 uses.
"""

from __future__ import annotations

import numpy as np

from mini_env.loader import EnvConfig
from mini_env.state import EnvState


# Orange accent colour (palette index 7) — hardcoded across renderer + editor,
# mirrors the orange band on LS20's two-tone `sfqyzhzkij` player sprite.
ACCENT = 7

# Reference "L" mark for rotation 0, inside a 4x4 tile.
#   . L . .
#   . L L .
#   . . . .
#   . . . .
_L_MARK_ROT0 = np.zeros((4, 4), dtype=np.uint8)
_L_MARK_ROT0[0, 1] = 1
_L_MARK_ROT0[1, 1] = 1
_L_MARK_ROT0[1, 2] = 1


def _rot_index(rotation: int) -> int:
    return {0: 0, 90: 1, 180: 2, 270: 3}[rotation]


def _rotated_l_mask(rotation: int) -> np.ndarray:
    """4x4 bool mask of the L-mark rotated CW by `rotation` degrees.

    np.rot90 rotates counter-clockwise by default — we want clockwise, so we
    pass negative k. The editor's JS port uses the same convention.
    """
    return np.rot90(_L_MARK_ROT0, k=-_rot_index(rotation))


def _draw_player(frame: np.ndarray, c: int, r: int, tile: int, body: int) -> None:
    """4x4 player tile: top 2 rows orange, bottom 2 rows blue. FIXED — never rotates.

    Mirrors LS20's `sfqyzhzkij` sprite (two rows of colour 12, three rows of
    colour 9 in 64-px space; we squash to 2+2 at tile=4).
    """
    y0, x0 = r * tile, c * tile
    frame[y0:y0 + tile, x0:x0 + tile] = body
    frame[y0:y0 + 2, x0:x0 + tile] = ACCENT


def _draw_goal(frame: np.ndarray, c: int, r: int, tile: int,
               body: int, rotation: int) -> None:
    """4x4 goal tile: blue body + 3-pixel orange L-mark rotated by `rotation`.

    Mirrors LS20's `kvynsvxbpi` shape sprite, colored per `GoalColor` and
    rotated per `GoalRotation`. The L-mark uses the same orientation
    convention as the pattern preview, so visually `goal-L == preview-L`
    means the puzzle is solved.
    """
    y0, x0 = r * tile, c * tile
    frame[y0:y0 + tile, x0:x0 + tile] = body
    mask = _rotated_l_mask(rotation)
    tile_view = frame[y0:y0 + tile, x0:x0 + tile]
    tile_view[mask == 1] = ACCENT


def render(state: EnvState, config: EnvConfig) -> np.ndarray:
    """Render the current state to a (32, 32) uint8 frame."""
    pal = config.palette
    tile = config.tile_px
    H = config.rows * tile          # 32
    W = config.cols * tile          # 32
    play_rows_px = config.play_rows * tile  # 28
    assert (H, W) == (32, 32), "renderer requires a 32x32 grid"

    # 1. Background
    frame = np.full((H, W), pal["bg"], dtype=np.uint8)

    # 2. Walls
    for (c, r) in config.walls:
        frame[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = pal["wall"]

    # 3. Cross (plus-sign)
    cc, cr = config.cross.col, config.cross.row
    frame[cr * tile + 1:cr * tile + 3, cc * tile:(cc + 1) * tile] = pal["cross"]
    frame[cr * tile:(cr + 1) * tile, cc * tile + 1:cc * tile + 3] = pal["cross"]

    # 4. Goal (blue body + rotated L-mark showing target rotation)
    _draw_goal(frame, config.goal.col, config.goal.row, tile,
               pal["player"], config.goal_rotation)

    # 5. Match-cue highlight: 1-px ring just OUTSIDE the goal tile.
    if config.show_match_cue and state.player_rotation == config.goal_rotation:
        gy0, gx0 = config.goal.row * tile, config.goal.col * tile
        top = gy0; bottom = gy0 + tile - 1
        left = gx0; right = gx0 + tile - 1
        ring_top, ring_bottom = top - 1, bottom + 1
        ring_left, ring_right = left - 1, right + 1
        if 0 <= ring_top < play_rows_px:
            frame[ring_top, max(ring_left, 0):min(ring_right, W - 1) + 1] = pal["highlight"]
        if 0 <= ring_bottom < play_rows_px:
            frame[ring_bottom, max(ring_left, 0):min(ring_right, W - 1) + 1] = pal["highlight"]
        if 0 <= ring_left < W:
            frame[max(ring_top, 0):min(ring_bottom, play_rows_px - 1) + 1, ring_left] = pal["highlight"]
        if 0 <= ring_right < W:
            frame[max(ring_top, 0):min(ring_bottom, play_rows_px - 1) + 1, ring_right] = pal["highlight"]

    # 6. Player (FIXED two-tone — no rotation in the tile itself)
    _draw_player(frame, state.player_c, state.player_r, tile, pal["player"])

    # 7. UI strip (rows 28..31) — LS20 layout: preview LEFT, energy bar RIGHT.
    ui_top = play_rows_px            # 28
    frame[ui_top:H, :] = pal["preview_bg"]

    # 7a. Pattern preview tile: rows 28..31, cols 0..3.
    pv_top, pv_left = ui_top, 0
    if state.denial_frames > 0:
        frame[pv_top:pv_top + 4, pv_left:pv_left + 4] = pal["denial_flash"]
    # L-mark in player colour (blue), rotated by player_rotation.
    mask = _rotated_l_mask(state.player_rotation)
    pv = frame[pv_top:pv_top + 4, pv_left:pv_left + 4]
    pv[mask == 1] = pal["player"]

    # 7b. Energy bar: row ui_top + 1 (= row 29), cols 4..31 (28 cols wide).
    bar_left = 4
    bar_width = W - bar_left          # 28
    filled = int(bar_width * max(0, state.step_counter) / config.step_limit)
    filled = max(0, min(bar_width, filled))
    if filled > 0:
        frame[ui_top + 1, bar_left:bar_left + filled] = pal["energy"]

    return frame
