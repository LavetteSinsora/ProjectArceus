"""
Simplified LS20 environment — standalone, no arcengine dependency.

A stripped-down version of the LS20 rotation-puzzle that preserves every core
mechanic but drastically reduces state space, making it easier for random
exploration to accidentally discover the reward signal:

  Frame  : 64×64 uint8, same size as the original (backward-compatible)
  Cells  : 8×8 px each (vs 5×5 in original) → coarser, fewer cells
  Grid   : 7 game rows × 8 cols + 1 UI row → 64 px tall × 64 px wide
  Inner  : rows 1-5, cols 1-6 = 30 playable cells  (original Level 1 ≈ 100)
  States : ~30 positions × 4 rotations × 2 target states ≈ 240 reachable

Mechanics preserved
───────────────────
  • Navigate a 2-D grid with 4-directional movement (up/down/left/right)
  • Step onto the cross tile  → rotation index cycles +1  (mod 4)
  • Step onto the target tile → WIN if rotation == goal, else nothing
  • Energy bar counts down each step; on depletion lose one life and respawn
  • 3 lives; exhaust all → terminal (game over)

Visual encoding (ARC colour palette, same indices as original)
──────────────────────────────────────────────────────────────
  Floor   : colour  0  (black)
  Wall    : colour  4  (yellow)
  Cross   : colour  3  (green) with colour-11 '+' highlight
  Target  : colour  9  (maroon) with goal-rotation colour as inner border
  Player  : colour encodes current rotation → [1=blue, 2=red, 6=magenta, 8=cyan]
  UI strip (rows 56-63):
    top 4 rows  — energy bar (colour 11 / colour 7) + life dots (colour 8 / 5)
    bottom 4 rows — current-rotation colour block (left) vs goal colour (right)

The rotation indicator lets the agent visually compare current vs goal:
"my player is BLUE, goal indicator is BLUE → rotations match → step on target!"

Interface (matches BaseArcEnv from env_wrapper.py)
───────────────────────────────────────────────────
  env.reset()                 → (64, 64) uint8 ndarray
  env.step(action_idx)        → ((64, 64) uint8, is_terminal: bool)
  env.n_actions               → 4
  env.available_actions       → [1, 2, 3, 4]  (always; 1-indexed like GameAction)
  env.level_completed         → bool
  env.won                     → bool
  env.frame_diff(f0, f1)      → (64, 64) float32  (UI rows 56-63 zeroed)
  env.patch_weights(f0, f1)   → (16,) float32 weights  (same shape as original)
  env.detect_moved_cell(f0,f1)→ int | None  (0-63 index in 8×8 fine grid)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# ── ARC colour palette indices ─────────────────────────────────────────────────
_BLACK       = 0    # floor
_BLUE        = 1    # rotation state 0  (0°)
_RED         = 2    # rotation state 1  (90°)
_GREEN       = 3    # cross / rotation-control tile
_YELLOW      = 4    # wall
_GRAY        = 5    # dead-life slot in UI
_MAGENTA     = 6    # rotation state 2  (180°)
_ORANGE      = 7    # spent energy in bar
_CYAN        = 8    # rotation state 3  (270°) / alive-life slot
_MAROON      = 9    # target zone
_LIGHT_GREEN = 11   # remaining energy / cross highlight

# One colour per rotation index — player colour equals current rotation state.
# Matching player colour with the goal indicator = rotation is correct.
_ROT_COLOR: list[int] = [_BLUE, _RED, _MAGENTA, _CYAN]

# ── Layout constants ───────────────────────────────────────────────────────────
_CELL     = 8    # pixels per cell side
_COLS     = 8    # grid columns  (cells)
_ROWS     = 7    # game rows     (cells); row 0 and row 6 are border walls
_UI_ROWS  = 1    # UI-strip rows (cells), appended below the game rows

FRAME_W: int = _COLS * _CELL                    # 64
FRAME_H: int = (_ROWS + _UI_ROWS) * _CELL       # 64
_UI_Y:   int = _ROWS * _CELL                    # 56 — first pixel row of UI strip

# ── Level definition ───────────────────────────────────────────────────────────

@dataclass
class LevelDef:
    """
    Configuration for one simplified-LS20 level.

    Coordinates are (row, col) in cell space (not pixels).
    The border walls (row 0, row 6, col 0, col 7) are added automatically.
    """
    player_start:   tuple[int, int]
    cross_pos:      tuple[int, int]
    target_pos:     tuple[int, int]
    start_rotation: int = 3                 # index into [0°, 90°, 180°, 270°]
    goal_rotation:  int = 0
    max_steps:      int = 30               # energy budget per life
    extra_walls:    list[tuple[int, int]] = field(default_factory=list)

    def all_walls(self) -> frozenset[tuple[int, int]]:
        border: set[tuple[int, int]] = {
            (r, c)
            for r in range(_ROWS)
            for c in range(_COLS)
            if r == 0 or r == _ROWS - 1 or c == 0 or c == _COLS - 1
        }
        return frozenset(border | set(self.extra_walls))


# Default level mirrors the spirit of LS20 Level 1:
#   start rotation 270° (index 3), goal 0° (index 0) → exactly one cross visit wins.
#   Player at centre, target near top, cross near bottom-right.
LEVEL_1 = LevelDef(
    player_start=(3, 3),
    cross_pos=(5, 5),
    target_pos=(1, 3),
    start_rotation=3,   # 270° — one click away from goal
    goal_rotation=0,    # 0°
    max_steps=30,
)


# ── Environment ────────────────────────────────────────────────────────────────

class SimplifiedLS20Env:
    """
    Simplified LS20: rotation-puzzle on a 7×8 cell grid, rendered as 64×64 px.

    Drop-in replacement for LS20Env (env_wrapper.py) — no arcengine required.
    Instantiate with an optional LevelDef; defaults to LEVEL_1.

        env = SimplifiedLS20Env()          # or SimplifiedLS20Env(my_level)
        frame = env.reset()                # (64, 64) uint8
        frame, done = env.step(0)          # action_idx 0-3
    """

    _MASKED_ROWS = slice(_UI_Y, FRAME_H)   # rows 56-63: zero in frame diffs

    def __init__(self, level: LevelDef = LEVEL_1) -> None:
        self._level  = level
        self._walls  = level.all_walls()
        self.n_actions: int = 4
        self._terminal = False
        self._won      = False
        self._reset_state()

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset episode; return initial (64, 64) uint8 frame."""
        self._reset_state()
        self._terminal = False
        self._won      = False
        return self._render()

    def step(self, action_idx: int) -> tuple[np.ndarray, bool]:
        """
        Execute one action and return (next_frame, is_terminal).

        action_idx (0-indexed):
            0 → up    (row - 1)
            1 → down  (row + 1)
            2 → left  (col - 1)
            3 → right (col + 1)
        """
        if self._terminal:
            return self._render(), True

        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action_idx]
        nr, nc  = self._player[0] + dr, self._player[1] + dc

        if (nr, nc) not in self._walls:
            self._player = (nr, nc)

            if (nr, nc) == self._level.cross_pos:
                # Entering the cross tile cycles rotation by one step
                self._rotation = (self._rotation + 1) % 4

            elif (nr, nc) == self._level.target_pos and not self._target_collected:
                if self._rotation == self._level.goal_rotation:
                    # Correct rotation → level complete
                    self._target_collected = True
                    self._won              = True
                    self._terminal         = True
                # Wrong rotation: player stands on target tile but nothing happens;
                # they can leave and try again after adjusting rotation.

        # Consume one energy step
        self._steps_left -= 1
        if not self._terminal and self._steps_left <= 0:
            self._lives -= 1
            if self._lives <= 0:
                self._terminal = True   # all lives exhausted → game over
            else:
                # Lose a life: full respawn (position, rotation, target, energy)
                self._player           = self._level.player_start
                self._rotation         = self._level.start_rotation
                self._target_collected = False
                self._steps_left       = self._level.max_steps

        return self._render(), self._terminal

    @property
    def available_actions(self) -> list[int]:
        """1-indexed action ids (always all four, like GameAction values)."""
        return [1, 2, 3, 4]

    @property
    def level_completed(self) -> bool:
        """True once the agent has collected the target with correct rotation."""
        return self._won

    @property
    def won(self) -> bool:
        return self._won

    # ── Mask-aware diff utilities (same contract as BaseArcEnv) ───────────────

    def frame_diff(self, f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
        """Pixel-wise absolute diff with UI rows 56-63 zeroed. Returns (64,64) float32."""
        d = np.abs(f1.astype(np.float32) - f0.astype(np.float32))
        d[self._MASKED_ROWS, :] = 0.0
        return d

    def patch_weights(self, f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
        """(64,64) frames → (16,) patch-change weights in [0,1] (4×4 patch grid of 16×16 px)."""
        pw = self.frame_diff(f0, f1).reshape(4, 16, 4, 16).mean(axis=(1, 3)).flatten()
        m  = float(pw.max())
        return (pw / m) if m > 1e-8 else np.zeros(16, dtype=np.float32)

    def detect_moved_cell(self, f0: np.ndarray, f1: np.ndarray) -> int | None:
        """
        Return 8×8 fine-grid cell index (0-63) with the most masked change.
        Returns None if max change < 2.0 (no significant movement).
        Each cell is 8×8 pixels, matching the game cell size.
        """
        cell_diff = self.frame_diff(f0, f1).reshape(8, 8, 8, 8).sum(axis=(1, 3))
        if cell_diff.max() < 2.0:
            return None
        r, c = np.unravel_index(cell_diff.argmax(), cell_diff.shape)
        return int(r * 8 + c)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self._player           = self._level.player_start
        self._rotation         = self._level.start_rotation
        self._target_collected = False
        self._steps_left       = self._level.max_steps
        self._lives            = 3

    def _render(self) -> np.ndarray:
        frame = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

        # ── Walls (solid YELLOW blocks) ────────────────────────────────────────
        for r, c in self._walls:
            _fill(frame, r, c, _YELLOW)

        # ── Cross tile: GREEN base with LIGHT_GREEN '+' cross-hair ────────────
        cr, cc = self._level.cross_pos
        y0, x0 = cr * _CELL, cc * _CELL
        frame[y0 : y0 + _CELL, x0 : x0 + _CELL] = _GREEN
        frame[y0 + 3 : y0 + 5, x0 : x0 + _CELL] = _LIGHT_GREEN   # horizontal bar
        frame[y0 : y0 + _CELL, x0 + 3 : x0 + 5] = _LIGHT_GREEN   # vertical bar

        # ── Target tile: MAROON with goal-rotation colour as inner ring ────────
        if not self._target_collected:
            tr, tc = self._level.target_pos
            goal_color = _ROT_COLOR[self._level.goal_rotation]
            y0, x0 = tr * _CELL, tc * _CELL
            frame[y0 : y0 + _CELL,       x0 : x0 + _CELL]       = _MAROON
            frame[y0 + 1 : y0 + _CELL - 1, x0 + 1 : x0 + _CELL - 1] = goal_color
            frame[y0 + 2 : y0 + _CELL - 2, x0 + 2 : x0 + _CELL - 2] = _MAROON

        # ── Player (drawn last → always visible on top of other tiles) ─────────
        pr, pc = self._player
        _fill(frame, pr, pc, _ROT_COLOR[self._rotation])

        # ── UI strip (rows 56-63 = cell row 7) ────────────────────────────────
        #  Top 4 px (rows 56-59): energy bar (56 cols) + life dots (8 cols)
        #  Bottom 4 px (rows 60-63): current-rotation block (left) vs goal (right)

        energy_frac = max(0.0, self._steps_left / self._level.max_steps)
        filled_px   = int(energy_frac * 56)              # 56 cols for the bar
        frame[_UI_Y : _UI_Y + 4, :filled_px]       = _LIGHT_GREEN
        frame[_UI_Y : _UI_Y + 4, filled_px : 56]   = _ORANGE

        # Life dots: 3 slots of 2 px width at cols 56, 59, 62
        for i in range(3):
            col = 56 + i * 3
            frame[_UI_Y : _UI_Y + 4, col : col + 2] = _CYAN if i < self._lives else _GRAY

        # Rotation indicator: current (left 8 cols) vs goal (right 8 cols)
        frame[_UI_Y + 4 : _UI_Y + 8, : _CELL]            = _ROT_COLOR[self._rotation]
        frame[_UI_Y + 4 : _UI_Y + 8, FRAME_W - _CELL :]  = _ROT_COLOR[self._level.goal_rotation]

        return frame


def _fill(frame: np.ndarray, row: int, col: int, color: int) -> None:
    """Fill the _CELL × _CELL block at grid position (row, col) with color."""
    y, x = row * _CELL, col * _CELL
    frame[y : y + _CELL, x : x + _CELL] = color
