"""MiniLS20Env — pure-numpy mini version of LS20 Level 1.

Attribute surface (matches JEPA/shared/env_wrapper.BaseArcEnv minus arcengine):

    env.reset() -> np.ndarray              # (32, 32) uint8
    env.step(action_idx) -> (frame, terminal_bool)
    env.n_actions: int                     # 4
    env.available_actions: list[int]       # [1, 2, 3, 4]
    env.level_completed: bool
    env.won: bool
    env._MASKED_ROWS: slice                # slice(28, 32)
    env.frame_diff(f0, f1)
    env.patch_weights(f0, f1)              # (16,) float32 in [0, 1] — 4x4 grid of 8x8 patches
    env.detect_moved_cell(f0, f1)          # 0..63 cell index in 8x8 fine grid of 4x4 px cells

Game rules — faithful to the LOCKED step() block in the build spec:

    ACTION1..4 = up, down, left, right
    Wall blocks → reduce denial_frames, terminal=False
    Goal gated → if rotations mismatch, set denial_frames=5, terminal=False
    Goal entry on matching rotation → won=True, terminal=True
    Cross → step onto it, rotation += 90 (mod 360), terminal=False
    Empty cell → step in, terminal=False
    step_counter reaches 0 → terminal=True (won stays False)
"""

from __future__ import annotations

import numpy as np

from mini_env.loader import EnvConfig, load_level
from mini_env.renderer import render
from mini_env.state import EnvState


# 0=up, 1=down, 2=left, 3=right  (1-indexed [1,2,3,4] for available_actions)
ACTION_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}

CELL_EMPTY = 0
CELL_WALL = 1
CELL_GOAL = 2
CELL_CROSS = 3


class MiniLS20Env:
    """Pure-numpy mini-LS20 with 32x32 pixel frames."""

    n_actions: int = 4
    _MASKED_ROWS: slice = slice(28, 32)  # UI strip rows in the 32x32 frame

    def __init__(self, level_path: str):
        self.config: EnvConfig = load_level(level_path)
        self.cols = self.config.cols
        self.play_rows = self.config.play_rows
        self.goal_rotation = self.config.goal_rotation

        # Build a (play_rows, cols) cell-type grid used by step().
        self.grid = np.zeros((self.play_rows, self.cols), dtype=np.int8)
        for (c, r) in self.config.walls:
            self.grid[r, c] = CELL_WALL
        gc, gr = self.config.goal.col, self.config.goal.row
        self.grid[gr, gc] = CELL_GOAL
        xc, xr = self.config.cross.col, self.config.cross.row
        self.grid[xr, xc] = CELL_CROSS

        self.state: EnvState | None = None
        # Public mirror attributes for ergonomic access (matches the spec
        # interface: env.player_c, env.player_r, env.step_counter, etc.)
        self.player_c = 0
        self.player_r = 0
        self.player_rotation = 0
        self.step_counter = 0
        self.denial_frames = 0
        self.won = False

        self.reset()

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def available_actions(self) -> list[int]:
        # 1-indexed action ids, mirroring arcengine GameAction values.
        return [1, 2, 3, 4]

    @property
    def level_completed(self) -> bool:
        return self.won

    def reset(self) -> np.ndarray:
        self.player_c = self.config.player_start.col
        self.player_r = self.config.player_start.row
        self.player_rotation = self.config.player_rotation
        self.step_counter = self.config.step_limit
        self.denial_frames = 0
        self.won = False
        self.state = EnvState(
            player_c=self.player_c,
            player_r=self.player_r,
            player_rotation=self.player_rotation,
            step_counter=self.step_counter,
            denial_frames=self.denial_frames,
            won=self.won,
        )
        return self._render()

    def step(self, action_idx: int) -> tuple[np.ndarray, bool]:
        if action_idx not in ACTION_DELTAS:
            raise ValueError(f"invalid action_idx {action_idx}; must be in 0..3")

        # 1. Decrement step counter; episode terminates if energy depleted.
        self.step_counter -= 1
        if self.step_counter <= 0:
            self._sync_state()
            return self._render(), True

        dc, dr = ACTION_DELTAS[action_idx]
        dest_c, dest_r = self.player_c + dc, self.player_r + dr

        # 2. Out-of-bounds = wall.
        if not (0 <= dest_c < self.cols and 0 <= dest_r < self.play_rows):
            if self.denial_frames > 0:
                self.denial_frames -= 1
            self._sync_state()
            return self._render(), False

        cell = int(self.grid[dest_r, dest_c])

        # 3. Wall blocks.
        if cell == CELL_WALL:
            if self.denial_frames > 0:
                self.denial_frames -= 1
            self._sync_state()
            return self._render(), False

        # 4. Goal: gated by rotation match.
        if cell == CELL_GOAL:
            if self.config.goal_gated and self.player_rotation != self.goal_rotation:
                self.denial_frames = 5
                self._sync_state()
                return self._render(), False
            self.player_c, self.player_r = dest_c, dest_r
            self.won = True
            self._sync_state()
            return self._render(), True

        # 5. Cross: step in, rotate +90.
        if cell == CELL_CROSS:
            self.player_c, self.player_r = dest_c, dest_r
            self.player_rotation = (self.player_rotation + 90) % 360
            if self.denial_frames > 0:
                self.denial_frames -= 1
            self._sync_state()
            return self._render(), False

        # 6. Empty cell.
        self.player_c, self.player_r = dest_c, dest_r
        if self.denial_frames > 0:
            self.denial_frames -= 1
        self._sync_state()
        return self._render(), False

    # ── Mask-aware diff utilities (adapted from JEPA env_wrapper for 32x32) ──

    def frame_diff(self, f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
        """Pixel-wise absolute difference; UI strip rows zeroed."""
        d = np.abs(f1.astype(np.float32) - f0.astype(np.float32))
        if self._MASKED_ROWS is not None:
            d[self._MASKED_ROWS, :] = 0.0
        return d

    def patch_weights(self, f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
        """(32,32) frames → (16,) patch change weights in [0, 1].

        4x4 grid of 8x8 patches. Mean over each patch, normalised to [0, 1].
        """
        pw = self.frame_diff(f0, f1).reshape(4, 8, 4, 8).mean(axis=(1, 3)).flatten()
        m = float(pw.max())
        return (pw / m) if m > 1e-8 else np.zeros(16, dtype=np.float32)

    def detect_moved_cell(self, f0: np.ndarray, f1: np.ndarray) -> int | None:
        """Return the 8x8 fine-grid cell index (0–63) with the most masked change.

        Returns None if no significant change detected (threshold < 2.0).
        Each cell is 4x4 pixels.
        """
        cell_diff = self.frame_diff(f0, f1).reshape(8, 4, 8, 4).sum(axis=(1, 3))
        if cell_diff.max() < 2.0:
            return None
        r, c = np.unravel_index(cell_diff.argmax(), cell_diff.shape)
        return int(r * 8 + c)

    # ── Private ──────────────────────────────────────────────────────────────

    def _sync_state(self) -> None:
        """Mirror env-level fields into the EnvState dataclass used by the renderer."""
        self.state = EnvState(
            player_c=self.player_c,
            player_r=self.player_r,
            player_rotation=self.player_rotation,
            step_counter=self.step_counter,
            denial_frames=self.denial_frames,
            won=self.won,
        )

    def _render(self) -> np.ndarray:
        if self.state is None:
            self._sync_state()
        return render(self.state, self.config)
