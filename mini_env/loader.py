"""Load and validate a JSON level definition for MiniLS20Env.

Schema (LOCKED — see mini_env/README.md):

    {
      "name": str,
      "grid_cells": [cols, rows],          # cols=8, rows=8 (last row = UI strip)
      "tile_px": int,                       # 4
      "step_limit": int,
      "palette": {bg, wall, player, goal_frame, cross,
                  preview_bg, highlight, energy, denial_flash},
      "walls": [[col, row], ...],
      "player_start": {"cell": [col, row], "rotation": int},
      "goal":         {"cell": [col, row], "rotation": int},
      "cross":        {"cell": [col, row]},
      "goal_gated":   bool,
      "show_match_cue": bool
    }

Coordinates: (col, row), zero-indexed cells. col in [0, cols),
row in [0, rows - 1)  -- the bottom row of cells is the UI strip and is NOT
addressable by walls/player/goal/cross.

The loader runs a BFS reachability check from the player start to ensure the
cross and the goal are not boxed in by walls.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


VALID_ROTATIONS = (0, 90, 180, 270)


@dataclass
class CellRef:
    col: int
    row: int

    @property
    def tuple(self) -> tuple[int, int]:
        return (self.col, self.row)


@dataclass
class EnvConfig:
    name: str
    cols: int
    rows: int                  # total rows including UI strip
    play_rows: int             # playable rows = rows - 1
    tile_px: int
    step_limit: int
    palette: dict
    walls: list[tuple[int, int]]
    player_start: CellRef
    player_rotation: int
    goal: CellRef
    goal_rotation: int
    cross: CellRef
    goal_gated: bool
    show_match_cue: bool
    source_path: str = ""

    # Convenient set for O(1) wall lookup.
    walls_set: frozenset[tuple[int, int]] = field(default_factory=frozenset)


def _validate_cell(label: str, cell: list, cols: int, play_rows: int) -> CellRef:
    if not (isinstance(cell, list) and len(cell) == 2):
        raise ValueError(f"{label}.cell must be [col, row], got {cell!r}")
    c, r = int(cell[0]), int(cell[1])
    if not (0 <= c < cols):
        raise ValueError(f"{label}.cell col={c} out of bounds [0, {cols})")
    if not (0 <= r < play_rows):
        raise ValueError(f"{label}.cell row={r} out of bounds [0, {play_rows}) "
                         f"(bottom row is reserved for the UI strip)")
    return CellRef(c, r)


def _validate_rotation(label: str, rot: int) -> int:
    rot = int(rot)
    if rot not in VALID_ROTATIONS:
        raise ValueError(f"{label}.rotation={rot} must be one of {VALID_ROTATIONS}")
    return rot


def _bfs_reachable(start: tuple[int, int],
                   walls: frozenset[tuple[int, int]],
                   cols: int, play_rows: int) -> set[tuple[int, int]]:
    seen = {start}
    q = deque([start])
    while q:
        c, r = q.popleft()
        for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nc, nr = c + dc, r + dr
            if not (0 <= nc < cols and 0 <= nr < play_rows):
                continue
            if (nc, nr) in walls or (nc, nr) in seen:
                continue
            seen.add((nc, nr))
            q.append((nc, nr))
    return seen


def load_level(path: str | Path) -> EnvConfig:
    """Read and validate a level JSON; return an EnvConfig."""
    p = Path(path)
    data = json.loads(p.read_text())

    name = str(data["name"])
    grid_cells = data["grid_cells"]
    if not (isinstance(grid_cells, list) and len(grid_cells) == 2):
        raise ValueError(f"grid_cells must be [cols, rows], got {grid_cells!r}")
    cols, rows = int(grid_cells[0]), int(grid_cells[1])
    play_rows = rows - 1  # bottom row reserved for UI strip
    tile_px = int(data["tile_px"])

    # Pixel size must equal 32x32 for the wrapper contract to hold.
    if cols * tile_px != 32 or rows * tile_px != 32:
        raise ValueError(
            f"grid_cells * tile_px must equal 32x32, "
            f"got {cols}*{tile_px}={cols * tile_px} x {rows}*{tile_px}={rows * tile_px}"
        )

    step_limit = int(data["step_limit"])
    if step_limit <= 0:
        raise ValueError(f"step_limit must be positive, got {step_limit}")

    palette = dict(data["palette"])
    required_palette_keys = {"bg", "wall", "player", "goal_frame", "cross",
                             "preview_bg", "highlight", "energy", "denial_flash"}
    missing = required_palette_keys - set(palette.keys())
    if missing:
        raise ValueError(f"palette missing keys: {sorted(missing)}")

    walls_raw = data.get("walls", [])
    walls = []
    for w in walls_raw:
        if not (isinstance(w, list) and len(w) == 2):
            raise ValueError(f"wall entry must be [col, row], got {w!r}")
        wc, wr = int(w[0]), int(w[1])
        if not (0 <= wc < cols and 0 <= wr < play_rows):
            raise ValueError(f"wall {(wc, wr)} out of playfield bounds")
        walls.append((wc, wr))
    walls_set = frozenset(walls)

    player_start = _validate_cell("player_start", data["player_start"]["cell"],
                                  cols, play_rows)
    player_rotation = _validate_rotation("player_start",
                                         data["player_start"]["rotation"])
    goal = _validate_cell("goal", data["goal"]["cell"], cols, play_rows)
    goal_rotation = _validate_rotation("goal", data["goal"]["rotation"])
    cross = _validate_cell("cross", data["cross"]["cell"], cols, play_rows)

    # Walls must not coincide with player / goal / cross.
    for label, ref in (("player_start", player_start),
                       ("goal", goal), ("cross", cross)):
        if ref.tuple in walls_set:
            raise ValueError(f"{label} cell {ref.tuple} collides with a wall")

    # Reachability: cross and goal must be reachable from the player start.
    reach = _bfs_reachable(player_start.tuple, walls_set, cols, play_rows)
    if cross.tuple not in reach:
        raise ValueError(f"cross at {cross.tuple} is unreachable from player_start "
                         f"{player_start.tuple} given the walls")
    if goal.tuple not in reach:
        raise ValueError(f"goal at {goal.tuple} is unreachable from player_start "
                         f"{player_start.tuple} given the walls")

    return EnvConfig(
        name=name,
        cols=cols, rows=rows, play_rows=play_rows, tile_px=tile_px,
        step_limit=step_limit, palette=palette,
        walls=walls, walls_set=walls_set,
        player_start=player_start, player_rotation=player_rotation,
        goal=goal, goal_rotation=goal_rotation,
        cross=cross,
        goal_gated=bool(data.get("goal_gated", True)),
        show_match_cue=bool(data.get("show_match_cue", True)),
        source_path=str(p),
    )
