"""Mutable per-episode state for MiniLS20Env.

The level layout (walls, goal cell, rotations, palette, step_limit) lives in
EnvConfig (immutable per episode). EnvState only carries what mutates during
play.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvState:
    player_c: int
    player_r: int
    player_rotation: int     # 0, 90, 180, or 270
    step_counter: int        # remaining energy; counts down to 0
    denial_frames: int = 0   # frames remaining for the "denial flash" preview tile
    won: bool = False
