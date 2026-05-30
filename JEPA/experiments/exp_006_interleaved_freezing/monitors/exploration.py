"""
Exploration metrics (metrics.md §6.2 / §6.3).

  - reachable_tile_coverage_pct: |V ∩ R| / |R| where V is the set of visited
    tile coordinates in an episode and R is the precomputed reachable set.
  - cross_hits_per_episode:      number of times the agent stepped onto a
    cross tile (the level's central mechanic).

Both metrics depend on a per-game JSON manifest:
    monitors/exploration_manifests/<game_id>.json
    {
      "reachable_tiles": [[r, c], ...] | null,
      "cross_tiles":     [[r, c], ...] | null,
      "tile_size":       int,
      "agent_color":     int | null    (optional: speeds up position derivation)
    }

If the manifest is missing or its fields are null, the metrics return nan
and a warning is printed *once*. The manifest must be filled in via a
separate scripted BFS / manual inspection pass (out of scope here).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

_MANIFEST_DIR = Path(__file__).parent / "exploration_manifests"
_WARNED: set = set()


def _load_manifest(game_id: str) -> dict:
    path = _MANIFEST_DIR / f"{game_id}.json"
    if not path.exists():
        if game_id not in _WARNED:
            print(f"[exploration] No manifest at {path} — metrics will be nan")
            _WARNED.add(game_id)
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as e:
        print(f"[exploration] Failed to parse {path}: {e}")
        return {}


def _tile_of(pos_rc: tuple, tile_size: int) -> tuple:
    r, c = pos_rc
    return (r // tile_size, c // tile_size)


def derive_agent_position(frame: np.ndarray, agent_color: int | None) -> tuple | None:
    """
    Locate the agent in a (64, 64) uint8 frame by its color index. Returns
    (row, col) of the centroid of matching pixels, or None if none found.

    LS20Env does not currently expose agent position directly, so we use a
    color heuristic. If `agent_color` is not provided in the manifest, the
    function returns None and downstream metrics are nan.
    """
    if agent_color is None:
        return None
    ys, xs = np.where(frame == agent_color)
    if len(ys) == 0:
        return None
    return (int(ys.mean()), int(xs.mean()))


class ExplorationTracker:
    """Per-episode tracker — accumulate visited tiles + cross hits."""

    def __init__(self, game_id: str):
        self.manifest = _load_manifest(game_id)
        self.tile_size = int(self.manifest.get("tile_size", 8))
        self.agent_color = self.manifest.get("agent_color")
        reach = self.manifest.get("reachable_tiles")
        crosses = self.manifest.get("cross_tiles")
        self._reachable = {tuple(t) for t in reach} if reach else None
        self._cross = {tuple(t) for t in crosses} if crosses else None
        self.reset()

    def reset(self) -> None:
        self._visited: set = set()
        self._cross_hits: int = 0

    def step(self, frame: np.ndarray) -> None:
        pos = derive_agent_position(frame, self.agent_color)
        if pos is None:
            return
        tile = _tile_of(pos, self.tile_size)
        self._visited.add(tile)
        if self._cross is not None and tile in self._cross:
            self._cross_hits += 1

    def coverage_pct(self) -> float:
        if self._reachable is None or not self._reachable:
            return float("nan")
        return len(self._visited & self._reachable) / len(self._reachable)

    def cross_hits(self) -> int | float:
        if self._cross is None:
            return float("nan")
        return self._cross_hits
