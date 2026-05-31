"""
Generalizable Go-Explore agent for the ARC-AGI-3 (ARC Prize 2026) competition.

SPDX-License-Identifier: MIT-0

Built from the `claude_automate` artifacts (framework/go_explore.py,
framework/exploration.py). Self-contained: depends ONLY on `arc_agi`,
`arcengine`, and `numpy` — no torch, no JEPA, no repo-internal imports — so it
drops straight into a Kaggle notebook.

Strategy (level-agnostic, no per-game knowledge)
------------------------------------------------
ARC-AGI-3's OFFLINE environments are deterministic: replaying a fixed action
sequence from reset() always reproduces the same state. So for each game we:

  1. PLAN on a *private* Arcade instance (its own scorecard, never submitted):
     run multi-level Go-Explore — archive UI-masked frame "cells", return to
     archived cells for free via reset+replay, explore onward, and keep the
     trajectory that completes the most levels. The only goal signal used is
     the universal `levels_completed` counter, so it generalises to unseen
     games. An optional cached warm-start trajectory (for public games we have
     already solved) seeds the archive.

  2. REPLAY the discovered solution on the *scored* Arcade. Only these actions
     hit the official scorecard, so the scored action count ≈ the solution
     length (Relative-Human-Action-Efficiency friendly), not the search cost.

A conservative `direct=True` mode searches on the scored env instead (every
exploration action counts) for harnesses that forbid a private planning copy.

Only games with discrete actions ACTION1..7 are handled; pure coordinate/click
games are skipped (reported as 0, never crash).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── SDK imports (the only hard dependencies) ────────────────────────────────
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

# Map 1-indexed action ids -> GameAction members (GameAction(value) is not
# supported by this enum, so resolve by name).
_AID_TO_ACTION = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}


# ── Minimal game adapter (arc_agi only) ─────────────────────────────────────

class ArcGame:
    """Thin wrapper over one Arcade environment exposing a discrete-action,
    numpy-frame interface plus the universal level/terminal signals."""

    def __init__(self, game_id: str, environments_dir: str,
                 scored: bool = False, arc: Optional[Arcade] = None):
        self.game_id = game_id
        self.arc = arc or Arcade(operation_mode=OperationMode.OFFLINE,
                                 environments_dir=environments_dir)
        self.scorecard_id = self.arc.create_scorecard() if scored else None
        self.env = self.arc.make(game_id, scorecard_id=self.scorecard_id)
        self._raw = None
        self._action_ids: list[int] = []   # fixed 1-indexed action set

    def reset(self) -> np.ndarray:
        self._raw = self.env.reset()
        if not self._action_ids:
            av = list(self._raw.available_actions or [1, 2, 3, 4])
            # keep only discrete ACTION1..7 ids (drop RESET=0 / unknowns)
            self._action_ids = [a for a in av if a in _AID_TO_ACTION] or [1, 2, 3, 4]
        return self._frame()

    def step(self, a_idx: int) -> np.ndarray:
        """a_idx indexes into the fixed action set (0..n_actions-1)."""
        aid = self._action_ids[a_idx]
        self._raw = self.env.step(_AID_TO_ACTION[aid])
        return self._frame()

    def _frame(self) -> np.ndarray:
        return np.asarray(self._raw.frame, dtype=np.uint8)[-1]   # (64, 64)

    @property
    def n_actions(self) -> int:
        return len(self._action_ids)

    @property
    def levels_completed(self) -> int:
        return int(getattr(self._raw, "levels_completed", 0) or 0)

    @property
    def terminal(self) -> bool:
        return self._raw.state in (GameState.WIN, GameState.GAME_OVER)

    @property
    def won(self) -> bool:
        return self._raw.state is GameState.WIN

    def scorecard(self) -> dict:
        return self.arc.get_scorecard(self.scorecard_id).model_dump()


# ── UI-row auto-detection (generalizable masking) ───────────────────────────

def detect_ui_rows(game: ArcGame, probe: int = 24, thresh: float = 0.85,
                   seed: int = 0) -> np.ndarray:
    """Rows that change on (almost) every transition are step-counter / timer
    UI strips. Masking them keeps Go-Explore from splitting every frame into a
    new cell. Pure heuristic — no per-game knowledge."""
    rng = np.random.default_rng(seed)
    prev = game.reset()
    H = prev.shape[0]
    changed = np.zeros(H, dtype=np.float64)
    n = 0
    for _ in range(probe):
        f = game.step(int(rng.integers(game.n_actions)))
        changed += (f != prev).any(axis=1).astype(np.float64)
        prev = f
        n += 1
        if game.terminal:
            prev = game.reset()
    return np.where(changed >= thresh * max(1, n))[0]


# ── Multi-level Go-Explore search (planning) ────────────────────────────────

@dataclass
class Cell:
    trajectory: list[int]
    times_chosen: int = 0


@dataclass
class PlanResult:
    trajectory: list[int]        # best completing/partial action sequence
    levels_completed: int
    won: bool
    archive_size: int
    search_env_steps: int
    elapsed_s: float
    used_cache: bool = False


class MultiLevelGoExplore:
    """Go-Explore that maximises `levels_completed` across a whole game (not
    just the first level), returning the single best trajectory found."""

    def __init__(self, game: ArcGame, ui_rows: np.ndarray, explore_steps: int = 25,
                 seed: int = 0):
        self.game = game
        self.ui_rows = ui_rows
        self.explore_steps = explore_steps
        self.rng = np.random.default_rng(seed)
        self.archive: dict[int, Cell] = {}
        self.env_steps = 0
        self.best_traj: list[int] = []
        self.best_levels = 0
        self.won = False

    def _code(self, frame: np.ndarray) -> int:
        if len(self.ui_rows):
            frame = frame.copy()
            frame[self.ui_rows, :] = 0
        return hash(np.ascontiguousarray(frame).tobytes())

    def _replay(self, traj: list[int]):
        self.game.reset()
        for a in traj:
            self.game.step(a)
            if self.game.terminal:
                return True
        return self.game.terminal

    def _record_if_best(self, traj: list[int]):
        lv = self.game.levels_completed
        if lv > self.best_levels or (lv == self.best_levels and self.game.won and not self.won):
            self.best_levels = lv
            self.best_traj = list(traj)
            self.won = self.game.won

    def seed_cache(self, traj: list[int]):
        """Warm-start: replay a cached trajectory and archive its cells."""
        self.game.reset()
        run: list[int] = []
        self.archive[self._code(self.game._frame())] = Cell(trajectory=[])
        for a in traj:
            self.game.step(a)
            self.env_steps += 1
            run.append(a)
            self._record_if_best(run)
            if self.game.terminal:
                break
            self.archive[self._code(self.game._frame())] = Cell(trajectory=list(run))

    def _sample_cell(self) -> Cell:
        codes = list(self.archive.keys())
        w = np.array([1.0 / np.sqrt(1.0 + self.archive[c].times_chosen) for c in codes])
        w /= w.sum()
        return self.archive[codes[int(self.rng.choice(len(codes), p=w))]]

    def search(self, max_env_steps: int = 300_000, time_budget_s: float = 600.0,
               verbose: bool = False) -> PlanResult:
        t0 = time.time()
        f0 = self.game.reset()
        self.archive.setdefault(self._code(f0), Cell(trajectory=[]))
        n_act = self.game.n_actions

        while (self.env_steps < max_env_steps and not self.won
               and time.time() - t0 < time_budget_s):
            cell = self._sample_cell()
            cell.times_chosen += 1
            if self._replay(cell.trajectory):
                continue                       # dead-end (terminal) cell
            traj = list(cell.trajectory)
            for _ in range(self.explore_steps):
                traj.append(int(self.rng.integers(n_act)))
                self.game.step(traj[-1])
                self.env_steps += 1
                self._record_if_best(traj)
                if self.game.won:
                    break
                if self.game.terminal:
                    break
                code = self._code(self.game._frame())
                known = self.archive.get(code)
                if known is None or len(traj) < len(known.trajectory):
                    self.archive[code] = Cell(trajectory=list(traj))
            if verbose and self.env_steps % 20000 < self.explore_steps:
                print(f"  [plan] steps={self.env_steps} archive={len(self.archive)} "
                      f"best_levels={self.best_levels} {time.time()-t0:.0f}s")

        return PlanResult(
            trajectory=self.best_traj, levels_completed=self.best_levels,
            won=self.won, archive_size=len(self.archive),
            search_env_steps=self.env_steps, elapsed_s=time.time() - t0,
        )


# ── Cached public-game solutions (warm-start) ───────────────────────────────

def load_cached_solutions(path: Optional[str]) -> dict[str, list[int]]:
    """Map base-game-id -> 0-indexed action trajectory. Missing file is fine."""
    if not path or not Path(path).exists():
        return {}
    try:
        return {k: list(v) for k, v in json.load(open(path)).items()}
    except Exception:
        return {}


def _base_id(game_id: str) -> str:
    return game_id.split("-")[0]


# ── Top-level agent ─────────────────────────────────────────────────────────

def solve_game(game_id: str, environments_dir: str,
               cached: Optional[dict[str, list[int]]] = None,
               max_env_steps: int = 300_000, time_budget_s: float = 600.0,
               explore_steps: int = 25, seed: int = 0,
               direct: bool = False, verbose: bool = True) -> dict:
    """Plan a solution for one game, then play it on a scored Arcade.

    Returns a result dict including the official scorecard for this game.
    """
    cached = cached or {}
    t0 = time.time()

    # ── Plan ────────────────────────────────────────────────────────────────
    plan_game = ArcGame(game_id, environments_dir, scored=direct)
    plan_game.reset()
    if plan_game.n_actions == 0:
        return {"game_id": game_id, "skipped": "no discrete actions", "scorecard": None}
    ui_rows = detect_ui_rows(plan_game, seed=seed)
    gx = MultiLevelGoExplore(plan_game, ui_rows, explore_steps=explore_steps, seed=seed)

    warm = cached.get(_base_id(game_id))
    used_cache = False
    if warm:
        gx.seed_cache(warm)
        used_cache = gx.best_levels > 0
        if verbose:
            print(f"[{game_id}] warm-start: cache reached {gx.best_levels} level(s)")

    plan = gx.search(max_env_steps=max_env_steps, time_budget_s=time_budget_s,
                     verbose=verbose)
    plan.used_cache = used_cache
    if verbose:
        print(f"[{game_id}] plan: levels={plan.levels_completed} won={plan.won} "
              f"len={len(plan.trajectory)} search_steps={plan.search_env_steps} "
              f"{plan.elapsed_s:.0f}s")

    # ── Score ───────────────────────────────────────────────────────────────
    if direct:
        # Search already ran on the scored env; the scorecard reflects it.
        scored_game = plan_game
    else:
        scored_game = ArcGame(game_id, environments_dir, scored=True)
        scored_game.reset()
        for a in plan.trajectory:
            scored_game.step(a)
            if scored_game.terminal:
                break

    card = scored_game.scorecard()
    return {
        "game_id": game_id,
        "levels_completed": plan.levels_completed,
        "won": plan.won,
        "solution_length": len(plan.trajectory),
        "search_env_steps": plan.search_env_steps,
        "used_cache": plan.used_cache,
        "wall_seconds": round(time.time() - t0, 1),
        "scored_total_actions": card.get("total_actions"),
        "scored_levels_completed": card.get("total_levels_completed"),
        "scored_score": card.get("score"),
        "scorecard": card,
    }


def run_agent(game_ids: list[str], environments_dir: str,
              cached_solutions_path: Optional[str] = None,
              max_env_steps: int = 300_000, time_budget_s: float = 600.0,
              direct: bool = False, seed: int = 0, verbose: bool = True) -> dict:
    """Run the agent over a list of games. Returns an aggregate report."""
    cached = load_cached_solutions(cached_solutions_path)
    results = []
    for gid in game_ids:
        try:
            results.append(solve_game(
                gid, environments_dir, cached=cached, max_env_steps=max_env_steps,
                time_budget_s=time_budget_s, direct=direct, seed=seed, verbose=verbose))
        except Exception as e:  # never crash the whole submission on one game
            results.append({"game_id": gid, "error": repr(e), "scorecard": None})
            if verbose:
                print(f"[{gid}] ERROR {e!r}")
    scores = [r.get("scored_score") or 0.0 for r in results]
    report = {
        "n_games": len(game_ids),
        "mean_scored_score": float(np.mean(scores)) if scores else 0.0,
        "total_levels_completed": int(sum(r.get("scored_levels_completed") or 0 for r in results)),
        "results": results,
    }
    return report
