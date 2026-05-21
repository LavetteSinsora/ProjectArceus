"""Go-Explore structured exploration (Ecoffet et al. 2021).

The generalizable answer to *detachment* — the failure mode where count-based
exploration with a neural policy covers the easy region and then has no
gradient to reach the hard frontier. Go-Explore sidesteps it by explicitly
remembering how to get back to every discovered state and always exploring
*from* the frontier.

It exploits a deterministic environment: replaying a fixed action sequence
from `reset()` always reproduces the same state, so any archived state can be
re-reached exactly, for free, with no policy.

Game-agnostic: a "cell" is just a hash of the UI-masked observation; the only
goal signal used is the universal `level_completed` flag.

Algorithm
---------
archive: cell_code -> Cell(trajectory of actions from reset that reaches it)
repeat:
    pick an archived cell (prefer cells chosen fewer times)
    new env; replay the cell's trajectory to return there
    take a burst of random actions; archive every new / shorter-reached cell
    if a `level_completed` transition occurs -> return that full trajectory
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cell:
    trajectory: list[int]          # actions from env.reset() that reach this cell
    times_chosen: int = 0
    depth: int = 0                 # len(trajectory) when first discovered


@dataclass
class SearchResult:
    solution: list[int] | None     # completing action sequence, or None
    archive_size: int
    env_steps: int
    iterations: int
    elapsed_s: float
    history: list = field(default_factory=list)   # (env_steps, archive_size)


class GoExplore:
    """Go-Explore search over a deterministic ARC environment.

    A single `env` instance is reused for the whole search: `env.reset()` is a
    full deterministic reset, so returning to any cell is just reset + replay.
    `masked_rows` is the env's UI-row slice, masked out before hashing so the
    step counter does not split every frame into its own cell.
    """

    def __init__(self, env, masked_rows=None, explore_steps: int = 25,
                 seed: int = 0):
        self.env = env
        self.masked_rows = masked_rows
        self.explore_steps = explore_steps
        self.rng = np.random.default_rng(seed)
        self.archive: dict[int, Cell] = {}
        self.env_steps = 0

    # ── cell hashing ─────────────────────────────────────────────────────────

    def cell_code(self, frame: np.ndarray) -> int:
        f = np.asarray(frame, dtype=np.uint8)
        if self.masked_rows is not None:
            f = f.copy()
            f[self.masked_rows, :] = 0
        return hash(np.ascontiguousarray(f).tobytes())

    # ── trajectory replay (deterministic env) ────────────────────────────────

    def _replay(self, trajectory: list[int]):
        """Reset the env and replay `trajectory`; return (frame, terminal)."""
        frame = self.env.reset()
        terminal = False
        for a in trajectory:
            frame, terminal = self.env.step(a)
            if terminal:
                break
        return frame, terminal

    # ── cell selection ───────────────────────────────────────────────────────

    def _sample_cell(self):
        """Pick a cell, biased toward cells chosen fewer times (the frontier)."""
        codes = list(self.archive.keys())
        weights = np.array(
            [1.0 / np.sqrt(1.0 + self.archive[c].times_chosen) for c in codes]
        )
        weights /= weights.sum()
        idx = int(self.rng.choice(len(codes), p=weights))
        return codes[idx], self.archive[codes[idx]]

    # ── main search ──────────────────────────────────────────────────────────

    def search(self, max_env_steps: int = 500_000,
               log_every: int = 20_000, verbose: bool = True) -> SearchResult:
        t0 = time.time()
        f0 = self.env.reset()
        self.archive[self.cell_code(f0)] = Cell(trajectory=[], depth=0)
        n_actions = self.env.n_actions

        solution = None
        iterations = 0
        history = []
        next_log = log_every

        while self.env_steps < max_env_steps and solution is None:
            iterations += 1
            _, cell = self._sample_cell()
            cell.times_chosen += 1

            frame, terminal = self._replay(cell.trajectory)
            if terminal:
                continue                       # dead-end cell — can't explore on

            traj = list(cell.trajectory)
            for _ in range(self.explore_steps):
                a = int(self.rng.integers(n_actions))
                frame, terminal = self.env.step(a)
                traj.append(a)
                self.env_steps += 1

                if self.env.level_completed:
                    solution = list(traj)
                    break

                code = self.cell_code(frame)
                known = self.archive.get(code)
                if known is None or len(traj) < len(known.trajectory):
                    self.archive[code] = Cell(trajectory=list(traj),
                                              depth=len(traj))
                if terminal:
                    break

            if verbose and self.env_steps >= next_log:
                print(f"[go-explore] env_steps={self.env_steps} "
                      f"archive={len(self.archive)} iters={iterations} "
                      f"elapsed={time.time()-t0:.0f}s")
                history.append((self.env_steps, len(self.archive)))
                next_log += log_every

        result = SearchResult(
            solution=solution, archive_size=len(self.archive),
            env_steps=self.env_steps, iterations=iterations,
            elapsed_s=time.time() - t0, history=history,
        )
        if verbose:
            status = (f"SOLVED in {len(solution)} actions"
                      if solution else "no solution found")
            print(f"[go-explore] {status} | archive={result.archive_size} "
                  f"env_steps={result.env_steps} elapsed={result.elapsed_s:.0f}s")
        return result


def collect_trajectory_frames(env, trajectory: list[int]):
    """Replay `trajectory` on `env` and return the (frame, action) pairs.

    Returns (frames, actions): `frames[i]` is the observation at which
    `actions[i]` was taken. Used to build the behavior-cloning dataset.
    """
    frame = env.reset()
    frames, actions = [], []
    for a in trajectory:
        frames.append(np.asarray(frame, dtype=np.uint8))
        actions.append(int(a))
        frame, terminal = env.step(a)
        if terminal:
            break
    return frames, actions
