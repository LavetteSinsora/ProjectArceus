"""exp_019 data collector.

Collects (frame, action, next_frame, terminal, completed) transitions from the
offline ARC env. Sources per level:
  - solution replay (cached Go-Explore solutions; supplies completion + modifier
    transitions),
  - random rollouts,
  - burst coverage: replay a random prefix of the solution, then random burst
    (reaches deeper states than pure random — Go-Explore-style coverage without
    re-running the search).

Run:  python3 collect_data.py <level_index0> <n_random_episodes> <out.npz>
Honest accounting: every env.reset() and env.step() is counted and saved.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "claude_automate"))

from framework.env_api import make_arc_env  # noqa: E402

SOLUTIONS = {
    0: _REPO / "claude_automate/experiments/solve_ls20_L1_20260518_104607/solution.json",
    1: _REPO / "claude_automate/experiments/solve_ls20_L2_20260518_104532/solution.json",
    2: _REPO / "claude_automate/experiments/solve_ls20_L3_20260518_110708/solution.json",
    3: _REPO / "claude_automate/experiments/solve_ls20_L4_20260518_110710/solution.json",
}


def load_solution(level: int) -> list[int]:
    d = json.load(open(SOLUTIONS[level]))
    return d["trajectory"] if isinstance(d, dict) else list(d)


class Recorder:
    def __init__(self):
        self.frames, self.actions, self.next_frames = [], [], []
        self.terminals, self.completeds = [], []
        self.real_steps = 0
        self.real_resets = 0

    def episode(self, env, actions, rng=None, random_after=None, max_steps=200):
        """Replay `actions`; if random_after is not None, continue with random
        actions up to max_steps after the scripted prefix."""
        obs = env.reset()
        self.real_resets += 1
        t = 0
        script = list(actions)
        while t < max_steps:
            if t < len(script):
                a = script[t]
            elif random_after is not None:
                a = int(rng.integers(0, 4))
            else:
                break
            nxt, term = env.step(a)
            self.real_steps += 1
            comp = bool(env.level_completed)
            self.frames.append(obs.copy())
            self.actions.append(a)
            self.next_frames.append(nxt.copy())
            self.terminals.append(bool(term))
            self.completeds.append(comp)
            obs = nxt
            t += 1
            if term:
                break
        return t

    def save(self, path):
        np.savez_compressed(
            path,
            frames=np.stack(self.frames).astype(np.uint8),
            actions=np.array(self.actions, dtype=np.int64),
            next_frames=np.stack(self.next_frames).astype(np.uint8),
            terminals=np.array(self.terminals, dtype=bool),
            completeds=np.array(self.completeds, dtype=bool),
            real_steps=self.real_steps,
            real_resets=self.real_resets,
        )


def main():
    level = int(sys.argv[1])
    n_random = int(sys.argv[2])
    out = sys.argv[3]
    t0 = time.time()
    rng = np.random.default_rng(level * 1000 + n_random)
    env = make_arc_env("ls20-9607627b", level_index=level)
    rec = Recorder()
    sol = load_solution(level)

    # 1) pure solution replay (twice: completion transitions are precious)
    rec.episode(env, sol)
    rec.episode(env, sol)
    # 2) random rollouts
    for _ in range(n_random):
        rec.episode(env, [], rng=rng, random_after=True)
    # 3) burst coverage from solution prefixes
    n_prefix = max(4, n_random // 2)
    for i in range(n_prefix):
        k = int(rng.integers(1, len(sol)))
        rec.episode(env, sol[:k], rng=rng, random_after=True)

    rec.save(out)
    n = len(rec.actions)
    ch = [int((rec.frames[i] != rec.next_frames[i]).sum()) for i in range(n)]
    print(json.dumps({
        "level0": level, "transitions": n,
        "real_steps": rec.real_steps, "real_resets": rec.real_resets,
        "completions": int(np.sum(rec.completeds)),
        "terminals": int(np.sum(rec.terminals)),
        "changed_cells_mean": float(np.mean(ch)),
        "changed_cells_max": int(np.max(ch)),
        "secs": round(time.time() - t0, 1),
    }))


if __name__ == "__main__":
    main()
