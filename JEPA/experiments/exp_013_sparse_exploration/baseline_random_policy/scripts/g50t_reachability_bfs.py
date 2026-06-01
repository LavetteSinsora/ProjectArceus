"""Exact forward-reachability BFS over the REAL g50t engine (analytic cross-check).

The static-lattice maze is not enough for g50t: the goal is gated behind
dynamic elements (buttons that open doors / move boxes). To get the exact
shortest-solution length and to confirm whether the goal is reachable AT ALL
within the per-life timer budget, we explore the reachable state graph by
driving the real engine.

State key = hash of the rendered 64x64 frame (captures player position AND the
full dynamic configuration: doors, boxes, conveyors, buttons). We BFS in
"player-move depth" (number of env actions). We do NOT model the timer here;
instead we report the shortest number of MOVES to first clear, and separately
the fact that a life dies at env-step 130. If shortest-moves > 129 the level is
unsolvable by ANY policy within one life (random or not).

Because cloning engine state is expensive, we re-derive each node by replaying
its action path from reset (paths are short, <= depth bound).

Usage:
    uv run python g50t_reachability_bfs.py --level 0 --max-depth 30
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "claude_automate"))

from claude_automate.framework.env_api import make_arc_env  # noqa: E402

GAME_ID = "g50t-5849a774"
N_ACTIONS = 5  # ACTION1..4 move, ACTION5 undo


def frame_key(env) -> bytes:
    return env._latest_raw_frame_bytes


def _replay(level_index: int, path: list[int]):
    """Build an env, reset, and replay `path`. Return (env, cleared, dead)."""
    env = make_arc_env(GAME_ID, level_index)
    env.reset()
    cleared = dead = False
    for a in path:
        _, term = env.step(a)
        if env.level_completed:
            cleared = True
            break
        if term:
            dead = True
            break
    return env, cleared, dead


def _key(env) -> bytes:
    # Render hash: the masked-out timer row (63) is excluded so the timer's
    # scroll does not explode the state space; everything else defines the
    # dynamic configuration.
    f = np.array(env._latest_raw.frame, dtype=np.uint8)[-1].copy()
    f[63, :] = 0
    return f.tobytes()


def bfs(level_index: int, max_depth: int):
    """BFS over reachable masked-frame states; return shortest moves-to-clear."""
    env0 = make_arc_env(GAME_ID, level_index)
    env0.reset()
    start_key = _key(env0)

    seen = {start_key}
    # queue holds (state_key, path) — path is the action sequence to reach it.
    q = deque([(start_key, [])])
    n_expanded = 0
    shortest_clear = None

    while q:
        key, path = q.popleft()
        if max_depth is not None and len(path) >= max_depth:
            continue
        n_expanded += 1
        for a in range(N_ACTIONS):
            env, cleared, dead = _replay(level_index, path + [a])
            if cleared:
                shortest_clear = len(path) + 1
                return {
                    "level_index": level_index,
                    "shortest_moves_to_clear": shortest_clear,
                    "states_seen": len(seen),
                    "states_expanded": n_expanded,
                    "solution_path": path + [a],
                }
            if dead:
                continue
            k = _key(env)
            if k not in seen:
                seen.add(k)
                q.append((k, path + [a]))
        if n_expanded % 200 == 0:
            print(f"  expanded={n_expanded} seen={len(seen)} frontier={len(q)}", flush=True)

    return {
        "level_index": level_index,
        "shortest_moves_to_clear": None,
        "states_seen": len(seen),
        "states_expanded": n_expanded,
        "note": f"no clear found within depth {max_depth}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=30)
    args = ap.parse_args()
    res = bfs(args.level, args.max_depth)
    print("\n=== g50t reachability BFS ===")
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
