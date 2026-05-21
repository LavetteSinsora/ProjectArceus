"""
probe_env.py — read-only environment reconnaissance for LS20 Level 1.

Verifies the environment contract the training framework will rely on:
  - frame shape / dtype / value range
  - episode length under a random policy
  - terminal + level_completed behaviour
  - how often a random policy completes the level (baseline)
  - which UI rows change every step (mask sanity check)

Does NOT train anything and does NOT modify any repo file. Run:
    cd "Code Repo"
    uv run python claude_automate/probe_env.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from JEPA.shared.env_wrapper import make_env  # noqa: E402


def build_env():
    from arc_agi import Arcade, OperationMode

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_REPO_ROOT / "environment_files"),
    )
    raw = arc.make("ls20-9607627b")
    return make_env(raw, "ls20-9607627b")


def run_random_episode(env, rng, max_steps=400):
    frame = env.reset()
    frames = [frame]
    steps = 0
    completed = False
    for _ in range(max_steps):
        a = int(rng.integers(env.n_actions))
        frame, terminal = env.step(a)
        frames.append(frame)
        steps += 1
        if env.level_completed:
            completed = True
        if terminal:
            break
    return frames, steps, completed, env.level_completed, env.won


def main():
    env = build_env()
    rng = np.random.default_rng(0)

    f0 = env.reset()
    print(f"frame: shape={f0.shape} dtype={f0.dtype} "
          f"min={f0.min()} max={f0.max()} unique={sorted(np.unique(f0).tolist())}")
    print(f"n_actions={env.n_actions}  available_actions={env.available_actions}")

    # Which rows change every step? (UI / step-counter detection)
    prev = env.reset()
    changed_rows = np.zeros(64, dtype=int)
    n_probe = 30
    for _ in range(n_probe):
        cur, _ = env.step(int(rng.integers(env.n_actions)))
        row_changed = (cur != prev).any(axis=1)
        changed_rows += row_changed.astype(int)
        prev = cur
    always_change = [r for r in range(64) if changed_rows[r] == n_probe]
    print(f"rows changing on all {n_probe} steps (UI rows): {always_change}")

    # Random-policy episode statistics
    lengths, n_complete, n_term = [], 0, Counter()
    n_ep = 25
    for i in range(n_ep):
        _, steps, completed, lvl, won = run_random_episode(env, rng)
        lengths.append(steps)
        n_complete += int(completed)
        n_term[("won" if won else "completed" if lvl else "dead")] += 1
    print(f"\nrandom policy over {n_ep} episodes:")
    print(f"  episode length: min={min(lengths)} max={max(lengths)} "
          f"mean={np.mean(lengths):.1f}")
    print(f"  level completed: {n_complete}/{n_ep} "
          f"({100*n_complete/n_ep:.0f}%)")
    print(f"  terminal kinds: {dict(n_term)}")


if __name__ == "__main__":
    main()
