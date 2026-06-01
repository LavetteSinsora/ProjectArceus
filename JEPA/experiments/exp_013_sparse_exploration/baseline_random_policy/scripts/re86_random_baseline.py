"""Random-policy exploration baseline for ARC-AGI-3 game re86 (re86-8af5384d).

For each level we run a uniform-random policy (each of the 5 legal actions with
prob 1/5) on the REAL arcengine and record env-steps to the first positive
extrinsic reward (= level clear, i.e. `levels_completed` increments / WIN).

re86 mechanics (parsed from environment_files/re86/8af5384d/re86.py):
  - 5 actions: ACTION1-4 move the SELECTED cross/plus piece by 3px
    (up / down / left / right), ACTION5 switches which piece is selected.
  - Every action (move, blocked move, or switch) decrements a per-life step
    budget ("StepCounter", shown as the row-63 bar). At 0 -> GAME_OVER.
  - ONE life: lose() sets GameState.GAME_OVER with no respawn.
  - Step budget per level: L1=100, L2=100, L3=200.
  - Win (jeiavrvavi): the movable pieces, when stamped onto the canvas, must
    exactly reproduce the hidden target template; on match next_level() fires
    and score (= levels_completed) increments.

Because each life is one episode (no respawn), per-life p == per-episode p.

Usage:
    uv run python JEPA/experiments/exp_013_sparse_exploration/baseline_random_policy/scripts/re86_random_baseline.py \
        --levels 0 1 2 --lives 20000 --seed 0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from claude_automate.framework.env_api import make_arc_env  # noqa: E402

GAME_ID = "re86-8af5384d"
N_ACTIONS = 5
# Per-level step budget per life (= "StepCounter" in re86.py levels list).
BUDGET = {0: 100, 1: 100, 2: 200}


def _latest_raw(env):
    return env._base._latest_raw if hasattr(env, "_base") else env._latest_raw


def _run_chunk(args):
    """Worker: run a chunk of lives on one level with its own seed. Returns
    (wins, total_steps, steps_to_win)."""
    level_index, n_lives, seed = args
    rng = np.random.default_rng(seed)
    res = run_level(level_index, n_lives, rng, verbose=False)
    return res["wins"], res["total_steps"], res["steps_to_win"]


def run_level_parallel(level_index: int, n_lives: int, base_seed: int, workers: int):
    """Split n_lives across `workers` processes; aggregate results."""
    per = [n_lives // workers] * workers
    for i in range(n_lives - sum(per)):
        per[i] += 1
    tasks = [(level_index, per[w], base_seed * 100003 + level_index * 9973 + w)
             for w in range(workers) if per[w] > 0]
    with Pool(processes=len(tasks)) as pool:
        chunks = pool.map(_run_chunk, tasks)
    wins = sum(c[0] for c in chunks)
    total_steps = sum(c[1] for c in chunks)
    steps_to_win = [s for c in chunks for s in c[2]]
    return {
        "level_index": level_index,
        "budget": BUDGET.get(level_index),
        "n_lives": n_lives,
        "wins": wins,
        "total_steps": total_steps,
        "steps_to_win": steps_to_win,
    }


def run_level(level_index: int, n_lives: int, rng: np.random.Generator, verbose: bool = True):
    """Run n_lives uniform-random episodes on `level_index`.

    Returns dict with: wins, total_steps, steps_to_win list, budget.
    Each life = one episode (game ends at GAME_OVER or WIN); per-life p == per-episode p.
    """
    env = make_arc_env(GAME_ID, level_index)
    wins = 0
    total_steps = 0
    steps_to_win: list[int] = []

    for _ in range(n_lives):
        env.reset()
        life_steps = 0
        won = False
        while True:
            a = int(rng.integers(0, N_ACTIONS))
            _, term = env.step(a)
            life_steps += 1
            total_steps += 1
            if term:
                raw = _latest_raw(env)
                # Success = a level was cleared (levels_completed incremented).
                if getattr(raw, "levels_completed", 0) >= 1 or env.level_completed:
                    won = True
                break
            # Safety: should always terminate by budget exhaustion.
            if life_steps > BUDGET.get(level_index, 1000) + 50:
                break
        if won:
            wins += 1
            steps_to_win.append(life_steps)

    return {
        "level_index": level_index,
        "budget": BUDGET.get(level_index),
        "n_lives": n_lives,
        "wins": wins,
        "total_steps": total_steps,
        "steps_to_win": steps_to_win,
    }


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def expected_steps(p_life: float, budget: int):
    """E[env steps to first reward] for a geometric process where each
    failed life costs `budget` steps and a winning life costs (on average)
    some fraction of the budget. We use the conservative/standard estimate:

        E[steps] ~= budget / p_life

    (each life is ~budget steps; the number of lives until first win is
    geometric with mean 1/p_life). This matches the LS20 convention of
    reporting total env steps summed across actors.
    """
    if p_life <= 0:
        return math.inf
    return budget / p_life


def report_level(res: dict):
    n = res["n_lives"]
    k = res["wins"]
    budget = res["budget"]
    p = k / n if n else 0.0
    lo, hi = wilson_ci(k, n)
    print(f"\n=== re86 level_index={res['level_index']} (L{res['level_index']+1}) ===")
    print(f"  budget/life = {budget} steps,  lives run = {n},  total env steps = {res['total_steps']:,}")
    print(f"  wins = {k}")
    if k > 0:
        sw = np.array(res["steps_to_win"])
        print(f"  steps-to-win: mean={sw.mean():.1f} min={sw.min()} max={sw.max()}")
        print(f"  per-life p = per-episode p = {p:.3e}  (Wilson 95% CI [{lo:.3e}, {hi:.3e}])")
        print(f"  E[env steps to first reward] ~= budget/p = {expected_steps(p, budget):,.0f}")
    else:
        # Censored: no wins observed. Report upper bound on p (Wilson hi),
        # hence lower bound on E[steps].
        print(f"  per-life p: 0 observed in {n} lives.")
        print(f"  Wilson 95% upper bound on p <= {hi:.3e}")
        print(f"  => E[env steps to first reward] >= budget/p_hi = {expected_steps(hi, budget):,.0f}")
        # Rule-of-three quick check.
        p3 = 3.0 / n
        print(f"  (rule-of-three: p <= {p3:.3e} => E >= {expected_steps(p3, budget):,.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lives", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    results = []
    for lvl in args.levels:
        t0 = time.time()
        if args.workers > 1:
            res = run_level_parallel(lvl, args.lives, args.seed, args.workers)
        else:
            res = run_level(lvl, args.lives, np.random.default_rng(args.seed))
        res["wall_s"] = time.time() - t0
        results.append(res)
        report_level(res)
        print(f"  (wall {res['wall_s']:.1f}s)")

    print("\n--- SUMMARY (E[env steps to first reward], uniform-random, n_actions=5) ---")
    for res in results:
        n, k, budget = res["n_lives"], res["wins"], res["budget"]
        if k > 0:
            p = k / n
            print(f"  L{res['level_index']+1}: p_life={p:.3e}  E[steps]~={expected_steps(p, budget):,.0f}")
        else:
            _, hi = wilson_ci(k, n)
            print(f"  L{res['level_index']+1}: 0/{n} wins  => p<= {hi:.2e}  E[steps] >= {expected_steps(hi, budget):,.0f}")


if __name__ == "__main__":
    main()
