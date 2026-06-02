"""Random-policy exploration baseline for ARC-AGI-3 game g50t (exp_013).

For a chosen level (0-indexed), run a uniform-random policy on the REAL engine
and estimate:
    p_life   = P(clear the level within one life / before the timer kills you)
    E[steps] = expected number of env steps until the FIRST positive extrinsic
               reward (level clear), under repeated lives.

Mechanics (verified from environment_files/g50t/5849a774/g50t.py):
  * 5 actions: ACTION1..4 = move player up/down/left/right by 6px (one tile);
    ACTION5 = undo last move (no-op if no move history). Every action costs a
    step and advances the timer.
  * Timer sprite `ppfvilwwnk` starts at x=0, width=64, scrolls 1px left every
    2 steps (game step counter `ucorwtereb % 2 == 0`). Loss fires when
    -x > width, i.e. x <= -65. That happens on env-step 130 exactly.
    => HARD BUDGET = 129 acting steps, death on step 130, regardless of action.
    Undo cannot extend the budget (timer is move-independent).
  * No lives system: a single death (timer or being crushed) ends the episode
    (GameState.GAME_OVER, terminal). So per-life == per-episode here.
  * Win: player center reaches the goal tile (`gilbljmfbc`); engine calls
    next_level(), which increments levels_completed (= the +reward signal).

Since each life is independent and capped at <=130 steps, with per-life clear
prob p:
    E[lives to first clear]      = 1 / p
    E[steps to first clear]      = sum over lives of E[steps in that life].
We measure steps-per-life directly in MC (a clearing life is shorter than a
dead life), so we report the empirical Ê[steps to first reward] together with
the simpler 1/p * E[steps|dead-life] decomposition as a cross-check.

Usage:
    uv run python g50t_random_baseline.py --level 0 --lives 200000 --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# Make claude_automate importable (env_api lives there).
_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "claude_automate"))

from claude_automate.framework.env_api import make_arc_env  # noqa: E402

GAME_ID = "g50t-5849a774"
MAX_STEPS_PER_LIFE = 130  # death fires on step 130; we cap the loop there.


def _game_of(env):
    """Reach the underlying ARCBaseGame through the (possibly Level-) wrapper."""
    base = getattr(env, "_base", env)
    return base._env._game


def run_level(level_index: int, n_lives: int, seed: int, verbose: bool = True):
    """Run n_lives independent random lives on `level_index`.

    Returns a dict of statistics.
    """
    rng = np.random.default_rng(seed)
    env = make_arc_env(GAME_ID, level_index)
    n_actions = env.n_actions

    clears = 0
    total_steps = 0  # total env steps spent across ALL lives (the exp_013 quantity)
    steps_when_cleared = []  # steps within the clearing life
    dead_life_lengths = []  # steps consumed by lives that did NOT clear
    first_clear_cumulative_step = None  # total steps at the moment of first clear

    t0 = time.time()
    for life in range(n_lives):
        env.reset()
        steps_this_life = 0
        cleared = False
        while steps_this_life < MAX_STEPS_PER_LIFE:
            a = int(rng.integers(0, n_actions))
            _, terminal = env.step(a)
            steps_this_life += 1
            total_steps += 1
            if env.level_completed:
                cleared = True
                break
            if terminal:  # GAME_OVER (timer/crush) without a clear
                break
        if cleared:
            clears += 1
            steps_when_cleared.append(steps_this_life)
            if first_clear_cumulative_step is None:
                first_clear_cumulative_step = total_steps
        else:
            dead_life_lengths.append(steps_this_life)

        if verbose and (life + 1) % 5000 == 0:
            el = time.time() - t0
            print(
                f"  L{level_index+1}: {life+1}/{n_lives} lives, "
                f"clears={clears}, p_hat={clears/(life+1):.3e}, "
                f"{(life+1)/el:.0f} lives/s",
                flush=True,
            )

    p_hat = clears / n_lives
    mean_dead_len = float(np.mean(dead_life_lengths)) if dead_life_lengths else float(MAX_STEPS_PER_LIFE)
    mean_clear_len = float(np.mean(steps_when_cleared)) if steps_when_cleared else float("nan")

    # E[steps to first reward], analytic from measured per-life quantities:
    #   first clear happens on life K ~ Geometric(p); the K-1 failed lives each
    #   cost mean_dead_len, the clearing life costs mean_clear_len.
    #   E[steps] = (1/p - 1) * mean_dead_len + mean_clear_len
    if p_hat > 0:
        e_steps_decomp = (1.0 / p_hat - 1.0) * mean_dead_len + mean_clear_len
    else:
        e_steps_decomp = float("inf")

    # Direct empirical estimate: total steps / number of clears (renewal-reward;
    # equals E[steps to first reward] in expectation when lives are i.i.d.).
    e_steps_direct = total_steps / clears if clears > 0 else float("inf")

    # Wilson 95% CI for p (robust at small counts).
    z = 1.96
    n = n_lives
    if clears > 0:
        denom = 1 + z**2 / n
        centre = (p_hat + z**2 / (2 * n)) / denom
        half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        p_lo, p_hi = max(0.0, centre - half), centre + half
    else:
        # Rule-of-three style upper bound when zero successes observed.
        p_lo, p_hi = 0.0, 3.0 / n

    return {
        "level_index": level_index,
        "n_lives": n_lives,
        "clears": clears,
        "p_life": p_hat,
        "p_life_ci95": (p_lo, p_hi),
        "mean_dead_life_steps": mean_dead_len,
        "mean_clear_life_steps": mean_clear_len,
        "total_steps": total_steps,
        "E_steps_to_first_reward_direct": e_steps_direct,
        "E_steps_to_first_reward_decomp": e_steps_decomp,
        "first_clear_cumulative_step": first_clear_cumulative_step,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0, help="0-indexed level (0=L1)")
    ap.add_argument("--lives", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = run_level(args.level, args.lives, args.seed)
    print("\n=== g50t random-policy baseline ===")
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
