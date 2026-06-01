"""Coverage-vs-entropy probe (READ-ONLY w.r.t. method code).

Central question for exp_013_1 (OCC / RND+ICM): on ls20 L1, the SOLVING seed kept
policy entropy ~1.34 (~ln4=1.386, near-uniform) and solved at 24k steps; the two
CENSORED seeds collapsed entropy (seed0 to ~0.0 for ~60k steps, seed1 ~0.7-0.8) and
never solved within 250k — i.e. WORSE than uniform random (random solves ~99% by 250k).

Hypothesis under test: **coverage != sequence**. A sub-max-entropy / committed policy
systematically UNDER-SAMPLES the specific winning sequence (hit rotation tile an odd #
of times THEN reach the goal within the energy budget), even if it covers many states.

This probe does NOT load any checkpoint (runs save none). Instead it isolates the CAUSAL
variable that differs across the 3 seeds — policy entropy — by rolling out a family of
stochastic policies on the REAL ls20 L1 engine at a fixed env-step budget, and measuring:
  * distinct (cell, rot, color) states visited       -> coverage breadth
  * fraction of lives that ever REACH the goal cell   -> "reach the goal component"
  * fraction of lives that ever HIT the rotation tile -> "reach the rotation component"
  * fraction of lives that reach goal with WRONG rot  -> "at goal but cannot chain"
  * WIN rate / steps-to-first-win                     -> the actual objective

Entropy is controlled by a per-life FIXED random action-preference vector p, softmaxed
at temperature tau and mixed with uniform by weight `mix`:  pi = (1-mix)*uniform + mix*p.
  * mix=0  -> uniform (entropy = ln4 = 1.386), the random baseline.
  * mix=1, low tau -> a committed, low-entropy policy (mimics the collapsed seed regime).
The per-life resampled preference is the closest cheap stand-in for "the policy commits to
SOME direction set" without a trained net; we sweep the resulting entropy and report it.

Usage:
  uv run python JEPA/experiments/exp_013_sparse_exploration/probes/coverage_vs_entropy.py \
      --lives 400 --budget-steps 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# claude_automate (real-engine env_api) lives at the Code-Repo root (parents[4]).
_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

GAME_ID = "ls20-9607627b"
N_ACTIONS = 4
PROBE_DIR = Path(__file__).resolve().parent


def _game(env):
    return env._base._env._game if hasattr(env, "_base") else env._env._game


def _level_info(env):
    g = _game(env)
    L = g.current_level
    return dict(
        goal_cells={(s.x, s.y) for s in L.get_sprites_by_tag("rjlbuycveu")},
        rot_cells={(s.x, s.y) for s in L.get_sprites_by_tag("rhsxkxzdjz")},
        goal_rot=list(g.ehwheiwsk)[0],
        goal_color=list(g.yjdexjsoa)[0],
    )


def _entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def make_policy(rng, mix, tau):
    """Return a sampled per-life action distribution and its entropy.
    mix=0 -> uniform; mix=1 -> softmax(pref/tau) for a random pref vector."""
    if mix <= 0:
        probs = np.ones(N_ACTIONS) / N_ACTIONS
        return probs, _entropy(probs)
    pref = rng.standard_normal(N_ACTIONS)
    sm = np.exp((pref - pref.max()) / max(tau, 1e-6))
    sm = sm / sm.sum()
    probs = (1 - mix) * (np.ones(N_ACTIONS) / N_ACTIONS) + mix * sm
    probs = probs / probs.sum()
    return probs, _entropy(probs)


def run_setting(level_index, mix, tau, n_lives, budget_steps, seed):
    """Roll `n_lives` independent lives; per life: fixed sampled policy, run until
    death/win/budget. Returns aggregate coverage + outcome stats."""
    from claude_automate.framework.env_api import make_arc_env

    rng = np.random.default_rng(seed)
    env = make_arc_env(GAME_ID, level_index)
    env.reset()
    info = _level_info(env)
    g = _game(env)

    global_states = set()           # distinct (cell,rot,color) across ALL lives (pooled coverage)
    wins = 0
    first_win_step = None
    reach_goal_lives = 0
    reach_goal_wrong_rot_lives = 0
    hit_rot_lives = 0
    entropies = []
    total_steps = 0

    for life in range(n_lives):
        env.reset()
        g = _game(env)
        prev_lives = g.aqygnziho
        probs, H = make_policy(rng, mix, tau)
        entropies.append(H)
        reached_goal = reached_goal_wrong = hit_rot = won = False
        for t in range(budget_steps):
            a = int(rng.choice(N_ACTIONS, p=probs))
            f, term = env.step(a)
            total_steps += 1
            cell = (g.gudziatsk.x, g.gudziatsk.y)
            rot = g.cklxociuu
            col = g.hiaauhahz
            global_states.add((cell[0], cell[1], rot, col))
            if cell in info["rot_cells"]:
                hit_rot = True
            if cell in info["goal_cells"]:
                reached_goal = True
                if not (rot == info["goal_rot"] and col == info["goal_color"]):
                    reached_goal_wrong = True
            if env.level_completed:
                won = True
                if first_win_step is None:
                    first_win_step = total_steps
                break
            if g.aqygnziho < prev_lives or term:   # life lost / game over
                break
        wins += int(won)
        reach_goal_lives += int(reached_goal)
        reach_goal_wrong_rot_lives += int(reached_goal_wrong)
        hit_rot_lives += int(hit_rot)

    return dict(
        mix=mix, tau=tau, n_lives=n_lives, budget_steps=budget_steps, seed=seed,
        mean_entropy=float(np.mean(entropies)),
        total_steps=total_steps,
        distinct_states_pooled=len(global_states),
        win_rate=wins / n_lives,
        wins=wins,
        first_win_step=first_win_step,
        reach_goal_rate=reach_goal_lives / n_lives,
        reach_goal_wrong_rot_rate=reach_goal_wrong_rot_lives / n_lives,
        hit_rot_rate=hit_rot_lives / n_lives,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--lives", type=int, default=400)
    ap.add_argument("--budget-steps", type=int, default=200,
                    help="max steps per life (energy budget caps a life ~43 moves anyway)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # (mix, tau) settings spanning near-uniform -> committed/low-entropy.
    settings = [
        (0.0, 1.0),     # uniform: H = ln4 = 1.386  (the random baseline; ~seed2 regime)
        (1.0, 1.0),     # mild commitment
        (1.0, 0.5),     # moderate commitment  (~seed1 regime, H~0.7-0.9)
        (1.0, 0.25),    # strong commitment
        (1.0, 0.12),    # near-deterministic   (~seed0 collapsed regime, H~0)
    ]
    rows = []
    for i, (mix, tau) in enumerate(settings):
        r = run_setting(args.level, mix, tau, args.lives, args.budget_steps,
                        seed=args.seed + 1000 * i)
        rows.append(r)
        print(f"mix={mix} tau={tau} H={r['mean_entropy']:.3f} | "
              f"win={r['win_rate']:.3f} ({r['wins']}/{args.lives}) "
              f"reachGoal={r['reach_goal_rate']:.3f} "
              f"goalWrongRot={r['reach_goal_wrong_rot_rate']:.3f} "
              f"hitRot={r['hit_rot_rate']:.3f} "
              f"distinct={r['distinct_states_pooled']} steps={r['total_steps']}")

    out = PROBE_DIR / "coverage_vs_entropy_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
