"""Random-policy exploration baseline for ARC-AGI-3 game tu93 (exp_013).

For tu93 levels 1/2/3 (0-indexed 0/1/2) we estimate the expected number of
environment steps a uniform-random policy needs before its FIRST positive
extrinsic reward (= first level clear, i.e. wrapper `level_completed` / engine
`levels_completed` increments).

Two estimators, cross-validated (cf. the LS20 L1 analysis):

  1. Monte-Carlo on the REAL engine (primary, always valid). A uniform-random
     policy (each of the 4 directional actions w.p. 1/4) is run for many
     episodes. tu93 has ONE life per episode (`lose()` -> GAME_OVER, no respawn),
     so per-life p == per-episode p. We record env-steps to the win and estimate
     p_hat and E[steps] = (mean steps per episode) / p_hat (geometric-restart
     model: independent episodes until the first success).

  2. Exact absorbing-Markov-chain DP on the parsed maze graph (cross-check).
     Valid for L1 which has NO moving obstacles -> the dynamics are a pure
     random walk on a finite node graph with a hard step budget; the chain is
     (node, steps_remaining) and we solve the per-episode win probability and
     the expected steps-to-absorption exactly. For L2/L3 moving obstacles make
     the state space huge and the per-life success non-Markov in node alone, so
     MC is primary and DP is reported as a no-obstacle upper bound on p only.

Mechanics recovered from environment_files/tu93/0768757b/tu93.py:
  * 4 actions ACTION1..4 = up / down / left / right (move 3px = half a node).
  * The maze sprite (tag 0005uvnhiglpvh) has pixels; ==2 are walkable "track"
    cells, ==0 are graph nodes. Node spacing = hcgctulqhn = 6px. A move in a
    direction is legal only if the cell 3px away (hwthhtvyki=3) is track; it
    then advances the player one full node (6px). Illegal moves are no-ops that
    STILL consume one step from the counter.
  * One external env.step() == one logical move == one decrement of the level
    step counter (the engine resolves its internal phase machine per
    perform_action).
  * Win: player node == exit node (tag 0015msvpvzxhqf) -> next_level(),
    levels_completed += 1. Lose: step counter hits 0, or player removed by an
    obstacle collision (tags 0001/0020/0023) -> GAME_OVER. Single life.
  * Step budgets: L1=50, L2=50, L3=35.

Run:
  uv run python JEPA/experiments/exp_013_headline_experiment/baseline_random_policy/scripts/tu93_random_baseline.py
  uv run python .../tu93_random_baseline.py --levels 0 1 2 --episodes 200000
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from claude_automate.framework.env_api import make_arc_env  # noqa: E402

GAME_ID = "tu93-0768757b"
NODE = 6   # hcgctulqhn  : node spacing (px)
HALF = 3   # hwthhtvyki  : per-action move / track-probe distance (px)
# action index (0-based) -> (dx, dy) of the *node* hop it produces
# A1 up(y-3), A2 down(y+3), A3 left(x-3), A4 right(x+3)
ACTION_DELTA = {0: (0, -NODE), 1: (0, NODE), 2: (-NODE, 0), 3: (NODE, 0)}
ACTION_PROBE = {0: (0, -HALF), 1: (0, HALF), 2: (-HALF, 0), 3: (HALF, 0)}


# ──────────────────────────────────────────────────────────────────────────
# Maze parsing + exact graph (for the no-obstacle DP and the shortest path).
# ──────────────────────────────────────────────────────────────────────────
def _engine_handle(env):
    """Return the underlying Tu93 game object regardless of LevelStartWrapper."""
    base = getattr(env, "_base", env)
    return base._env._game


def parse_level_graph(level_index: int):
    """Return (start_rel, exit_rel, neighbors_fn, n_actions, budget, maze_px).

    Coordinates are *relative* to the maze sprite origin, in (x, y) px.
    neighbors_fn(node) -> list of (next_node, action_idx) legal node hops.
    """
    env = make_arc_env(GAME_ID, level_index)
    env.reset()
    g = _engine_handle(env)
    maze = g.current_level.get_sprites_by_tag("0005uvnhiglpvh")[0]
    player = g.current_level.get_sprites_by_tag("0017unajnymcki")[0]
    ex = g.current_level.get_sprites_by_tag("0015msvpvzxhqf")[0]
    px = np.array(maze.pixels)
    H, W = px.shape
    start = (player.x - maze.x, player.y - maze.y)
    exit_rel = (ex.x - maze.x, ex.y - maze.y)
    budget = int(g.ksulgrfyqx.yhzmaedply)

    def track(x, y):  # px[row=y, col=x] == 2 is walkable track
        return 0 <= y < H and 0 <= x < W and px[y, x] == 2

    def neighbors(node):
        x, y = node
        out = []
        for a, (pdx, pdy) in ACTION_PROBE.items():
            if track(x + pdx, y + pdy):
                ndx, ndy = ACTION_DELTA[a]
                out.append(((x + ndx, y + ndy), a))
        return out

    return start, exit_rel, neighbors, env.n_actions, budget, px


def shortest_path(level_index: int):
    start, exit_rel, neighbors, n_actions, budget, _ = parse_level_graph(level_index)
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == exit_rel:
            break
        for nn, a in neighbors(cur):
            if nn not in prev:
                prev[nn] = (cur, a)
                q.append(nn)
    if exit_rel not in prev:
        return None, None
    path = []
    c = exit_rel
    while prev[c] is not None:
        p, a = prev[c]
        path.append(a)
        c = p
    path.reverse()
    return path, len(prev)


# ──────────────────────────────────────────────────────────────────────────
# Exact DP (no-obstacle, valid for L1; upper bound on p for L2/L3).
# ──────────────────────────────────────────────────────────────────────────
def dp_no_obstacle(level_index: int, n_actions: int = 4):
    """Exact per-episode win prob & E[steps] for the pure random walk on the
    node graph with a hard step budget and *no* obstacles.

    State = (node, steps_remaining). Each step: pick action uniformly in
    {0..n_actions-1}; legal moves hop a node (illegal = stay), both cost 1 step.
    Win when node == exit. Out of steps -> lose. Returns (p_win, E_steps_to_win
    conditioned-on-eventual-success-via-restarts, p_per_episode).
    """
    start, exit_rel, neighbors, _, budget, _ = parse_level_graph(level_index)

    # enumerate reachable nodes
    nodes = set([start])
    q = deque([start])
    adj = {}
    while q:
        cur = q.popleft()
        moves = neighbors(cur)
        # transition per action: legal -> target node, else stay
        trans = []
        nbr = {a: nn for nn, a in moves}
        for a in range(n_actions):
            nxt = nbr.get(a, cur)
            trans.append(nxt)
            if nxt not in nodes:
                nodes.add(nxt)
                q.append(nxt)
        adj[cur] = trans

    # P[win | node, s steps remaining]; exit is absorbing-win.
    # W(node, 0)=0 unless node==exit (already-at-exit wins immediately at hop).
    # The engine checks exit AFTER the move, so reaching exit with the move that
    # uses the last step still wins. We treat W(exit, s)=1 for all s>=0.
    node_list = list(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)
    W = np.zeros(N)  # s = 0 layer
    for n in node_list:
        if n == exit_rel:
            W[idx[n]] = 1.0
    # also expected steps to win (conditional) is awkward with restarts; we
    # instead return per-episode p and let the caller use the geometric model.
    for s in range(1, budget + 1):
        Wn = np.zeros(N)
        for n in node_list:
            if n == exit_rel:
                Wn[idx[n]] = 1.0
                continue
            tgt = adj[n]
            acc = 0.0
            for a in range(n_actions):
                acc += W[idx[tgt[a]]]
            Wn[idx[n]] = acc / n_actions
        W = Wn
    p_episode = float(W[idx[start]])
    return p_episode, budget


# ──────────────────────────────────────────────────────────────────────────
# Monte-Carlo on the real engine.
# ──────────────────────────────────────────────────────────────────────────
def monte_carlo(level_index: int, episodes: int, seed: int = 0, max_steps: int | None = None):
    """Run `episodes` uniform-random episodes; return dict of stats.

    Each episode: reset -> act uniformly until terminal. Record whether it was a
    win (level_completed) and how many env-steps it took. tu93 = 1 life, so a
    terminal is either a win or a death/timeout (both end the episode).
    """
    rng = np.random.default_rng(seed)
    env = make_arc_env(GAME_ID, level_index)
    n_actions = env.n_actions
    if max_steps is None:
        max_steps = 1000  # hard safety cap; real budget is <= 60

    wins = 0
    steps_to_win = []
    steps_per_episode = []
    for _ in range(episodes):
        env.reset()
        steps = 0
        won = False
        while True:
            a = int(rng.integers(n_actions))
            _, term = env.step(a)
            steps += 1
            if env.level_completed:
                won = True
                break
            if term or steps >= max_steps:
                break
        steps_per_episode.append(steps)
        if won:
            wins += 1
            steps_to_win.append(steps)

    p_hat = wins / episodes
    mean_steps_per_ep = float(np.mean(steps_per_episode))
    # Wilson 95% CI for p
    z = 1.959964
    if episodes > 0:
        denom = 1 + z * z / episodes
        centre = (p_hat + z * z / (2 * episodes)) / denom
        half = (z * math.sqrt(p_hat * (1 - p_hat) / episodes + z * z / (4 * episodes ** 2))) / denom
        ci = (max(0.0, centre - half), min(1.0, centre + half))
    else:
        ci = (0.0, 1.0)

    # Geometric-restart expected env-steps to first reward:
    #   E[steps] = mean_steps_per_episode / p   (independent restarts)
    e_steps = (mean_steps_per_ep / p_hat) if p_hat > 0 else math.inf
    # CI on E[steps] from the CI on p (steps/episode ~ tight by comparison)
    e_lo = (mean_steps_per_ep / ci[1]) if ci[1] > 0 else math.inf
    e_hi = (mean_steps_per_ep / ci[0]) if ci[0] > 0 else math.inf

    return {
        "level_index": level_index,
        "episodes": episodes,
        "wins": wins,
        "p_hat": p_hat,
        "p_ci95": ci,
        "mean_steps_per_episode": mean_steps_per_ep,
        "mean_steps_to_win": float(np.mean(steps_to_win)) if steps_to_win else None,
        "E_steps_to_first_reward": e_steps,
        "E_steps_ci95": (e_lo, e_hi),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for lvl in args.levels:
        print(f"\n===== tu93 level_index={lvl} (Level {lvl + 1}) =====")
        path, n_nodes = shortest_path(lvl)
        if path is not None:
            print(f"  shortest solution (no-obstacle BFS): {len(path)} moves; "
                  f"reachable nodes={n_nodes}")
        else:
            print("  no path found in BFS (obstacles may block static graph)")
        p_dp, budget = dp_no_obstacle(lvl)
        print(f"  step budget/life = {budget}")
        print(f"  EXACT no-obstacle DP per-episode p_win = {p_dp:.6e}")
        mc = monte_carlo(lvl, args.episodes, seed=args.seed)
        print(f"  MC ({mc['episodes']} eps): wins={mc['wins']}  "
              f"p_hat={mc['p_hat']:.6e}  CI95={tuple(f'{x:.3e}' for x in mc['p_ci95'])}")
        print(f"  mean steps/episode = {mc['mean_steps_per_episode']:.3f}")
        print(f"  E[env steps to first reward] = {mc['E_steps_to_first_reward']:.1f}  "
              f"CI95=({mc['E_steps_ci95'][0]:.0f}, {mc['E_steps_ci95'][1]:.0f})")


if __name__ == "__main__":
    main()
