"""LS20 random-policy baseline: exact forward-DP + real-engine Monte-Carlo.

Computes E[env steps to first positive extrinsic reward] under a uniform-random
policy for LS20 Levels 1, 2, 3 (level_index 0, 1, 2).

Method (see baseline_random_policy/METHODOLOGY.md):
  1. Build the REAL arcengine game via make_arc_env and extract, per level, the
     exact deterministic transition of the player state under each of the 4
     directional actions. State that gates the win:
         (cell, rotation_idx, color_idx, pickups_remaining_bitmask)
     (shape is fixed for L1/L2/L3 — no shape tiles).
  2. Forward DP over the per-life energy budget gives the exact per-life win
     probability p_life and E[steps]. Energy is consumed 1 unit per real move
     (StepsDecrement units of the on-screen counter); a wrong-state goal-bump is
     a FREE no-op (no energy, no move); an energy pickup REFILLS the counter and
     costs no energy on that step.
  3. Monte-Carlo on the real engine cross-checks p_life (tight for L1; for
     L2/L3, p is astronomically small so MC only confirms the order of
     magnitude / that no win is observed in feasible sample sizes).

Verified engine facts (probed live, see the writeup):
  * 1 wrapper step == 1 logical move; cell size = 5 px.
  * Action mapping (0-indexed into [A1,A2,A3,A4]): 0=up, 1=down, 2=left, 3=right.
  * On-screen counter starts at StepCounter (42). Each real move decrements it
    by StepsDecrement (L1: 1; L2/L3: default 2). Life lost when the post-
    decrement counter < 0. Win-check precedes life-loss, so the fatal move still
    offers a win chance.  => opportunities/life = floor(StepCounter/dec)+1
    (L1: 43, L2/L3: 22).
  * Rotation tile: rot=(rot+1)%4, lands on tile, costs energy.
  * Color tile:    color=(color+1)%4, lands on tile, costs energy.
  * Energy pickup: refills counter to full, consumed (restored on death),
    that step costs no energy.
  * Goal in matching (shape,color,rot): WIN. Goal in wrong state: blocked,
    free no-op (no energy, no move).
  * 3 lives; death restores position, rotation/color, energy, and pickups.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Geometry / transition extraction from the real engine
# ---------------------------------------------------------------------------
CELL = 5
# action 0-indexed -> (dx_cells, dy_cells): 0=up,1=down,2=left,3=right
ACT_DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
ROT_MAP = {0: 0, 90: 1, 180: 2, 270: 3}
COLORS = [12, 9, 14, 8]  # tnkekoeuk: color value -> index = position


def _game_from_env(env):
    return env._base._env._game if hasattr(env, "_base") else env._env._game


def extract_level(level_index: int):
    """Return a dict describing the level geometry & gating from the engine."""
    from claude_automate.framework.env_api import make_arc_env

    env = make_arc_env("ls20-9607627b", level_index)
    env.reset()
    g = _game_from_env(env)
    L = g.current_level

    def pos(tag):
        return {(s.x, s.y) for s in L.get_sprites_by_tag(tag)}

    walls = pos("ihdgageizm") | pos("rjlbuycveu")  # goal tile also blocks if not matching
    # but the goal tile must be steppable when matching -> handle separately.
    goal_cells = pos("rjlbuycveu")
    rot_cells = pos("rhsxkxzdjz")
    color_cells = pos("soyhouuebz")
    shape_cells = pos("ttfwljgohq")
    pickup_cells = sorted(pos("npxgalaybz"))
    hard_walls = pos("ihdgageizm")

    info = dict(
        level_index=level_index,
        start=(g.gudziatsk.x, g.gudziatsk.y),
        start_rot=g.cklxociuu,
        start_color=g.hiaauhahz,
        start_shape=g.fwckfzsyc,
        goal_cells=sorted(goal_cells),
        goal_rot=list(g.ehwheiwsk),
        goal_color=list(g.yjdexjsoa),
        goal_shape=list(g.ldxlnycps),
        rot_cells=sorted(rot_cells),
        color_cells=sorted(color_cells),
        shape_cells=sorted(shape_cells),
        pickup_cells=pickup_cells,
        hard_walls=hard_walls,
        step_counter=L.get_data("StepCounter"),
        steps_decrement=(2 if L.get_data("StepsDecrement") is None
                         else L.get_data("StepsDecrement")),
        grid_size=L.grid_size,
    )
    return info


# ---------------------------------------------------------------------------
# Exact forward DP over a single life
# ---------------------------------------------------------------------------
def solve_level(info, verbose=True):
    """Forward DP: returns (p_life, E_steps_per_life, opportunities, diagnostics).

    State = (x, y, rot, color, pickup_mask). shape fixed.
    Energy budget handled as DP time horizon: each real move costs 1 'turn';
    a life has `opportunities` turns where a win can be scored (the win-check
    happens before energy runs out). Free no-ops (wrong-goal bump / wall bump)
    do NOT consume a turn? -- they DO consume a turn for wall-bumps (energy is
    spent) but NOT for wrong-goal bumps (energy not spent).

    To keep the DP exact we track the on-screen counter explicitly (integer),
    because energy pickups refill it. A 'turn' is one action. We track:
        counter value c (0..StepCounter), and whether this action is the last
        survivable one.
    """
    hard = info["hard_walls"]
    goal = set(info["goal_cells"])
    rot_cells = set(info["rot_cells"])
    color_cells = set(info["color_cells"])
    pickups = info["pickup_cells"]
    pidx = {c: i for i, c in enumerate(pickups)}
    gx0, gy0 = info["grid_size"]
    dec = info["steps_decrement"]
    SC = info["step_counter"]

    g_rot = info["goal_rot"][0]
    g_col = info["goal_color"][0]
    g_shape = info["goal_shape"][0]
    shape = info["start_shape"]
    shape_ok = (shape == g_shape)

    sx, sy = info["start"]

    def in_bounds(x, y):
        return 0 <= x < gx0 and 0 <= y < gy0

    # Transition for one action from a (cell,rot,color) ignoring energy.
    # Returns (nx,ny,nrot,ncol, kind) where kind in:
    #   'move'  normal move (costs energy)
    #   'rot'   stepped on rotation tile (costs energy)
    #   'color' stepped on color tile (costs energy)
    #   'pickup' stepped on pickup (refills, no energy cost) -> handled w/ mask
    #   'win'   stepped on goal in matching state (terminal)
    #   'noop_free'  wrong-goal bump: no move, no energy
    #   'noop_wall'  wall/oob bump: no move, BUT energy is spent
    def classify(x, y, rot, col, dx, dy, mask):
        nx, ny = x + dx * CELL, y + dy * CELL
        if not in_bounds(nx, ny) or (nx, ny) in hard:
            return (x, y, rot, col, "noop_wall")
        if (nx, ny) in goal:
            if shape_ok and rot == g_rot and col == g_col:
                return (nx, ny, rot, col, "win")
            else:
                return (x, y, rot, col, "noop_free")
        if (nx, ny) in rot_cells:
            return (nx, ny, (rot + 1) % 4, col, "rot")
        if (nx, ny) in color_cells:
            return (nx, ny, rot, (col + 1) % 4, "color")
        if (nx, ny) in pidx and not (mask >> pidx[(nx, ny)]) & 1:
            return (nx, ny, rot, col, "pickup")
        return (nx, ny, rot, col, "move")

    # Forward DP over actions within a life.
    # dist[(x,y,rot,col,mask,counter)] = probability mass of being in that state
    # *before* taking an action, alive. We propagate until all mass is dead.
    # Win prob accumulates whenever an action lands on goal in matching state.
    n_pick = len(pickups)
    full_mask = 0  # 0 = no pickup taken yet
    start_state = (sx, sy, info["start_rot"], info["start_color"], 0, SC)

    from collections import defaultdict
    cur = {start_state: 1.0}
    p_win = 0.0
    # expected number of actions taken before death-or-win, weighted
    e_actions = 0.0
    # safety cap on iterations; also stop when surviving mass is negligible
    max_iters = 20000
    it = 0
    while cur and it < max_iters:
        if sum(cur.values()) < 1e-15:
            break
        it += 1
        nxt = defaultdict(float)
        for (x, y, rot, col, mask, c), pmass in cur.items():
            # each action takes one env step (one action). Count it.
            e_actions += pmass  # one action about to be spent by everyone alive
            for a, (dx, dy) in ACT_DELTA.items():
                ap = pmass * 0.25
                nx, ny, nrot, ncol, kind = classify(x, y, rot, col, dx, dy, mask)
                if kind == "win":
                    p_win += ap
                    continue
                if kind == "noop_free":
                    # no energy spent, no move. Survives, same counter.
                    nxt[(x, y, rot, col, mask, c)] += ap
                    continue
                # all other kinds spend an action; energy bookkeeping:
                if kind == "pickup":
                    nmask = mask | (1 << pidx[(nx, ny)])
                    nc = SC  # refill, no decrement this step
                    nxt[(nx, ny, nrot, ncol, nmask, nc)] += ap
                    continue
                # move / rot / color / noop_wall: decrement counter
                # mfyzdfvxsm: if c>=0: c-=dec; alive iff resulting c>=0
                nc = c - dec
                if nc < 0:
                    # life ends on this action (death). No survival mass.
                    continue
                if kind == "noop_wall":
                    nxt[(x, y, rot, col, mask, nc)] += ap
                else:  # move/rot/color
                    nxt[(nx, ny, nrot, ncol, mask, nc)] += ap
        cur = dict(nxt)

    return p_win, e_actions, it


def per_game(p_life, lives=3):
    return 1.0 - (1.0 - p_life) ** lives


# ---------------------------------------------------------------------------
# Real-engine Monte-Carlo cross-check
# ---------------------------------------------------------------------------
def monte_carlo(level_index, n_lives, seed=0, max_steps_per_life=200):
    """Run uniform-random lives on the real engine; count wins.

    Returns (wins, lives_run, total_steps)."""
    import numpy as np
    from claude_automate.framework.env_api import make_arc_env

    rng = np.random.default_rng(seed)
    env = make_arc_env("ls20-9607627b", level_index)
    wins = 0
    total_steps = 0
    lives_run = 0
    env.reset()
    g = _game_from_env(env)
    lives_run = 1
    while lives_run <= n_lives:
        a = int(rng.integers(0, 4))
        f, term = env.step(a)
        total_steps += 1
        raw = env._base._latest_raw if hasattr(env, "_base") else env._env._latest_raw
        if getattr(raw, "levels_completed", 0) >= 1 or env.level_completed:
            wins += 1
            env.reset()
            g = _game_from_env(env)
            lives_run += 1
            continue
        # detect life loss: aqygnziho decreased OR game over
        if g.aqygnziho < 3 - 0:  # we instead reset per life by tracking
            pass
        if term:  # GAME_OVER after 3 lives
            env.reset()
            g = _game_from_env(env)
            lives_run += 1
            continue
        if total_steps > n_lives * max_steps_per_life:
            break
    return wins, lives_run, total_steps


def monte_carlo_perlife(level_index, n_lives, seed=0):
    """Cleaner MC: simulate independent single lives by resetting the engine
    each life and manually enforcing the energy budget (so we count per-life
    success rather than per-3-life-game)."""
    import numpy as np
    from claude_automate.framework.env_api import make_arc_env

    rng = np.random.default_rng(seed)
    env = make_arc_env("ls20-9607627b", level_index)
    g = None
    wins = 0
    steps_total = 0
    for life in range(n_lives):
        env.reset()
        g = _game_from_env(env)
        prev_lives = g.aqygnziho
        won = False
        while True:
            a = int(rng.integers(0, 4))
            f, term = env.step(a)
            steps_total += 1
            raw = env._base._latest_raw if hasattr(env, "_base") else env._env._latest_raw
            if getattr(raw, "levels_completed", 0) >= 1:
                won = True
                break
            if g.aqygnziho < prev_lives or term:
                # a life was lost (engine respawned) or game over -> end this "life"
                break
        if won:
            wins += 1
    return wins, n_lives, steps_total


# ---------------------------------------------------------------------------
def report(level_index):
    info = extract_level(level_index)
    p_life, e_actions, iters = solve_level(info)
    opportunities = info["step_counter"] // info["steps_decrement"] + 1
    p_g = per_game(p_life, 3)
    # E[steps to first win] = E[lives until win] * E[steps per life]
    # steps per life ~= opportunities (worst-case full life). Use e_actions/life
    # average from DP as the per-life action count if it loses; but for E[steps]
    # the dominant term is (1/p_life)*steps_per_failed_life.
    steps_per_life = opportunities  # a failing life burns the full budget
    e_steps = (1.0 / p_life) * steps_per_life if p_life > 0 else float("inf")
    print(f"\n===== LS20 Level {level_index+1} (level_index={level_index}) =====")
    print(f" start cell {info['start']} rot{info['start_rot']} "
          f"color{info['start_color']} shape{info['start_shape']}")
    print(f" goal {info['goal_cells']} need rot{info['goal_rot']} "
          f"color{info['goal_color']} shape{info['goal_shape']}")
    print(f" rot tiles {info['rot_cells']}  color tiles {info['color_cells']}")
    print(f" pickups {info['pickup_cells']}")
    print(f" StepCounter {info['step_counter']} decrement {info['steps_decrement']} "
          f"-> {opportunities} opportunities/life")
    print(f" DP iters {iters}")
    print(f" p_life      = {p_life:.4e}  (~1 in {1/p_life:,.0f})" if p_life > 0
          else " p_life      = 0 (no win reachable within budget)")
    print(f" p_game(3L)  = {p_g:.4e}")
    print(f" E[steps→win]= {e_steps:,.0f}" if p_life > 0 else " E[steps]=inf")
    return info, p_life, p_g, e_steps, opportunities


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--mc-lives", type=int, default=0,
                    help="if >0, run this many MC lives per level as a cross-check")
    args = ap.parse_args()

    results = {}
    for li in args.levels:
        info, p_life, p_g, e_steps, opp = report(li)
        results[li] = (p_life, e_steps)
        if args.mc_lives > 0:
            w, n, s = monte_carlo_perlife(li, args.mc_lives, seed=li)
            phat = w / n
            print(f" MC: {w}/{n} wins  p̂={phat:.4e}  ({s:,} steps)")
