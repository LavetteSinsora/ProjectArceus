"""Offline single-run Stochastic Goose runner — subprocess-friendly for PARALLEL sweeps.

Runs one (game, seed) offline (no API key) and writes <out>/goose_<game>_seed<seed>.json
INCREMENTALLY (overwritten with latest progress on every level-up / every --save-every
actions / every --save-secs) so a disconnect loses nothing. Launch many of these in
parallel (see the Colab cell) to parallelize the sweep.

Cutoff matches how we benchmark our own methods: a PER-LEVEL env-step budget
(--max-actions-per-level). Goose resets its model+buffer on every level advance, so each
level gets a fresh budget exactly like our per-level PPO/ICM/RND runs. The level being
attempted is abandoned (censored) once it burns its per-level budget. --max-actions is a
hard total ceiling and --max-minutes a wall-clock safety net (both secondary).

Output is DIRECTLY consumable by the exp_014 staircase (no conversion step). Per (game, seed)
it writes, under <out>:
  goose_<game>_seed<seed>_L<L>/result.json   — one per attempted level, the exact schema the
                                               staircase reads (method/game/level_index/seed/
                                               env_steps_to_first_reward/solved), with the
                                               INCREMENTAL steps-to-clear that level.
  goose_<game>_seed<seed>.json               — small per-run summary (cumulative level_steps,
                                               done flag) for skip-done + the aggregation cell.
Copy the goose_*_L*/ dirs into JEPA/.../exp_014_figures_and_results/data/goose/runs/ and run
staircase.py — the 'goose' series appears with no extra step.

    uv run python replication/card_stochastic_goose/goose_offline_run.py \
        --game ls20 --seed 0 --out /tmp/goose --max-actions-per-level 600000
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import agent` works

import torch  # noqa: E402
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402
from agent import Action  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="offline single goose run")
    p.add_argument("--game", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--max-actions", type=int, default=300_000,
                   help="hard total env-step ceiling across all levels (safety)")
    p.add_argument("--max-actions-per-level", type=int, default=0,
                   help="env-step budget per level (0=disabled); matches our per-level method budgets")
    p.add_argument("--max-minutes", type=float, default=120.0)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--save-secs", type=float, default=120.0)
    p.add_argument("--torch-threads", type=int, default=0,
                   help="cap intra-op CPU threads (>0) to avoid oversubscription when running in parallel")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.torch_threads > 0:
        torch.set_num_threads(a.torch_threads)

    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    gid = next((e.game_id for e in arc.get_environments() if e.game_id.startswith(a.game)), a.game)
    env = arc.make(gid)
    frame = env.observation_space or env.step(GameAction.RESET)
    torch.manual_seed(a.seed)
    ag = Action(game_id=gid, max_minutes=a.max_minutes)

    NLEV = 3
    budget = a.max_actions_per_level or a.max_actions
    path = os.path.join(a.out, f"goose_{a.game}_seed{a.seed}.json")
    # level_steps[k] = cumulative env-steps when levels_completed reached k (k>=1 is "cleared level k").
    level_steps, steps, t0, last, last_save = {}, 0, time.time(), -1, 0.0
    censored_level = None  # 1-indexed level abandoned because it burned its per-level budget

    def write_results(done):
        # Per-run summary (skip-done + aggregation cell).
        json.dump({"agent": "stochastic_goose", "game": a.game, "game_id": gid, "seed": a.seed,
                   "levels_completed": last, "total_actions": steps, "level_steps": level_steps,
                   "max_actions_per_level": a.max_actions_per_level, "censored_level": censored_level,
                   "wall_seconds": time.time() - t0, "done": done}, open(path, "w"), indent=2)
        # Staircase-ready per-level result.json (incremental steps-to-clear; no conversion needed).
        cleared = sorted(L for L in level_steps if L >= 1)
        attempted_top = min((cleared[-1] if cleared else 0) + 1, NLEV)  # incl. the in-progress level
        for L in range(1, attempted_top + 1):
            solved = L in level_steps
            prev_cum = level_steps.get(L - 1, 0)  # cumulative steps at which level L began
            if solved:
                inc = max(level_steps[L] - prev_cum, 1)
                steps_to, total, cens = inc, inc, False
            else:  # in-progress or (when done) abandoned at budget
                steps_to, total, cens = None, steps - prev_cum, done
            d = os.path.join(a.out, f"goose_{a.game}_seed{a.seed}_L{L}")
            os.makedirs(d, exist_ok=True)
            json.dump({"exp_name": f"goose_{a.game}_seed{a.seed}_L{L}", "method": "goose",
                       "game": a.game, "level_index": L - 1, "seed": a.seed,
                       "env_steps_to_first_reward": steps_to, "solved": solved, "censored": cens,
                       "total_env_steps": total, "max_env_steps": budget,
                       "agent": "stochastic_goose", "game_id": gid},
                      open(os.path.join(d, "result.json"), "w"), indent=2)

    while steps < a.max_actions and (time.time() - t0) < a.max_minutes * 60:
        if frame.state is GameState.WIN:
            break
        act = ag.choose_action([frame], frame)
        frame = (env.step(act, data={"x": int(act.action_data.x), "y": int(act.action_data.y)})
                 if act == GameAction.ACTION6 else env.step(act))
        if frame is None:
            continue
        steps += 1
        lvl = getattr(frame, "levels_completed", last)
        if lvl is not None and lvl > last:
            for L in range(last + 1, lvl + 1):
                if L >= 1:
                    level_steps[L] = steps
            last = lvl
            print(f"[{a.game} s{a.seed}] cleared level {lvl} @ step {steps}", flush=True)
            write_results(False); last_save = time.time()
        # Per-level env-step cutoff — abandon the current level once it burns its budget (matches our methods).
        # Current level began when level `last` cleared (level_steps[last]); for level 1 that is step 0.
        if a.max_actions_per_level and (steps - level_steps.get(last, 0)) >= a.max_actions_per_level:
            censored_level = last + 1
            print(f"[{a.game} s{a.seed}] level {censored_level} CENSORED @ {steps} "
                  f"({a.max_actions_per_level} steps/level budget spent)", flush=True)
            break
        if steps % a.save_every == 0 or (time.time() - last_save) > a.save_secs:
            write_results(False); last_save = time.time()
            print(f"[{a.game} s{a.seed}] step {steps} levels={last} "
                  f"fps={steps/(time.time()-t0):.1f}  (saved)", flush=True)
    write_results(True)
    print(f"[{a.game} s{a.seed}] DONE levels={last} steps={steps} censored={censored_level} -> {path}", flush=True)


if __name__ == "__main__":
    main()
