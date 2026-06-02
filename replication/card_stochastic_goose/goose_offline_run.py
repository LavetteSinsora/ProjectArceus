"""Offline single-run Stochastic Goose runner — subprocess-friendly for PARALLEL sweeps.

Runs one (game, seed) offline (no API key) and writes <out>/goose_<game>_seed<seed>.json
INCREMENTALLY (overwritten with latest progress on every level-up / every --save-every
actions / every --save-secs) so a disconnect loses nothing. Launch many of these in
parallel (see the Colab cell) to parallelize the sweep.

    uv run python replication/card_stochastic_goose/goose_offline_run.py \
        --game ls20 --seed 0 --out /tmp/goose --max-actions 200000
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
    p.add_argument("--max-actions", type=int, default=300_000)
    p.add_argument("--max-minutes", type=float, default=120.0)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--save-secs", type=float, default=120.0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    gid = next((e.game_id for e in arc.get_environments() if e.game_id.startswith(a.game)), a.game)
    env = arc.make(gid)
    frame = env.observation_space or env.step(GameAction.RESET)
    torch.manual_seed(a.seed)
    ag = Action(game_id=gid, max_minutes=a.max_minutes)

    path = os.path.join(a.out, f"goose_{a.game}_seed{a.seed}.json")
    level_steps, steps, t0, last, last_save = {}, 0, time.time(), -1, 0.0

    def dump(done):
        json.dump({"agent": "stochastic_goose", "game": a.game, "game_id": gid, "seed": a.seed,
                   "levels_completed": last, "total_actions": steps, "level_steps": level_steps,
                   "wall_seconds": time.time() - t0, "done": done}, open(path, "w"), indent=2)

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
                level_steps[L] = steps
            last = lvl
            print(f"[{a.game} s{a.seed}] cleared level {lvl} @ step {steps}", flush=True)
            dump(False); last_save = time.time()
        if steps % a.save_every == 0 or (time.time() - last_save) > a.save_secs:
            dump(False); last_save = time.time()
            print(f"[{a.game} s{a.seed}] step {steps} levels={last} "
                  f"fps={steps/(time.time()-t0):.1f}  (saved)", flush=True)
    dump(True)
    print(f"[{a.game} s{a.seed}] DONE levels={last} steps={steps} -> {path}", flush=True)


if __name__ == "__main__":
    main()
