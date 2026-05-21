"""solve_wm.py — model-based few-shot solving (NOT VIABLE with the current WM).

⚠️  STATUS (exp 006, see RESEARCH_LOG.md): this script is kept for the record
but does NOT produce a working solve with the world model trained by
`pretrain_wm.py`. The model reaches ~99% next-frame *pixel* accuracy but
**0% exact-frame accuracy** — and this script's Go-Explore uses *exact* frame
hashing and *exact* imagined replay, so imagined rollouts diverge immediately.
The `completed` head also fails to transfer (too few training examples).
Model-based planning here would need a substantially more accurate model.

Intended idea (unrealised): use a `FrameWorldModel` to run Go-Explore in
imagination (zero real env steps), then verify candidates in the real env —
solving a held-out level with a few real episodes instead of a from-scratch
search. The dynamics model transfers at the *aggregate* level (99% pixel acc
on unseen levels) but not *exactly* enough for exact-replay planning.

    cd "Code Repo"
    uv run python claude_automate/solve_wm.py --wm <wm_dir>/wm.pt --level 3
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from claude_automate.framework.config import Config
from claude_automate.framework.distill import distill_trajectory_recurrent
from claude_automate.framework.env_api import make_arc_env
from claude_automate.framework.go_explore import (
    GoExplore, collect_trajectory_frames,
)
from claude_automate.framework.networks import RecurrentActorCritic
from claude_automate.framework.world_model import FrameWorldModel, ModelEnv

_EXP_DIR = Path(__file__).resolve().parent / "experiments"


def pick_device(pref):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def verify_in_real(env, trajectory):
    """Replay `trajectory` in the REAL env. Returns (completed, real_steps)."""
    env.reset()
    steps = 0
    for a in trajectory:
        _, terminal = env.step(int(a))
        steps += 1
        if env.level_completed:
            return True, steps
        if terminal:
            break
    return False, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=str, required=True, help="path to wm.pt")
    ap.add_argument("--game-id", type=str, default="ls20-9607627b")
    ap.add_argument("--level", type=int, default=3, help="0-indexed level")
    ap.add_argument("--imagined-budget", type=int, default=500_000,
                    help="max imagined (model) env steps for the search")
    ap.add_argument("--max-verify", type=int, default=40,
                    help="max candidate trajectories to verify in the real env")
    ap.add_argument("--explore-steps", type=int, default=25)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.wm, map_location=device, weights_only=False)
    model = FrameWorldModel(n_colors=ckpt["n_colors"],
                            n_actions=ckpt["n_actions"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cfg = Config()
    cfg.game_id, cfg.level_index = args.game_id, args.level
    game_short = args.game_id.split("-")[0]
    real_env = make_arc_env(args.game_id, args.level)
    masked_rows = getattr(real_env, "_MASKED_ROWS", None)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _EXP_DIR / f"solvewm_{game_short}_L{args.level + 1}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[solve-wm] game={args.game_id} level_index={args.level} "
          f"device={device} run_dir={run_dir}")

    # ── 1. one real reset to seed the imagined env ───────────────────────────
    real_steps_used = 0
    init_frame = real_env.reset()           # the only "free" real interaction

    # ── 2. Go-Explore IN IMAGINATION (zero real env steps) ───────────────────
    model_env = ModelEnv(model, init_frame, n_actions=real_env.n_actions,
                         device=device, masked_rows=masked_rows,
                         max_steps=cfg.max_episode_steps)
    explorer = GoExplore(model_env, masked_rows=masked_rows,
                         explore_steps=args.explore_steps, seed=args.seed)
    result = explorer.search(max_env_steps=args.imagined_budget)
    print(f"[solve-wm] imagined search: archive={result.archive_size} "
          f"imagined_steps={result.env_steps} "
          f"model_flagged_solution={'yes' if result.solution else 'no'}")

    # ── 3. build candidate list, verify in the REAL env ──────────────────────
    candidates, seen = [], set()
    if result.solution:
        candidates.append(result.solution)
        seen.add(tuple(result.solution))
    # deepest archived trajectories — the goal is most likely a deep state
    for cell in sorted(explorer.archive.values(),
                       key=lambda c: len(c.trajectory), reverse=True):
        key = tuple(cell.trajectory)
        if key in seen or not key:
            continue
        seen.add(key)
        candidates.append(cell.trajectory)
        if len(candidates) >= args.max_verify:
            break

    solution = None
    n_verified = 0
    for traj in candidates:
        completed, steps = verify_in_real(real_env, traj)
        real_steps_used += steps
        n_verified += 1
        if completed:
            solution = list(traj)
            print(f"[solve-wm] VERIFIED solve on candidate #{n_verified} "
                  f"({len(solution)} actions)")
            break

    summary = {
        "game_id": args.game_id, "level_index": args.level,
        "wm_checkpoint": args.wm,
        "imagined_search_steps": result.env_steps,
        "imagined_archive_size": result.archive_size,
        "candidates_verified": n_verified,
        "real_env_steps_used": real_steps_used,
        "solved": solution is not None,
    }

    # ── 4. distil the verified solution into a recurrent policy ──────────────
    if solution is not None:
        frames, actions = collect_trajectory_frames(real_env, solution)
        policy = RecurrentActorCritic(
            n_actions=real_env.n_actions, n_colors=cfg.n_colors,
            hidden_dim=cfg.hidden_dim, frame_size=cfg.frame_size).to(device)
        dstats = distill_trajectory_recurrent(policy, frames, actions, cfg,
                                              device, epochs=4000)
        torch.save({"model": policy.state_dict(), "config": cfg.to_dict(),
                    "solution_length": len(solution)}, run_dir / "policy.pt")
        summary["solution_length"] = len(solution)
        summary["distill_train_acc"] = dstats["train_acc"]
        (run_dir / "solution.json").write_text(json.dumps(
            {"trajectory": solution, "length": len(solution)}, indent=2))

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 62)
    if solution is not None:
        print(f"[solve-wm] SOLVED level {args.level + 1} via world model")
        print(f"  imagined search: {result.env_steps} model steps "
              f"(0 real env steps)")
        print(f"  REAL env steps used (verification only): "
              f"{real_steps_used}  over {n_verified} candidate(s)")
    else:
        print(f"[solve-wm] NOT solved — {n_verified} candidates verified, "
              f"{real_steps_used} real env steps spent")
    print(f"  artifacts: {run_dir}")
    print("=" * 62)


if __name__ == "__main__":
    main()
