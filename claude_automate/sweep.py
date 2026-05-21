"""sweep.py — run the Go-Explore solver across many levels and games.

Extends the working `solve.py` search (Go-Explore + recurrent distillation) to
the remaining LS20 levels and to every level of TU93, RE86 and G50T. Each
level runs in its own subprocess; all artifacts and a combined
`sweep_summary.json` go into ONE folder: experiments/sweep_<timestamp>/.

    cd "Code Repo"
    uv run python claude_automate/sweep.py
    uv run python claude_automate/sweep.py --sweep-dir <existing>   # resume

Resumable: a level whose `<sweep_dir>/<tag>/summary.json` already exists is
skipped. Each subprocess runs in its own process group, and on timeout the
WHOLE group is killed (so no orphaned grandchild processes survive — a plain
`subprocess` timeout only kills the `uv` launcher, not the Python it spawned).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from claude_automate.framework.env_api import make_arc_env

_EXP_DIR = Path(__file__).resolve().parent / "experiments"
_SOLVE = Path(__file__).resolve().parent / "solve.py"

# game short name -> (full id, first level index to sweep when not --all)
_GAMES = {
    "ls20": ("ls20-9607627b", 4),   # L1-L4 already solved -> start at L5
    "tu93": ("tu93-0768757b", 1),   # L1 already solved    -> start at L2
    "re86": ("re86-8af5384d", 0),   # untested             -> all levels
    "g50t": ("g50t-5849a774", 0),   # untested             -> all levels
}


def n_levels(game_id: str) -> int:
    return len(make_arc_env(game_id, 0)._env._game._levels)


def run_solve(gid: str, lvl: int, out_dir: Path, max_env_steps: int,
              timeout: int) -> str:
    """Run solve.py in its own process group; kill the whole group on timeout
    so no grandchild process is orphaned. Returns 'completed' or 'timeout'."""
    proc = subprocess.Popen(
        ["uv", "run", "python", str(_SOLVE),
         "--game-id", gid, "--level", str(lvl),
         "--max-env-steps", str(max_env_steps),
         "--device", "cpu", "--out-dir", str(out_dir)],
        cwd=str(_REPO_ROOT), start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        proc.wait(timeout=timeout)
        return "completed"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-env-steps", type=int, default=800_000,
                    help="Go-Explore search budget per level")
    ap.add_argument("--per-level-timeout", type=int, default=2400,
                    help="hard wall-clock cap per level (seconds)")
    ap.add_argument("--all", action="store_true",
                    help="include already-solved levels too")
    ap.add_argument("--sweep-dir", type=str, default=None,
                    help="resume into an existing sweep_<ts> folder")
    args = ap.parse_args()

    tasks = []
    for short, (gid, start) in _GAMES.items():
        first = 0 if args.all else start
        for lvl in range(first, n_levels(gid)):
            tasks.append((short, gid, lvl))

    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        if not sweep_dir.is_absolute():
            sweep_dir = _EXP_DIR / sweep_dir
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = _EXP_DIR / f"sweep_{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_dir / "sweep_summary.json"
    print(f"[sweep] {len(tasks)} levels; dir={sweep_dir}", flush=True)

    results = []
    t_start = time.time()
    for i, (short, gid, lvl) in enumerate(tasks, 1):
        tag = f"{short}_L{lvl + 1}"
        out_dir = sweep_dir / tag
        sjson = out_dir / "summary.json"
        rec = {"game": short, "game_id": gid, "level_index": lvl,
               "tag": tag, "level_human": lvl + 1}

        if sjson.exists():                       # resume — already solved
            rec["wall_s"] = None
            rec["status"] = "resumed"
        else:
            print(f"[sweep] ({i}/{len(tasks)}) {tag} — solving ...", flush=True)
            t0 = time.time()
            outcome = run_solve(gid, lvl, out_dir, args.max_env_steps,
                                args.per_level_timeout)
            rec["wall_s"] = round(time.time() - t0, 1)
            if outcome == "timeout":
                rec["status"] = "timeout"

        if sjson.exists():
            s = json.loads(sjson.read_text())
            rec["solved"] = bool(s.get("solved", False))
            rec["solution_length"] = s.get("solution_length")
            rec["search_env_steps"] = s.get("search_env_steps")
            rec["eval_completion_rate"] = s.get("eval_completion_rate")
            rec.setdefault("status", "solved" if rec["solved"] else "unsolved")
        else:
            rec["solved"] = False
            rec.setdefault("status", "no_summary")

        results.append(rec)
        summary_path.write_text(json.dumps({
            "max_env_steps": args.max_env_steps,
            "elapsed_s": round(time.time() - t_start, 1),
            "n_done": len(results), "n_total": len(tasks),
            "n_solved": sum(r["solved"] for r in results),
            "results": results,
        }, indent=2))
        print(f"[sweep] {tag}: {rec['status']} solved={rec['solved']} "
              f"len={rec.get('solution_length')} wall={rec['wall_s']}s",
              flush=True)

    n_solved = sum(r["solved"] for r in results)
    print(f"\n[sweep] DONE — {n_solved}/{len(results)} solved "
          f"in {round((time.time()-t_start)/60,1)} min  →  {summary_path}")


if __name__ == "__main__":
    main()
