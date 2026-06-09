"""exp_013 encoder×leak ABLATION driver (the §4.4 table). Runs the 5 rows of the
2×2 (+ pixel floor) through the SAME exp_013_1b harness, varying only --phi-mode and
--leak so the agent/PPO/hypers are held fixed:

    R1 full        phi=icm    leak=0.05   (ICM φ warm-up auto-on)
    R2 -leak       phi=icm    leak=0.0    (vanilla RND on φ)      ← no data yet
    R3 -φ rand     phi=frozen leak=0.05   (random-encoder leaky RND)
    R4 -φ rand,-lk phi=frozen leak=0.0    (true vanilla RND)
    R5 -φ pixels   phi=pixel  leak=0.05   (raw-pixel RND, no encoder) ← no data yet

PRIORITY: R2 + R5 (the cells we have zero data for). Sequential by default (MPS is
GPU-bound → one proc saturates it; raise --concurrency only if you measure headroom).

    # priority pass, L1 × 4 games × 3 seeds, locally:
    uv run python -m JEPA.experiments.exp_017_ablation_study.ablation_2x2 \
        --rows R2 R5 --games ls20 tu93 re86 g50t --seeds 0 1 2

Runs land in the exp_013_1b harness `runs/` dir; pull them into this study's curated
`data/` archive with `collect.py`, then build the §4.4 table with `aggregate.py`.
Aggregates from result.json by (phi_mode, leak) — robust to identical exp_name prefixes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

EXP1 = "JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.run"
BASE = "JEPA/experiments/exp_013_headline_experiment"
RUNS1 = f"{BASE}/exp_013_1b_leaky_rnd_on_icm_phi/runs"

# row -> (phi_mode, leak). leak is a float; None means "harness default".
ROWS = {
    "R1": ("icm", 0.05),
    "R2": ("icm", 0.0),
    "R3": ("frozen", 0.05),
    "R4": ("frozen", 0.0),
    "R5": ("pixel", 0.05),
}
# per-cell env-step caps (mirror sweep.py / the random-baseline scales).
CAPS = {("ls20", 0): 200_000, ("ls20", 1): 300_000, ("tu93", 0): 600_000,
        ("tu93", 1): 300_000, ("re86", 0): 1_000_000, ("g50t", 0): 300_000}
DEFAULT_CAP = 300_000


def main():
    ap = argparse.ArgumentParser(description="exp_013 encoder×leak ablation (§4.4 table)")
    ap.add_argument("--rows", nargs="+", default=["R2", "R5"], choices=list(ROWS),
                    help="ablation rows to run (default: the two cells with no data)")
    ap.add_argument("--games", nargs="+", default=["ls20", "tu93", "re86", "g50t"],
                    choices=["ls20", "tu93", "re86", "g50t"])
    ap.add_argument("--levels", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cap", type=int, default=None, help="uniform max-env-steps override")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--warmup-episodes", type=int, default=None,
                    help="override ICM warm-up episodes (icm rows only)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--logdir", default="/tmp/exp013_ablation_logs")
    args = ap.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    def cap_for(g, l):
        return args.cap if args.cap is not None else CAPS.get((g, l), DEFAULT_CAP)

    # Build the job list. Priority rows first, then by (game, level, seed).
    jobs = []
    for row in args.rows:
        phi, leak = ROWS[row]
        for g in args.games:
            for l in args.levels:
                for s in args.seeds:
                    extra = ["--game", g, "--level", str(l), "--seed", str(s),
                             "--phi-mode", phi, "--leak", str(leak),
                             "--max-env-steps", str(cap_for(g, l))]
                    if args.n_envs:
                        extra += ["--n-envs", str(args.n_envs)]
                    if args.warmup_episodes is not None and phi == "icm":
                        extra += ["--warmup-episodes", str(args.warmup_episodes)]
                    name = f"{row}_{phi}_leak{leak}_{g}_L{l+1}_s{s}"
                    jobs.append((name, extra))

    print(f"ablation: rows={args.rows} games={args.games} levels={args.levels} "
          f"seeds={args.seeds} → {len(jobs)} runs, concurrency={args.concurrency}", flush=True)
    t0 = time.time()

    def run_proc(name, extra):
        ts = time.time()
        with open(f"{args.logdir}/{name}.log", "w") as f:
            rc = subprocess.run([sys.executable, "-m", EXP1] + extra,
                                stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"[{'ok ' if rc == 0 else 'FAIL'}] {name:40s} {(time.time()-ts)/60:5.1f} min", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(lambda j: run_proc(*j), jobs))
    print(f"\nABLATION DONE in {(time.time()-t0)/60:.1f} min", flush=True)

    # Aggregate every 1b run by (phi_mode, leak, game, level) from result.json.
    rows = [json.load(open(f)) for f in glob.glob(f"{RUNS1}/*/result.json")]
    agg = defaultdict(list)
    for r in rows:
        key = (r.get("phi_mode", "?"), float(r.get("leak", -1)), r["game"], r["level_index"])
        agg[key].append(r)
    print(f"\n{'phi':>7} {'leak':>5} {'game':>5} {'lvl':>3} {'n':>3} {'solved':>7} {'median_steps':>13}")
    print("-" * 52, flush=True)
    for k in sorted(agg):
        rs = agg[k]
        sv = np.array([r["env_steps_to_first_reward"] for r in rs if r["solved"]], float)
        med = f"{np.median(sv):,.0f}" if sv.size else "—"
        print(f"{k[0]:>7} {k[1]:>5g} {k[2]:>5} {k[3]+1:>3} {len(rs):>3} "
              f"{len(sv)}/{len(rs):<5} {med:>13}", flush=True)


if __name__ == "__main__":
    main()
