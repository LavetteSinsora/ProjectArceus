"""Run the exp_013 new-methods sweep as a SCRIPT (for Colab chaining / unattended runs).

Mirrors colab_sweep_exp013.ipynb but as one CLI, so you can queue it behind a still-running
notebook with a tiny cell and save results to Drive (no interactive download needed):

    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/ProjectArceus
    !git pull --ff-only -q && pip -q install -e . --no-deps
    !python -m JEPA.experiments.exp_013_sparse_exploration.sweep --save-dir /content/drive/MyDrive/exp013_results

Methods: A (frozen-φ), B (icm-φ), B-xfer (L1→L2 transfer), C (additive control), D (lookahead).
ICM/RND baselines are NOT re-run. Stop-on-first-reward; per-cell caps; intrinsic-only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

EXP1 = "JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.run"
EXP2 = "JEPA.experiments.exp_013_sparse_exploration.exp_013_2_rnd_icm_additive.run"
EXP4 = "JEPA.experiments.exp_013_sparse_exploration.exp_013_4_disagreement.run"
EXP5 = "JEPA.experiments.exp_013_sparse_exploration.exp_013_5_lookahead.run"
BASE = "JEPA/experiments/exp_013_sparse_exploration"
RUNS1 = f"{BASE}/exp_013_1_rnd_icm/runs"

CAPS = {("ls20", 0): 200_000, ("ls20", 1): 300_000, ("tu93", 0): 600_000, ("re86", 0): 1_000_000}
RANDOM_E = {("ls20", 0): "~50k", ("ls20", 1): "inf", ("tu93", 0): "~500k", ("re86", 0): "~2.0M"}


def main():
    ap = argparse.ArgumentParser(description="exp_013 new-methods sweep driver")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--run-d", dest="run_d", action="store_true", default=True)
    ap.add_argument("--no-d", dest="run_d", action="store_false")
    ap.add_argument("--run-disagree", action="store_true")
    ap.add_argument("--save-dir", default=None, help="zip results here (e.g. a mounted Drive dir)")
    ap.add_argument("--logdir", default="/tmp/exp013_sweep_logs")
    args = ap.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    SEEDS = args.seeds
    TRANSFER_SRC, TRANSFER_DST, COMPARE = ("ls20", 0), ("ls20", 1), [("ls20", 0)]

    def base_args(g, l):
        a = ["--game", g, "--level", str(l), "--max-env-steps", str(CAPS[(g, l)])]
        if args.n_envs:
            a += ["--n-envs", str(args.n_envs)]
        return a

    def run_proc(name, module, extra):
        t0 = time.time()
        with open(f"{args.logdir}/{name}.log", "w") as f:
            rc = subprocess.run([sys.executable, "-m", module] + extra,
                                stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"[{'ok ' if rc == 0 else 'FAIL'}] {name:44s} {(time.time()-t0)/60:5.1f} min", flush=True)
        return rc

    def latest_ckpt(exp_name):
        cks = sorted(glob.glob(f"{RUNS1}/{exp_name}_*/checkpoints/step_*.pt"))
        return cks[-1] if cks else None

    t0 = time.time()
    # Phase 1 — transfer source: B (icm) on ls20 L1, sequential (φ ckpts needed for Phase 2).
    sg, sl = TRANSFER_SRC
    for s in SEEDS:
        run_proc(f"B_{sg}_L{sl+1}_s{s}", EXP1, base_args(sg, sl) + ["--seed", str(s)])
    src = {s: latest_ckpt(f"exp013_1_rndicm_icm_{sg}_L{sl+1}_seed{s}") for s in SEEDS}
    print("source φ ckpts:", {s: (p is not None) for s, p in src.items()}, flush=True)

    # Phase 2 — A / C / D on compare cells; B-random + B-xfer (+ D) on the transfer-dst cell.
    jobs = []
    for (g, l) in COMPARE:
        for s in SEEDS:
            sd = ["--seed", str(s)]
            jobs.append((f"A_frozen_{g}_L{l+1}_s{s}", EXP1, ["--phi-mode", "frozen"] + base_args(g, l) + sd))
            jobs.append((f"C_additive_{g}_L{l+1}_s{s}", EXP2, base_args(g, l) + sd))
            if args.run_d:
                jobs.append((f"D_lookahead_{g}_L{l+1}_s{s}", EXP5, base_args(g, l) + sd))
            if args.run_disagree:
                jobs.append((f"E_disagree_{g}_L{l+1}_s{s}", EXP4, base_args(g, l) + sd))
    dg, dl = TRANSFER_DST
    for s in SEEDS:
        sd = ["--seed", str(s)]
        jobs.append((f"B_random_{dg}_L{dl+1}_s{s}", EXP1, base_args(dg, dl) + sd))
        if src.get(s):
            jobs.append((f"B_xfer_{dg}_L{dl+1}_s{s}", EXP1, base_args(dg, dl) + sd + ["--init-phi-ckpt", src[s]]))
        if args.run_d:
            jobs.append((f"D_lookahead_{dg}_L{dl+1}_s{s}", EXP5, base_args(dg, dl) + sd))
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(lambda j: run_proc(*j), jobs))
    print(f"\nSWEEP DONE in {(time.time()-t0)/60:.1f} min", flush=True)

    # Aggregate vs the random benchmark.
    files = []
    for d in ["exp_013_1_rnd_icm", "exp_013_2_rnd_icm_additive", "exp_013_4_disagreement", "exp_013_5_lookahead"]:
        files += glob.glob(f"{BASE}/{d}/runs/*/result.json")
    rows = [json.load(open(f)) for f in files]

    def tag(r):
        m = r.get("method")
        if m == "rnd_icm_additive": return "C_additive"
        if m == "disagreement": return "E_disagreement"
        if m == "lookahead_mcts": return "D_lookahead"
        n = r["exp_name"]
        if "_frozen_" in n: return "A_rnd_leak_frozen"
        if "_icm_xfer_" in n: return "B_xfer"
        return "B_rnd_leak_icm"

    agg = defaultdict(list)
    for r in rows:
        agg[(r["game"], r["level_index"], tag(r))].append(r)
    print(f"\n{'game':>5} {'lvl':>3} {'method':>18} {'n':>3} {'solved':>7} {'median':>10} {'rand_E':>8}")
    print("-" * 64, flush=True)
    for k in sorted(agg, key=lambda x: (x[0], x[1], x[2])):
        rs = agg[k]; sv = np.array([r["env_steps_to_first_reward"] for r in rs if r["solved"]], float)
        med = f"{np.median(sv):,.0f}" if sv.size else "—"
        print(f"{k[0]:>5} {k[1]+1:>3} {k[2]:>18} {len(rs):>3} {len(sv)}/{len(rs):<5} {med:>10} {RANDOM_E.get((k[0],k[1]),'?'):>8}", flush=True)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        z = shutil.make_archive(os.path.join(args.save_dir, "exp013_sweep_results"), "zip", BASE)
        print("\nsaved results zip ->", z, flush=True)


if __name__ == "__main__":
    main()
