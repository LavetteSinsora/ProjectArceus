"""Run the exp_013 new-methods sweep as a SCRIPT — Colab-chainable and SPLITTABLE across machines.

Methods per cell: A (frozen-φ), B (icm-φ), C (additive control), D (lookahead). Plus the
ls20 L1->L2 φ-TRANSFER pair (B-random vs B-xfer) when ls20 is in --games. ICM/RND baselines
are NOT re-run. Stop-on-first-reward; per-cell caps; intrinsic-only.

Split the work across two machines by environment, e.g.:
    # Colab (forward envs):
    !python -m JEPA.experiments.exp_013_sparse_exploration.sweep --games ls20 tu93 --save-dir /content/drive/MyDrive/exp013_results
    # Mac (last envs):
    uv run python -m JEPA.experiments.exp_013_sparse_exploration.sweep --games g50t re86 --save-dir ~/exp013_results
Aggregate later by merging both machines' result.json / CSV.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

EXP1 = "JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.run"
EXP2 = "JEPA.experiments.exp_013_sparse_exploration.exp_013_2_rnd_icm_additive.run"
EXP4 = "JEPA.experiments.exp_013_sparse_exploration.exp_013_4_disagreement.run"
EXP5 = "JEPA.experiments.exp_013_sparse_exploration.exp_013_5_lookahead.run"
BASE = "JEPA/experiments/exp_013_sparse_exploration"
RUNS1 = f"{BASE}/exp_013_1_rnd_icm/runs"

CAPS = {("ls20", 0): 200_000, ("ls20", 1): 300_000, ("tu93", 0): 600_000,
        ("re86", 0): 1_000_000, ("g50t", 0): 300_000}
DEFAULT_CAP = 300_000
RANDOM_E = {("ls20", 0): "~50k", ("ls20", 1): "inf", ("tu93", 0): "~500k",
            ("re86", 0): "~2.0M", ("g50t", 0): "inf"}


def main():
    ap = argparse.ArgumentParser(description="exp_013 new-methods sweep (splittable across machines)")
    ap.add_argument("--games", nargs="+", default=["ls20"], choices=["ls20", "tu93", "re86", "g50t"])
    ap.add_argument("--levels", type=int, nargs="+", default=[0], help="levels (0-indexed) for the A/B/C/D comparison")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--transfer", dest="transfer", action="store_true", default=True,
                    help="run the ls20 L1->L2 φ-transfer pair (only if ls20 in --games & B in --methods)")
    ap.add_argument("--no-transfer", dest="transfer", action="store_false")
    ap.add_argument("--methods", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D", "E"],
                    help="A=frozen-φ RND, B=icm-φ RND, C=additive, D=lookahead, E=disagreement")
    ap.add_argument("--cap", type=int, default=None, help="uniform max-env-steps override for every cell")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--save-dir", default=None, help="zip results here (e.g. a mounted Drive dir)")
    ap.add_argument("--zip-each", action="store_true",
                    help="re-zip results to --save-dir after EVERY run (crash/disconnect-safe progress)")
    ap.add_argument("--logdir", default="/tmp/exp013_sweep_logs")
    args = ap.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    ziplock = threading.Lock()

    def zip_progress(tag=""):
        """Re-zip ONLY the lightweight results (result.json / metrics.jsonl / config / figs);
        skip model checkpoints (*.pt, ~11MB each) so the incremental zip stays small + fast."""
        if not args.save_dir:
            return
        os.makedirs(args.save_dir, exist_ok=True)
        out = os.path.join(args.save_dir, "exp013_progress.zip")
        root_parent = os.path.dirname(BASE.rstrip("/"))
        with ziplock:
            tmp = out + ".tmp"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _dirs, files in os.walk(BASE):
                    if "checkpoints" in root.split(os.sep) or "__pycache__" in root:
                        continue
                    for fn in files:
                        if fn.endswith((".pt", ".pyc")):
                            continue
                        fp = os.path.join(root, fn)
                        zf.write(fp, os.path.relpath(fp, root_parent))
            os.replace(tmp, out)
        print(f"     ↳ progress zip updated -> {out}  ({os.path.getsize(out)/1e6:.1f} MB, {tag})", flush=True)

    SEEDS = args.seeds
    games = args.games
    M = set(args.methods)
    do_transfer = args.transfer and ("ls20" in games) and ("B" in M)
    cells = [(g, l) for g in games for l in args.levels]

    def base_args(g, l):
        cap = args.cap if args.cap is not None else CAPS.get((g, l), DEFAULT_CAP)
        a = ["--game", g, "--level", str(l), "--max-env-steps", str(cap)]
        if args.n_envs:
            a += ["--n-envs", str(args.n_envs)]
        return a

    def run_proc(name, module, extra):
        t0 = time.time()
        with open(f"{args.logdir}/{name}.log", "w") as f:
            rc = subprocess.run([sys.executable, "-m", module] + extra,
                                stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"[{'ok ' if rc == 0 else 'FAIL'}] {name:44s} {(time.time()-t0)/60:5.1f} min", flush=True)
        if args.zip_each:
            zip_progress(name)
        return rc

    def latest_ckpt(exp_name):
        cks = sorted(glob.glob(f"{RUNS1}/{exp_name}_*/checkpoints/step_*.pt"))
        return cks[-1] if cks else None

    cap_str = f"cap={args.cap}" if args.cap is not None else "cap=per-cell"
    print(f"sweep: games={games} levels={args.levels} seeds={SEEDS} methods={sorted(M)} "
          f"transfer={do_transfer} {cap_str}", flush=True)
    t0 = time.time()

    # Phase 1 — ls20 transfer source: B (icm) on ls20 L1 FIRST (sequential; φ ckpts for Phase 2).
    src = {}
    if do_transfer:
        for s in SEEDS:
            run_proc(f"B_ls20_L1_s{s}", EXP1, base_args("ls20", 0) + ["--seed", str(s)])
        src = {s: latest_ckpt(f"exp013_1_rndicm_icm_ls20_L1_seed{s}") for s in SEEDS}
        print("source φ ckpts:", {s: (p is not None) for s, p in src.items()}, flush=True)

    # Phase 2 — A / B / C / D on each cell (+ the ls20 L1->L2 transfer pair).
    jobs = []
    for (g, l) in cells:
        for s in SEEDS:
            sd = ["--seed", str(s)]
            if "A" in M:
                jobs.append((f"A_frozen_{g}_L{l+1}_s{s}", EXP1, ["--phi-mode", "frozen"] + base_args(g, l) + sd))
            if "C" in M:
                jobs.append((f"C_additive_{g}_L{l+1}_s{s}", EXP2, base_args(g, l) + sd))
            if "D" in M:
                jobs.append((f"D_lookahead_{g}_L{l+1}_s{s}", EXP5, base_args(g, l) + sd))
            if "E" in M:
                jobs.append((f"E_disagree_{g}_L{l+1}_s{s}", EXP4, base_args(g, l) + sd))
            if "B" in M and not (do_transfer and (g, l) == ("ls20", 0)):  # B on ls20 L1 already ran in Phase 1
                jobs.append((f"B_icm_{g}_L{l+1}_s{s}", EXP1, base_args(g, l) + sd))
    if do_transfer:
        for s in SEEDS:
            sd = ["--seed", str(s)]
            jobs.append((f"B_random_ls20_L2_s{s}", EXP1, base_args("ls20", 1) + sd))
            if src.get(s):
                jobs.append((f"B_xfer_ls20_L2_s{s}", EXP1, base_args("ls20", 1) + sd + ["--init-phi-ckpt", src[s]]))
            if "D" in M:
                jobs.append((f"D_lookahead_ls20_L2_s{s}", EXP5, base_args("ls20", 1) + sd))

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
