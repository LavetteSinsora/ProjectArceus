"""Local Stochastic Goose baseline sweep — full configured budgets, 8-HOUR HARD CAP, MPS.

Adapted from colab_goose_sweep.ipynb for local (no Drive, no clone) execution on the M3 Pro.
Runs the FULL configured per-level env-step budgets (600k / 1M) faithfully — the governing
limit here is WALL-CLOCK: the whole sweep is bounded to ~8 hours.

EFFICIENCY (why this is ~4-5x faster than a naive CPU run):
  Profiling showed the batch-64 training backward pass is ~95% of per-step compute (3.0 s on
  CPU vs 0.02 s for everything else). On MPS that step drops to ~0.32 s — single-process
  throughput goes 1.6 -> ~15 FPS. The MPS GPU is saturated by ONE process (aggregate is
  ~16 FPS at any concurrency), so we run only 2 jobs at a time: that hits peak GPU throughput
  while keeping just two replay buffers (~3.6 GB each, CPU RAM) resident on this 18 GB machine.
  MPS also matches the original Stochastic Goose run (it trained on GPU, not CPU).

Aggregate sweep budget is therefore fixed at ~8h x 16 FPS ~= 460k env-steps total. Spread over
8 jobs that is ~55k steps/job — still far below the 600k/level budget, so every level will be
wall-clock-censored (the "run-as-is, 8h cap" mode). The runner saves INCREMENTALLY every
SAVE_SECS, so a force-kill at the deadline loses at most a few seconds of progress.

Scheduling: 8 jobs (4 games x 2 seeds), 2 concurrent -> 4 waves. Each job is wall-clock-capped
so all four waves finish inside 8h, with a global hard kill as a backstop. Results land in
goose_local_results/ next to this file.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "goose_offline_run.py"
OUT = HERE / "goose_local_results"
LOG_DIR = OUT / "logs"
OUT.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

GAMES = ["ls20", "tu93", "re86", "g50t"]
SEEDS = [0, 1]
NLEV = 3

# FULL configured per-level env-step budgets (match exp_014 method budgets). NOT reached locally
# — the 8h wall-clock cap censors first. Kept faithful per the chosen "run-as-is" mode.
PER_LEVEL_BUDGET = {"ls20": 600_000, "tu93": 600_000, "re86": 1_000_000, "g50t": 600_000}
MAX_ACTIONS = {g: PER_LEVEL_BUDGET[g] * NLEV for g in GAMES}

# ── 8-HOUR HARD CAP + scheduling ──────────────────────────────────────────────
DEVICE = "mps"             # the ~9x lever; falls back is the runner's --device auto
CONCURRENCY = 2            # peak GPU throughput at 2 (GPU-bound); keeps RAM safe (2 buffers)
JOB_CAP_MIN = 115          # per-job wall-clock: 4 waves x 115 = 460 min = 7.67 h < 8 h
GLOBAL_HARD_MIN = 478      # backstop force-kill for the whole sweep (margin under 8h=480)
SAVE_SECS = 60             # frequent incremental saves so a force-kill loses <=60 s
SAVE_EVERY = 2000

start = time.time()
global_deadline = start + GLOBAL_HARD_MIN * 60


def launch(g, s):
    log_path = LOG_DIR / f"log_{g}_seed{s}.txt"
    # Cap this job to whichever is sooner: its own slice, or the global deadline.
    cap_min = min(JOB_CAP_MIN, max(1.0, (global_deadline - time.time()) / 60))
    cmd = [sys.executable, str(RUNNER),
           "--game", g, "--seed", str(s),
           "--out", str(OUT),
           "--device", DEVICE,
           "--max-actions-per-level", str(PER_LEVEL_BUDGET[g]),
           "--max-actions", str(MAX_ACTIONS[g]),
           "--max-minutes", f"{cap_min:.1f}",
           "--save-every", str(SAVE_EVERY),
           "--save-secs", str(SAVE_SECS)]
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return {"g": g, "s": s, "proc": proc, "log": lf, "t0": time.time()}


def main():
    print(f"=== Goose LOCAL sweep START {datetime.now():%Y-%m-%d %H:%M:%S} "
          f"(device={DEVICE}, {CONCURRENCY} concurrent, hard cap ~8h) ===", flush=True)
    print(f"    results → {OUT}", flush=True)

    pending = [(g, s) for g in GAMES for s in SEEDS]
    running, summary = [], []

    while (pending or running) and time.time() < global_deadline:
        # Top up the pool.
        while pending and len(running) < CONCURRENCY and time.time() < global_deadline:
            g, s = pending.pop(0)
            running.append(launch(g, s))
            print(f"  ▶ launched {g} seed{s}  ({len(running)} running, {len(pending)} queued) "
                  f"@ {datetime.now():%H:%M:%S}", flush=True)
        time.sleep(10)
        # Reap finishers (a job ends when it hits its wall-clock cap and writes a clean final).
        still = []
        for j in running:
            if j["proc"].poll() is None:
                still.append(j)
                continue
            j["log"].close()
            el = (time.time() - j["t0"]) / 60
            st = "ok" if j["proc"].returncode == 0 else f"rc={j['proc'].returncode}"
            print(f"  ✔ {j['g']} seed{j['s']}: {st} ({el:.1f} min)", flush=True)
            summary.append({"game": j["g"], "seed": j["s"], "status": st, "elapsed_min": round(el, 1)})
        running = still

    # Global backstop: force-kill anything still alive (incremental saves are on disk).
    if running:
        print(f"  ⏰ global {GLOBAL_HARD_MIN}-min cap — terminating {len(running)} straggler(s)", flush=True)
        for j in running:
            j["proc"].send_signal(signal.SIGTERM)
        time.sleep(10)
        for j in running:
            if j["proc"].poll() is None:
                j["proc"].kill()
            try:
                j["log"].close()
            except Exception:
                pass
            summary.append({"game": j["g"], "seed": j["s"], "status": "capped",
                            "elapsed_min": round((time.time() - j["t0"]) / 60, 1)})
    for g, s in pending:  # never got to start (shouldn't happen within 8h, but record it)
        summary.append({"game": g, "seed": s, "status": "not_started", "elapsed_min": 0})

    json.dump(summary, open(OUT / "sweep_summary.json", "w"), indent=2)
    total_h = (time.time() - start) / 3600
    print(f"\n=== Goose LOCAL sweep DONE {datetime.now():%Y-%m-%d %H:%M:%S} ({total_h:.2f} h) ===", flush=True)
    for r in summary:
        print(f"  {r['game']} seed{r['seed']}: {r['status']} ({r['elapsed_min']} min)", flush=True)
    print(f"\nResults + per-level result.json dirs → {OUT}", flush=True)
    print(f"Summary → {OUT / 'sweep_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
