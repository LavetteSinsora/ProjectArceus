"""exp_014_7 — multi-seed sweep runner.

Runs diagnose.py over a range of seeds (each writes its own
results/encoder_leak_series_<game>_L<n>_seed<k>.npz), then calls aggregate.py.
The RND target is seed-dependent (each seed is a different random ruler) and the
online IDM init is seed-dependent too, so any leak/separation claim needs several
seeds — this produces the median/IQR the lineage kept lacking.

Per-seed runs are independent; --jobs>1 runs them as parallel subprocesses (each
is single-threaded and CPU-cheap aside from --pixel-onehot). Extra args after `--`
are forwarded verbatim to diagnose.py.

Run:
    uv run python -m JEPA.experiments.exp_014_figures_and_results.\
exp_014_7_encoder_leak_comparison.run_sweep --seeds 0 1 2 3 4 --jobs 3 -- --updates-free 30
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MODULE = ("JEPA.experiments.exp_014_figures_and_results."
          "exp_014_7_encoder_leak_comparison.diagnose")


def _run_one(seed: int, extra: list[str]) -> tuple[int, int]:
    cmd = [sys.executable, "-m", MODULE, "--seed", str(seed),
           "--suffix", f"_seed{seed}", "--no-plot", *extra]
    print(f"[sweep] seed {seed}: {' '.join(cmd[2:])}", flush=True)
    p = subprocess.run(cmd)
    return seed, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=1, help="parallel subprocesses")
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--no-aggregate", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="args after `--` forwarded to diagnose.py")
    a = ap.parse_args()
    extra = a.extra[1:] if a.extra and a.extra[0] == "--" else a.extra
    extra = ["--game", a.game, "--level", str(a.level), *extra]

    if a.jobs <= 1:
        results = [_run_one(s, extra) for s in a.seeds]
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(_run_one, a.seeds, [extra] * len(a.seeds)))
    bad = [s for s, rc in results if rc != 0]
    print(f"[sweep] done. ok={[s for s,_ in results if s not in bad]}  failed={bad}")

    if not a.no_aggregate and len(bad) < len(a.seeds):
        from JEPA.experiments.exp_014_figures_and_results.exp_014_7_encoder_leak_comparison.aggregate import (
            aggregate,
        )
        HERE = Path(__file__).resolve().parent
        aggregate(HERE / "results", HERE / "figures", a.game, a.level,
                  [s for s in a.seeds if s not in bad])


if __name__ == "__main__":
    main()
