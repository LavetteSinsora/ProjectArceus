"""Leak-cadence / leak-strength sweep: is the leak helping or hurting?

Scans the exp_013_1b harness `runs/` (+ any `--extra` unzipped Colab zip), keeps the
chosen encoder (default icm-φ), and pivots median env-steps-to-first-reward by
(leak μ, leak_per cadence, game). Also copies the lightweight files into
data/leak_sweep/<phi>_<per>_mu<μ>/<game>_seed<n>/ so the sweep is version-controlled.

    uv run python -m JEPA.experiments.exp_017_ablation_study.leak_sweep
    uv run python -m JEPA.experiments.exp_017_ablation_study.leak_sweep --extra /tmp/colab_unzipped
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from collections import defaultdict
from math import inf
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "leak_sweep"
HARNESS_RUNS = (HERE.parents[0] / "exp_013_headline_experiment"
                / "exp_013_1b_leaky_rnd_on_icm_phi" / "runs")


def main():
    ap = argparse.ArgumentParser(description="leak-cadence/strength sweep pivot")
    ap.add_argument("--phi", default="icm", choices=["icm", "frozen", "pixel"])
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--min-steps", type=int, default=1000)
    args = ap.parse_args()

    roots = [str(HARNESS_RUNS)]
    for e in args.extra:
        roots += glob.glob(str(Path(e) / "**" / "exp_013_1b_leaky_rnd_on_icm_phi" / "runs"),
                           recursive=True) or [str(e)]
    roots = [r for r in dict.fromkeys(roots) if Path(r).exists()]

    # best run per (leak, per, game, seed)
    best: dict[tuple, tuple[int, Path, dict]] = {}
    for root in roots:
        for rj in glob.glob(f"{root}/**/result.json", recursive=True):
            try:
                r = json.load(open(rj))
            except Exception:
                continue
            if r.get("phi_mode") != args.phi:
                continue
            if r.get("total_env_steps", 0) < args.min_steps:
                continue
            key = (float(r.get("leak", 0.0)), r.get("leak_per", "minibatch"),
                   r["game"], r["seed"])
            steps = r.get("total_env_steps", 0)
            if key not in best or steps > best[key][0]:
                best[key] = (steps, Path(rj).parent, r)

    # archive + aggregate
    cells = defaultdict(list)
    for (leak, per, game, seed), (_s, src, r) in best.items():
        dst = DATA / f"{args.phi}_{per}_mu{leak}" / f"{game}_seed{seed}"
        dst.mkdir(parents=True, exist_ok=True)
        for fn in ("result.json", "config.json", "metrics.jsonl"):
            if (src / fn).exists():
                shutil.copy2(src / fn, dst / fn)
        cells[(leak, per, game)].append(r)

    games = sorted({g for (_l, _p, g) in cells})
    conds = sorted({(l, p) for (l, p, _g) in cells})
    print(f"\nLEAK SWEEP — encoder φ = {args.phi}   (median env-steps to first reward; solved/total)\n")
    hdr = f"{'μ':>6} {'cadence':>10}" + "".join(f"{g:>16}" for g in games)
    print(hdr); print("-" * len(hdr))
    for (leak, per) in conds:
        cellstrs = []
        for g in games:
            rs = cells.get((leak, per, g), [])
            if not rs:
                cellstrs.append("—"); continue
            sv = np.array([r["env_steps_to_first_reward"] for r in rs if r["solved"]], float)
            cellstrs.append(f"∞ (0/{len(rs)})" if not sv.size
                            else f"{np.median(sv):,.0f} ({sv.size}/{len(rs)})")
        eff = leak if per == "update" else leak * 4  # ~minibatches×μ effective/update
        print(f"{leak:>6g} {per:>10}" + "".join(f"{c:>16}" for c in cellstrs)
              + f"   (~{eff:.0%}/upd)")
    print(f"\narchived → {DATA.relative_to(HERE.parents[2])}")
    print("note: 'minibatch' cadence forgets ~minibatches×μ per PPO update; 'update' forgets exactly μ.")


if __name__ == "__main__":
    main()
