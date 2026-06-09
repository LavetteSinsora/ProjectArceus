"""Ingest raw ablation runs → this study's curated data/ archive.

Scans the exp_013_1b harness `runs/` dir (always) plus any EXTRA dirs you pass
(e.g. an unzipped Colab results zip), classifies each run into an ablation ROW by
(phi_mode, leak), and copies the lightweight files (result.json / config.json /
metrics.jsonl) into data/<ROW>/<game>_L<lvl>_seed<seed>/. De-dups per cell, keeping
the run with the most env-steps (the real run over a smoke/partial). Idempotent.

    uv run python -m JEPA.experiments.exp_017_ablation_study.collect
    uv run python -m JEPA.experiments.exp_017_ablation_study.collect --extra /tmp/colab_unzipped
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
HARNESS_RUNS = (HERE.parents[0] / "exp_013_headline_experiment"
                / "exp_013_1b_leaky_rnd_on_icm_phi" / "runs")

# (phi_mode, leak) -> row label. leak compared with tolerance.
def classify(phi_mode: str, leak: float) -> str | None:
    leaky = leak > 1e-9
    return {
        ("icm", True): "R1_icm_leak0.05",
        ("icm", False): "R2_icm_leak0.0",
        ("frozen", True): "R3_frozen_leak0.05",
        ("frozen", False): "R4_frozen_leak0.0",
        ("pixel", True): "R5_pixel_leak0.05",
        ("pixel", False): "R5b_pixel_leak0.0",   # not in the plan, but classify if seen
    }.get((phi_mode, leaky))


def main():
    ap = argparse.ArgumentParser(description="ingest ablation runs into data/")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra run-root dirs to also scan (e.g. an unzipped Colab zip)")
    ap.add_argument("--min-steps", type=int, default=1000, help="skip smoke stubs below this")
    args = ap.parse_args()

    roots = [str(HARNESS_RUNS)]
    for e in args.extra:
        # accept either a runs/ dir or a zip-root containing the harness path
        roots += glob.glob(str(Path(e)), recursive=False)
        roots += glob.glob(str(Path(e) / "**" / "exp_013_1b_leaky_rnd_on_icm_phi" / "runs"),
                           recursive=True)
    roots = [r for r in dict.fromkeys(roots) if Path(r).exists()]
    print("scanning roots:")
    for r in roots:
        print("  ", r)

    # best run per (row, game, level, seed)
    best: dict[tuple, tuple[int, Path, dict]] = {}
    for root in roots:
        for rj in glob.glob(f"{root}/**/result.json", recursive=True):
            try:
                r = json.load(open(rj))
            except Exception:
                continue
            if "phi_mode" not in r:                 # only final-code runs carry phi_mode
                continue
            if r.get("total_env_steps", 0) < args.min_steps:
                continue
            row = classify(r["phi_mode"], float(r.get("leak", 0.0)))
            if row is None:
                continue
            key = (row, r["game"], r["level_index"], r["seed"])
            steps = r.get("total_env_steps", 0)
            if key not in best or steps > best[key][0]:
                best[key] = (steps, Path(rj).parent, r)

    DATA.mkdir(exist_ok=True)
    n = 0
    for (row, game, lvl, seed), (_steps, src, r) in sorted(best.items()):
        dst = DATA / row / f"{game}_L{lvl + 1}_seed{seed}"
        dst.mkdir(parents=True, exist_ok=True)
        for fn in ("result.json", "config.json", "metrics.jsonl"):
            if (src / fn).exists():
                shutil.copy2(src / fn, dst / fn)
        n += 1
    print(f"\ningested {n} runs → {DATA.relative_to(HERE.parents[2])}")
    # quick per-row tally
    by_row: dict[str, int] = {}
    for (row, *_rest) in best:
        by_row[row] = by_row.get(row, 0) + 1
    for row in sorted(by_row):
        print(f"   {row:<22} {by_row[row]} cells")


if __name__ == "__main__":
    main()
