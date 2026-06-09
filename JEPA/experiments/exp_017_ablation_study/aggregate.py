"""Build the §4.4 ablation table from the curated data/ archive.

Prints a console table and writes table.md + table.csv (median env-steps-to-first-reward
per row×game, with solved-count). Cells with no solve show "∞ (n/N)"; missing cells "—".

    uv run python -m JEPA.experiments.exp_017_ablation_study.aggregate
"""

from __future__ import annotations

import csv
import glob
import json
from math import inf
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GAMES = ["ls20", "tu93", "re86", "g50t"]
LEVEL = 0  # L1 — the discriminating axis

# Row display order + human labels for the table.
ROW_ORDER = ["R1_icm_leak0.05", "R2_icm_leak0.0", "R3_frozen_leak0.05",
             "R4_frozen_leak0.0", "R5_pixel_leak0.05"]
ROW_LABEL = {
    "R1_icm_leak0.05":    "Leaky RND on φ (full)   [icm, μ=.05]",
    "R2_icm_leak0.0":     "− leak (vanilla RND/φ)  [icm, μ=0]",
    "R3_frozen_leak0.05": "− φ random enc          [frozen, μ=.05]",
    "R4_frozen_leak0.0":  "− φ rand, − leak (RND)  [frozen, μ=0]",
    "R5_pixel_leak0.05":  "− φ raw pixels          [pixel, μ=.05]",
}
# uniform-random-policy E[steps] per game @ L1 (reference floor).
RANDOM_E = {"ls20": 49_843, "tu93": 500_000, "re86": inf, "g50t": inf}


def cell(row: str, game: str):
    rs = []
    for rj in glob.glob(str(DATA / row / f"{game}_L{LEVEL + 1}_seed*" / "result.json")):
        try:
            rs.append(json.load(open(rj)))
        except Exception:
            pass
    if not rs:
        return None
    sv = np.array([r["env_steps_to_first_reward"] for r in rs if r["solved"]], float)
    return {"n": len(rs), "nsolved": int(sv.size),
            "median": float(np.median(sv)) if sv.size else inf,
            "iqr": (float(np.percentile(sv, 25)), float(np.percentile(sv, 75))) if sv.size else None}


def fmt(c):
    if c is None:
        return "—"
    if c["nsolved"] == 0:
        return f"∞ (0/{c['n']})"
    return f"{c['median']:,.0f} ({c['nsolved']}/{c['n']})"


def main():
    # console
    hdr = f"{'ablation row':<40}" + "".join(f"{g:>16}" for g in GAMES)
    print(hdr); print("-" * len(hdr))
    print(f"{'random policy (reference)':<40}" +
          "".join(f"{('%s' % ('∞' if RANDOM_E[g]==inf else f'{RANDOM_E[g]:,.0f}')):>16}" for g in GAMES))
    print("-" * len(hdr))
    md = ["| Row | " + " | ".join(GAMES) + " |",
          "|---|" + "---|" * len(GAMES)]
    csv_rows = [["row"] + GAMES]
    for row in ROW_ORDER:
        cells = [cell(row, g) for g in GAMES]
        print(f"{ROW_LABEL[row]:<40}" + "".join(f"{fmt(c):>16}" for c in cells))
        md.append("| " + ROW_LABEL[row].strip() + " | " + " | ".join(fmt(c) for c in cells) + " |")
        csv_rows.append([row] + [("" if c is None else
                                  ("inf" if c["nsolved"] == 0 else f"{c['median']:.0f}")) for c in cells])
    print("\ncell = median env-steps-to-first-reward (solved/total); ∞ = none solved; — = no data")

    (HERE / "table.md").write_text("\n".join(md) + "\n")
    with open(HERE / "table.csv", "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"\nwrote {HERE.name}/table.md and table.csv")


if __name__ == "__main__":
    main()
