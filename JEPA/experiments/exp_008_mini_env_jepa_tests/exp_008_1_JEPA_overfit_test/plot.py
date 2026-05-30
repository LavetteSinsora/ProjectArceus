"""Plot the exp_008_1 results.

Reads results/per_checkpoint.csv and writes a 4-panel PNG to results/overfit.png:
    (a) L_JEPA          per source (log y)
    (b) L_IDM           per source (log y)
    (c) IDM accuracy    per source (linear y)
    (d) random/trained loss-ratio gap for both terms (log y)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_1_JEPA_overfit_test.config import Config


def _load_overall_rows(csv_path: Path) -> list[dict]:
    """Return only action='all' rows. Drop the duplicate 'final(976)' row
    (same numbers as update 976; cleaner x-axis as ints)."""
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r["action"] != "all":
                continue
            upd = r["update"]
            if upd.startswith("final"):
                continue
            r["update"] = int(upd)
            r["jepa_mse"] = float(r["jepa_mse"])
            r["idm_ce"] = float(r["idm_ce"])
            r["idm_acc"] = float(r["idm_acc"])
            rows.append(r)
    return rows


def main():
    cfg = Config()
    out_dir = Path(cfg.output_dir)
    rows = _load_overall_rows(out_dir / "per_checkpoint.csv")

    tr = sorted([r for r in rows if r["source"] == "trained"], key=lambda r: r["update"])
    rn = sorted([r for r in rows if r["source"] == "random"],  key=lambda r: r["update"])
    x_tr = [r["update"] for r in tr]
    x_rn = [r["update"] for r in rn]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) L_JEPA
    ax = axes[0, 0]
    ax.plot(x_tr, [r["jepa_mse"] for r in tr], "o-", color="#1f77b4", label="trained source")
    ax.plot(x_rn, [r["jepa_mse"] for r in rn], "s-", color="#d62728", label="random source")
    ax.set_yscale("log")
    ax.set_xlabel("training update")
    ax.set_ylabel("L_JEPA  (predictor MSE)")
    ax.set_title("JEPA loss vs source")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # (b) L_IDM
    ax = axes[0, 1]
    ax.plot(x_tr, [r["idm_ce"] for r in tr], "o-", color="#1f77b4", label="trained source")
    ax.plot(x_rn, [r["idm_ce"] for r in rn], "s-", color="#d62728", label="random source")
    ax.set_yscale("log")
    ax.set_xlabel("training update")
    ax.set_ylabel("L_IDM  (cross-entropy)")
    ax.set_title("IDM loss vs source")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # (c) IDM accuracy
    ax = axes[1, 0]
    ax.plot(x_tr, [r["idm_acc"] for r in tr], "o-", color="#1f77b4", label="trained source")
    ax.plot(x_rn, [r["idm_acc"] for r in rn], "s-", color="#d62728", label="random source")
    ax.axhline(0.25, color="k", lw=0.7, ls="--", alpha=0.6, label="chance (0.25)")
    ax.set_xlabel("training update")
    ax.set_ylabel("IDM accuracy")
    ax.set_title("IDM accuracy vs source")
    ax.set_ylim(0.2, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (d) gap ratios
    ax = axes[1, 1]
    jepa_gap = [rn_r["jepa_mse"] / tr_r["jepa_mse"] for tr_r, rn_r in zip(tr, rn)]
    idm_gap  = [rn_r["idm_ce"]   / tr_r["idm_ce"]   for tr_r, rn_r in zip(tr, rn)]
    ax.plot(x_tr, jepa_gap, "o-", color="#2ca02c", label="L_JEPA  random / trained")
    ax.plot(x_tr, idm_gap,  "s-", color="#9467bd", label="L_IDM   random / trained")
    ax.set_yscale("log")
    ax.set_xlabel("training update")
    ax.set_ylabel("loss ratio  random / trained")
    ax.set_title("Overfit gap")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle("exp_008_1 — JEPA overfit test (7_4 checkpoint sweep, 50k transitions/source)",
                 fontsize=13)
    fig.tight_layout()

    out_path = out_dir / "overfit.png"
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
