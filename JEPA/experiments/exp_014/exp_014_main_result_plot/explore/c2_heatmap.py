"""Candidate 2 — log-steps heatmap (methods on rows, game×level on columns).

color = log10(median env-steps to first reward), darker = cheaper (better).
Unreachable cells are hatched grey with an "∞". Each solved cell prints its median
(e.g. 33k) and the solve fraction (7/11) so partial success and variance-of-coverage
are visible. The random-policy row sits at the bottom as the reference baseline.

  uv run python explore/c2_heatmap.py
"""
from __future__ import annotations

from math import isfinite, log10
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

import _data as D

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

ROWS = ["icm", "rnd", "random"]            # learned first, reference last
COLS = [(g, lv) for g in D.GAMES for lv in range(D.NLEV)]
VMIN, VMAX = log10(1e3), log10(2.5e6)


def main():
    stats = D.load_baselines()
    cmap = plt.get_cmap("YlGnBu")
    norm = Normalize(VMIN, VMAX)

    fig, ax = plt.subplots(figsize=(13.2, 4.2))
    for j, (g, lv) in enumerate(COLS):
        for i, mth in enumerate(ROWS):
            c = D.cell(g, mth, lv, stats)
            y = len(ROWS) - 1 - i
            if c["reached"]:
                # darker = cheaper -> invert: low steps -> high color value
                v = 1.0 - norm(log10(c["pos"]))
                face = cmap(v)
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor=face, edgecolor="white", lw=1.5))
                txt = D.fmt_steps(c["pos"])
                lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                tc = "white" if lum < 0.5 else "#222"
                ax.text(j + 0.5, y + 0.60, txt, ha="center", va="center",
                        fontsize=9, color=tc, fontweight="bold")
                if c["ntot"]:
                    ax.text(j + 0.5, y + 0.27, f"{c['nsolved']}/{c['ntot']}", ha="center",
                            va="center", fontsize=7.2, color=tc)
            else:
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor="#eceff1",
                                       edgecolor="white", lw=1.5, hatch="////"))
                ax.text(j + 0.5, y + 0.5, "∞", ha="center", va="center",
                        fontsize=13, color="#90a4ae", fontweight="bold")

    ax.set_xlim(0, len(COLS))
    ax.set_ylim(0, len(ROWS))
    # column ticks: level labels, plus game group brackets below
    ax.set_xticks([j + 0.5 for j in range(len(COLS))])
    ax.set_xticklabels([D.LEVELS[lv] for (_, lv) in COLS], fontsize=9)
    ax.set_yticks([len(ROWS) - 1 - i + 0.5 for i in range(len(ROWS))])
    ax.set_yticklabels([D.PRETTY[m] for m in ROWS], fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # game group labels under the level row
    for gi, g in enumerate(D.GAMES):
        cx = gi * D.NLEV + D.NLEV / 2
        ax.text(cx, -0.55, g, ha="center", va="top", fontsize=11, fontweight="bold")
        if gi > 0:
            ax.axvline(gi * D.NLEV, color="#b0bec5", lw=1.6)
    ax.set_ylim(-0.7, len(ROWS))

    sm = plt.cm.ScalarMappable(cmap=cmap.reversed(), norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.012)
    cb.set_label("env-steps to first reward (log)", fontsize=9)
    cb.set_ticks([log10(1e3), log10(1e4), log10(1e5), log10(1e6)])
    cb.set_ticklabels(["1k", "10k", "100k", "1M"])

    ax.set_title("Steps to first reward — darker = cheaper (better);  hatched ∞ = never solved\n"
                 "cell = median over seeds · small n/N = seeds solved",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    p = OUT / "c2_heatmap.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
