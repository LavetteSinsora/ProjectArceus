"""Candidate 1 — faceted forest / dot-and-band small-multiples (log-x).

One panel per game (2x2). Within a panel, rows = the three levels (L1 top .. L3 bottom),
and within each level row ICM and RND get their own dot+interval (median, p25-p75 across
seeds). A grey diamond marks the uniform-random reference. Unreachable (0 seeds solved or
random=inf) cells become an "∞ / never" marker pinned in a dedicated right-hand gutter.

Why this is the headline candidate:
  * log-x absorbs the 2k -> 2M dynamic range in ONE shared axis.
  * median + p25-p75 bar communicates seed variance directly.
  * solve-fraction is printed next to each dot (e.g. 7/11), so partial success is explicit.
  * the random diamond gives an at-a-glance "did learning help?" per cell.
  * ∞ is a real, separate visual state, not a fudged large number.

  uv run python explore/c1_forest.py
"""
from __future__ import annotations

from math import isfinite
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _data as D

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

XMIN, XMAX = 1e3, 3e6          # shared log axis covering tu93-L2 (~2k) .. re86 random (2M)
INF_X = XMAX * 3.2             # gutter centre for ∞ markers (off the data axis)


def panel(ax, game, stats):
    # vertical layout: 3 levels, each with two method sub-rows (icm above rnd)
    lvl_gap = 1.0
    sub = 0.22
    yticks, ylabels = [], []
    for lv in range(D.NLEV):
        base = (D.NLEV - 1 - lv) * lvl_gap          # L1 at top
        yticks.append(base)
        ylabels.append(D.LEVELS[lv])

        # random reference diamond (or ∞)
        rc = D.cell(game, "random", lv, stats)
        if rc["reached"]:
            ax.scatter([rc["pos"]], [base], marker="D", s=46, color=D.COLORS["random"],
                       edgecolor="white", lw=0.8, zorder=4)
        else:
            ax.scatter([INF_X], [base], marker="x", s=40, color=D.COLORS["random"], zorder=4)

        for k, mth in enumerate(("icm", "rnd")):
            y = base + (sub if mth == "icm" else -sub)
            c = D.cell(game, mth, lv, stats)
            col = D.COLORS[mth]
            if c["reached"]:
                ax.plot([c["lo"], c["hi"]], [y, y], color=col, lw=2.6,
                        solid_capstyle="round", alpha=0.55, zorder=3)
                ax.scatter([c["pos"]], [y], s=54, color=col, edgecolor="white",
                           lw=0.8, zorder=5)
                ax.text(c["hi"] * 1.18, y, f"{c['nsolved']}/{c['ntot']}", color=col,
                        fontsize=7.2, va="center", ha="left", zorder=6)
            else:
                ax.scatter([INF_X], [y], marker=">", s=48, color=col, zorder=5)
                if c["ntot"]:
                    ax.text(INF_X * 1.4, y, f"0/{c['ntot']}", color=col, fontsize=7.2,
                            va="center", ha="left", zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(XMIN, INF_X * 2.4)
    ax.set_ylim(-0.6, (D.NLEV - 1) * lvl_gap + 0.6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    # ∞ gutter
    ax.axvspan(XMAX * 1.6, INF_X * 2.4, color="#f2f3f4", zorder=0)
    ax.axvline(XMAX * 1.6, color="#cfd4d6", lw=1.0, zorder=1)
    ax.text(INF_X, ax.get_ylim()[1] - 0.12, "∞ never", fontsize=7.5, color="#7f8c8d",
            ha="center", va="top", style="italic")
    ax.set_xticks([1e3, 1e4, 1e5, 1e6])
    ax.set_xticklabels(["1k", "10k", "100k", "1M"], fontsize=8)
    ax.grid(axis="x", color="#eaecee", lw=0.8, zorder=0)
    ax.set_title(game, fontsize=12, fontweight="bold", loc="left", pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    stats = D.load_baselines()
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0), sharex=False)
    for ax, g in zip(axes.flat, D.GAMES):
        panel(ax, g, stats)
    for ax in axes[-1]:
        ax.set_xlabel("env-steps to first reward  (log scale)", fontsize=9)

    handles = [
        Line2D([0], [0], marker="o", color=D.COLORS["icm"], lw=2.6, label="ICM (median, p25–p75)"),
        Line2D([0], [0], marker="o", color=D.COLORS["rnd"], lw=2.6, label="RND (median, p25–p75)"),
        Line2D([0], [0], marker="D", color=D.COLORS["random"], lw=0, label="random-policy reference"),
        Line2D([0], [0], marker=">", color="#566573", lw=0, label="∞  never solved (0 seeds)"),
    ]
    fig.legend(handles=handles, ncol=4, frameon=False, fontsize=9.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Steps to first reward across 4 ARC games × 3 levels  —  lower = better, left = better",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.935, "median over seeds; bar = p25–p75; n/N = seeds solved; "
             "diamond = uniform-random reference; ∞ = unreachable",
             ha="center", fontsize=8.5, color="#566573")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    p = OUT / "c1_forest.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
