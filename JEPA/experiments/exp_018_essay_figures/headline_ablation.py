"""Headline figure built ENTIRELY from the exp_017 ablation data (all final-code, L1).

A 2x2 grid (one panel per game) of horizontal bars: median environment steps to first
reward (log x) per method, annotated with seeds solved (k/N); never-solved = ∞ marker at
the axis edge. Consistent by construction — every bar is the same code generation as the
ablation table. Writes PNG + PDF here (and via --out-dir elsewhere).

    uv run python -m JEPA.experiments.exp_018_essay_figures.headline_ablation
"""
from __future__ import annotations

import argparse
import glob
import json
from math import inf
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[0] / "exp_017_ablation_study" / "data"
GAMES = ["ls20", "tu93", "re86", "g50t"]

# (label, color, data dir under exp_017/data ; None = random-policy reference)
METHODS = [
    ("Full (Leaky RND on $\\phi^w$)", "#55A868", "leak_sweep/icm_minibatch_mu0.01"),
    ("$-$ leak (vanilla RND)",        "#4C72B0", "R2_icm_leak0.0"),
    ("$-\\,\\phi^w$ (random $\\phi$)", "#DD8452", "leak_sweep/frozen_minibatch_mu0.01"),
    ("$-\\,\\phi^w$ (pixel $\\phi$)",  "#8172B3", "leak_sweep/pixel_minibatch_mu0.01"),
    ("$-$ harness (random)",          "#b0b0b0", None),
]
RANDOM_E = {"ls20": 49_843, "tu93": 500_000, "re86": inf, "g50t": inf}


def cell(src, game):
    if src is None:
        e = RANDOM_E[game]
        return (e, 1, 1) if e != inf else (inf, 0, 1)
    rs = [json.load(open(f)) for f in glob.glob(str(DATA / src / f"{game}_*seed*" / "result.json"))]
    rs = [r for r in rs if r["level_index"] == 0]
    if not rs:
        return None
    sv = np.array([r["env_steps_to_first_reward"] for r in rs if r["solved"]], float)
    return (float(np.median(sv)), int(sv.size), len(rs)) if sv.size else (inf, 0, len(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    XMAX = 1.2e6                                  # axis cap; ∞ bars drawn at the edge
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))
    ypos = np.arange(len(METHODS))[::-1]          # Full on top
    for ax, game in zip(axes.flat, GAMES):
        for y, (label, color, src) in zip(ypos, METHODS):
            c = cell(src, game)
            ref = src is None
            alpha = 0.45 if ref else 0.9
            if c is None:
                continue
            med, ns, n = c
            if med == inf:                        # never solved → ∞ marker at edge
                ax.barh(y, XMAX, color=color, alpha=0.12, height=0.66)
                ax.text(XMAX * 0.97, y, r"$\infty$", va="center", ha="right",
                        fontsize=12, color=color)
            else:
                ax.barh(y, med, color=color, alpha=alpha, height=0.66,
                        edgecolor="white", lw=0.6)
                ax.text(med * 1.15, y, f"{med/1000:.0f}k" + (f" ({ns}/{n})" if not ref else ""),
                        va="center", ha="left", fontsize=8.0, color="#333")
        ax.set_xscale("log")
        ax.set_xlim(8e3, XMAX * 1.6)
        ax.set_ylim(-0.6, len(METHODS) - 0.4)
        ax.set_yticks(ypos)
        ax.set_yticklabels([m[0] for m in METHODS], fontsize=8.5)
        ax.set_title(game, fontsize=11, fontweight="medium")
        ax.set_xlabel("env steps to first reward (log)", fontsize=8.5)
        ax.grid(axis="x", which="both", alpha=0.18)
        ax.tick_params(axis="x", labelsize=7.5)

    fig.suptitle("Component ablation — steps to first reward (L1, lower is better)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / "figures" / "headline_ablation_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    if args.out_dir:
        import shutil
        d = Path(args.out_dir); d.mkdir(parents=True, exist_ok=True)
        for f in (out, out.with_suffix(".pdf")):
            shutil.copy2(f, d / f.name)
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
