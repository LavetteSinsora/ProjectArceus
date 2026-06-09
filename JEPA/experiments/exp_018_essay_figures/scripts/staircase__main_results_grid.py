"""Headline figures — cumulative level-attainment staircase, one PER ENVIRONMENT, two versions:
  {game}_baselines.png     — random + ICM + RND only
  {game}_with_methods.png  — baselines + our methods (leaky RND / on ICM-φ / additive)

x = cumulative env-steps; y = level being attempted (L1=axis). A method rises at the accumulated
step-cost of clearing each level; never-cleared levels run into an ∞ break. No title/subtitle.
Reads the centralized data/ archive.

    uv run python JEPA/experiments/exp_014_figures_and_results/exp_014_0_main_result_plot/staircase.py
"""

from __future__ import annotations

import glob
import json
from math import inf, isfinite
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GAMES = ["ls20", "tu93", "re86", "g50t"]
NLEV = 3
# uniform-random-policy E[steps] per level (baseline_random_policy/SUMMARY.md); inf = unreachable
RANDOM_E = {"ls20": [49_843, inf, inf], "tu93": [500_000, 2_173, inf],
            "re86": [inf, inf, inf], "g50t": [inf, inf, inf]}   # re86: random never clears L1
# Muted (seaborn-deep) palette, fixed per method across all figures. Our method (B) is the
# hero: green, drawn thickest. color = method; line style = condition.
COLORS = {"random": "#b0b0b0", "icm": "#DD8452", "rnd": "#4C72B0",
          "A": "#8172B3", "B": "#55A868", "C": "#64B5CD", "goose": "#DA8BC3"}
LABELS = {"random": "random", "icm": "ICM", "rnd": "RND",
          "A": "leaky RND", "B": r"Leaky RND w/ $\phi^w$", "C": "additive RND+ICM",
          "goose": "Stochastic Goose"}
DIR = {"A": "leaky_rnd", "B": "leaky_rnd_on_icm_phi", "C": "additive_rnd_icm", "goose": "goose"}


def _load(method):
    if method in ("icm", "rnd"):
        out = []
        for f in glob.glob(str(DATA / "baseline_icm_rnd" / "runs" / "*" / "result.json")):
            r = json.load(open(f))
            if r.get("method") == method:
                out.append(r)
        return out
    return [json.load(open(f)) for f in glob.glob(str(DATA / DIR[method] / "runs" / "*" / "result.json"))]


RESULTS = {m: _load(m) for m in ["icm", "rnd", "A", "B", "C", "goose"]}


def cell(method, game, lv):
    if method == "random":
        e = RANDOM_E[game][lv]
        return dict(reached=isfinite(e), pos=e, lo=e, hi=e, nsolved=None, ntot=None)
    rs = [r for r in RESULTS[method] if r["game"] == game and r["level_index"] == lv]
    sv = np.array([r["env_steps_to_first_reward"] for r in rs if r["solved"]], float)
    if sv.size:
        return dict(reached=True, pos=float(np.median(sv)), lo=float(np.percentile(sv, 25)),
                    hi=float(np.percentile(sv, 75)), nsolved=int(sv.size), ntot=len(rs))
    return dict(reached=False, pos=inf, lo=inf, hi=inf, nsolved=0, ntot=len(rs))


def build_path(game, method):
    treads, risers, cum_prev, stuck = [], [], 0.0, None
    for j in range(NLEV):
        info = cell(method, game, j)
        if not info["reached"]:
            stuck = j
            break
        cum = cum_prev + info["pos"]
        treads.append((cum_prev, cum, j))
        risers.append((cum, j, cum_prev + info["lo"], cum_prev + info["hi"]))
        cum_prev = cum
    else:
        treads.append((cum_prev, cum_prev, NLEV))
    return treads, risers, stuck, cum_prev


def plot_env(game, methods, outname):
    paths, fmax = {}, 0.0
    for m in methods:
        t, r, s, c = build_path(game, m)
        paths[m] = (t, r, s)
        fmax = max(fmax, c, *([x[3] for x in r] or [0]))
    fmax = fmax or 1.0
    x_break, x_inf, xr = fmax * 1.10, fmax * 1.26, fmax * 1.33

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for y in range(NLEV):
        ax.axhline(y, color="#d5dbdb", lw=1.0, zorder=0)
    # Order so the line that reaches a new level FIRST (smallest first-riser x) is on TOP
    # (highest vertical offset + drawn over the others). Never-clears → bottom.
    def first_clear(m):
        return paths[m][1][0][0] if paths[m][1] else inf
    ranked = sorted(methods, key=first_clear)                          # fastest first
    offvals = np.linspace(0.09, -0.09, len(methods)) if len(methods) > 1 else [0.0]
    dy_map = {m: offvals[i] for i, m in enumerate(ranked)}
    z_map = {m: 3 + 2 * (len(methods) - i) for i, m in enumerate(ranked)}   # fastest → highest zorder
    for m in methods:
        t, r, s = paths[m]
        dy, z = dy_map[m], z_map[m]
        c = COLORS[m]
        ref = (m == "random")
        ls = "--" if ref else "-"
        lw = 1.8 if ref else 2.3
        for (cum, yf, lo, hi) in r:
            if not ref and hi > lo:
                ax.fill_betweenx([yf + dy, yf + 1 + dy], lo, hi, color=c, alpha=0.11, lw=0, zorder=1)
        for i, (x0, x1, y) in enumerate(t):
            ax.plot([x0, x1], [y + dy, y + dy], color=c, lw=lw, ls=ls, solid_capstyle="round", zorder=z)
            if i < len(r):
                cum = r[i][0]
                ax.plot([cum, cum], [y + dy, y + 1 + dy], color=c, lw=lw, ls=ls, zorder=z)
        if s is not None:
            xs = t[-1][1] if t else 0.0
            ax.plot([xs, x_break], [s + dy, s + dy], color=c, lw=lw, ls=ls, alpha=0.9, zorder=z)
            ax.plot([x_break * 1.04, x_inf], [s + dy, s + dy], color=c, lw=lw, ls=":", alpha=0.6, zorder=z)
            ax.scatter([x_inf], [s + dy], color=c, marker=">", s=42, zorder=z + 1)
        elif t:
            ax.scatter([t[-1][0]], [NLEV + dy], color=c, marker="*", s=150, edgecolor="white", lw=0.7, zorder=z + 1)

    ax.axvline(x_break, color="#b2babb", lw=1.0, ls=(0, (2, 3)), zorder=2)
    for yb in (-0.30, -0.40):
        ax.plot([x_break - fmax * 0.012, x_break + fmax * 0.012], [yb, yb + 0.06],
                color="k", lw=1.0, clip_on=False, zorder=6)
    ax.set_xlim(-fmax * 0.02, xr)
    ax.set_ylim(-0.45, NLEV + 0.5)
    ax.set_yticks(range(NLEV + 1))
    ax.set_yticklabels(["L1", "L2", "L3", "✓ all"])
    ax.set_xlabel(f"cumulative environment steps ({game})")
    rt = [v for v in ax.get_xticks() if 0 <= v <= x_break]
    ax.set_xticks(rt + [x_inf])
    ax.set_xticklabels([(f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v/1000)}k") if v > 0 else "0" for v in rt] + ["∞"])
    ax.tick_params(axis="x", labelsize=8)
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=2.3, ls="--" if m == "random" else "-",
                          label=LABELS[m]) for m in ranked]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    p = OUT / outname
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _draw_env(ax, game, methods):
    """Draw one environment's staircase onto a provided axis (no legend / no axis-title;
    those are shared at the figure level by plot_grid)."""
    paths, fmax = {}, 0.0
    for m in methods:
        t, r, s, c = build_path(game, m)
        paths[m] = (t, r, s)
        fmax = max(fmax, c, *([x[3] for x in r] or [0]))
    fmax = fmax or 1.0
    x_break, x_inf, xr = fmax * 1.10, fmax * 1.26, fmax * 1.33

    for y in range(NLEV):
        ax.axhline(y, color="#d5dbdb", lw=1.0, zorder=0)

    def first_clear(m):
        return paths[m][1][0][0] if paths[m][1] else inf
    ranked = sorted(methods, key=first_clear)
    offvals = np.linspace(0.09, -0.09, len(methods)) if len(methods) > 1 else [0.0]
    dy_map = {m: offvals[i] for i, m in enumerate(ranked)}
    z_map = {m: 3 + 2 * (len(methods) - i) for i, m in enumerate(ranked)}
    for m in methods:
        t, r, s = paths[m]
        dy, z = dy_map[m], z_map[m]
        c = COLORS[m]
        ref = (m == "random")
        ls = "--" if ref else "-"
        lw = 1.8 if ref else (3.0 if m == "B" else 2.2)   # B (ours) is the hero line
        for (cum, yf, lo, hi) in r:
            if not ref and hi > lo:
                ax.fill_betweenx([yf + dy, yf + 1 + dy], lo, hi, color=c, alpha=0.12, lw=0, zorder=1)
        for i, (x0, x1, y) in enumerate(t):
            ax.plot([x0, x1], [y + dy, y + dy], color=c, lw=lw, ls=ls, solid_capstyle="round", zorder=z)
            if i < len(r):
                cum = r[i][0]
                ax.plot([cum, cum], [y + dy, y + 1 + dy], color=c, lw=lw, ls=ls, zorder=z)
        if s is not None:
            xs = t[-1][1] if t else 0.0
            ax.plot([xs, x_break], [s + dy, s + dy], color=c, lw=lw, ls=ls, alpha=0.9, zorder=z)
            ax.plot([x_break * 1.04, x_inf], [s + dy, s + dy], color=c, lw=lw, ls=":", alpha=0.6, zorder=z)
            ax.scatter([x_inf], [s + dy], color=c, marker=">", s=38, zorder=z + 1)
        elif t:
            ax.scatter([t[-1][0]], [NLEV + dy], color=c, marker="*", s=150, edgecolor="white", lw=0.7, zorder=z + 1)

    ax.axvline(x_break, color="#b2babb", lw=1.0, ls=(0, (2, 3)), zorder=2)
    for yb in (-0.30, -0.40):
        ax.plot([x_break - fmax * 0.012, x_break + fmax * 0.012], [yb, yb + 0.06],
                color="k", lw=1.0, clip_on=False, zorder=6)
    ax.set_xlim(-fmax * 0.02, xr)
    ax.set_ylim(-0.45, NLEV + 0.5)
    ax.set_yticks(range(NLEV + 1))
    ax.set_yticklabels(["L1", "L2", "L3", "✓ all"])
    rt = [v for v in ax.get_xticks() if 0 <= v <= x_break]
    ax.set_xticks(rt + [x_inf])
    ax.set_xticklabels([(f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v/1000)}k") if v > 0 else "0" for v in rt] + ["∞"])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(game, fontsize=11, fontweight="medium")


def plot_grid(methods, outname):
    """One 2x2 figure (a panel per environment) with a single shared legend and shared
    axis titles. This is the headline results figure for the write-up."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    for ax, game in zip(axes.flat, GAMES):
        _draw_env(ax, game, methods)
    # Fixed legend order (ours first), shared across all panels.
    order = ["B"] + (["goose"] if "goose" in methods else []) + ["rnd", "icm", "random"]
    order = [m for m in order if m in methods]
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=3.0 if m == "B" else 2.3,
                          ls="--" if m == "random" else "-", label=LABELS[m]) for m in order]
    fig.legend(handles=handles, frameon=False, fontsize=10, loc="upper center",
               ncol=len(order), bbox_to_anchor=(0.5, 1.005))
    fig.supxlabel("cumulative environment steps", fontsize=11)
    fig.supylabel("level reached", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    p = OUT / outname
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")     # vector PDF for the writeup
    plt.close(fig)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="also copy PNG+PDF to this dir")
    args = ap.parse_args()
    # Goose appears once its result.json land in data/goose/runs/ (written directly by the runner).
    has_goose = len(RESULTS["goose"]) > 0
    # Headline figure: 2x2 grid, our method (B) vs baselines. Variant A (frozen-φ leaky RND) dropped.
    methods = ["random", "icm", "rnd", "B"] + (["goose"] if has_goose else [])
    p = plot_grid(methods, "main_results_grid.png")
    print(p)
    if args.out_dir:
        import shutil
        from pathlib import Path as _P
        dst = _P(args.out_dir); dst.mkdir(parents=True, exist_ok=True)
        for f in (p, p.with_suffix(".pdf")):
            shutil.copy2(f, dst / f.name)
        print(f"copied PNG+PDF -> {dst}")


if __name__ == "__main__":
    main()
