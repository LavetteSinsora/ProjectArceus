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
COLORS = {"random": "#7f8c8d", "icm": "#e67e22", "rnd": "#2e86de",
          "A": "#8e44ad", "B": "#27ae60", "C": "#16a085"}
LABELS = {"random": "random", "icm": "ICM", "rnd": "RND",
          "A": "leaky RND", "B": "leaky RND on ICM-φ", "C": "additive RND+ICM"}
DIR = {"A": "leaky_rnd", "B": "leaky_rnd_on_icm_phi", "C": "additive_rnd_icm"}


def _load(method):
    if method in ("icm", "rnd"):
        out = []
        for f in glob.glob(str(DATA / "baseline_icm_rnd" / "runs" / "*" / "result.json")):
            r = json.load(open(f))
            if r.get("method") == method:
                out.append(r)
        return out
    return [json.load(open(f)) for f in glob.glob(str(DATA / DIR[method] / "runs" / "*" / "result.json"))]


RESULTS = {m: _load(m) for m in ["icm", "rnd", "A", "B", "C"]}


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
    ax.set_xlabel("cumulative environment steps")
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


def main():
    for g in GAMES:
        print(plot_env(g, ["random", "icm", "rnd"], f"{g}_baselines.png"))
        print(plot_env(g, ["random", "icm", "rnd", "A", "B"], f"{g}_with_methods.png"))


if __name__ == "__main__":
    main()
