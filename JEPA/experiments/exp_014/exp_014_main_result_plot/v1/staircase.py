"""HEADLINE figure — cumulative level-attainment staircase, one per environment.

Idea: treat exploration across levels as CONTINUED. x = cumulative env-steps; y = the level
currently being attempted (L1 = the x-axis). A method stays on the L_k bar while it works on
level k, then steps UP at the *accumulated* step-cost of clearing it. Levels never cleared run
off into an ∞ break on the right. Each method is a colored staircase; seed spread is the shaded
band at each riser. The dashed gray line is the uniform-random policy reference.

  uv run python JEPA/experiments/exp_014/exp_014_main_result_plot/staircase.py

Caveat: levels were run as INDEPENDENT stop-on-first-reward experiments, so the cumulative is
the additive composite Σ_j (steps to clear level j), not a single continued run.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from math import inf, isfinite
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENTS = Path(__file__).resolve().parents[3]          # JEPA/experiments (archived under v1/)
RUNS = EXPERIMENTS / "exp_013_sparse_exploration" / "runs"  # baseline icm/rnd calibration
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GAMES = ["ls20", "tu93", "re86", "g50t"]
NLEV = 3

# Uniform-random-policy E[steps] per level (baseline_random_policy/SUMMARY.md); inf = unreachable.
RANDOM_E = {
    "ls20": [49_843, inf, inf],
    "tu93": [500_000, 2_173, inf],
    "re86": [2_000_000, inf, inf],     # L2/L3 censored (>=1.56M / >=3.12M) -> inf
    "g50t": [inf, inf, inf],
}
COLORS = {"random": "#7f8c8d", "icm": "#e67e22", "rnd": "#2e86de",
          "A": "#8e44ad", "B": "#27ae60", "C": "#16a085", "D": "#c0392b"}
PRETTY = {"random": "random policy", "icm": "ICM", "rnd": "RND"}


def load_baselines():
    rows = []
    for f in glob.glob(str(RUNS / "*" / "result.json")):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    by = defaultdict(list)
    for r in rows:
        if r.get("method") in ("icm", "rnd"):
            by[(r["game"], r["method"], r["level_index"])].append(r)
    stats = {}
    for k, rs in by.items():
        sv = np.array([r["env_steps_to_first_reward"] for r in rs if r["solved"]], float)
        if sv.size:
            stats[k] = dict(reached=True, pos=float(np.median(sv)),
                            lo=float(np.percentile(sv, 25)), hi=float(np.percentile(sv, 75)),
                            nsolved=int(sv.size), ntot=len(rs))
        else:
            stats[k] = dict(reached=False, pos=inf, lo=inf, hi=inf, nsolved=0, ntot=len(rs))
    return stats


def method_levels(game, method, stats):
    out = []
    for lv in range(NLEV):
        if method == "random":
            e = RANDOM_E[game][lv]
            out.append(dict(reached=isfinite(e), pos=e, lo=e, hi=e, nsolved=None, ntot=None))
        else:
            out.append(stats.get((game, method, lv),
                                 dict(reached=False, pos=inf, lo=inf, hi=inf, nsolved=0, ntot=0)))
    return out


def build_path(levels):
    """-> treads [(x0,x1,y)], risers [(cum,yfrom,lo,hi,frac,label)], stuck_level, cum_finite_max."""
    treads, risers = [], []
    cum_prev, stuck = 0.0, None
    for j, info in enumerate(levels):
        if not info["reached"]:
            stuck = j
            break
        cum = cum_prev + info["pos"]
        treads.append((cum_prev, cum, j))
        if info["nsolved"] is None:
            frac, label = 1.0, ""
        else:
            frac = info["nsolved"] / max(info["ntot"], 1)
            label = f"  {info['nsolved']}/{info['ntot']}"
        risers.append((cum, j, cum_prev + info["lo"], cum_prev + info["hi"], frac, label))
        cum_prev = cum
    else:
        treads.append((cum_prev, cum_prev, NLEV))
    return treads, risers, stuck, cum_prev


def plot_game(game, stats, methods):
    paths, finite_max = {}, 0.0
    for m in methods:
        treads, risers, stuck, cum = build_path(method_levels(game, m, stats))
        paths[m] = (treads, risers, stuck)
        finite_max = max(finite_max, cum, *([r[3] for r in risers] or [0]))
    finite_max = finite_max or 1.0
    x_break, x_inf, xr = finite_max * 1.10, finite_max * 1.26, finite_max * 1.33

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for y in range(NLEV):
        ax.axhline(y, color="#d5dbdb", lw=1.0, zorder=0)

    offs = np.linspace(-0.09, 0.09, len(methods)) if len(methods) > 1 else [0.0]
    for idx, m in enumerate(methods):
        treads, risers, stuck = paths[m]
        dy = offs[idx]                                              # vertical jitter so overlapping treads/∞ all show
        c = COLORS.get(m, "#34495e")
        ref = (m == "random")
        ls = "--" if ref else "-"
        lw = 1.8 if ref else 2.6
        for (cum, yfrom, lo, hi, frac, label) in risers:           # seed-spread bands
            if not ref and hi > lo:
                ax.fill_betweenx([yfrom + dy, yfrom + 1 + dy], lo, hi, color=c, alpha=0.13, lw=0, zorder=1)
        for i, (x0, x1, y) in enumerate(treads):
            frac = risers[i][4] if i < len(risers) else 1.0
            a = 1.0 if ref else (0.45 + 0.55 * frac)
            ax.plot([x0, x1], [y + dy, y + dy], color=c, lw=lw, ls=ls, alpha=a, solid_capstyle="round", zorder=3)
            if i < len(risers):
                cum, _, _, _, _, label = risers[i]
                ax.plot([cum, cum], [y + dy, y + 1 + dy], color=c, lw=lw, ls=ls, alpha=a, zorder=3)
                if label:
                    ax.text(cum, y + 0.5 + dy, label, color=c, fontsize=7.5, va="center", ha="left")
        if stuck is not None:                                       # ∞ tail
            x_start = treads[-1][1] if treads else 0.0
            ax.plot([x_start, x_break], [stuck + dy, stuck + dy], color=c, lw=lw, ls=ls, alpha=0.9, zorder=3)
            ax.plot([x_break * 1.04, x_inf], [stuck + dy, stuck + dy], color=c, lw=lw, ls=":", alpha=0.6, zorder=3)
            ax.scatter([x_inf], [stuck + dy], color=c, marker=">", s=55, zorder=4)
        elif treads:                                                # cleared everything
            ax.scatter([treads[-1][0]], [NLEV + dy], color=c, marker="*", s=170,
                       edgecolor="white", lw=0.8, zorder=5)

    ax.axvline(x_break, color="#b2babb", lw=1.0, ls=(0, (2, 3)), zorder=2)   # break marker
    for yb in (-0.30, -0.40):
        ax.plot([x_break - finite_max * 0.012, x_break + finite_max * 0.012], [yb, yb + 0.06],
                color="k", lw=1.0, clip_on=False, zorder=6)
    ax.set_xlim(-finite_max * 0.02, xr)
    ax.set_ylim(-0.45, NLEV + 0.5)
    ax.set_yticks(range(NLEV + 1))
    ax.set_yticklabels(["L1", "L2", "L3", "✓ all"])
    ax.set_xlabel("cumulative environment steps")
    ax.set_title(f"{game} — cumulative steps to clear each level\n"
                 "↑ more levels cleared   ·   → more steps   ·   best = reach ✓all soonest (upper-left);  ∞ = never solved",
                 fontsize=10)

    realticks = [t for t in ax.get_xticks() if 0 <= t <= x_break]
    ax.set_xticks(realticks + [x_inf])
    ax.set_xticklabels([(f"{t/1e6:.1f}M" if t >= 1e6 else f"{int(t/1000)}k") if t > 0 else "0"
                        for t in realticks] + ["∞"])
    ax.tick_params(axis="x", labelsize=8)

    handles = [plt.Line2D([0], [0], color=COLORS.get(m, "#34495e"), lw=2.6,
                          ls="--" if m == "random" else "-", label=PRETTY.get(m, m)) for m in methods]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    p = OUT / f"staircase_{game}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    stats = load_baselines()
    methods = ["random", "icm", "rnd"]
    for g in GAMES:
        print("wrote", plot_game(g, stats, methods))


if __name__ == "__main__":
    main()
