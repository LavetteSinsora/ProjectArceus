"""Shared data loader for explore/ candidates.

Reuses v1/staircase.py's load_baselines() and RANDOM_E EXACTLY (no re-derivation):
we import the v1 module by path and call into it, then reshape into a tidy table of
cells keyed by (game, method, level). Every number on every candidate figure comes
from here -> from the real exp_013 result.json runs.
"""

from __future__ import annotations

import importlib.util
from math import inf, isfinite
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "v1" / "staircase.py"

_spec = importlib.util.spec_from_file_location("v1_staircase", V1)
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

load_baselines = _v1.load_baselines
RANDOM_E = _v1.RANDOM_E
GAMES = _v1.GAMES            # ["ls20","tu93","re86","g50t"]
NLEV = _v1.NLEV              # 3
LEVELS = ["L1", "L2", "L3"]

# Publication palette (colourblind-safe-ish, distinct from each other).
COLORS = {
    "random": "#9aa0a6",   # neutral grey reference
    "icm": "#e8702a",      # warm orange
    "rnd": "#1f77b4",      # blue
    "A": "#8e44ad", "B": "#27ae60", "C": "#16a085", "D": "#c0392b",
}
PRETTY = {"random": "random policy", "icm": "ICM", "rnd": "RND"}
METHODS = ["random", "icm", "rnd"]


def cell(game, method, level, stats):
    """Uniform per-cell record for any method (random or learned)."""
    if method == "random":
        e = RANDOM_E[game][level]
        return dict(reached=isfinite(e), pos=e, lo=e, hi=e,
                    nsolved=(None if not isfinite(e) else None),
                    ntot=None, frac=(1.0 if isfinite(e) else 0.0))
    s = stats.get((game, method, level),
                  dict(reached=False, pos=inf, lo=inf, hi=inf, nsolved=0, ntot=8))
    ntot = max(s["ntot"], 1)
    return dict(reached=s["reached"], pos=s["pos"], lo=s["lo"], hi=s["hi"],
                nsolved=s["nsolved"], ntot=s["ntot"], frac=s["nsolved"] / ntot)


def all_cells(stats):
    """List of dicts, one per (game, method, level)."""
    out = []
    for g in GAMES:
        for mth in METHODS:
            for lv in range(NLEV):
                c = cell(g, mth, lv, stats)
                c.update(game=g, method=mth, level=lv)
                out.append(c)
    return out


def fmt_steps(x):
    if not isfinite(x):
        return "∞"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}k"
    return f"{x:.0f}"
