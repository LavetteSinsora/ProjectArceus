"""Emit the headline ablation table (operating point μ=0.01) as console + LaTeX.

Single-path ablation: Full → −leak → −φ(random) → −φ(pixels) → −harness. Each cell is
median env-steps to first reward (k) with (solved/total); ∞ = none solved; \\TBD = no data
yet. Pulls straight from the curated data/, so once the missing rows are ingested
(collect.py / leak_sweep.py) just re-run this and \\input the .tex — no hand-pasting.

    uv run python -m JEPA.experiments.exp_017_ablation_study.make_table
"""
from __future__ import annotations

import glob
import json
from math import inf
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GAMES = ["ls20", "tu93", "re86", "g50t"]
GAME_TEX = {"ls20": "LS20", "tu93": "TU93", "re86": "RE86", "g50t": "G50T"}
LEVEL = 0

# row: (LaTeX label, setting, data glob under data/ ; None=random-policy reference)
ROWS = [
    (r"\textbf{Full} (Leaky RND on $\phi^w$)", "leak_sweep/icm_minibatch_mu0.01"),
    (r"\quad $-$ leak (vanilla RND)",          "R2_icm_leak0.0"),
    (r"\quad $-\,\phi^w$ (random $\phi$)",     "leak_sweep/frozen_minibatch_mu0.01"),
    (r"\quad $-\,\phi^w$ (pixel $\phi$)",      "leak_sweep/pixel_minibatch_mu0.01"),
    (r"$-$ harness (random policy)",           None),
]
RANDOM_E = {"ls20": 49_843, "tu93": 500_000, "re86": inf, "g50t": inf}


def cell(glob_dir: str, game: str):
    """median env-steps among solved (inf if none solved), or None if no runs (TBD)."""
    rs = []
    for f in glob.glob(str(DATA / glob_dir / f"{game}_*seed*" / "result.json")):
        r = json.load(open(f))
        if r["level_index"] == LEVEL:
            rs.append(r)
    if not rs:
        return None
    sv = np.array([r["env_steps_to_first_reward"] for r in rs if r["solved"]], float)
    return float(np.median(sv)) if sv.size else inf


def fmt(v, is_min, tex_mode):
    """Format one numeric cell; bold (LaTeX) / *star* (console) if it's the column best."""
    if v is None:
        return r"\TBD" if tex_mode else "TBD"
    s = r"$\infty$" if v == inf else f"{v/1000:.0f}k"
    if v == inf:
        return s
    if is_min:
        return (r"\textbf{" + s + "}") if tex_mode else f"*{s}*"
    return s


def plain(s):
    return (s.replace(r"\mu{=}", "μ=").replace(r"\phi^w", "φʷ").replace(r"\phi", "φ")
             .replace(r"\textbf{", "").replace(r"\quad ", "  ").replace("}", "")
             .replace(r"\,", "").replace("$", "").replace(r"\infty", "∞")
             .replace(r"\TBD", "TBD").replace("---", "—"))


def main():
    # numeric grid: per-row values per game (median, inf, or None=TBD)
    grid = []   # (label, [vals], is_ref)
    for label, src in ROWS:
        if src is None:
            grid.append((label, [RANDOM_E[g] for g in GAMES], True))
        else:
            grid.append((label, [cell(src, g) for g in GAMES], False))
    # per-column minimum over the METHOD rows only (the random reference is not "best")
    col_min = []
    for j in range(len(GAMES)):
        finite = [row[1][j] for row in grid if not row[2]
                  and row[1][j] not in (None, inf)]
        col_min.append(min(finite) if finite else None)

    def is_min(v, j, ref):
        return (not ref and v is not None and col_min[j] is not None
                and abs(v - col_min[j]) < 1e-9)

    # console
    print(f"{'variant':<36}" + "".join(f"{GAME_TEX[g]:>12}" for g in GAMES))
    print("-" * (36 + 12 * 4))
    for label, vals, ref in grid:
        print(f"{plain(label):<36}"
              + "".join(f"{plain(fmt(v, is_min(v, j, ref), False)):>12}"
                        for j, v in enumerate(vals)))
    body = [(label, [fmt(v, is_min(v, j, ref), True)
                     for j, v in enumerate(vals)], ref)
            for label, vals, ref in grid]

    # LaTeX
    L = [
        r"\newcommand{\TBD}{\textcolor{gray}{--}}  % put once in preamble",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{Component ablation} on four ARC-AGI-3 games (L1). Cells are the "
        r"median environment steps to first reward over $N$ seeds (lower is better); "
        r"\textbf{bold} = fastest in each column, $\infty$ = never solved. Agent (CNN\,+\,PPO) "
        r"and all hyperparameters are fixed; we vary only the novelty representation $\phi$ and "
        r"the leak $\mu$. $-$ harness replaces the intrinsic bonus with an undirected policy.}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Variant & LS20 & TU93 & RE86 & G50T \\",
        r"    \midrule",
    ]
    for label, cells, ref in body:
        if ref:
            L.append(r"    \midrule")
        L.append(f"    {label} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = HERE / "table.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"\nwrote {out.relative_to(HERE.parents[2])}")


if __name__ == "__main__":
    main()
