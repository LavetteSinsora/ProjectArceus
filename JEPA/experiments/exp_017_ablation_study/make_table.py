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
    (r"\textbf{Full} (Leaky RND on $\phi^w$)", r"icm, $\mu{=}.01$",
     "leak_sweep/icm_minibatch_mu0.01"),
    (r"\quad $-$ leak",                         r"icm, $\mu{=}0$",
     "R2_icm_leak0.0"),
    (r"\quad $-\,\phi$ (random enc.)",          r"frozen, $\mu{=}.01$",
     "leak_sweep/frozen_minibatch_mu0.01"),
    (r"\quad $-\,\phi$ (raw pixels)",           r"pixel, $\mu{=}.01$",
     "leak_sweep/pixel_minibatch_mu0.01"),
    (r"$-$ harness (random policy)",            r"---", None),
]
RANDOM_E = {"ls20": 49_843, "tu93": 500_000, "re86": inf, "g50t": inf}


def cell(glob_dir: str, game: str):
    rs = []
    for f in glob.glob(str(DATA / glob_dir / f"{game}_*seed*" / "result.json")):
        r = json.load(open(f))
        if r["level_index"] == LEVEL:
            rs.append(r)
    if not rs:
        return None
    sv = np.array([r["env_steps_to_first_reward"] for r in rs if r["solved"]], float)
    return (float(np.median(sv)) if sv.size else inf, int(sv.size), len(rs))


def tex(c, ref=False):
    if ref:                                   # random-policy reference: just E[steps]
        return r"$\infty$" if c == inf else f"{c/1000:.0f}k"
    if c is None:
        return r"\TBD"
    md, ns, n = c
    return f"$\\infty$\\,({ns}/{n})" if ns == 0 else f"{md/1000:.0f}k\\,({ns}/{n})"


def plain(s):
    return (s.replace(r"\mu{=}", "μ=").replace(r"\phi^w", "φʷ").replace(r"\phi", "φ")
             .replace(r"\textbf{", "").replace(r"\quad ", "  ").replace("}", "")
             .replace(r"\,", " ").replace("$", "").replace(r"\infty", "∞")
             .replace(r"\TBD", "TBD").replace("---", "—"))


def main():
    # console
    print(f"{'variant':<34}{'setting':<16}" + "".join(f"{GAME_TEX[g]:>14}" for g in GAMES))
    print("-" * (34 + 16 + 14 * 4))
    body = []
    for label, setting, src in ROWS:
        cells = ([cell(src, g) for g in GAMES] if src is not None
                 else [RANDOM_E[g] for g in GAMES])
        ref = src is None
        print(f"{plain(label):<34}{plain(setting):<16}"
              + "".join(f"{plain(tex(c, ref)):>14}" for c in cells))
        body.append((label, setting, [tex(c, ref) for c in cells], ref))

    # LaTeX
    L = [
        r"\newcommand{\TBD}{\textcolor{gray}{--}}  % put once in preamble",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{Component ablation} on four ARC-AGI-3 games (L1). Cells are the "
        r"median environment steps to first reward (lower is better) with seeds solved "
        r"($k/N$); $\infty$ = never solved. Agent (CNN\,+\,PPO) and all hyperparameters are "
        r"fixed; only the novelty representation $\phi$ and leak $\mu$ change. $-$ harness "
        r"replaces the intrinsic bonus with an undirected policy.}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    Variant & Setting & LS20 & TU93 & RE86 & G50T \\",
        r"    \midrule",
    ]
    for i, (label, setting, cells, ref) in enumerate(body):
        if ref:
            L.append(r"    \midrule")
        L.append(f"    {label} & {setting} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = HERE / "table.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"\nwrote {out.relative_to(HERE.parents[2])}")


if __name__ == "__main__":
    main()
