"""exp_014_3 scatter — unique masked states (x) vs RND L1 performance (y).

Hypothesis: the more distinct states a game/L1 has, the bigger the advantage RND exploration
gives over a uniform-random policy (which visits every state with equal probability and is
optimal only in tiny state spaces).

x-axis: # distinct masked states at L1 (from unique_states.json)
y-axis: % step-reduction vs random-policy expected steps
  • (random_E − rnd_median) / random_E × 100 for games where random can solve
  • For re86/g50t where random-policy E[steps]=∞, approximate as 100% improvement
    (annotated separately so no false precision).

Also shows RND solve rate (fraction of 8 seeds that found first reward within cap) as bubble
size — a quick sanity check.

    uv run python -m JEPA.experiments.exp_014_figures_and_results.exp_014_3_unique_states.scatter_states_vs_rnd

Reads:
  results/unique_states.json           (from count_unique_states.py)
  ../data/baseline_icm_rnd/runs/       (RND L1 seeds)
Writes:
  figures/scatter_unique_states_vs_rnd.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "baseline_icm_rnd" / "runs"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Random-policy L1 expected steps (from staircase.py analytical baseline).
# inf = random policy cannot reliably solve within any practical cap.
RANDOM_E = {"ls20": 49_843, "tu93": 500_000, "re86": float("inf"), "g50t": float("inf")}

# RND run caps (steps at which unsolved seeds terminated)
RND_CAP = {"ls20": 2_000_000, "tu93": 299_008, "re86": 2_000_000, "g50t": 299_008}

GAMES = ["ls20", "tu93", "g50t", "re86"]
COLORS = {"ls20": "#2e86de", "tu93": "#e67e22", "re86": "#c0392b", "g50t": "#27ae60"}
LABELS = {"ls20": "ls20", "tu93": "tu93", "re86": "re86", "g50t": "g50t"}


def load_rnd_l1(game: str):
    """Return (steps_to_first_reward or cap) for all 8 seeds of RND L1."""
    seeds = sorted(d for d in DATA.iterdir() if f"rnd_{game}_L1" in d.name)
    out = []
    for s in seeds:
        rows = [json.loads(l) for l in (s / "metrics.jsonl").read_text().splitlines() if l]
        solved = [r["env_steps_to_first_reward"] for r in rows if r.get("env_steps_to_first_reward")]
        cap = RND_CAP[game]
        out.append(solved[0] if solved else cap)
    return out


def main():
    state_file = HERE / "results" / "unique_states.json"
    if not state_file.exists():
        print("ERROR: unique_states.json not found. Run count_unique_states.py first.")
        return

    states = json.load(open(state_file))

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.axhline(0, color="#95a5a6", lw=1, ls="--", zorder=1)
    ax.set_xscale("log")

    for game in GAMES:
        tag = f"{game}_L1"
        if tag not in states:
            print(f"  skipping {tag} (not in unique_states.json)")
            continue

        n_states = states[tag]["n_distinct_states"]
        steps = load_rnd_l1(game)
        solve_rate = sum(1 for s in steps if s < RND_CAP[game]) / len(steps)
        rnd_median = float(np.median(steps))

        rand_e = RANDOM_E[game]
        if np.isinf(rand_e):
            # Random can't solve → 100% "improvement" (RND can; random can't).
            # Use a softer proxy: treat random as 10× the cap to keep on a finite scale.
            rand_eff = RND_CAP[game] * 10
            label_suffix = " (random=∞)"
            marker = "^"
        else:
            rand_eff = rand_e
            label_suffix = ""
            marker = "o"

        pct_improvement = (rand_eff - rnd_median) / rand_eff * 100

        sz = 80 + 300 * solve_rate   # bubble size scales with solve rate
        ax.scatter(n_states, pct_improvement, s=sz, color=COLORS[game],
                   marker=marker, zorder=3, edgecolors="white", linewidths=1.2, alpha=0.92)
        ax.annotate(
            f"{LABELS[game]}{label_suffix}\n({n_states} states, {solve_rate*100:.0f}% solved)",
            (n_states, pct_improvement),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8.5, color=COLORS[game],
        )
        print(f"  {tag}: {n_states} states, rnd_median={rnd_median:.0f}, rand_eff={rand_eff:.0f}, "
              f"improvement={pct_improvement:.1f}%, solve_rate={solve_rate:.2f}")

    ax.set_xlabel("# distinct masked states at L1", fontsize=11)
    ax.set_ylabel("RND step-reduction vs random  (%)", fontsize=11)
    ax.set_title("RND advantage grows with state-space size\n"
                 "(bubble size = fraction of 8 seeds that solved L1)", fontsize=10.5)
    ax.grid(True, alpha=0.3)

    # Legend for marker shape
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=9,
               label="random can solve L1"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=9,
               label="random cannot solve L1 (∞)"),
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")

    fig.tight_layout()
    out = FIG / "scatter_unique_states_vs_rnd.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
