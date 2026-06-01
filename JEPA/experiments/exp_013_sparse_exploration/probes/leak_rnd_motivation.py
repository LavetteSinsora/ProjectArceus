"""Leaky-RND MOTIVATION figures — controlled experiments on the REAL RNDPhi network.

Demonstrates, with our actual architecture (orthogonal-init MLPs, 3-layer predictor vs
2-layer target, Adam lr=1e-4, the L2-to-init leak), the two claims behind leaky RND:

  (1) SATURATION: standard RND's novelty collapses to a floor after a few visits, so a
      state visited 100× and one visited 1000× look identical → no count resolution
      among visited states.
  (2) THE FIX = FORGETTING: leaking the predictor toward its init turns the one-way error
      ratchet into a RECENCY signal — a state's novelty REGENERATES once you stop visiting
      it, so the bonus never permanently saturates.

States are fixed random φ-vectors (dim = trunk_dim), matching what RND sees in exp_013_1
(novelty is computed on φ, not pixels). CPU-only. Writes 3 PNGs + a numbers JSON.

    uv run python JEPA/experiments/exp_013_sparse_exploration/probes/leak_rnd_motivation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root for `import JEPA`
from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.rnd_phi import RNDPhi  # noqa: E402

DEVICE = torch.device("cpu")
DIM = 256
LR = 1e-4            # matches cfg.rnd_lr
LEAK = 0.01          # matches cfg.leak
STEPS_PER_VISIT = 4  # one "visit" = the batch-amplified update a recurring state gets in a
                     # real rollout (a state recurs ~dozens of times per 2048-step batch),
                     # so saturation is fast — not the artificially-slow 1-sample-1-step regime.
OUT = Path(__file__).resolve().parent / "leak_rnd_figs"
OUT.mkdir(parents=True, exist_ok=True)


def state(gen: torch.Generator) -> torch.Tensor:
    return torch.randn(1, DIM, generator=gen)


def fresh(leak: float, seed: int = 0):
    torch.manual_seed(seed)
    rnd = RNDPhi(dim=DIM, hidden=256, out=256, leak=leak).to(DEVICE)
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=LR)
    return rnd, opt


def visit(rnd: RNDPhi, opt, phi: torch.Tensor) -> None:
    """One 'visit' = STEPS_PER_VISIT distillation steps on phi, then the leak (once)."""
    for _ in range(STEPS_PER_VISIT):
        opt.zero_grad(set_to_none=True)
        rnd.distill_loss(phi).backward()
        opt.step()
    rnd.apply_leak()


# ----------------------------------------------------------------------------- Exp A & B
def saturation_curve(n_visits: int = 2000):
    """Visit ONE fixed state repeatedly; record novelty(s) before each visit."""
    gen = torch.Generator().manual_seed(123)
    s = state(gen)
    curves = {}
    for leak, key in [(0.0, "standard"), (LEAK, "leaky")]:
        rnd, opt = fresh(leak, seed=0)
        nov = np.empty(n_visits, dtype=np.float64)
        for i in range(n_visits):
            nov[i] = float(rnd.novelty(s))
            visit(rnd, opt, s)
        curves[key] = nov
    return curves


# ----------------------------------------------------------------------------- Exp C
def cycling_experiment(k_states: int = 10, rounds: int = 300):
    """The ARC regime: a SMALL fixed set of states (cf. ls20's ~43 reachable masked boards)
    the agent cycles through repeatedly. Once all are learned there are no 'new' states to
    cause interference-forgetting, so standard RND's novelty dies PERMANENTLY. Does the leak
    keep it alive? Track (a) mean novelty across the set vs round, and (b) the steady-state
    novelty-vs-recency profile (novelty as a function of how many visits ago a state was last seen)."""
    gen = torch.Generator().manual_seed(7)
    states = [state(gen) for _ in range(k_states)]
    out = {}
    for leak, key in [(0.0, "standard"), (LEAK, "leaky")]:
        rnd, opt = fresh(leak, seed=1)
        mean_per_round = np.empty(rounds, dtype=np.float64)
        for r in range(rounds):
            novs = []
            for k in range(k_states):
                novs.append(float(rnd.novelty(states[k])))
                visit(rnd, opt, states[k])
            mean_per_round[r] = float(np.mean(novs))
        # steady-state recency profile: after the last round, states[k] was last visited
        # (k_states-1-k) visits ago → larger lag = staler.
        profile = np.array([float(rnd.novelty(states[k])) for k in range(k_states)], dtype=np.float64)
        lag = np.array([k_states - 1 - k for k in range(k_states)], dtype=np.float64)
        order = np.argsort(lag)
        out[key] = {"mean_per_round": mean_per_round, "lag": lag[order], "profile": profile[order]}
    return out, k_states


# ----------------------------------------------------------------------------- figures
def _roll_med(a, w=15):
    """Rolling median — suppresses float32 jitter at the ~1e-13 floor (Adam nudging a
    perfectly-fit target); shows the typical novelty, not the FP noise."""
    half = w // 2
    pad = np.pad(a, half, mode="edge")
    return np.array([np.median(pad[i:i + w]) for i in range(len(a))])


def fig_saturation(curves):
    std, leaky = curves["standard"], curves["leaky"]
    x = np.arange(1, len(std) + 1)
    # scaled 1/sqrt(N) "ideal count novelty" reference (MBIE-EB), matched at N=1
    ideal = std[0] / np.sqrt(x)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, _roll_med(std), label="standard RND", lw=2.2, color="#c0392b")
    ax.plot(x, _roll_med(leaky), label=f"leaky RND (μ={LEAK})", lw=2.2, color="#2471a3")
    ax.plot(x, ideal, "--", color="#7f8c8d", lw=1.6, label=r"ideal count novelty $\propto 1/\sqrt{N}$")
    xmax = 300
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, xmax)
    ax.set_ylim(1e-15, 5)
    ax.set_xlabel("visits to the state")
    ax.set_ylabel("RND novelty  ½·‖P(φ)−T(φ)‖²")
    ax.set_title("Standard RND saturates to ~0; among visited states it has no resolution")
    ax.axvspan(50, xmax, color="#fbeeee", alpha=0.6)
    ax.text(120, 3e-7, "dead zone:\nnovelty ≈ 0 for every\nstate visited > ~50×\n(see fig.2 for N=100 vs 1000)",
            ha="center", va="center", fontsize=8.5, color="#c0392b")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_saturation.png", dpi=150)
    plt.close(fig)


def fig_discrimination(curves):
    std = curves["standard"]
    Ns = [1, 10, 100, 1000]
    rnd_vals = [std[n - 1] for n in Ns]
    ideal = [std[0] / np.sqrt(n) for n in Ns]
    xi = np.arange(len(Ns))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(xi - w / 2, rnd_vals, w, label="standard RND", color="#c0392b")
    ax.bar(xi + w / 2, ideal, w, label=r"ideal $1/\sqrt{N}$ count", color="#7f8c8d")
    ax.set_xticks(xi)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("novelty bonus")
    ax.set_title("RND can't grade visited states: novelty(100) ≈ novelty(1000)")
    for i, v in enumerate(rnd_vals):
        ax.text(xi[i] - w / 2, v, f"{v:.1e}", ha="center", va="bottom", fontsize=8, color="#c0392b")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_discrimination.png", dpi=150)
    plt.close(fig)


def fig_signal_survival(cyc, k):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, color in [("standard", "#c0392b"), ("leaky", "#2471a3")]:
        m = cyc[key]["mean_per_round"]
        lbl = "standard RND" if key == "standard" else f"leaky RND (μ={LEAK})"
        ax.plot(np.arange(1, len(m) + 1), m, lw=2.2, color=color, label=lbl)
    ax.set_xlabel(f"round (one round = revisit all {k} states once)")
    ax.set_ylabel("mean novelty over the state set")
    ax.set_title(f"Cycling a small fixed set of {k} states (the ARC regime)\n"
                 "standard RND's signal dies; leaky RND holds a live floor")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_signal_survival.png", dpi=150)
    plt.close(fig)


def fig_recency_profile(cyc, k):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, color in [("standard", "#c0392b"), ("leaky", "#2471a3")]:
        lbl = "standard RND" if key == "standard" else f"leaky RND (μ={LEAK})"
        ax.plot(cyc[key]["lag"], cyc[key]["profile"], "o-", lw=2.0, ms=5, color=color, label=lbl)
    ax.set_xlabel("visits since this state was last seen (staleness)")
    ax.set_ylabel("novelty (steady state)")
    ax.set_title("Leaky RND encodes RECENCY: staler states score higher\n"
                 "standard RND is flat at ~0 — it can't tell stale from fresh")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_recency_profile.png", dpi=150)
    plt.close(fig)


def main():
    print("[leak_rnd] running saturation curve ...")
    curves = saturation_curve()
    print("[leak_rnd] running cycling experiment ...")
    cyc, k = cycling_experiment()

    fig_saturation(curves)
    fig_discrimination(curves)
    fig_signal_survival(cyc, k)
    fig_recency_profile(cyc, k)

    std = curves["standard"]
    std_floor = float(cyc["standard"]["mean_per_round"][-1])
    leaky_floor = float(cyc["leaky"]["mean_per_round"][-1])
    nums = {
        "novelty_initial": float(std[0]),
        "novelty_N10": float(std[9]),
        "novelty_N100": float(std[99]),
        "novelty_N1000": float(std[999]),
        "rel_drop_N100_pct": float(100 * (1 - std[99] / std[0])),
        "rel_drop_N1000_pct": float(100 * (1 - std[999] / std[0])),
        "discrimination_100_vs_1000_ratio": float(std[99] / max(std[999], 1e-30)),
        "cycling_mean_novelty_floor_standard": std_floor,
        "cycling_mean_novelty_floor_leaky": leaky_floor,
        "cycling_leaky_over_standard": float(leaky_floor / max(std_floor, 1e-30)),
        "recency_profile_standard": [float(v) for v in cyc["standard"]["profile"]],
        "recency_profile_leaky": [float(v) for v in cyc["leaky"]["profile"]],
    }
    (OUT / "numbers.json").write_text(json.dumps(nums, indent=2))
    print(json.dumps(nums, indent=2))
    print("wrote figures + numbers ->", OUT)


if __name__ == "__main__":
    main()
