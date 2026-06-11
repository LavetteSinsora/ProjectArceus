"""Recency, side-by-side: a VISITED state vs an ABANDONED state, over a 100k–120k
environment-step window, for standard RND vs leaky RND (μ=0.01, 0.05).

Two real, well-separated LS20-L2 masked states are both memorized before the window
(predictor error ~0 by step 100k). Then, across the window:
  * State A is KEPT frequently visited  → distilled every update.
  * State B is ABANDONED                → never distilled again.
A leak fires once per update (the real loop's cadence).

What the figure shows (y = RND novelty, log):
  * Leaky RND: the ABANDONED state's novelty BOUNCES UP (the leak pulls the predictor
    back toward init where it is no longer refit); the VISITED state stays flat-low
    (it keeps being refit). → novelty tracks RECENCY.
  * Standard RND (μ=0): BOTH stay flat-low — abandoning a state does nothing, the
    predictor never forgets it. → novelty tracks lifetime visits, not recency.

solid = frequently visited (A) · dashed = not visited / abandoned (B). No title.
Real env, real RNDPhi, frozen projection, timer-masked board identity. CPU-only.

Run:
    uv run python -m JEPA.experiments.exp_014_figures_and_results.\
exp_014_1_rnd_saturation.single_state_recency_probe.visited_vs_abandoned
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi
from JEPA.experiments.exp_014_figures_and_results.exp_014_1_rnd_saturation import diagnose as D

DEVICE = torch.device("cpu")
SEED = 0
MU_VALUES = [0.0, 0.01, 0.05]                 # standard + two leaks
PROJ_DIM, RND_LR = 256, 1e-4
PRE_ROAM_STEPS = 600
WARMUP_INNER, WARMUP_UPDATES = 60, 40         # fully memorize BOTH states by step 100k, so the
                                              # VISITED line is genuinely flat in the window (standard
                                              # RND has no floor → it converges to ~machine-zero)
DISPLAY_FLOOR = 1e-8                           # clamp for the log plot (standard RND ≈ 0); like the
                                              # saturation figure's np.clip — keeps a compact y-range
WINDOW_INNER = 20                             # distill steps/update on the visited state in the window
WINDOW_UPDATES = 120                          # updates spanning the plotted step window
STEP_START, STEP_END = 100_000, 120_000       # x-axis env-step window

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
COLORS = {0.0: "#d62728", 0.01: "#2ca02c", 0.05: "#9467bd"}
MLABEL = {0.0: "standard RND (μ=0)", 0.01: "leaky μ=0.01", 0.05: "leaky μ=0.05"}
_TAG = "ls20_L2"


def pick_two_states(envs, rng, W):
    """Pre-roam → two frequently-visited, well-separated real masked states.
    A = most-visited; B = farthest from A in projection space (so distilling A
    barely perturbs B → B's behaviour reflects the leak, not interference)."""
    visit, exemplar = defaultdict(int), {}
    for _ in range(PRE_ROAM_STEPS):
        nobs = envs.step(rng.integers(0, envs.n_actions, size=envs.n_envs))[0]
        m = D.mask_board(nobs)
        for i in range(envs.n_envs):
            k = D.state_key(m[i]); visit[k] += 1; exemplar.setdefault(k, m[i])
    ranked = sorted(visit, key=visit.get, reverse=True)
    pool = ranked[: max(12, 2)]
    feats = D.project_states(np.stack([exemplar[k] for k in pool]), W).numpy()
    b = int(np.argmax(np.linalg.norm(feats - feats[0], axis=1)))      # farthest from A(=pool[0])
    A, B = exemplar[pool[0]], exemplar[pool[b]]
    return A, B, visit[pool[0]], visit[pool[b]]


def main():
    FIG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    envs = VecLS20EnvLevel(env_name=D.GAME, n_envs=D.N_ENVS,
                           max_episode_steps=D.MAX_EPISODE_STEPS, seed=SEED, level_index=1)
    rng = np.random.default_rng(SEED)
    W = D.build_projection(seed=SEED + 1)

    A_m, B_m, visA, visB = pick_two_states(envs, rng, W)
    A_feat = D.project_states(A_m[None], W).to(DEVICE)               # (1, dim) visited
    B_feat = D.project_states(B_m[None], W).to(DEVICE)               # (1, dim) abandoned
    both = torch.cat([A_feat, B_feat], 0)
    print(f"[visited_vs_abandoned] A pre-roam visits={visA}  B pre-roam visits={visB}")

    rnds = {}
    for mu in MU_VALUES:
        torch.manual_seed(SEED + 100)
        r = RNDPhi(dim=PROJ_DIM, hidden=D.RND_HIDDEN, out=D.RND_OUT, leak=mu).to(DEVICE)
        rnds[mu] = (r, torch.optim.Adam(r.predictor.parameters(), lr=RND_LR))

    orig_nov = float(np.mean([float(rnds[mu][0].novelty(A_feat).item()) for mu in MU_VALUES]))

    # ── warm-up (pre-window): memorize BOTH A and B → ~0 by step 100k ───────
    for _ in range(WARMUP_UPDATES):
        for r, opt in rnds.values():
            for _e in range(WARMUP_INNER):
                opt.zero_grad(); r.distill_loss(both).backward(); opt.step()
            r.apply_leak()

    # ── window 100k→120k: keep visiting A; abandon B; measure both ──────────
    x_steps = np.linspace(STEP_START, STEP_END, WINDOW_UPDATES)
    novA = {mu: [] for mu in MU_VALUES}
    novB = {mu: [] for mu in MU_VALUES}
    for _ in range(WINDOW_UPDATES):
        for mu, (r, opt) in rnds.items():
            novA[mu].append(float(r.novelty(A_feat).item()))         # measure before update
            novB[mu].append(float(r.novelty(B_feat).item()))
            for _e in range(WINDOW_INNER):                           # A is visited → keep distilling A
                opt.zero_grad(); r.distill_loss(A_feat).backward(); opt.step()
            r.apply_leak()                                           # B is abandoned (never distilled)

    # ── figure (no title) ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    for mu in MU_VALUES:
        c = COLORS[mu]
        ax.plot(x_steps, np.clip(novA[mu], DISPLAY_FLOOR, None), color=c, lw=2.2, ls="-", alpha=0.95)
        ax.plot(x_steps, np.clip(novB[mu], DISPLAY_FLOOR, None), color=c, lw=2.2, ls="--", alpha=0.95)
    ax.axhline(orig_nov, color="#7f8c8d", lw=1, ls=":", alpha=0.8)
    ax.text(STEP_START, orig_nov, " original novelty", color="#7f8c8d",
            fontsize=8, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(DISPLAY_FLOOR * 0.6, orig_nov * 3)
    ax.set_xlim(STEP_START, STEP_END)
    ax.set_xlabel("environment steps (ls20)")
    ax.set_ylabel("RND novelty  ½·mean (P−T)²")
    ax.grid(True, which="both", alpha=0.25)
    # two legends: colour = method, linestyle = visited/abandoned
    meth_h = [plt.Line2D([0], [0], color=COLORS[mu], lw=2.2, label=MLABEL[mu]) for mu in MU_VALUES]
    style_h = [plt.Line2D([0], [0], color="#444", lw=2.2, ls="-", label="frequently visited"),
               plt.Line2D([0], [0], color="#444", lw=2.2, ls="--", label="not visited (abandoned)")]
    leg1 = ax.legend(handles=meth_h, frameon=False, fontsize=8.5, loc="center left")
    ax.add_artist(leg1)
    ax.legend(handles=style_h, frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    out = FIG / f"visited_vs_abandoned_{_TAG}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)

    # ── numbers + json ──────────────────────────────────────────────────────
    nums = {}
    for mu in MU_VALUES:
        nums[str(mu)] = {
            "visited_start": novA[mu][0], "visited_end": novA[mu][-1],
            "abandoned_start": novB[mu][0], "abandoned_end": novB[mu][-1],
            "abandoned_rise_ratio": novB[mu][-1] / max(novB[mu][0], 1e-30),
        }
    (RES / f"visited_vs_abandoned_{_TAG}.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "game": "ls20", "level_index": 1, "mu_values": MU_VALUES,
        "step_window": [STEP_START, STEP_END], "window_updates": WINDOW_UPDATES,
        "A_pre_roam_visits": int(visA), "B_pre_roam_visits": int(visB),
        "original_novelty": orig_nov, "by_mu": nums,
        "novA": novA, "novB": novB, "x_steps": x_steps.tolist(), "figure": str(out),
    }, indent=2))
    print("\n=== abandoned-state novelty over the window (start → end) ===")
    for mu in MU_VALUES:
        n = nums[str(mu)]
        tag = "standard RND" if mu == 0.0 else f"leaky μ={mu}"
        print(f"  {tag:<16} visited {n['visited_start']:.2e}→{n['visited_end']:.2e}   "
              f"abandoned {n['abandoned_start']:.2e}→{n['abandoned_end']:.2e} "
              f"({n['abandoned_rise_ratio']:.1f}×)")


if __name__ == "__main__":
    main()
