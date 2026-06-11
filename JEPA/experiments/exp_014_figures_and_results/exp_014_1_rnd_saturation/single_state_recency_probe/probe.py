"""exp_014_1 / single-state recency probe — RND error REGENERATES when a state
stops being visited.

The cleanest, most controlled isolation of the leaky-RND mechanism. Where
exp_014_2 abandons a state inside a full real roam, here we strip everything else
away and drive ONE anchor state by hand, in "updates" (a block of distill steps
followed by ONE leak — the real loop's cadence: distill, then a single per-update
forget):

  * Phase MEMORIZE (distill ON the anchor): for MEMORIZE_UPDATES updates we
    distill the predictor MEMORIZE_INNER steps on a single real masked board state
    A. A's RND error is driven DOWN to ~machine-zero — A becomes "known".
  * Phase ABANDON (distill on OTHER states): we never touch A again. Each update we
    pick a *different* random real state, do ABANDON_INNER=1 distill step on it
    (the real loop's rnd_epochs=1) + the one leak, then re-measure A's error. A is
    "not visited recently".

Standard RND (μ=0): A's error stays low — the predictor never forgets it, so A is
permanently "known" and yields no exploration signal if revisited.
Leaky RND (μ>0): A's error BOUNCES BACK UP as the predictor leaks toward its
random init while it is busy fitting the other states → novelty regenerates, at a
rate that grows with μ, climbing back toward A's original (un-trained) novelty.
That recovered error is the renewable, recency-based exploration signal standard
RND lacks.

ISOLATING THE LEAK (two deliberate, principled choices)
-------------------------------------------------------
A single shared predictor fit on a tiny state support has a cross-state
*interference* floor: distilling other states perturbs A's prediction even with no
leak. We keep that interference below the leak signal so the figure attributes the
regeneration to the leak, not to interference:
  1. The "other" states are a well-separated pool (greedy farthest-point in the
     frozen-projection feature space, seeded with A), so fitting them barely
     touches A directly. They are still real, frequently-visited states; "random" =
     a random draw from that separated pool each update.
  2. ABANDON_LR ≪ RND_LR. The leak is learning-rate-INDEPENDENT (a multiplicative
     shrink of the weights toward init), while interference scales with the distill
     lr; a small abandon lr drives residual interference toward zero (μ=0 stays
     flat/dead) without touching the leaky lines. At full lr (ABANDON_LR=RND_LR)
     the leaky lines are unchanged but μ=0 also creeps up to ~1e-3 via pure
     interference — the same regime as exp_014_2.

One shared anchor, one shared sequence of "other" states, identical RND init across
every μ — the ONLY difference between the lines is the leak. Real env, real RNDPhi,
timer-masked board identity. CPU-only. matplotlib Agg.

Run:
    uv run python -m JEPA.experiments.exp_014_figures_and_results.\
exp_014_1_rnd_saturation.single_state_recency_probe.probe
"""
from __future__ import annotations

import argparse
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
from JEPA.experiments.exp_014_figures_and_results.exp_014_1_rnd_saturation.diagnose import (
    mask_board, state_key, build_projection, project_states,
)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20")
    p.add_argument("--level", type=int, default=1, help="0-indexed level (1=L2, 2=L3)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


_ARGS = _parse_args()
DEVICE = torch.device("cpu")
GAME, LEVEL_INDEX, SEED = _ARGS.game, _ARGS.level, _ARGS.seed
N_ENVS, MAX_EP = 16, 200
PROJ_DIM, RND_LR = 256, 1e-4
ABANDON_LR = 1e-7                          # abandon-phase distill lr. The leak is lr-INDEPENDENT
                                          # (a multiplicative pull on weights), while cross-state
                                          # interference scales with lr; a small abandon lr pushes
                                          # interference below the leak signal so the leak is the
                                          # ISOLATED reviver of the anchor (μ=0 stays dead).
MU_VALUES = [0.0, 0.001, 0.01, 0.05]      # 0.0 = standard RND
PRE_ROAM_STEPS = 600                       # gather the pool of distinct masked states
MEMORIZE_INNER = 50                        # distill steps per memorize update (heavy: A is visited a lot)
MEMORIZE_UPDATES = 40                      # distill ON the anchor → drive its error to ~machine-zero
ABANDON_INNER = 1                          # distill steps per abandon update = the real loop's rnd_epochs=1
ABANDON_UPDATES = 200                      # distill on OTHER random states → watch A bounce up
N_OTHER = 12                               # size of the well-separated "other" pool we draw from

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
COLORS = {0.0: "#d62728", 0.001: "#1f77b4", 0.01: "#2ca02c", 0.05: "#9467bd"}  # match exp_014_1 saturation fig
LABELS = {0.0: "standard RND (μ=0)", 0.001: "leaky μ=0.001",
          0.01: "leaky μ=0.01", 0.05: "leaky μ=0.05"}
_TAG = f"{GAME}_L{LEVEL_INDEX + 1}"


def gather_pool(envs, rng):
    """Uniform-random pre-roam → distinct masked states + visit counts.

    Returns (pool_masked: (P,64,64) uint8, visit_counts: (P,) int) sorted by
    visit count descending, so pool[0] is the most-visited (our anchor)."""
    visit, exemplar = defaultdict(int), {}
    for _ in range(PRE_ROAM_STEPS):
        nobs = envs.step(rng.integers(0, envs.n_actions, size=envs.n_envs))[0]
        m = mask_board(nobs)
        for i in range(envs.n_envs):
            k = state_key(m[i])
            visit[k] += 1
            exemplar.setdefault(k, m[i])
    ranked = sorted(visit.items(), key=lambda kv: kv[1], reverse=True)
    pool_masked = np.stack([exemplar[k] for k, _ in ranked])
    counts = np.array([c for _, c in ranked], dtype=int)
    return pool_masked, counts


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    envs = VecLS20EnvLevel(env_name=GAME, n_envs=N_ENVS, max_episode_steps=MAX_EP,
                           seed=SEED, level_index=LEVEL_INDEX)
    W = build_projection(SEED)

    # ── pool of real distinct masked states; anchor = most-visited ──────────
    pool_masked, counts = gather_pool(envs, rng)
    n_states = len(pool_masked)
    feats_all = project_states(pool_masked, W).to(DEVICE)              # (P, PROJ_DIM)
    anchor_feat = feats_all[0:1]                                       # (1, PROJ_DIM) most-visited

    # "other" pool: greedy farthest-point selection in feature space, seeded with
    # the anchor (index 0), so the others are well-separated from A and from each
    # other → fitting them barely perturbs A, isolating the leak as A's reviver.
    f = feats_all.cpu().numpy()
    chosen = [0]
    while len(chosen) < min(1 + N_OTHER, n_states):
        d = np.min([np.linalg.norm(f - f[c], axis=1) for c in chosen], axis=0)
        d[chosen] = -1.0
        chosen.append(int(np.argmax(d)))
    other_idx = chosen[1:]                                            # exclude the anchor
    other_feats = feats_all[other_idx]                               # (N_OTHER, PROJ_DIM)
    n_other = other_feats.shape[0]
    print(f"[recency/{_TAG}] {n_states} distinct masked states; "
          f"anchor pre-roam visits={counts[0]}; {n_other} well-separated 'other' states")

    # Shared, pre-drawn sequence of "other" states for the ABANDON phase — identical
    # for every μ so the ONLY thing that differs between the lines is the leak.
    abandon_seq = rng.integers(0, n_other, size=ABANDON_UPDATES)

    # ── one RND per μ, IDENTICAL init across μ ──────────────────────────────
    rnds = {}
    for mu in MU_VALUES:
        torch.manual_seed(SEED)                # identical target + predictor init for all μ
        r = RNDPhi(dim=PROJ_DIM, hidden=256, out=256, leak=mu).to(DEVICE)
        rnds[mu] = (r, torch.optim.Adam(r.predictor.parameters(), lr=RND_LR))

    # err_hist[mu] = anchor error measured once per update (BEFORE that update's
    # distill), across both phases. Each update = n_inner distill steps + 1 leak.
    err_hist = {mu: [] for mu in MU_VALUES}

    def update(r, opt, feat, n_inner):
        for _ in range(n_inner):
            opt.zero_grad()
            r.distill_loss(feat).backward()
            opt.step()
        r.apply_leak()                                                  # ONE leak per update

    # ── Phase MEMORIZE: distill ON the anchor (heavy fit; A becomes "known") ─
    for _ in range(MEMORIZE_UPDATES):
        for mu, (r, opt) in rnds.items():
            err_hist[mu].append(float(r.novelty(anchor_feat).item()))   # measure A before update
            update(r, opt, anchor_feat, MEMORIZE_INNER)

    # ── Phase ABANDON: rnd_epochs=1 on OTHER random states; A is never touched ─
    for _, opt in rnds.values():                        # drop to the small abandon lr
        for g in opt.param_groups:
            g["lr"] = ABANDON_LR
    for t in range(ABANDON_UPDATES):
        feat = other_feats[abandon_seq[t]:abandon_seq[t] + 1]           # one random other state
        for mu, (r, opt) in rnds.items():
            err_hist[mu].append(float(r.novelty(anchor_feat).item()))   # measure A before update
            update(r, opt, feat, ABANDON_INNER)

    for mu in MU_VALUES:                                                # final post-loop reading
        err_hist[mu].append(float(rnds[mu][0].novelty(anchor_feat).item()))

    # ── figure: anchor error vs update, memorize | abandon ──────────────────
    MEMORIZE_STEPS = MEMORIZE_UPDATES        # x-axis is in updates
    ABANDON_STEPS = ABANDON_UPDATES
    n = MEMORIZE_UPDATES + ABANDON_UPDATES + 1
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.axvspan(0, MEMORIZE_STEPS, color="#fdebd0", alpha=0.6)
    ax.axvspan(MEMORIZE_STEPS, n, color="#eaf2f8", alpha=0.6)
    ax.text(MEMORIZE_STEPS / 2, 0.96, "MEMORIZE A\n(distill on A → error ↓)",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.5, color="#9c640c")
    ax.text(MEMORIZE_STEPS + ABANDON_STEPS * 0.5, 0.96,
            "ABANDON A\n(distill on OTHER states → A error ↑?)",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.5, color="#1f618d")
    ax.axvline(MEMORIZE_STEPS, color="#566573", lw=1, ls="--")
    for mu in MU_VALUES:
        y = np.clip(err_hist[mu], 1e-16, None)
        ax.plot(x, y, lw=2.0, color=COLORS[mu], label=LABELS[mu], alpha=0.95)
    ax.set_yscale("log")
    ax.set_xlabel("update  (memorize: distill on A · abandon: 1 distill on another state, + 1 leak each)")
    ax.set_ylabel("RND error at anchor A   ½·mean (P−T)²")
    ax.set_title(
        "A state not visited recently regenerates its RND error under the leak\n"
        f"real {GAME.upper()}-L{LEVEL_INDEX + 1} ({n_states} states) — "
        "standard RND stays 'known'; leaky bounces back up")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    out = FIG / f"single_state_recency_{_TAG}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # ── overlay-style figure: matches exp_014_1 `rnd_saturation_vs_visits` ───
    # Same colors / log-y novelty / single state, so it can sit beside (or be
    # overlaid on) one saturation panel. Phase 1 (state visited) = the SATURATION
    # drop; phase 2 (state not visited recently) = where standard RND stays flat
    # and DEAD but the leaky curve turns BACK UP toward A's original novelty.
    init_nov = float(np.mean([err_hist[mu][0] for mu in MU_VALUES]))   # A's un-trained novelty ceiling
    figs, axs = plt.subplots(figsize=(7.6, 4.7))
    axs.axvspan(0, MEMORIZE_STEPS, color="#f6f6f6", alpha=0.9)
    axs.axvline(MEMORIZE_STEPS, color="#566573", lw=1, ls="--")
    axs.axhline(init_nov, color="#7f8c8d", lw=1, ls=":", alpha=0.8)
    axs.text(n * 0.985, init_nov, " A's original novelty", color="#7f8c8d",
             fontsize=8, ha="right", va="bottom")
    axs.text(MEMORIZE_STEPS / 2, 0.27, "state VISITED\n→ novelty\nsaturates ↓",
             transform=axs.get_xaxis_transform(), ha="center", va="center",
             fontsize=8.5, color="#566573")
    axs.text(MEMORIZE_STEPS + ABANDON_STEPS * 0.52, 0.30,
             "state NOT visited recently\n→ leaky novelty revives ↑  (μ=0 stays dead)",
             transform=axs.get_xaxis_transform(), ha="center", va="center",
             fontsize=8.5, color="#566573")
    for mu in MU_VALUES:
        y = np.clip(err_hist[mu], 1e-16, None)
        axs.plot(x, y, lw=1.9, color=COLORS[mu], label=LABELS[mu], alpha=0.92)
    axs.set_yscale("log")
    axs.set_xlabel("update  (state visited ·│· state abandoned)")
    axs.set_ylabel("RND novelty  (½·mean (P−T)²)")     # matches the saturation figure's y-label
    axs.set_title(
        "Single-state recency: RND novelty saturates when visited, then REVIVES when not\n"
        f"real {GAME.upper()}-L{LEVEL_INDEX + 1} — standard RND (μ=0) saturates flat and stays dead; "
        "leaky RND climbs back up",
        fontsize=10)
    axs.grid(True, which="both", alpha=0.25)
    axs.legend(frameon=False, fontsize=8.5, loc="lower right")
    figs.tight_layout()
    out_sat = FIG / f"single_state_saturate_then_revive_{_TAG}.png"
    figs.savefig(out_sat, dpi=150)
    plt.close(figs)
    print("wrote", out_sat)

    # ── key numbers + results json ──────────────────────────────────────────
    mem_end = MEMORIZE_STEPS - 1          # last reading taken DURING the memorize phase
    nums = {}
    for mu in MU_VALUES:
        floor = float(np.min(err_hist[mu][:MEMORIZE_STEPS]))   # how low A got while memorized
        e0 = float(err_hist[mu][0])                            # A's novelty before any training
        e_mem = float(err_hist[mu][mem_end])
        e_end = float(err_hist[mu][-1])                        # A after the abandon phase
        nums[str(mu)] = {
            "err_initial": e0,
            "err_memorize_floor": floor,
            "err_end_memorize": e_mem,
            "err_end_abandon": e_end,
            "regen_ratio_vs_floor": e_end / max(floor, 1e-30),
            "recovered_frac_of_initial": (e_end - floor) / max(e0 - floor, 1e-30),
        }
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "probe": "single_state_recency",
        "game": GAME, "level_index": LEVEL_INDEX, "level_human": LEVEL_INDEX + 1,
        "seed": SEED, "device": str(DEVICE),
        "n_distinct_states": int(n_states), "anchor_pre_roam_visits": int(counts[0]),
        "proj_dim": PROJ_DIM, "rnd_lr": RND_LR, "abandon_lr": ABANDON_LR,
        "mu_values": MU_VALUES,
        "memorize_inner": MEMORIZE_INNER, "abandon_inner": ABANDON_INNER,
        "memorize_updates": MEMORIZE_UPDATES, "abandon_updates": ABANDON_UPDATES,
        "n_other_pool": n_other,
        "by_mu": nums,
        "err_hist": {str(mu): err_hist[mu] for mu in MU_VALUES},
        "figure": str(out),
        "figure_saturate_then_revive": str(out_sat),
    }
    (RES / f"single_state_recency_{_TAG}.json").write_text(json.dumps(summary, indent=2))

    print("\n=== anchor error: memorize-floor → end-of-abandon (recency regeneration) ===")
    for mu in MU_VALUES:
        d = nums[str(mu)]
        tag = "standard RND" if mu == 0.0 else f"leaky μ={mu}"
        print(f"  {tag:<16} floor={d['err_memorize_floor']:.2e} → "
              f"end={d['err_end_abandon']:.2e}  ({d['regen_ratio_vs_floor']:.1f}× rebound)")
    print("wrote", out)


if __name__ == "__main__":
    main()
