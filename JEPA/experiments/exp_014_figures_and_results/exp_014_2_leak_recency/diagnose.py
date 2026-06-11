"""exp_014_2 — The LEAK EFFECT: an abandoned state's RND error REGENERATES.

Direct demonstration of why leaky RND is a *recency* (not lifetime) signal, on a REAL LS20-L2
state. We pick one frequently-visited masked board state A, then:

  * Phase 1 (train ON A): the agent visits A; we distill the predictor on A (among others), so
    A's RND error is driven DOWN (A becomes "known").
  * Phase 2 (ABANDON A): the agent has moved on — we keep roaming/training on every OTHER visited
    state but NEVER on A again, and measure A's error every update (continuously feeding A's
    representation through target+predictor).

Standard RND (μ=0): A's error stays low forever — the predictor never forgets it.
Leaky RND (μ>0): A's error climbs back up as the predictor leaks toward its random init →
novelty REGENERATES (rate grows with μ). That recovered error is the renewable exploration
signal standard RND lacks.

One shared visit stream across all μ (identical roam); the ONLY difference is the leak. Real env,
real RNDPhi, timer-masked board identity, deduped distinct states. CPU-only. matplotlib Agg.

    uv run python -m JEPA.experiments.exp_014_figures_and_results.exp_014_2_leak_recency.diagnose
"""
from __future__ import annotations

import json
from collections import defaultdict
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20")
    p.add_argument("--level", type=int, default=1, help="0-indexed level (1=L2, 2=L3)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


_ARGS = _parse_args()
DEVICE = torch.device("cpu")
GAME, LEVEL_INDEX = _ARGS.game, _ARGS.level
SEED = _ARGS.seed
N_ENVS, ROLLOUT_STEPS, MAX_EP = 16, 128, 200
PROJ_DIM, RND_LR, RND_EPOCHS = 256, 1e-4, 1   # 1 = faithful to the real loop (rnd_epochs=1)
MU_VALUES = [0.0, 0.01, 0.05]
PRE_ROAM_STEPS = 600          # pick A = most-visited distinct masked state
PHASE1_UPDATES = 40           # train ON A → drive its error down
PHASE2_UPDATES = 140          # ABANDON A → watch its error regenerate (leaky) / stay low (std)
HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
FIG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
COLORS = {0.0: "#c0392b", 0.01: "#2ca02c", 0.05: "#9467bd"}
LABELS = {0.0: "standard RND (μ=0)", 0.01: "leaky μ=0.01", 0.05: "leaky μ=0.05"}
_LEVEL_TAG = f"{GAME}_L{LEVEL_INDEX + 1}"


def roam_unique(envs, rng):
    """One update of roaming → dict {state_key: masked_board} of DISTINCT visited states
    (dedup keeps the projected batch tiny: only ~43 distinct masked boards exist)."""
    seen = {}
    for _ in range(ROLLOUT_STEPS):
        a = rng.integers(0, envs.n_actions, size=envs.n_envs)
        nobs, _r, dones, _i = envs.step(a)
        m = mask_board(nobs)
        for i in range(envs.n_envs):
            if not dones[i]:
                seen.setdefault(state_key(m[i]), m[i])
    return seen


def main():
    rng = np.random.default_rng(SEED)
    envs = VecLS20EnvLevel(env_name=GAME, n_envs=N_ENVS, max_episode_steps=MAX_EP,
                           seed=SEED, level_index=LEVEL_INDEX)
    W = build_projection(SEED)

    # --- pre-roam: pick A = most-visited distinct masked state ---
    visit, exemplar = defaultdict(int), {}
    for _ in range(PRE_ROAM_STEPS):
        nobs = envs.step(rng.integers(0, envs.n_actions, size=envs.n_envs))[0]
        m = mask_board(nobs)
        for i in range(envs.n_envs):
            k = state_key(m[i]); visit[k] += 1; exemplar.setdefault(k, m[i])
    A_key = max(visit, key=visit.get)
    A_feat = project_states(exemplar[A_key][None], W).to(DEVICE)        # (1, PROJ_DIM)
    print(f"[exp_014_2/{_LEVEL_TAG}] picked A (pre-roam visits={visit[A_key]}); {len(visit)} distinct masked states")

    # --- one RND per μ, IDENTICAL init ---
    rnds = {}
    for mu in MU_VALUES:
        torch.manual_seed(SEED)                       # identical target+predictor init across μ
        r = RNDPhi(dim=PROJ_DIM, hidden=256, out=256, leak=mu).to(DEVICE)
        rnds[mu] = (r, torch.optim.Adam(r.predictor.parameters(), lr=RND_LR))

    err_hist = {mu: [] for mu in MU_VALUES}
    a_seen_flags = []
    for u in range(PHASE1_UPDATES + PHASE2_UPDATES):
        phase2 = u >= PHASE1_UPDATES
        for mu, (r, _) in rnds.items():               # measure A BEFORE this update's distill
            err_hist[mu].append(float(r.novelty(A_feat).item()))
        seen = roam_unique(envs, rng)
        a_seen_flags.append(A_key in seen)
        keys = [k for k in seen if not (phase2 and k == A_key)]   # abandon A in phase 2
        if keys:
            feats = project_states(np.stack([seen[k] for k in keys]), W).to(DEVICE)
        for mu, (r, opt) in rnds.items():
            if keys:
                for _ in range(RND_EPOCHS):
                    opt.zero_grad(); r.distill_loss(feats).backward(); opt.step()
            r.apply_leak()                            # per-update forget, exactly as the real loop
        if u % 20 == 0:
            print(f"[exp_014_2/{_LEVEL_TAG}] u{u} {'P2-abandon' if phase2 else 'P1-train '} "
                  + " ".join(f"μ{mu}={err_hist[mu][-1]:.2e}" for mu in MU_VALUES), flush=True)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    n = PHASE1_UPDATES + PHASE2_UPDATES
    x = np.arange(n)
    ax.axvspan(0, PHASE1_UPDATES, color="#fdebd0", alpha=0.6)
    ax.axvspan(PHASE1_UPDATES, n, color="#eaf2f8", alpha=0.6)
    ax.text(PHASE1_UPDATES / 2, 0.94, "Phase 1: visiting A\n(error driven down)",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color="#9c640c")
    ax.text(PHASE1_UPDATES + PHASE2_UPDATES * 0.5, 0.94, "Phase 2: A ABANDONED\n(agent moved on)",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color="#1f618d")
    ax.axvline(PHASE1_UPDATES, color="#566573", lw=1, ls="--")
    for mu in MU_VALUES:
        ax.plot(x, err_hist[mu], lw=2.2, color=COLORS[mu], label=LABELS[mu])
    ax.set_yscale("log")
    ax.set_xlabel("update")
    ax.set_ylabel("RND error at state A  ½·‖P(φ)−T(φ)‖²")
    n_states = len(visit)
    _title = (
        "An abandoned state's novelty REGENERATES under the leak\n"
        f"real {GAME.upper()}-L{LEVEL_INDEX+1} ({n_states} states) — standard RND stays 'known'; leaky climbs back"
        if LEVEL_INDEX <= 1 else
        "Leaky RND maintains novelty floor; standard RND collapses via generalization\n"
        f"real {GAME.upper()}-L{LEVEL_INDEX+1} ({n_states} states) — "
        f"{err_hist[0.0][-1]:.1e} (std) vs {err_hist[0.05][-1]:.1e} (μ=0.05) at end"
    )
    ax.set_title(_title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    out = FIG / f"leak_recency_abandoned_state_{_LEVEL_TAG}.png"
    fig.savefig(out, dpi=150); plt.close(fig)

    nums = {}
    for mu in MU_VALUES:
        p1, p2 = err_hist[mu][PHASE1_UPDATES - 1], err_hist[mu][-1]
        nums[str(mu)] = {"err_end_phase1": p1, "err_end_phase2": p2, "regen_ratio": p2 / max(p1, 1e-30)}
    json.dump({"game": GAME, "level_index": LEVEL_INDEX, "n_distinct_states": len(visit),
               "A_pre_roam_visits": int(visit[A_key]), "phase1": PHASE1_UPDATES,
               "phase2": PHASE2_UPDATES, "rnd_epochs": RND_EPOCHS, "by_mu": nums,
               "err_hist": {str(mu): err_hist[mu] for mu in MU_VALUES}},
              open(RES / f"leak_recency_results_{_LEVEL_TAG}.json", "w"), indent=2)
    print("\n=== regeneration after abandonment (errA: end-P1 → end-P2) ===")
    for mu in MU_VALUES:
        print(f"  μ={mu}: {nums[str(mu)]['err_end_phase1']:.2e} → {nums[str(mu)]['err_end_phase2']:.2e} "
              f"({nums[str(mu)]['regen_ratio']:.1f}×)")
    print("wrote", out)


if __name__ == "__main__":
    main()
