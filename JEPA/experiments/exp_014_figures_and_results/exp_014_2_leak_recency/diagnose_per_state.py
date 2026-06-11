"""exp_014_2 — Per-state novelty figures for the leak-recency experiment.

Same controlled abandonment as diagnose.py:
  Phase 1: random-policy roam; ALL distinct states including A are trained on.
  Phase 2: A is abandoned; all OTHER states keep being visited and trained on.

NEW in this script:
  • Novelty is measured for EVERY distinct masked state at the START of each rollout
    (before distillation), giving one measurement per 2048 env-steps per state.
  • Visit events are logged with their EXACT cumulative env-step position
    (update * N_ENVS * ROLLOUT_STEPS + rollout_step * N_ENVS + env_idx).
  • One figure is saved per state, showing:
      - x-axis: cumulative env steps
      - y-axis: novelty ½·‖P(φ)−T(φ)‖² (log scale)
      - One curve per μ (shared roam, only the predictor differs)
      - Rug plot at y_min: tick marks at each env-step where this state was visited
      - Vertical dashed line + shading for Phase 1 / Phase 2 boundary
      - Annotation: state index, visit count by phase, μ=0 vs μ=0.05 end-ratio

The user picks the figures where the signal is clearest: a long visit-free Phase 2
gap with the leaky curve climbing while μ=0 stays flat.

    uv run python -m JEPA.experiments.exp_014_figures_and_results.exp_014_2_leak_recency.diagnose_per_state
    uv run ... --level 1   # L2 (43 states — more variety in visit patterns)
    uv run ... --level 2   # L3 (12 states — cleanest abandonment for state A)
"""
from __future__ import annotations

import argparse
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


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20")
    p.add_argument("--level", type=int, default=2, help="0-indexed (2=L3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase1", type=int, default=60,
                   help="updates where A is visited (drives novelty down)")
    p.add_argument("--phase2", type=int, default=260,
                   help="updates where A is ABANDONED (watch novelty trend)")
    p.add_argument("--pure-abandon", action="store_true",
                   help="Phase 2: NO distillation on ANY state — only leak acts. "
                        "Gives the clearest rising-novelty signal: μ=0 stays flat, "
                        "μ>0 climbs toward random init.")
    return p.parse_args()


DEVICE = torch.device("cpu")
MU_VALUES = [0.0, 0.01, 0.05]
COLORS = {0.0: "#c0392b", 0.01: "#2ca02c", 0.05: "#9467bd"}
LABELS = {0.0: "std RND (μ=0)", 0.01: "leaky μ=0.01", 0.05: "leaky μ=0.05"}
N_ENVS = 16
ROLLOUT_STEPS = 128
STEPS_PER_UPDATE = N_ENVS * ROLLOUT_STEPS    # 2048
PRE_ROAM_STEPS = 600
PROJ_DIM = 256
RND_LR = 1e-4
RND_EPOCHS = 1


def main():
    cfg = _parse()
    GAME, LEVEL_INDEX, SEED = cfg.game, cfg.level, cfg.seed
    PHASE1, PHASE2 = cfg.phase1, cfg.phase2
    PURE_ABANDON = cfg.pure_abandon
    N_TOTAL = PHASE1 + PHASE2

    rng = np.random.default_rng(SEED)
    envs = VecLS20EnvLevel(env_name=GAME, n_envs=N_ENVS, max_episode_steps=200,
                           seed=SEED, level_index=LEVEL_INDEX)
    W = build_projection(SEED)

    HERE = Path(__file__).resolve().parent
    LEVEL_TAG = f"{GAME}_L{LEVEL_INDEX+1}"
    suffix = "_pure_abandon" if PURE_ABANDON else ""
    FIG_DIR = HERE / "figures" / f"per_state_{LEVEL_TAG}{suffix}"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR = HERE / "results"
    RES_DIR.mkdir(exist_ok=True)

    # ── pre-roam: pick A = most-visited distinct state ──────────────────────
    visit_count: dict[bytes, int] = defaultdict(int)
    exemplar: dict[bytes, np.ndarray] = {}
    for _ in range(PRE_ROAM_STEPS):
        a = rng.integers(0, envs.n_actions, size=N_ENVS)
        nobs = envs.step(a)[0]
        m = mask_board(nobs)
        for i in range(N_ENVS):
            k = state_key(m[i]); visit_count[k] += 1; exemplar.setdefault(k, m[i])
    A_key = max(visit_count, key=visit_count.get)
    state_keys = list(exemplar.keys())                 # stable ordering
    state_idx = {k: i for i, k in enumerate(state_keys)}
    print(f"[per_state/{GAME}_L{LEVEL_INDEX+1}] {len(state_keys)} distinct states; "
          f"A = state_{state_idx[A_key]} (pre-roam visits={visit_count[A_key]})")

    # ── initialise one RND per μ (identical init across all μ) ───────────────
    rnds: dict[float, tuple[RNDPhi, torch.optim.Adam]] = {}
    for mu in MU_VALUES:
        torch.manual_seed(SEED)
        r = RNDPhi(dim=PROJ_DIM, hidden=256, out=256, leak=mu).to(DEVICE)
        rnds[mu] = (r, torch.optim.Adam(r.predictor.parameters(), lr=RND_LR))

    # storage: novelty_series[state_key][mu] = [(cumstep, novelty), ...]
    novelty_series: dict[bytes, dict[float, list[tuple[int, float]]]] = {
        k: {mu: [] for mu in MU_VALUES} for k in state_keys
    }
    # train_steps[state_key] = [cumstep, ...] — steps where state was in the TRAINING BATCH
    # (not physical visits; shows when distillation actually happened → Phase 2 is empty for A)
    train_steps: dict[bytes, list[int]] = defaultdict(list)

    # ── main loop ─────────────────────────────────────────────────────────────
    for u in range(N_TOTAL):
        phase2 = u >= PHASE1
        cum_step_base = u * STEPS_PER_UPDATE   # cumulative env steps at rollout start

        # 1. measure novelty for ALL known states BEFORE this rollout's distillation
        feat_dict = {}
        for k in state_keys:
            feat_dict[k] = project_states(exemplar[k][None], W).to(DEVICE)  # (1, PROJ_DIM)
        for k in state_keys:
            for mu, (r, _) in rnds.items():
                with torch.no_grad():
                    nov = r.novelty(feat_dict[k]).item()
                novelty_series[k][mu].append((cum_step_base, nov))

        # 2. collect rollout
        seen_this: dict[bytes, np.ndarray] = {}
        for s in range(ROLLOUT_STEPS):
            a = rng.integers(0, envs.n_actions, size=N_ENVS)
            nobs, _r, dones, _i = envs.step(a)
            m = mask_board(nobs)
            for i in range(N_ENVS):
                if not dones[i]:
                    k = state_key(m[i])
                    if k in exemplar:
                        seen_this.setdefault(k, m[i])

        # 3. distil on visited states
        # pure_abandon: Phase 2 skips ALL training (only leak acts) → cleanest rising signal
        # normal: Phase 2 skips only A (leak vs generalization from neighbours)
        if phase2 and PURE_ABANDON:
            keys_to_train = []
        else:
            keys_to_train = [k for k in seen_this if not (phase2 and k == A_key)]
        if keys_to_train:
            feats = project_states(
                np.stack([seen_this[k] for k in keys_to_train]), W).to(DEVICE)
            for mu, (r, opt) in rnds.items():
                for _ in range(RND_EPOCHS):
                    opt.zero_grad(); r.distill_loss(feats).backward(); opt.step()
        # record training-batch membership — each trained state gets ONE tick at the
        # midpoint of this rollout (represents "distillation happened here")
        tick_step = cum_step_base + STEPS_PER_UPDATE // 2
        for k in keys_to_train:
            train_steps[k].append(tick_step)

        # 4. apply leak
        for mu, (r, _) in rnds.items():
            r.apply_leak()

        if u % 40 == 0:
            nov_A = {mu: novelty_series[A_key][mu][-1][1] for mu in MU_VALUES}
            print(f"  u{u:>4} {'P2-abandon' if phase2 else 'P1-train '} "
                  + " ".join(f"μ{mu}={nov_A[mu]:.2e}" for mu in MU_VALUES), flush=True)

    # ── save per-state figures ─────────────────────────────────────────────────
    total_steps = N_TOTAL * STEPS_PER_UPDATE
    phase_boundary_step = PHASE1 * STEPS_PER_UPDATE

    saved = []
    for k in state_keys:
        idx = state_idx[k]
        is_A = (k == A_key)
        tsteps = np.array(train_steps[k])     # training-batch membership ticks

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.axvspan(0, phase_boundary_step, color="#fdebd0", alpha=0.5)
        ax.axvspan(phase_boundary_step, total_steps, color="#eaf2f8", alpha=0.5)
        ax.axvline(phase_boundary_step, color="#566573", lw=1, ls="--")

        for mu in MU_VALUES:
            xs = [x for x, _ in novelty_series[k][mu]]
            ys = [y for _, y in novelty_series[k][mu]]
            ax.plot(xs, ys, lw=2.2, color=COLORS[mu], label=LABELS[mu], zorder=3)

        # rug plot: training-batch ticks — x in data coords, y in axes fraction (0=bottom)
        # Each tick = one distillation event for this state.
        # Phase 1: regular ticks; Phase 2 for A: empty (abandonment).
        if len(tsteps) > 0:
            from matplotlib.transforms import blended_transform_factory
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.vlines(tsteps, 0, 0.06, transform=trans,
                      color="#2c3e50", alpha=0.55, linewidth=1.4, zorder=4)

        # count ticks in each phase
        n_p1 = int(np.sum(tsteps < phase_boundary_step)) if len(tsteps) > 0 else 0
        n_p2 = int(np.sum(tsteps >= phase_boundary_step)) if len(tsteps) > 0 else 0

        # phase labels
        ax.text(phase_boundary_step * 0.5, 0.96,
                f"Phase 1: state trained\n({n_p1} distil events)",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color="#9c640c")
        if PURE_ABANDON:
            p2_label = "ALL training stopped\n(leak only — μ>0 recovers)"
        elif is_A:
            p2_label = "A ABANDONED\n(0 distil events)"
        else:
            p2_label = f"state trained\n({n_p2} distil events)"
        ax.text(phase_boundary_step + (total_steps - phase_boundary_step) * 0.5, 0.96,
                f"Phase 2: {p2_label}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color="#1f618d")

        ax.set_yscale("log")
        ax.set_xlabel("cumulative env steps", fontsize=10)
        ax.set_ylabel("RND novelty  ½·‖P(φ)−T(φ)‖²", fontsize=9)

        end_std = novelty_series[k][0.0][-1][1]
        end_lk = novelty_series[k][0.05][-1][1]
        ratio = end_lk / max(end_std, 1e-30)
        title_tag = " ★ A (most-trained → abandoned)" if is_A else f" state {idx}"
        ax.set_title(f"{GAME.upper()}-L{LEVEL_INDEX+1}{title_tag}\n"
                     f"end-P2: μ=0.05 is {ratio:.1f}× higher than std RND", fontsize=10)

        ax.grid(True, which="both", alpha=0.2)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        fig.tight_layout()

        fname = FIG_DIR / f"state_{idx:02d}{'_A' if is_A else ''}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        saved.append(str(fname))
        print(f"  saved {fname.name}  (ratio={ratio:.1f}×, "
              f"train_P1={n_p1}, train_P2={n_p2})")

    # ── summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "game": GAME, "level_index": LEVEL_INDEX,
        "n_states": len(state_keys),
        "A_index": state_idx[A_key],
        "phase1_updates": PHASE1, "phase2_updates": PHASE2,
        "states": [
            {
                "idx": state_idx[k],
                "is_A": k == A_key,
                "train_phase1": int(np.sum(np.array(train_steps[k]) < phase_boundary_step)) if train_steps[k] else 0,
                "train_phase2": int(np.sum(np.array(train_steps[k]) >= phase_boundary_step)) if train_steps[k] else 0,
                "end_novelty": {str(mu): novelty_series[k][mu][-1][1] for mu in MU_VALUES},
                "ratio_05_vs_0": novelty_series[k][0.05][-1][1] / max(novelty_series[k][0.0][-1][1], 1e-30),
            }
            for k in state_keys
        ],
    }
    json.dump(summary, open(RES_DIR / f"per_state_{GAME}_L{LEVEL_INDEX+1}.json", "w"), indent=2)
    print(f"\nSaved {len(saved)} figures to {FIG_DIR}")
    print("States ranked by μ=0.05 vs μ=0 end-ratio (highest = clearest signal):")
    ranked = sorted(summary["states"], key=lambda s: s["ratio_05_vs_0"], reverse=True)
    for s in ranked[:5]:
        flag = " ← A" if s["is_A"] else ""
        print(f"  state_{s['idx']:02d}: ratio={s['ratio_05_vs_0']:.1f}×, "
              f"train_P1={s['train_phase1']}, train_P2={s['train_phase2']}{flag}")


if __name__ == "__main__":
    main()
