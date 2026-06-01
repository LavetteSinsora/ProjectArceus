"""exp_014_1 — RND saturation vs. the leaky-RND fix, on REAL env visitation.

Headline evidence figure for the "leaky RND" innovation.

THE CLAIM
---------
Standard RND (no leak, μ=0) is a one-way error ratchet: once the predictor has
fit a state's random target, that state's novelty *permanently* collapses toward
machine zero, no matter how exploration evolves. Leaky RND adds a per-update
shrink-to-init of the predictor (θ_P ← (1−μ)θ_P + μ·θ_P^init), turning the
ratchet into a visitation-RATE signal that holds a positive floor: a state that
keeps being re-visited keeps being (mildly) re-learned, but between re-fits the
leak nudges its prediction back toward the random init, so novelty never dies.

THE FIGURE
----------
For a handful of the most-visited *masked board states* on LS20 Level 2 (board
identity with the energy/step-timer UI rows masked out), we plot, on a log-y
axis:
    x = cumulative #visits of that state
    y = RND novelty measured at that state (BEFORE that rollout's distill step)
one line per method: standard RND (μ=0) and leaky RND for μ ∈ {0.001,0.01,0.05}.
Expected: μ=0 plunges toward ~1e-12..1e-15 as visits → ~1000 (dead); leaky lines
hold a floor that rises with μ.

DATA COLLECTION (all REAL, no synthetic states)
-----------------------------------------------
* Env: LS20 Level 2 via the shared `VecLS20EnvLevel(level_index=1)` (read-only).
* Policy: uniform-random actions, 16 envs. ONE shared action/visit stream so the
  visit trajectory is byte-identical across every μ — the ONLY difference is the
  leak.
* State identity: mask obs rows 60-63 (the step-timer / energy bar, which marches
  every step regardless of the agent — verified by probes/signal_redundancy.py),
  then hash the masked 64x64 color-index board to a canonical id. The timer is
  NEVER counted.
* RND input: a FIXED random projection of the masked board. one-hot(16 colors)
  over the masked 64x64 board → flatten → a single frozen random Linear to
  dim=256. Identical (seeded) across all μ, so per-state inputs are identical;
  only the predictor's leak differs.
* RND engine: the real `RNDPhi` (exp_013_1) — same target/predictor MLPs, same
  `distill_loss`, same `apply_leak()`. Matches the real loop: per rollout
  (rollout_steps=128 × n_envs=16) we (1) measure novelty on every chosen-state
  visit BEFORE updating, (2) distill the predictor 1 epoch at lr=1e-4 on the
  visited states, (3) `apply_leak()` once. Leak is per-update; visits per-step.

CPU-only (two other agents train on MPS). matplotlib Agg.

Run:
    uv run python -m \
      JEPA.experiments.exp_014.exp_014_1_rnd_saturation_diagnosis.diagnose
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
from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.rnd_phi import RNDPhi


# ── fixed config (mirrors the real exp_013_1 loop on the knobs that matter) ──
DEVICE = torch.device("cpu")
GAME = "ls20"
LEVEL_INDEX = 1               # LS20 Level 2
N_ENVS = 16
ROLLOUT_STEPS = 128           # 128 * 16 = 2048 env-steps / update (real loop)
MAX_EPISODE_STEPS = 200
SEED = 0

N_COLORS = 16
FRAME = 64
TIMER_ROW0 = 60               # rows 60-63 = step-timer/energy UI → masked out
PROJ_DIM = 256                # frozen-random projection dim = RND feature dim

RND_HIDDEN = 256
RND_OUT = 256
RND_LR = 1e-4                 # matches exp_013_1 config.rnd_lr
RND_EPOCHS = 100              # inner distill steps per update; large enough that the
                              # predictor truly MEMORISES the visited support each update
                              # so the μ=0 ratchet drives a re-visited state's novelty
                              # toward machine-zero (the saturation we are proving). The
                              # leak still fires ONCE per update (per-update forget,
                              # exactly as RNDPhi.apply_leak is called in the real loop):
                              # for μ>0 it re-injects a constant error each update that the
                              # next update's distill cannot fully remove → a positive floor.
MU_VALUES = [0.0, 0.001, 0.01, 0.05]   # 0.0 = standard RND
N_CHOSEN_STATES = 4
TARGET_VISITS = 1000          # we want the chosen states to pass this on the x-axis
VISIT_FLOOR = 50              # pre-roam visit threshold for the candidate pool
                              # (states this frequent in the pre-roam blow past
                              # 1000 visits in the main roam)

PRE_ROAM_STEPS = 600          # per-env steps in the pre-roam to pick chosen states
# RND saturation is driven by the number of PREDICTOR DISTILL STEPS (one per
# update, exactly as the real exp_013_1 loop), not by raw env-steps. A top state
# is visited ~250-300×/update, so we run a fixed, generous number of updates so
# (a) the chosen states blow well past TARGET_VISITS on the x-axis and (b) the
# predictor receives enough distill steps for the standard (μ=0) ratchet to
# drive novelty toward machine-zero — which is the phenomenon we are proving.
N_UPDATES = 250               # ≈ 250*2048 = 512k env-steps, ~250 distill steps

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"


def mask_board(frame: np.ndarray) -> np.ndarray:
    """Zero the step-timer/energy UI rows (60-63). frame: (...,64,64) uint8."""
    m = frame.copy()
    m[..., TIMER_ROW0:, :] = 0
    return m


def state_key(masked_frame: np.ndarray) -> bytes:
    return masked_frame.tobytes()


def build_projection(seed: int) -> torch.Tensor:
    """Frozen random linear map: one-hot board (64*64*16) → PROJ_DIM.

    Returned as a (in_dim, PROJ_DIM) weight matrix; we apply it manually so we
    can feed it the sparse one-hot board cheaply. Seeded → identical for all μ.
    """
    g = torch.Generator().manual_seed(seed)
    in_dim = FRAME * FRAME * N_COLORS
    # Orthogonal-ish scaled Gaussian; normalize by sqrt(in_dim) so outputs are
    # O(1) regardless of board size (keeps novelty in a sane numeric range).
    W = torch.randn(in_dim, PROJ_DIM, generator=g) / (in_dim ** 0.5)
    return W


def project_states(masked_frames: np.ndarray, W: torch.Tensor) -> torch.Tensor:
    """masked_frames: (B,64,64) uint8 color indices → (B, PROJ_DIM) features.

    one-hot(16) per cell → flatten → @ W. Done densely (B is small per call)."""
    B = masked_frames.shape[0]
    flat = masked_frames.reshape(B, FRAME * FRAME).astype(np.int64)  # (B, 4096)
    oh = np.zeros((B, FRAME * FRAME, N_COLORS), dtype=np.float32)
    rows = np.arange(B)[:, None]
    cols = np.arange(FRAME * FRAME)[None, :]
    oh[rows, cols, flat] = 1.0
    x = torch.from_numpy(oh.reshape(B, FRAME * FRAME * N_COLORS))
    return x @ W


def pre_roam_pick_states(envs: VecLS20EnvLevel, rng: np.random.Generator,
                         W: torch.Tensor):
    """Short uniform-random pre-roam → high-visit, MUTUALLY DISTINCT masked states.

    We don't just take the top-N by visit count: on LS20 L2 the very hottest
    masked states are near-duplicate agent positions whose frozen-projection
    features sit almost on top of each other, so one shared RND predictor cannot
    fit them independently (mutual interference floors novelty for ALL μ and
    hides the leak). Instead we greedily pick, from the frequently-visited states
    (visit ≥ VISIT_FLOOR), the N whose projected features are maximally far apart
    — each is then individually memorisable, so the μ=0 ratchet can drive it to
    machine-zero while the leak holds a μ-floor. All picks are still REAL,
    frequently-visited states (they comfortably exceed 1000 visits in the roam).
    """
    visit: dict[bytes, int] = defaultdict(int)
    exemplar: dict[bytes, np.ndarray] = {}
    for _ in range(PRE_ROAM_STEPS):
        a = rng.integers(0, envs.n_actions, size=envs.n_envs)
        nobs, _r, dones, _info = envs.step(a)
        m = mask_board(nobs)
        for i in range(envs.n_envs):
            if dones[i]:
                continue
            k = state_key(m[i])
            visit[k] += 1
            if k not in exemplar:
                exemplar[k] = nobs[i].copy()

    ranked = sorted(visit.items(), key=lambda kv: kv[1], reverse=True)
    # candidate pool: frequently-visited states (guarantees they pass 1000 visits)
    pool = [k for k, c in ranked if c >= VISIT_FLOOR]
    if len(pool) < N_CHOSEN_STATES:                 # fallback: just take the top-N
        pool = [k for k, _c in ranked[:max(N_CHOSEN_STATES, 8)]]
    pool_masked = np.stack([
        np.frombuffer(k, dtype=np.uint8).reshape(FRAME, FRAME) for k in pool
    ])
    feats = project_states(pool_masked, W).numpy()  # (P, dim)

    # greedy farthest-point selection, seeded with the most-visited state.
    chosen_i = [0]
    while len(chosen_i) < min(N_CHOSEN_STATES, len(pool)):
        d = np.min([np.linalg.norm(feats - feats[c], axis=1) for c in chosen_i],
                   axis=0)
        d[chosen_i] = -1.0
        chosen_i.append(int(np.argmax(d)))
    chosen = [pool[i] for i in chosen_i]
    return chosen, {k: visit[k] for k in chosen}, {k: exemplar[k] for k in chosen}


def describe_state(raw_frame: np.ndarray) -> dict:
    """Cheap human-readable descriptor of a masked board state.

    We don't have a semantic decoder, so we summarize the playfield (rows above
    the timer): the color histogram, the location of the rarest non-zero color,
    and the centroid + bounding box of the *active pattern* pixels (every
    non-background, non-wall color). On LS20 the chosen states share a color
    histogram but differ in WHERE the pattern sits / how it is rotated — these
    geometry fields capture that (agent location / rotation level)."""
    play = raw_frame[:TIMER_ROW0, :]                  # exclude timer rows
    colors, counts = np.unique(play, return_counts=True)
    hist = {int(c): int(n) for c, n in zip(colors, counts)}
    # rarest non-background color (background = most common, usually 0)
    nz = [(c, n) for c, n in zip(colors, counts)]
    nz_sorted = sorted(nz, key=lambda cn: cn[1])
    rare_color = int(nz_sorted[0][0])
    ys, xs = np.where(play == rare_color)
    rare_loc = [int(ys.mean()), int(xs.mean())] if len(ys) else None
    # background = most-common color; "pattern" = everything else
    bg = int(colors[int(np.argmax(counts))])
    pys, pxs = np.where((play != bg) & (play != 0))
    if len(pys):
        pattern_centroid = [round(float(pys.mean()), 1), round(float(pxs.mean()), 1)]
        pattern_bbox = [int(pys.min()), int(pxs.min()), int(pys.max()), int(pxs.max())]
    else:
        pattern_centroid, pattern_bbox = None, None
    return {
        "color_hist": hist,
        "rarest_color": rare_color,
        "rarest_color_mean_loc_yx": rare_loc,
        "pattern_centroid_yx": pattern_centroid,
        "pattern_bbox_yxyx": pattern_bbox,
        "n_distinct_colors": len(colors),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[exp_014_1] device={DEVICE}  ls20 L{LEVEL_INDEX + 1}  "
          f"n_envs={N_ENVS}  rollout={ROLLOUT_STEPS}  mu={MU_VALUES}")

    envs = VecLS20EnvLevel(env_name=GAME, n_envs=N_ENVS,
                           max_episode_steps=MAX_EPISODE_STEPS, seed=SEED,
                           level_index=LEVEL_INDEX)

    # Single shared visit/action stream for the WHOLE script (pre-roam + main).
    rng = np.random.default_rng(SEED)

    # ── 1. frozen random projection (shared across all μ AND used to pick
    #        mutually-distinct chosen states) ────────────────────────────────
    W = build_projection(seed=SEED + 1)

    # ── 2. pre-roam: pick high-visit, mutually-distinct chosen states ──────
    chosen_keys, pre_counts, exemplars = pre_roam_pick_states(envs, rng, W)
    chosen_idx = {k: j for j, k in enumerate(chosen_keys)}
    descriptions = [describe_state(exemplars[k]) for k in chosen_keys]
    print("[exp_014_1] chosen states (pre-roam counts over "
          f"{PRE_ROAM_STEPS * N_ENVS} steps):")
    for j, k in enumerate(chosen_keys):
        print(f"   state {j}: pre-roam visits={pre_counts[k]}  desc={descriptions[j]}")

    # Precompute the projected feature for each chosen state (fixed input).
    chosen_masked = np.stack([
        np.frombuffer(k, dtype=np.uint8).reshape(FRAME, FRAME) for k in chosen_keys
    ])
    chosen_feats = project_states(chosen_masked, W).to(DEVICE)   # (N_chosen, PROJ_DIM)

    # ── 3. one RND instance per μ (identical init via seeding) ─────────────
    rnds, opts = {}, {}
    for mu in MU_VALUES:
        torch.manual_seed(SEED + 100)        # identical target+predictor init for all μ
        r = RNDPhi(dim=PROJ_DIM, hidden=RND_HIDDEN, out=RND_OUT, leak=mu).to(DEVICE)
        rnds[mu] = r
        opts[mu] = torch.optim.Adam(r.predictor.parameters(), lr=RND_LR)

    # ── 4. main roam ──────────────────────────────────────────────────────
    # Global cumulative visit counter (masked-state id → count). Shared by all μ
    # because the action stream is shared → identical trajectories.
    visit: dict[bytes, int] = defaultdict(int)
    # records[mu][state_j] = list of (cum_visit_count, novelty)
    records = {mu: [[] for _ in range(N_CHOSEN_STATES)] for mu in MU_VALUES}

    total_env_steps = 0
    passed_target_at = None
    for update in range(1, N_UPDATES + 1):
        # Measure novelty at the chosen states for every μ BEFORE this update's
        # distill step (the real loop measures the intrinsic reward, then trains).
        with torch.no_grad():
            chosen_nov = {mu: rnds[mu].novelty(chosen_feats).cpu().numpy()
                          for mu in MU_VALUES}

        # Record ONE point per chosen state per update: (cumulative visits SO FAR,
        # novelty now). One re-fit per update = one leak per update, so the
        # x-axis (visits) advances at the per-update cadence at which the leak
        # competes with re-learning — exactly the regime the leak is designed for.
        for j, k in enumerate(chosen_keys):
            c = max(visit[k], 1)
            for mu in MU_VALUES:
                records[mu][j].append((c, float(chosen_nov[mu][j])))

        step_masked_batches = []
        for _ in range(ROLLOUT_STEPS):
            a = rng.integers(0, envs.n_actions, size=N_ENVS)
            nobs, _r, dones, _info = envs.step(a)
            total_env_steps += N_ENVS
            m = mask_board(nobs)
            keep = ~dones
            batch = m[keep]
            step_masked_batches.append(batch)
            for fr in batch:
                visit[fr.tobytes()] += 1

        # Distill each predictor on this rollout's visited states, then leak ONCE.
        # We deduplicate the rollout to its DISTINCT masked states so every visited
        # state gets equal gradient weight per update (the small ~50-state support
        # otherwise lets the few hottest states dominate and starves saturation on
        # the rest). RND_EPOCHS inner steps/update give the standard (μ=0) ratchet
        # enough force to drive a memorised state's novelty toward machine-zero —
        # the saturation we are proving. The leak is still applied once per update.
        if step_masked_batches:
            roll = np.concatenate(step_masked_batches, axis=0)        # (M,64,64)
            uniq = np.unique(roll.reshape(roll.shape[0], -1), axis=0)
            uniq = uniq.reshape(-1, FRAME, FRAME)                     # distinct states
            feats = project_states(uniq, W).to(DEVICE)               # (U, PROJ_DIM)
            for mu in MU_VALUES:
                r, opt = rnds[mu], opts[mu]
                for _e in range(RND_EPOCHS):
                    opt.zero_grad()
                    loss = r.distill_loss(feats)
                    loss.backward()
                    opt.step()
                r.apply_leak()

        # track when all chosen states first pass TARGET_VISITS (informational);
        # we KEEP roaming so the predictor saturates over many distill steps.
        min_visits = min(visit[k] for k in chosen_keys)
        if passed_target_at is None and min_visits >= TARGET_VISITS:
            passed_target_at = update
        if update == 1 or update % 25 == 0 or update == N_UPDATES:
            # mean current standard-RND novelty over chosen states (saturation gauge)
            cur_std = float(np.mean(chosen_nov[0.0]))
            print(f"   upd {update:>4}  env_steps={total_env_steps:>9}  "
                  f"min_chosen_visits={min_visits:>6}  distinct_masked={len(visit)}  "
                  f"std_RND_nov={cur_std:.3e}")

    final_visits = {chosen_idx[k]: visit[k] for k in chosen_keys}
    max_reached = max(final_visits.values())
    min_reached = min(final_visits.values())
    print(f"[exp_014_1] final chosen-state visits: {final_visits}  "
          f"(min={min_reached}, max={max_reached})  total_env_steps={total_env_steps}")

    # ── 5. key numbers: novelty at ~100 and ~1000 visits per μ per state ───
    def nov_near(rec, target):
        """rec = list of (count, nov); return nov of the record with count
        closest to target (and the actual count used)."""
        if not rec:
            return None, None
        arr = np.array(rec)
        idx = int(np.argmin(np.abs(arr[:, 0] - target)))
        return float(arr[idx, 1]), int(arr[idx, 0])

    key_numbers = {}
    for mu in MU_VALUES:
        key_numbers[str(mu)] = {}
        for j in range(N_CHOSEN_STATES):
            n100, c100 = nov_near(records[mu][j], 100)
            n1000, c1000 = nov_near(records[mu][j], 1000)
            key_numbers[str(mu)][f"state_{j}"] = {
                "nov_at_~100": n100, "count_used_100": c100,
                "nov_at_~1000": n1000, "count_used_1000": c1000,
            }

    # ── 6. THE FIGURE: small-multiples, one panel per chosen state ─────────
    colors = {0.0: "#d62728", 0.001: "#1f77b4", 0.01: "#2ca02c", 0.05: "#9467bd"}
    labels = {0.0: "standard RND (μ=0)", 0.001: "leaky μ=0.001",
              0.01: "leaky μ=0.01", 0.05: "leaky μ=0.05"}

    ncol = 2
    nrow = int(np.ceil(N_CHOSEN_STATES / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.4 * nrow),
                             squeeze=False)
    for j in range(N_CHOSEN_STATES):
        ax = axes[j // ncol][j % ncol]
        for mu in MU_VALUES:
            rec = records[mu][j]
            if not rec:
                continue
            arr = np.array(rec)
            # clamp the floor so log scale renders true-zero novelties
            y = np.clip(arr[:, 1], 1e-16, None)
            ax.plot(arr[:, 0], y, color=colors[mu], lw=1.6,
                    label=labels[mu], alpha=0.9)
        ax.set_yscale("log")
        ax.set_xlabel("cumulative visits to this state")
        ax.set_ylabel("RND novelty (½·mean (P−T)²)")
        d = descriptions[j]
        ax.set_title(f"chosen state {j}  (final visits={final_visits[j]})\n"
                     f"pattern centroid y,x={d['pattern_centroid_yx']}",
                     fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8, loc="lower left")
    # hide unused panels
    for j in range(N_CHOSEN_STATES, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(
        "RND saturation vs. the leaky-RND fix — real LS20 L2 visitation\n"
        "standard RND novelty plunges to machine-zero as a state is re-visited; "
        "leaky RND holds a μ-dependent floor",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    main_fig = FIG_DIR / "rnd_saturation_vs_visits.png"
    fig.savefig(main_fig, dpi=140)
    plt.close(fig)
    print(f"[exp_014_1] wrote headline figure: {main_fig}")

    # supporting figure: all states overlaid for μ=0 vs μ=0.01 (clarity)
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
    for j in range(N_CHOSEN_STATES):
        for mu in (0.0, 0.01):
            rec = records[mu][j]
            if not rec:
                continue
            arr = np.array(rec)
            y = np.clip(arr[:, 1], 1e-16, None)
            ls = "-" if mu == 0.01 else "--"
            ax2.plot(arr[:, 0], y, ls, lw=1.3,
                     color=colors[mu], alpha=0.7,
                     label=(labels[mu] if j == 0 else None))
    ax2.set_yscale("log")
    ax2.set_xlabel("cumulative visits to this state")
    ax2.set_ylabel("RND novelty")
    ax2.set_title("μ=0 (dashed) collapses; μ=0.01 (solid) holds a floor — all chosen states")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    supp_fig = FIG_DIR / "rnd_saturation_overlay.png"
    fig2.savefig(supp_fig, dpi=140)
    plt.close(fig2)
    print(f"[exp_014_1] wrote supporting figure: {supp_fig}")

    # raw-RND-ONLY figure (μ=0; the "problem" figure — no leak lines)
    fig3, ax3 = plt.subplots(figsize=(7.4, 4.8))
    for j in range(N_CHOSEN_STATES):
        rec = records[0.0][j]
        if not rec:
            continue
        arr = np.array(rec)
        y = np.clip(arr[:, 1], 1e-16, None)
        ax3.plot(arr[:, 0], y, lw=1.7, color="#c0392b", alpha=0.8,
                 label=f"state {j} (final visits={final_visits[j]})")
    ax3.set_yscale("log")
    ax3.set_xlabel("cumulative visits to this state")
    ax3.set_ylabel("RND novelty  ½·mean (P−T)²")
    ax3.set_title("Standard RND (no leak): novelty collapses toward zero as a state is re-visited\n"
                  "real LS20 L2 visitation — the saturation problem")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(fontsize=8, loc="upper right")
    raw_fig = FIG_DIR / "rnd_saturation_standard_only.png"
    fig3.tight_layout()
    fig3.savefig(raw_fig, dpi=140)
    plt.close(fig3)
    print(f"[exp_014_1] wrote raw-RND-only figure: {raw_fig}")

    # save the raw per-(μ,state) series so any subset can be re-plotted instantly (no re-roam)
    series = {str(mu): {str(j): records[mu][j] for j in range(N_CHOSEN_STATES)} for mu in MU_VALUES}
    (RES_DIR / "raw_series.json").write_text(json.dumps(series))
    print(f"[exp_014_1] wrote raw series -> {RES_DIR / 'raw_series.json'}")

    # ── 7. dump results json ───────────────────────────────────────────────
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(DEVICE),
        "game": GAME, "level_index": LEVEL_INDEX, "level_human": LEVEL_INDEX + 1,
        "n_envs": N_ENVS, "rollout_steps": ROLLOUT_STEPS, "seed": SEED,
        "mu_values": MU_VALUES, "proj_dim": PROJ_DIM, "rnd_lr": RND_LR,
        "timer_rows_masked": [TIMER_ROW0, FRAME - 1],
        "n_chosen_states": N_CHOSEN_STATES,
        "total_env_steps": total_env_steps,
        "n_updates": N_UPDATES,
        "all_chosen_passed_1000_at_update": passed_target_at,
        "final_chosen_visits": {str(j): int(v) for j, v in final_visits.items()},
        "chosen_state_descriptions": {str(j): descriptions[j]
                                      for j in range(N_CHOSEN_STATES)},
        "key_numbers": key_numbers,
        "headline_figure": str(main_fig),
        "supporting_figure": str(supp_fig),
    }
    res_path = RES_DIR / "rnd_saturation_results.json"
    res_path.write_text(json.dumps(summary, indent=2))
    print(f"[exp_014_1] wrote results: {res_path}")

    # ── 8. print the headline numbers ──────────────────────────────────────
    print("\n========== KEY NUMBERS (novelty @ ~100 / ~1000 visits) ==========")
    for mu in MU_VALUES:
        tag = "standard RND" if mu == 0.0 else f"leaky μ={mu}"
        n100s = [key_numbers[str(mu)][f"state_{j}"]["nov_at_~100"]
                 for j in range(N_CHOSEN_STATES)]
        n1000s = [key_numbers[str(mu)][f"state_{j}"]["nov_at_~1000"]
                  for j in range(N_CHOSEN_STATES)]
        m100 = float(np.mean([x for x in n100s if x is not None]))
        m1000 = float(np.mean([x for x in n1000s if x is not None]))
        print(f"  {tag:<16}  mean nov @~100 = {m100:.3e}   @~1000 = {m1000:.3e}")
    return summary


if __name__ == "__main__":
    main()
