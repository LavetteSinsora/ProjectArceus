"""Overlay for exp_014_1's `rnd_saturation_vs_visits_1state.png`: the SAME single
real state, SAME cumulative-visits x-axis — extended to show that when the state
stops being trained on, the leaky novelty REVIVES while standard RND stays dead.

This is the real-roam companion to `probe.py` (which isolates the mechanism with a
hand-driven anchor). Here we reproduce the actual saturation curve and continue it:

  * Phase SATURATE — byte-for-byte the exp_014_1 saturation loop (same seed, same
    uniform-random LS20-L2 roam, same `RNDPhi`, same frozen projection, same
    RND_EPOCHS=100 + one leak per update). The most-visited masked state's novelty
    is driven down exactly as in `rnd_saturation_vs_visits_1state.png`.
  * Phase REVIVE — the agent has moved on, so the state leaves the predictor's
    training set: we KEEP roaming (the random roam still passes through it, so its
    cumulative-visit count keeps climbing along the SAME x-axis) but EXCLUDE it from
    every distill batch, while the per-update leak keeps firing. Standard RND (μ=0)
    just sits where it was left; leaky RND climbs back up toward the state's original
    novelty — visible on the very same cumulative-visits axis.

Because phase SATURATE re-runs the exp_014_1 loop with the same seed, the left half
of this figure overlays `rnd_saturation_vs_visits_1state.png`; the right half is the
new revival evidence.

CPU-only. matplotlib Agg.

Run:
    uv run python -m JEPA.experiments.exp_014_figures_and_results.\
exp_014_1_rnd_saturation.single_state_recency_probe.saturate_then_revive_real_roam
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
# Mirror exp_014_1 EXACTLY by importing its constants + helpers (single source of truth).
from JEPA.experiments.exp_014_figures_and_results.exp_014_1_rnd_saturation import diagnose as D

DEVICE = torch.device("cpu")
ABANDON_UPDATES = 150          # extra roam updates AFTER the state leaves the training set
HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
COLORS = {0.0: "#d62728", 0.001: "#1f77b4", 0.01: "#2ca02c", 0.05: "#9467bd"}
LABELS = {0.0: "standard RND (μ=0)", 0.001: "leaky μ=0.001",
          0.01: "leaky μ=0.01", 0.05: "leaky μ=0.05"}
_TAG = f"{D.GAME}_L{D.LEVEL_INDEX + 1}"


def smooth(y, w=7):
    """Geometric rolling mean over a window (matches the thick smoothed lines in the
    reference figure); operates in log space so it reads well on the log-y axis."""
    y = np.clip(np.asarray(y, float), 1e-16, None)
    ly = np.log(y)
    k = np.ones(w) / w
    s = np.convolve(ly, k, mode="same")
    # fix the convolution edges (mode="same" shrinks the window at the ends)
    for i in range(w // 2):
        s[i] = ly[: i + w // 2 + 1].mean()
        s[-(i + 1)] = ly[-(i + w // 2 + 1):].mean()
    return np.exp(s)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(D.SEED)
    np.random.seed(D.SEED)

    envs = VecLS20EnvLevel(env_name=D.GAME, n_envs=D.N_ENVS,
                           max_episode_steps=D.MAX_EPISODE_STEPS, seed=D.SEED,
                           level_index=D.LEVEL_INDEX)
    rng = np.random.default_rng(D.SEED)
    W = D.build_projection(seed=D.SEED + 1)                       # same projection as exp_014_1

    # anchor = the most-visited masked state (= state 0 of the saturation figure).
    chosen_keys, pre_counts, exemplars = D.pre_roam_pick_states(envs, rng, W)
    anchor_key = chosen_keys[0]
    anchor_masked = np.frombuffer(anchor_key, dtype=np.uint8).reshape(D.FRAME, D.FRAME)
    anchor_feat = D.project_states(anchor_masked[None], W).to(DEVICE)      # (1, PROJ_DIM)
    print(f"[saturate_then_revive/{_TAG}] anchor pre-roam visits={pre_counts[anchor_key]}")

    # one RND per μ, identical init — mirrors exp_014_1.
    rnds, opts = {}, {}
    for mu in D.MU_VALUES:
        torch.manual_seed(D.SEED + 100)
        r = RNDPhi(dim=D.PROJ_DIM, hidden=D.RND_HIDDEN, out=D.RND_OUT, leak=mu).to(DEVICE)
        rnds[mu] = r
        opts[mu] = torch.optim.Adam(r.predictor.parameters(), lr=D.RND_LR)

    init_nov = float(np.mean([float(rnds[mu].novelty(anchor_feat).item()) for mu in D.MU_VALUES]))

    visit: dict[bytes, int] = defaultdict(int)
    records = {mu: [] for mu in D.MU_VALUES}                      # (cum_anchor_visits, novelty)
    total_updates = D.N_UPDATES + ABANDON_UPDATES
    abandon_at_visits = None                                      # x where the state leaves training

    for update in range(1, total_updates + 1):
        abandoned = update > D.N_UPDATES
        with torch.no_grad():
            nov = {mu: float(rnds[mu].novelty(anchor_feat).item()) for mu in D.MU_VALUES}
        c = max(visit[anchor_key], 1)
        for mu in D.MU_VALUES:
            records[mu].append((c, nov[mu]))
        if abandoned and abandon_at_visits is None:
            abandon_at_visits = c

        # roam exactly as exp_014_1 (count visits to every masked state, incl. anchor).
        step_batches = []
        for _ in range(D.ROLLOUT_STEPS):
            a = rng.integers(0, envs.n_actions, size=D.N_ENVS)
            nobs, _r, dones, _info = envs.step(a)
            m = D.mask_board(nobs)
            batch = m[~dones]
            step_batches.append(batch)
            for fr in batch:
                visit[fr.tobytes()] += 1

        if step_batches:
            roll = np.concatenate(step_batches, axis=0)
            uniq = np.unique(roll.reshape(roll.shape[0], -1), axis=0).reshape(-1, D.FRAME, D.FRAME)
            if abandoned:                                        # the state has LEFT the training set
                uniq = uniq[np.any(uniq.reshape(len(uniq), -1) != anchor_masked.reshape(-1), axis=1)]
            if len(uniq):
                feats = D.project_states(uniq, W).to(DEVICE)
                for mu in D.MU_VALUES:
                    r, opt = rnds[mu], opts[mu]
                    for _e in range(D.RND_EPOCHS):
                        opt.zero_grad()
                        r.distill_loss(feats).backward()
                        opt.step()
                    r.apply_leak()

        if update == 1 or update % 25 == 0 or update == D.N_UPDATES or update == total_updates:
            tag = "REVIVE " if abandoned else "saturate"
            print(f"   {tag} upd {update:>4}  anchor_visits={c:>7}  "
                  + " ".join(f"μ{mu}={nov[mu]:.2e}" for mu in D.MU_VALUES), flush=True)

    # ── figure: same axes/colors as rnd_saturation_vs_visits_1state.png ──────
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    if abandon_at_visits is not None:
        ax.axvspan(abandon_at_visits, max(records[0.0][-1][0], abandon_at_visits),
                   color="#eaf2f8", alpha=0.7)
        ax.axvline(abandon_at_visits, color="#566573", lw=1.1, ls="--")
        ax.text(abandon_at_visits, 0.97, " state leaves the training set\n (agent moves on)",
                transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=8.5, color="#1f618d")
    ax.axhline(init_nov, color="#7f8c8d", lw=1, ls=":", alpha=0.8)
    ax.text(0.5, init_nov, " original novelty", color="#7f8c8d", fontsize=8,
            ha="left", va="bottom")
    for mu in D.MU_VALUES:
        arr = np.array(records[mu])
        x, y = arr[:, 0], np.clip(arr[:, 1], 1e-16, None)
        ax.plot(x, y, color=COLORS[mu], lw=0.7, alpha=0.25)               # raw
        ax.plot(x, smooth(y), color=COLORS[mu], lw=2.2, label=LABELS[mu])  # smoothed
    ax.set_yscale("log")
    ax.set_xlabel("cumulative visits to this state")
    ax.set_ylabel("RND novelty  ½·mean (P−T)²")
    ax.set_title(
        "RND saturation, then REVIVAL — single state, real LS20 L2\n"
        "left: novelty collapses as the state is visited (= the saturation figure);  "
        "right: once it leaves the\ntraining set, standard RND stays dead but leaky RND climbs back up",
        fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    out = FIG / f"saturate_then_revive_real_roam_{_TAG}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

    # ── numbers + json ──────────────────────────────────────────────────────
    nums = {}
    for mu in D.MU_VALUES:
        arr = np.array(records[mu])
        sat_floor = float(arr[:D.N_UPDATES, 1].min())
        end = float(arr[-1, 1])
        nums[str(mu)] = {"sat_floor": sat_floor, "end_after_revive": end,
                         "revive_ratio": end / max(sat_floor, 1e-30)}
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "probe": "saturate_then_revive_real_roam",
        "game": D.GAME, "level_index": D.LEVEL_INDEX, "level_human": D.LEVEL_INDEX + 1,
        "seed": D.SEED, "rnd_epochs": D.RND_EPOCHS, "rnd_lr": D.RND_LR,
        "mu_values": D.MU_VALUES, "saturate_updates": D.N_UPDATES,
        "abandon_updates": ABANDON_UPDATES, "abandon_at_visits": abandon_at_visits,
        "anchor_original_novelty": init_nov, "by_mu": nums,
        "records": {str(mu): records[mu] for mu in D.MU_VALUES},
        "figure": str(out),
    }
    (RES / f"saturate_then_revive_real_roam_{_TAG}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== single-state real-roam: saturation floor → end after revival ===")
    for mu in D.MU_VALUES:
        d = nums[str(mu)]
        tag = "standard RND" if mu == 0.0 else f"leaky μ={mu}"
        print(f"  {tag:<16} floor={d['sat_floor']:.2e} → end={d['end_after_revive']:.2e}  "
              f"({d['revive_ratio']:.1f}×)")


if __name__ == "__main__":
    main()
