"""Evidence for the IDM-encoder feature-scale claim.

Panel A (logged proxy, L1 run): idm_mean_pairwise_l2 vs novelty_raw_mean vs
rnd_distill_loss over training — shows they co-move (rise to ~u19 then relax),
and novelty ∝ scale².  NOTE: this is a PAIRWISE-DISTANCE proxy on 5 probe states,
not the literal ‖h‖, and it is NOT monotonically increasing (it peaks then decays).

Panel B (DIRECT, from checkpoints): load each saved IDM encoder, encode a FIXED
set of harvested states, and plot the literal mean output norm ‖h‖ (and mean
pairwise L2) vs checkpoint step. This directly confirms whether the encoder's
output magnitude grows during training.

Run:
    uv run python -m JEPA.experiments.exp_016_organic_leaky_rnd_icm.\
exp_016_0_naive_baseline.probes.feature_scale_evidence
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from JEPA.experiments.exp_016_organic_leaky_rnd_icm.exp_016_0_naive_baseline.config import Config
from JEPA.experiments.exp_016_organic_leaky_rnd_icm.exp_016_0_naive_baseline.tracker import IDMEncoder
from JEPA.experiments.exp_016_organic_leaky_rnd_icm.exp_016_0_naive_baseline.diagnostics import (
    harvest_states, encode_all,
)

HERE = Path(__file__).resolve().parents[1]
L1_RUN = HERE / "runs" / "exp016_0_naive_ls20_L1_seed0_20260606_193526"
# checkpoint set with the longest trajectory (13 ckpts, 20k→250k)
CKPT_DIR = HERE / "checkpoints" / "exp016_0_naive_tu93_L3_seed0_20260606_212547"
DEVICE = torch.device("cpu")


def panel_a(ax):
    m = [json.loads(l) for l in open(L1_RUN / "metrics.jsonl")]
    step = np.array([r["step"] for r in m])
    pl = np.array([r["idm_mean_pairwise_l2"] for r in m])
    nv = np.array([r["novelty_raw_mean"] for r in m])
    rl = np.array([r["rnd_distill_loss"] for r in m])
    ax.plot(step, pl, label="idm_mean_pairwise_l2 (scale proxy)", lw=2)
    ax.plot(step, nv, label="novelty_raw_mean", lw=1.5, ls="--")
    ax.plot(step, rl, label="rnd_distill_loss", lw=1.5, ls=":")
    ax.set_yscale("log"); ax.set_xlabel("env steps"); ax.set_ylabel("value (log)")
    ax.set_title("A. L1 logged proxy — co-moves, peaks ~u19 then RELAXES\n"
                 "(not monotonic; novelty ∝ scale²)")
    ax.legend(fontsize=8)
    print("[A] L1 proxy: peak pairwise_l2 = %.1f at step %d; final = %.1f"
          % (pl.max(), step[pl.argmax()], pl[-1]))


def panel_b(ax):
    ckpts = sorted(CKPT_DIR.glob("step_*.pt"))
    if not ckpts:
        ax.text(0.5, 0.5, "no checkpoints", ha="center"); return
    cfg0 = Config(**torch.load(ckpts[0], map_location=DEVICE,
                               weights_only=False)["config"])
    # fixed state set, harvested once (same states for every checkpoint)
    reg, _ = harvest_states(cfg0.game, cfg0.level_index, cfg0.seed,
                            cfg0.probe_roam_steps, cfg0.n_envs,
                            tuple(cfg0.timer_mask_rows), cfg0.n_probe_states)
    states = reg.all_masked()
    print(f"[B] {cfg0.game} L{cfg0.level_index+1}: {len(states)} fixed states, "
          f"{len(ckpts)} checkpoints")
    steps, norms, pair = [], [], []
    for c in ckpts:
        ck = torch.load(c, map_location=DEVICE, weights_only=False)
        cfg = Config(**ck["config"])
        idm = IDMEncoder(cfg.n_actions, cfg.n_colors, cfg.frame_size,
                         cfg.trunk_dim, cfg.idm_hidden).to(DEVICE)
        idm.load_state_dict(ck["idm"]); idm.eval()
        h = encode_all(idm.encode_masked, states, DEVICE)        # (S, D)
        steps.append(ck["step"])
        norms.append(float(h.norm(dim=-1).mean()))
        S = h.shape[0]
        d = torch.cdist(h, h)
        pair.append(float(d[~torch.eye(S, dtype=bool)].mean()))
        print(f"    step {ck['step']:>7}: mean ||h|| = {norms[-1]:8.2f}  "
              f"mean pairwise L2 = {pair[-1]:8.2f}")
    ax.plot(steps, norms, "o-", label="mean ‖h‖ (literal output norm)", lw=2)
    ax.plot(steps, pair, "s--", label="mean pairwise L2", lw=1.5)
    ax.set_xlabel("checkpoint env step"); ax.set_ylabel("magnitude")
    ax.set_title(f"B. DIRECT from checkpoints ({cfg0.game} L{cfg0.level_index+1})\n"
                 "literal IDM output norm on a fixed state set")
    ax.legend(fontsize=8)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.tight_layout()
    out = HERE / "findings" / "feature_scale_evidence.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
