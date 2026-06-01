"""04 — Two mechanism figures:
  fig4_eta_squeeze: r^i_mean (from metrics) vs the PPO entropy-bonus yardstick
     (c_ent=0.01) over training, L1 & L2, showing r^i is crushed far below the
     PPO signal within the first few updates because eta was frozen against the
     UNTRAINED forward error.
  fig5_death_depth: mean_episode_steps over evals — L2 dies at a fixed ~66-step
     depth (truncation_rate=0), a behavioural plateau the weak r^i never breaks.
"""
from __future__ import annotations
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
FIG = os.path.join(ROOT, "figures")
L1 = "JEPA/experiments/exp_011_ls20_icm/exp_011_0_icm_baseline/runs/*/metrics.jsonl"
L2 = "JEPA/experiments/exp_011_ls20_icm/exp_011_2_icm_ls20_l2/runs/*/metrics.jsonl"
C_ENT = 0.01


def load(pat):
    return [[json.loads(l) for l in open(f)] for f in sorted(glob.glob(pat))]


l1 = load(L1); l2 = load(L2)

# ---- fig4: r^i vs yardstick ----
fig, ax = plt.subplots(figsize=(8, 4.6))
for rows in l2:
    s = [r["step"] for r in rows]; v = [r["intrinsic_reward_mean"] for r in rows]
    ax.plot(s, v, color="tab:red", alpha=0.6, lw=1.2)
for rows in l1:
    s = [r["step"] for r in rows]; v = [r["intrinsic_reward_mean"] for r in rows]
    ax.plot(s, v, color="tab:green", alpha=0.6, lw=1.2)
ax.axhline(C_ENT, color="k", ls="--", lw=1.4, label="calibration target = entropy-bonus scale c_ent = 0.01")
ax.axhline(1.0, color="purple", ls=":", lw=1.4, label="terminal reward = +1")
ax.set_yscale("log")
ax.set_xlabel("env steps"); ax.set_ylabel("mean intrinsic reward r^i")
ax.set_title("eta is frozen against the UNTRAINED forward error: once the model\n"
             "learns, r^i is crushed ~60-100x below the PPO entropy bonus (in BOTH levels)")
from matplotlib.lines import Line2D
h = [Line2D([0],[0],color="tab:green",lw=2,label="L1 (solved)"),
     Line2D([0],[0],color="tab:red",lw=2,label="L2 (failed)")]
ax.legend(handles=h + ax.get_legend_handles_labels()[0], fontsize=8, loc="upper right")
ax.grid(alpha=0.25, which="both")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_eta_squeeze.png"), dpi=110, bbox_inches="tight")
print("saved fig4")

# ---- fig5: death depth (mean_episode_steps over evals) ----
fig, ax = plt.subplots(figsize=(8, 4.6))
for rows in l2:
    ev = [r for r in rows if r.get("success_rate") is not None]
    ax.plot([r["step"] for r in ev], [r["mean_episode_steps"] for r in ev],
            "o-", color="tab:red", alpha=0.6, lw=1.4)
for rows in l1:
    ev = [r for r in rows if r.get("success_rate") is not None]
    ax.plot([r["step"] for r in ev], [r["mean_episode_steps"] for r in ev],
            "o-", color="tab:green", alpha=0.6, lw=1.4)
ax.axhline(66, color="tab:red", ls=":", lw=1, alpha=0.7)
ax.axhline(13, color="tab:green", ls=":", lw=1, alpha=0.7)
ax.text(20000, 70, "L2 dies at fixed ~66 steps (trunc_rate=0)", color="tab:red", fontsize=9)
ax.text(220000, 16, "L1 converges to 13-step optimal solve", color="tab:green", fontsize=9)
ax.set_xlabel("env steps"); ax.set_ylabel("mean eval episode length (steps)")
ax.set_title("Behavioural plateau: L2 episodes terminate at a fixed depth, never truncating —\n"
             "the policy keeps dying at the same point; weak r^i never restructures it")
ax.legend(handles=[Line2D([0],[0],color="tab:green",lw=2,label="L1 (solved)"),
                   Line2D([0],[0],color="tab:red",lw=2,label="L2 (failed)")], fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_death_depth.png"), dpi=110, bbox_inches="tight")
print("saved fig5")
