"""exp_016_0 dashboard — render one PNG that makes every logged signal inspectable.

Reads a run's metrics.jsonl + state_novelty.jsonl and draws a multi-panel figure
grouped exactly like SYSTEM_CARD §5: policy/REINFORCE, coverage, novelty/count,
normalization-ablation, IDM/encoder, drift, and the full-state novelty landscape.

Run (defaults to the latest run dir):
    uv run python -m JEPA.experiments.exp_016_organic_leaky_rnd_icm.\
exp_016_0_naive_baseline.dashboard
    uv run ... .dashboard --run <path/to/run_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _latest_run() -> Path:
    runs = Path(__file__).resolve().parent / "runs"
    cand = [d for d in runs.iterdir() if (d / "metrics.jsonl").exists()]
    return max(cand, key=lambda d: (d / "metrics.jsonl").stat().st_mtime)


def _load(run: Path):
    m = [json.loads(l) for l in open(run / "metrics.jsonl")]
    sn = [json.loads(l) for l in open(run / "state_novelty.jsonl")] \
        if (run / "state_novelty.jsonl").exists() else []
    return m, sn


def _col(m, k):
    return np.array([r.get(k, np.nan) for r in m], dtype=np.float64)


def _frr(m):
    for r in m:
        v = r.get("env_steps_to_first_reward")
        if v is not None:
            return float(v), r["step"]
    return None, None


def _vline(ax, x):
    if x is not None:
        ax.axvline(x, color="crimson", ls="--", lw=1.2, alpha=0.8,
                   label="first reward")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    args = p.parse_args()
    run = Path(args.run) if args.run else _latest_run()
    m, sn = _load(run)
    step = _col(m, "step")
    frr_steps, _ = _frr(m)

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(5, 3, figure=fig, hspace=0.38, wspace=0.22,
                  height_ratios=[1, 1, 1, 1, 1.4])
    fig.suptitle(f"exp_016_0 naive leaky-RND-on-IDM  —  {run.name}\n"
                 f"first reward @ {frr_steps if frr_steps else 'NONE'} env-steps "
                 f"(random LS20-L1 ≈ 50k)", fontsize=15, y=0.995)

    # ── Row 0: policy / REINFORCE ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(step, _col(m, "entropy"), color="tab:blue")
    ax.axhline(np.log(4), color="gray", ls=":", lw=1, label="ln(4) uniform")
    _vline(ax, frr_steps); ax.set_title("Policy entropy (F3 collapse)")
    ax.set_xlabel("env steps"); ax.set_ylabel("entropy"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    pa = np.array([r["per_action_prob"] for r in m])           # (U, n_actions)
    for a in range(pa.shape[1]):
        ax.plot(step, pa[:, a], label=f"a{a}")
    _vline(ax, frr_steps); ax.set_title("Per-action probability (collapse target)")
    ax.set_xlabel("env steps"); ax.set_ylabel("mean π(a)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(step, _col(m, "grad_norm"), color="tab:purple", label="grad_norm")
    ax.set_ylabel("grad_norm", color="tab:purple")
    ax2 = ax.twinx()
    ax2.plot(step, _col(m, "return_variance"), color="tab:orange", label="return_var")
    ax2.set_ylabel("return variance", color="tab:orange")
    _vline(ax, frr_steps); ax.set_title("REINFORCE variance (F1)")
    ax.set_xlabel("env steps")

    # ── Row 1: coverage ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(step, _col(m, "cumulative_unique_states"), color="tab:green")
    _vline(ax, frr_steps); ax.set_title("Cumulative unique states (coverage F9)")
    ax.set_xlabel("env steps"); ax.set_ylabel("# distinct masked states")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(step, _col(m, "unique_states_this_update"), label="unique/update")
    ax.plot(step, _col(m, "new_states_this_update"), label="new/update")
    ax.plot(step, _col(m, "unique_per_episode"), label="unique/episode (loop)")
    _vline(ax, frr_steps); ax.set_title("Exploration rate / looping")
    ax.set_xlabel("env steps"); ax.set_ylabel("count"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(step, _col(m, "noop_fraction"), color="tab:red")
    _vline(ax, frr_steps); ax.set_title("No-op / wall-bump fraction")
    ax.set_xlabel("env steps"); ax.set_ylabel("fraction"); ax.set_ylim(0, 1)

    # ── Row 2: novelty / count ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(step, _col(m, "novelty_raw_mean"), label="mean")
    ax.plot(step, _col(m, "novelty_floor"), label="floor (min)", ls="--")
    ax.set_yscale("log"); _vline(ax, frr_steps)
    ax.set_title("Raw novelty: mean & floor (F4 saturation)")
    ax.set_xlabel("env steps"); ax.set_ylabel("novelty (log)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    nm = _col(m, "novelty_raw_mean"); nsd = _col(m, "novelty_raw_std")
    ax.plot(step, nsd / (nm + 1e-9), color="tab:brown")
    _vline(ax, frr_steps); ax.set_title("Novelty resolution  (std / mean across states)")
    ax.set_xlabel("env steps"); ax.set_ylabel("coefficient of variation")

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(step, _col(m, "rnd_distill_loss"), color="tab:cyan")
    ax.set_yscale("log"); _vline(ax, frr_steps)
    ax.set_title("RND predictor distill loss"); ax.set_xlabel("env steps")

    # ── Row 3: normalization ablation + IDM ────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(step, _col(m, "novelty_raw_mean"), label="raw novelty mean")
    ax.plot(step, _col(m, "run_std"), label="running std (divisor)")
    ax.plot(step, _col(m, "reward_norm_mean"), label="z-scored reward mean")
    ax.set_yscale("symlog"); _vline(ax, frr_steps)
    ax.set_title("Reward normalization — is z-score needed?")
    ax.set_xlabel("env steps"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(step, _col(m, "return_raw_std"), label="return raw std")
    ax.plot(step, _col(m, "return_norm_std"), label="return norm std (=1)")
    ax.plot(step, _col(m, "return_raw_mean"), label="return raw mean", ls="--")
    _vline(ax, frr_steps); ax.set_title("Return normalization — is ÷std needed?")
    ax.set_xlabel("env steps"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 2])
    ax.plot(step, _col(m, "inverse_acc_onpolicy"), label="inv_acc on-policy")
    ax.plot(step, _col(m, "inverse_acc_holdout"), label="inv_acc held-out")
    ax.axhline(0.25, color="gray", ls=":", lw=1, label="chance")
    _vline(ax, frr_steps); ax.set_title("IDM controllability (F2/encoder)")
    ax.set_xlabel("env steps"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # ── Row 4: drift (left) + state-novelty landscape (heatmap, wide) ──────
    ax = fig.add_subplot(gs[4, 0])
    ax.plot(step, _col(m, "drift_idm_rel_l2"), label="IDM enc")
    ax.plot(step, _col(m, "drift_actor_rel_l2"), label="actor enc")
    ax.plot(step, _col(m, "drift_idm_over_pairdist"), label="IDM drift÷pairdist", ls="--")
    ax.set_yscale("log"); _vline(ax, frr_steps)
    ax.set_title("Encoder drift (F2)"); ax.set_xlabel("env steps"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[4, 1:])
    if sn:
        S = len(sn[-1]["novelty"])
        U = len(sn)
        mat = np.full((U, S), np.nan)
        for u, row in enumerate(sn):
            nv = row["novelty"]
            mat[u, :len(nv)] = nv
        order = np.argsort(-np.array(sn[-1]["visits"]))       # most-visited first
        mat = mat[:, order]
        im = ax.imshow(np.log10(mat.T + 1e-6), aspect="auto", origin="lower",
                       cmap="viridis", interpolation="nearest",
                       extent=[0, U, 0, S])
        frr_u = next((i for i, r in enumerate(m)
                      if r.get("env_steps_to_first_reward") is not None), None)
        if frr_u is not None:
            ax.axvline(frr_u, color="crimson", ls="--", lw=1.2)
        ax.set_title("Full-state novelty landscape  (rows=states sorted by visits, "
                     "log₁₀ novelty;  bright=novel)")
        ax.set_xlabel("update"); ax.set_ylabel("state (most-visited at bottom)")
        fig.colorbar(im, ax=ax, fraction=0.025, label="log₁₀ novelty")
    else:
        ax.text(0.5, 0.5, "no state_novelty.jsonl", ha="center")

    out = run / "dashboard.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[dashboard] wrote {out}")
    print(f"[dashboard] first reward @ {frr_steps} env-steps | "
          f"final entropy {_col(m,'entropy')[-1]:.3f} | "
          f"final noop {_col(m,'noop_fraction')[-1]:.2f} | "
          f"final inv_acc(hold) {_col(m,'inverse_acc_holdout')[-1]:.2f} | "
          f"coverage {int(_col(m,'cumulative_unique_states')[-1])}")


if __name__ == "__main__":
    main()
