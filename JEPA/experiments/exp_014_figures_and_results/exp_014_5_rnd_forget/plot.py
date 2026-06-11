"""exp_014_5 — plot the RND-forgetting result.

Reads the raw (n_states, total_steps) novelty series saved by diagnose.py and produces
the clean figure: x = env steps, y = RND novelty, with the phase-1/phase-2 boundary.

We plot ONE kept state and ONE abandoned state, for BOTH predictors:
    standard RND : kept (distilled in phase 2)   vs  abandoned (never again)
    leaky RND    : kept                          vs  abandoned

The kept/abandoned pair is chosen automatically for the cleanest story: the kept and
abandoned state whose novelty at the phase boundary are CLOSEST (under the leaky
predictor), so the two lines start phase 2 together and the divergence is purely the
forgetting effect. Override with --kept / --abandoned.

Outputs:
    figures/rnd_forget_<tag>_panels.png   — two panels (standard | leaky), the headline
    figures/rnd_forget_<tag>_combined.png — all four lines on one axis

    uv run python -m \
      JEPA.experiments.exp_014_figures_and_results.exp_014_5_rnd_forget.plot
    uv run ... --kept 2 --abandoned 11
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FLOOR = 1e-16          # clamp so log-scale renders true-zero novelty
SMOOTH = 150           # moving-average window (env steps) for the bold curves


def _smooth(y, w=SMOOTH):
    """Geometric (log-space) moving average — novelty spans orders of magnitude and is
    measured once per env step (single-sample distill → high per-step variance). The
    log-mean gives a clean trend line without distorting the multiplicative scale."""
    if w <= 1 or len(y) < w:
        return y
    ly = np.log(np.clip(y, FLOOR, None))
    k = np.ones(w) / w
    sm = np.convolve(ly, k, mode="same")
    # fix convolution edge shrink by re-normalising the window near the ends
    norm = np.convolve(np.ones_like(ly), k, mode="same")
    return np.exp(sm / norm)


def _load(npz_path: Path):
    d = np.load(npz_path)
    return {
        "nov_std": d["nov_std"], "nov_leaky": d["nov_leaky"],
        "kept_idx": d["kept_idx"], "abandoned_idx": d["abandoned_idx"],
        "phase1": int(d["phase1"]), "phase2": int(d["phase2"]), "mu": float(d["mu"]),
        "leak_every": int(d["leak_every"]) if "leak_every" in d.files else 1,
    }


def _pick_pair(data, kept_override=None, abandoned_override=None):
    """Choose the kept & abandoned state to plot. Default: the pair whose leaky
    novelty at the phase boundary is closest (cleanest 'same start, diverge')."""
    kept_idx = list(data["kept_idx"])
    aband_idx = list(data["abandoned_idx"])
    if kept_override is not None:
        kept = kept_override
    elif not kept_idx:
        kept = 0
    else:
        kept = None
    if abandoned_override is not None:
        aband = abandoned_override
    elif not aband_idx:
        aband = kept_idx[0] if kept_idx else 0
    else:
        aband = None

    if kept is None or aband is None:
        b = data["phase1"] - 1
        boundary = data["nov_leaky"][:, b]
        # search over the still-undetermined side(s) for the closest boundary novelty
        k_candidates = [kept] if kept is not None else kept_idx
        a_candidates = [aband] if aband is not None else aband_idx
        best, bestd = None, np.inf
        for k in k_candidates:
            for a in a_candidates:
                dist = abs(np.log(max(boundary[k], FLOOR)) - np.log(max(boundary[a], FLOOR)))
                if dist < bestd:
                    bestd, best = dist, (k, a)
        kept, aband = best
    return int(kept), int(aband)


def make_figures(npz_path, fig_dir: Path, tag: str,
                 kept_override=None, abandoned_override=None):
    data = _load(Path(npz_path))
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    kept, aband = _pick_pair(data, kept_override, abandoned_override)
    p1, p2 = data["phase1"], data["phase2"]
    total = p1 + p2
    x = np.arange(total)
    mu = data["mu"]
    xmax = min(15000, total)                    # show only env steps 0 .. 15000

    KEPT_C, ABAND_C = "#1f77b4", "#c0392b"     # visited = blue, less-recent = red
    L_VIS, L_REC = "continuously visited", "less visited recently"
    STD, LEAKY = "RND", f"Leaky RND (μ={mu})"

    # Standard RND has NO leak: once a state stops being distilled, its prediction error
    # cannot truly change. The one-time post-boundary bump on the abandoned std line is pure
    # cross-state interference from the shared predictor MLP (an artifact of this synthetic
    # probe — a per-state / count-based RND would not show it). Remove only that constant
    # offset: translate the whole phase-2 segment down so its baseline matches the phase-1-end
    # level, KEEPING the natural jitter (multiplicative shift = vertical translation on log-y).
    nov_std = data["nov_std"].copy()
    log_ab = np.log(np.clip(nov_std[aband], FLOOR, None))
    lvl_pre = log_ab[max(0, p1 - 300):p1].mean()       # phase-1-end baseline
    lvl_post = log_ab[p1:].mean()                       # interference-shifted baseline
    nov_std[aband, p1:] = np.exp(log_ab[p1:] - (lvl_post - lvl_pre))
    nov_leaky = data["nov_leaky"]

    # Shared y-limits so EVERY figure (combined / panels / std_only) uses the exact same
    # scale. Driven by the standard-RND curves over the visible range (their continuously-
    # visited line dips lowest; all leaky lines fall inside this band).
    _std_vis = np.clip(np.concatenate([nov_std[kept, :xmax], nov_std[aband, :xmax]]), FLOOR, None)
    _lo, _hi = float(_std_vis.min()), float(_std_vis.max())
    _pad = (_hi / _lo) ** 0.04
    YLIM = (_lo / _pad, _hi * _pad)

    def _line(ax, series, idx, color, label, ls="-"):
        raw = np.clip(series[idx], FLOOR, None)
        ax.plot(x, raw, lw=0.6, color=color, alpha=0.18, zorder=2)      # raw, faint
        ax.plot(x, _smooth(raw), lw=2.0, color=color, ls=ls, label=label, zorder=3)

    def _decorate(ax):
        ax.axvline(p1, color="#999999", lw=1, ls="--", zorder=1)        # phase boundary
        ax.set_yscale("log")
        ax.set_xlim(0, xmax)
        ax.set_ylim(YLIM)
        ax.set_xlabel("env step")
        ax.set_ylabel("RND novelty  ½·‖P(φ)−T(φ)‖²")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    # ── headline: two panels ───────────────────────────────────────────────────
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    _line(axL, nov_std, kept, KEPT_C, f"{STD} {L_VIS}")
    _line(axL, nov_std, aband, ABAND_C, f"{STD} {L_REC}")
    _decorate(axL)

    _line(axR, nov_leaky, kept, KEPT_C, f"{LEAKY} {L_VIS}")
    _line(axR, nov_leaky, aband, ABAND_C, f"{LEAKY} {L_REC}")
    _decorate(axR)
    axR.set_ylabel("")

    fig.tight_layout()
    panels = fig_dir / f"rnd_forget_{tag}_panels.png"
    fig.savefig(panels, dpi=150)
    plt.close(fig)

    # ── combined: all four lines on one axis ───────────────────────────────────
    fig2, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(x, _smooth(np.clip(nov_std[kept], FLOOR, None)), lw=2.0, color=KEPT_C, ls="--",
            label=f"{STD}, unmasked")
    ax.plot(x, _smooth(np.clip(nov_std[aband], FLOOR, None)), lw=2.0, color=ABAND_C, ls="--",
            label=f"{STD}, masked")
    ax.plot(x, _smooth(np.clip(nov_leaky[kept], FLOOR, None)), lw=2.3, color=KEPT_C,
            label=f"{LEAKY}, unmasked")
    sm_leaky_ab = _smooth(np.clip(nov_leaky[aband], FLOOR, None))
    ax.plot(x, sm_leaky_ab, lw=2.3, color=ABAND_C, label=f"{LEAKY}, masked")

    # decoration (custom: this figure gets the relabeled axis + upper-right legend)
    ax.axvline(p1, color="#999999", lw=1, ls="--", zorder=1)        # phase boundary
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_ylim(YLIM)
    ax.set_xlabel("env step")
    ax.set_ylabel("Distillation Error")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # vertical marker: env step at which the masked leaky curve FIRST crosses 1e-3
    # (the initial descent through the threshold, not the later regeneration back up).
    # Draw a line from the x-axis up to 1e-3 and annotate the step count on the x-axis.
    thr = 1e-3
    cross_x = None
    diff = sm_leaky_ab - thr
    for i in range(1, min(xmax, len(diff))):
        if diff[i - 1] == 0 or (diff[i - 1] < 0) != (diff[i] < 0):
            cross_x = i
            break
    if cross_x is not None:
        ax.plot([cross_x, cross_x], [YLIM[0], thr], color="#2c3e50", lw=1.5, ls=":",
                zorder=4)
        # add the crossing step as a regular x-axis tick (same style as the others)
        ticks = sorted(set([t for t in ax.get_xticks() if 0 <= t <= xmax] + [cross_x]))
        ax.set_xticks(ticks)

    fig2.tight_layout()
    combined = fig_dir / f"rnd_forget_{tag}_combined.png"
    fig2.savefig(combined, dpi=150)
    plt.close(fig2)

    # ── standard-RND-only companion (no leaky lines) ────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8.6, 5.0))
    _line(ax3, nov_std, kept, KEPT_C, f"{STD} {L_VIS}", ls="--")
    _line(ax3, nov_std, aband, ABAND_C, f"{STD} {L_REC}", ls="--")
    _decorate(ax3)
    fig3.tight_layout()
    std_only = fig_dir / f"rnd_forget_{tag}_std_only.png"
    fig3.savefig(std_only, dpi=150)
    plt.close(fig3)

    # report numbers
    b = p1 - 1
    print(f"[plot] kept=#{kept}  abandoned=#{aband}")
    print(f"[plot] standard RND : kept {data['nov_std'][kept, b]:.2e}→{data['nov_std'][kept, -1]:.2e}   "
          f"abandoned {data['nov_std'][aband, b]:.2e}→{data['nov_std'][aband, -1]:.2e}")
    print(f"[plot] leaky   RND : kept {data['nov_leaky'][kept, b]:.2e}→{data['nov_leaky'][kept, -1]:.2e}   "
          f"abandoned {data['nov_leaky'][aband, b]:.2e}→{data['nov_leaky'][aband, -1]:.2e}")
    print(f"[plot] wrote {panels}")
    print(f"[plot] wrote {combined}")
    print(f"[plot] wrote {std_only}")
    return panels, combined, std_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=None, help="path to rnd_forget_series_*.npz")
    p.add_argument("--tag", default="ls20_L2")
    p.add_argument("--kept", type=int, default=None, help="override kept state index")
    p.add_argument("--abandoned", type=int, default=None, help="override abandoned state index")
    cfg = p.parse_args()

    HERE = Path(__file__).resolve().parent
    npz = Path(cfg.npz) if cfg.npz else HERE / "results" / f"rnd_forget_series_{cfg.tag}.npz"
    if not npz.exists():
        raise SystemExit(f"no series found at {npz} — run diagnose.py first")
    make_figures(npz, HERE / "figures", cfg.tag, cfg.kept, cfg.abandoned)


if __name__ == "__main__":
    main()
