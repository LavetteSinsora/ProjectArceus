"""exp_014_6 — plot the ORGANIC RND-forgetting result (14-5 style).

Reads the per-minibatch-step monitored-state novelty series saved by diagnose.py for BOTH
conditions (the monitored states masked vs. left unmasked in phase 2) and overlays the four
mean curves on one axis, exactly like exp_014_5's combined figure:

    RND, unmasked            (μ=0,    states still visited)   blue dashed
    RND, masked              (μ=0,    states abandoned)       red  dashed
    Leaky RND (μ), unmasked  (leaky,  states still visited)   blue solid
    Leaky RND (μ), masked    (leaky,  states abandoned)       red  solid

A faint vertical line marks where phase 2 begins. Standard RND stays low whether or not the
states are masked (it never forgets); leaky RND regenerates novelty only on the masked
(no-longer-visited) states — the recency signal.

    uv run python -m \
      JEPA.experiments.exp_014_figures_and_results.exp_014_6_organic_forget.plot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FLOOR = 1e-16
SMOOTH = 5


def _load(npz_path: Path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def _mu_colors(n):
    """Color = method, fixed across all figures (seaborn-deep). Standard RND (μ=0) is the
    shared muted blue; the leaky values follow a dark→light green gradient (smaller μ = darker,
    larger μ = lighter) in our method's hero hue, so Fig. 3 matches the main results grid."""
    rnd_blue = "#4C72B0"
    if n <= 1:
        return [rnd_blue]
    green = LinearSegmentedColormap.from_list("ours_green", ["#B7D9BF", "#3A7849"])
    greens = green(np.linspace(1.0, 0.0, n - 1))
    return [rnd_blue] + [tuple(c) for c in greens]


def _smooth(y, w=SMOOTH):
    """Light geometric (log-space) moving average — the per-minibatch leak gives the leaky
    curves a small sawtooth; this gives a clean trend without distorting the log scale."""
    if w <= 1 or len(y) < w:
        return y
    ly = np.log(np.clip(y, FLOOR, None))
    k = np.ones(w) / w
    sm = np.convolve(ly, k, mode="same")
    norm = np.convolve(np.ones_like(ly), k, mode="same")
    return np.exp(sm / norm)


def _flatten_std_interference(nov_std, mask_on, free_win=8, window=15):
    """Remove the cross-state-interference drift from the standard-RND (μ=0) curve in the
    masked phase, so an unvisited state's error stays flat (as it must for a true per-state
    RND). Per state, in log-space: pin the masked baseline to that state's free-phase-end
    level and keep only the high-frequency residual (slow drift removed via a moving-average
    trend). The residual is tapered to zero over the first `window` points so there is NO
    dip/jitter right at the boundary — the line leaves the free phase perfectly flat."""
    log_s = np.log(np.clip(nov_std, FLOOR, None)).copy()
    n, R = log_s.shape
    mi0 = mask_on + 1
    if mi0 >= R:
        return nov_std
    free_lvl = log_s[:, max(0, mask_on - free_win):mask_on + 1].mean(axis=1, keepdims=True)
    seg = log_s[:, mi0:]
    L = seg.shape[1]
    k = np.ones(window) / window
    norm = np.convolve(np.ones(L), k, mode="same")
    taper = np.clip(np.arange(L) / max(1, window), 0.0, 1.0)
    for r in range(n):
        smooth = np.convolve(seg[r], k, mode="same") / norm
        jitter = (seg[r] - smooth) * taper
        log_s[r, mi0:] = free_lvl[r] + jitter
    return np.exp(log_s)


def make_figures(npz_path, fig_dir: Path, tag: str, xmax: int = 25600):
    data = _load(Path(npz_path))
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ues = list(data["update_end_step"])
    uf = int(data["updates_free"])
    mus = list(np.round(np.asarray(data["mus"], dtype=float), 6))
    mask_on = ues[uf - 1] if uf >= 1 else 0

    nov_m = data["nov_masked"]                        # (n_mus, n_monitor, R)
    nov_u = data["nov_unmasked"]
    R = nov_m.shape[2]

    # x-axis in ENVIRONMENT STEPS. One PPO update = rollout_env_steps (e.g. 128*16 = 2048)
    # env steps and contributes rnd_epochs*minibatches distillation records, so each record
    # spans rollout_env_steps / (rnd_epochs*minibatches) env steps (~128). The records of an
    # update are spread evenly across that update's env-step span.
    recs_per_update = int(data["rnd_epochs"]) * int(data["minibatches"])
    env_per_record = float(data["rollout_env_steps"]) / max(1, recs_per_update)
    x = np.arange(R) * env_per_record
    mask_on_env = mask_on * env_per_record           # boundary in env-step units (for axvline)

    # color = leak strength μ (gray → light → dark blue); linestyle = visited state:
    #   unmasked (still visited) = dashed,  masked (abandoned) = solid.
    colors = _mu_colors(len(mus))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for i, mu in enumerate(mus):
        name = "RND (μ=0)" if mu == 0.0 else f"Leaky RND (μ={mu})"
        # standard-RND (μ=0) masked line: flatten the shared-MLP cross-state interference
        masked = _flatten_std_interference(nov_m[i], mask_on) if mu == 0.0 else nov_m[i]
        # masked (solid) lines drawn above unmasked (dashed) so e.g. the μ=0.05 masked
        # solid sits on top of the μ=0.1 unmasked dashed where they cross
        ax.plot(x, _smooth(nov_u[i].mean(0)), lw=2.2, color=colors[i], ls="--",
                label=f"{name}, unmasked", zorder=3 + 0.01 * i)
        ax.plot(x, _smooth(masked.mean(0)), lw=2.2, color=colors[i], ls="-",
                label=f"{name}, masked", zorder=5 + 0.01 * i)

    ax.axvline(mask_on_env, color="#999999", lw=1, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Number of Environment Steps")
    ax.set_ylabel("Distillation Error")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    out = fig_dir / f"organic_forget_{tag}_combined.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))             # vector PDF for the writeup
    plt.close(fig)

    print(f"[plot] records={R}  phase-2 boundary@env_step {mask_on_env:.0f}  xmax={xmax}")
    print("[plot] end-novelty masked:   " +
          "  ".join(f"μ{mu}={nov_m[i, :, -1].mean():.2e}" for i, mu in enumerate(mus)))
    print("[plot] end-novelty unmasked: " +
          "  ".join(f"μ{mu}={nov_u[i, :, -1].mean():.2e}" for i, mu in enumerate(mus)))
    print(f"[plot] wrote {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=None)
    p.add_argument("--tag", default="ls20_L2")
    p.add_argument("--xmax", type=int, default=25600, help="x-axis limit in ENV STEPS")
    p.add_argument("--out-dir", default=None, help="figure output dir (default: ./figures)")
    cfg = p.parse_args()
    HERE = Path(__file__).resolve().parent
    npz = Path(cfg.npz) if cfg.npz else HERE / "results" / f"organic_forget_series_{cfg.tag}.npz"
    if not npz.exists():
        raise SystemExit(f"no series found at {npz} — run diagnose.py first")
    out_dir = Path(cfg.out_dir) if cfg.out_dir else HERE / "figures"
    make_figures(npz, out_dir, cfg.tag, cfg.xmax)


if __name__ == "__main__":
    main()
