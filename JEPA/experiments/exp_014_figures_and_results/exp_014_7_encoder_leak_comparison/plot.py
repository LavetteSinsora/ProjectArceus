"""exp_014_7 figures — encoder comparison of RND counting + cross-state leakage.

Reads results/encoder_leak_series_<tag>.npz (from diagnose.py) and writes:

  1. encoder_scatter_<tag>.png    novelty vs cumulative visit count, log–log, one
       panel per encoder + a pooled overlay. Drivers (o) and probes (X) marked
       distinctly. Per-state driver curves that separate = real resolution.
  2. encoder_probe_leak_<tag>.png the LEAK headline. Probe novelty over updates,
       one panel per encoder. Probes are (near-)never visited, so any decay is the
       count leaking from the distilled drivers. Flat-high = no leak (idm);
       decaying = leak (pixel/linproj/random).
  3. encoder_geometry_<tag>.png   monitored-set separation (scale-free chord L2) and
       raw cosine over updates, one line per encoder.
  4. encoder_influence_<tag>.png  the N×N cross-talk heatmap per encoder (final
       snapshot): infl[i,j] = fractional novelty drop on j from distilling i.
       Diagonal-dominant = clean counting; dense off-diagonal = leak.
  5. encoder_novelty_over_updates_<tag>.png  per-state novelty over updates (all
       states), one panel per encoder; masked phase shaded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# seaborn-deep hues, consistent with the shared figure palette
ENC_COLOR = {"pixel": "#C44E52", "linproj": "#8172B3", "random": "#DD8452", "idm": "#4C72B0"}
FLOOR = 1e-12


def _load(npz_path: Path) -> dict:
    d = dict(np.load(npz_path, allow_pickle=True))
    d["encoder_names"] = [str(x) for x in d["encoder_names"]]
    return d


def _dims(d: dict) -> dict:
    names = d["encoder_names"]
    if "encoder_dims" in d:
        return {n: int(v) for n, v in zip(names, d["encoder_dims"])}
    return {n: 0 for n in names}


def scatter_figure(d: dict, fig_dir: Path, tag: str) -> Path:
    names = d["encoder_names"]; nov = d["nov"]; cum = d["cum_visits"]
    is_probe = d.get("is_probe", np.zeros(nov.shape[1], dtype=int))
    n_enc, n_mon, _ = nov.shape
    cmap = plt.get_cmap("viridis", n_mon); dims = _dims(d)
    fig, axes = plt.subplots(1, n_enc + 1, figsize=(4.3 * (n_enc + 1), 4.2))
    for e, name in enumerate(names):
        ax = axes[e]
        for s in range(n_mon):
            x = np.maximum(cum[s], 0.5); y = np.maximum(nov[e, s], FLOOR)
            mk = "X" if is_probe[s] else "o"
            ax.scatter(x, y, s=26, color=cmap(s), alpha=0.85, marker=mk,
                       edgecolors="white", linewidths=0.4, zorder=3,
                       label=f"{'probe' if is_probe[s] else 'driver'} {s}")
            ax.plot(x, y, color=cmap(s), lw=0.8, alpha=0.35, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cumulative visit count")
        if e == 0:
            ax.set_ylabel("RND novelty"); ax.legend(fontsize=6, frameon=False, loc="lower left")
        ax.set_title(f"{name}  (D={dims.get(name, 0)})", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    ax = axes[-1]
    for e, name in enumerate(names):
        ax.scatter(np.maximum(cum.reshape(-1), 0.5), np.maximum(nov[e].reshape(-1), FLOOR),
                   s=15, color=ENC_COLOR.get(name, "#333"), alpha=0.5, label=name)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("cumulative visit count")
    ax.set_title("all encoders (pooled)", fontsize=10); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"exp_014_7 — novelty vs visit count by encoder  [{tag}]   "
                 "(o = driver, X = probe)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = fig_dir / f"encoder_scatter_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); return out


def probe_leak_figure(d: dict, fig_dir: Path, tag: str) -> Path:
    names = d["encoder_names"]; nov = d["nov"]
    is_probe = d.get("is_probe", np.zeros(nov.shape[1], dtype=int))
    probe_idx = np.where(is_probe == 1)[0]; driver_idx = np.where(is_probe == 0)[0]
    n_enc = nov.shape[0]; x = np.arange(nov.shape[2])
    updates_free = int(d["updates_free"]); updates_masked = int(d["updates_masked"])
    fig, axes = plt.subplots(1, n_enc, figsize=(4.4 * n_enc, 4.2), sharey=True)
    if n_enc == 1:
        axes = [axes]
    for e, name in enumerate(names):
        ax = axes[e]
        # driver envelope (grey) for reference
        if len(driver_idx):
            dmean = np.maximum(nov[e, driver_idx].mean(0), FLOOR)
            ax.plot(x, dmean, color="#999", lw=1.2, ls="--", label="drivers (mean)")
        for s in probe_idx:
            ax.plot(x, np.maximum(nov[e, s], FLOOR), lw=1.8, marker="o", ms=3,
                    label=f"probe {s}")
        if updates_masked > 0:
            ax.axvspan(updates_free + 0.5, x[-1] - 0.5, color="#bbb", alpha=0.25)
        ax.set_yscale("log"); ax.set_xlabel("update"); ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if e == 0:
            ax.set_ylabel("probe RND novelty"); ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"exp_014_7 — PROBE novelty (≈never visited) over updates  [{tag}]\n"
                 "probes decay only if the count LEAKS from the distilled drivers — "
                 "flat = no leak (good), decaying = leak", fontsize=11, y=1.03)
    fig.tight_layout()
    out = fig_dir / f"encoder_probe_leak_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); return out


def geometry_figure(d: dict, fig_dir: Path, tag: str) -> Path:
    names = d["encoder_names"]
    l2u = d.get("mean_l2_unit", d["mean_l2"]); cos = d["mean_cos"]
    x = np.arange(l2u.shape[1])
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2))
    for e, name in enumerate(names):
        c = ENC_COLOR.get(name, "#333")
        axl.plot(x, l2u[e], color=c, lw=2, marker="o", ms=3, label=name)
        axr.plot(x, cos[e], color=c, lw=2, marker="o", ms=3, label=name)
    axl.set_xlabel("update"); axl.set_ylabel("mean unit-norm (chord) L2 distance")
    axl.set_title("monitored-set separation, scale-free (higher = better)", fontsize=10)
    axl.grid(alpha=0.3); axl.legend(fontsize=8, frameon=False)
    axr.set_xlabel("update"); axr.set_ylabel("mean pairwise cosine similarity")
    axr.set_title("monitored-set similarity (lower = better)", fontsize=10)
    axr.grid(alpha=0.3); axr.legend(fontsize=8, frameon=False)
    fig.suptitle(f"exp_014_7 — representation geometry of the monitored states  [{tag}]",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = fig_dir / f"encoder_geometry_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); return out


def influence_figure(d: dict, fig_dir: Path, tag: str):
    infl = d.get("influence")
    if infl is None or infl.shape[1] == 0:
        return None
    names = d["encoder_names"]; n_enc = infl.shape[0]; n_mon = infl.shape[2]
    is_probe = d.get("is_probe", np.zeros(n_mon, dtype=int))
    snap = infl[:, -1]                                   # final snapshot (n_enc,N,N)
    d_idx = np.where(is_probe == 0)[0]; p_idx = np.where(is_probe == 1)[0]
    fig, axes = plt.subplots(1, n_enc, figsize=(3.7 * n_enc, 3.9))
    if n_enc == 1:
        axes = [axes]
    labels = [f"{'P' if is_probe[i] else 'D'}{i}" for i in range(n_mon)]
    for e, name in enumerate(names):
        ax = axes[e]
        im = ax.imshow(snap[e], vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(n_mon)); ax.set_yticks(range(n_mon))
        ax.set_xticklabels(labels, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("measured on state j");
        if e == 0:
            ax.set_ylabel("distilled state i")
        leak = (snap[e][np.ix_(d_idx, p_idx)].mean()
                if len(d_idx) and len(p_idx) else snap[e][~np.eye(n_mon, dtype=bool)].mean())
        ax.set_title(f"{name}\nleak(drv→probe)={leak:+.2f}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"exp_014_7 — influence / cross-talk: novelty drop on j from "
                 f"distilling i  [{tag}]\n"
                 "diagonal-dominant = clean per-state counting; dense = leak "
                 "(D=driver, P=probe)", fontsize=10.5, y=1.04)
    fig.tight_layout()
    out = fig_dir / f"encoder_influence_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); return out


def novelty_over_updates_figure(d: dict, fig_dir: Path, tag: str) -> Path:
    names = d["encoder_names"]; nov = d["nov"]; n_enc, n_mon, T1 = nov.shape
    is_probe = d.get("is_probe", np.zeros(n_mon, dtype=int))
    updates_free = int(d["updates_free"]); updates_masked = int(d["updates_masked"])
    cmap = plt.get_cmap("viridis", n_mon); x = np.arange(T1)
    fig, axes = plt.subplots(1, n_enc, figsize=(4.4 * n_enc, 4.2), sharey=True)
    if n_enc == 1:
        axes = [axes]
    for e, name in enumerate(names):
        ax = axes[e]
        for s in range(n_mon):
            ls = ":" if is_probe[s] else "-"
            ax.plot(x, np.maximum(nov[e, s], FLOOR), color=cmap(s), lw=1.5, ls=ls,
                    marker="o", ms=2.5, label=f"{'P' if is_probe[s] else 'D'}{s}")
        if updates_masked > 0:
            ax.axvspan(updates_free + 0.5, T1 - 0.5, color="#bbb", alpha=0.25)
        ax.set_yscale("log"); ax.set_xlabel("update"); ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if e == 0:
            ax.set_ylabel("RND novelty"); ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"exp_014_7 — per-state novelty over updates  [{tag}]  "
                 "(solid = driver, dotted = probe)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = fig_dir / f"encoder_novelty_over_updates_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); return out


def make_figures(npz_path, fig_dir, tag: str) -> None:
    npz_path = Path(npz_path); fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    d = _load(npz_path)
    for fn in (scatter_figure, probe_leak_figure, geometry_figure, influence_figure,
               novelty_over_updates_figure):
        out = fn(d, fig_dir, tag)
        if out is not None:
            print(f"[exp_014_7] wrote {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("npz"); p.add_argument("--tag", default=None)
    a = p.parse_args(); npz = Path(a.npz)
    make_figures(npz, npz.parent.parent / "figures",
                 a.tag or npz.stem.replace("encoder_leak_series_", ""))
