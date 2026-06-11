"""exp_014_7 — aggregate multiple seeds into median/IQR figures + a summary.

Loads encoder_leak_series_<game>_L<n>_seed<k>.npz for the given seeds, stacks the
per-seed series (seeds differ in env stream, RND target, and IDM init), and writes:

  agg_geometry_<tag>.png   chord-L2 + raw cosine, median line + IQR band per encoder
  agg_probe_leak_<tag>.png probe novelty over updates, median + IQR per encoder
  agg_leak_bar_<tag>.png   final influence off-diagonal (leak) per encoder, median
                           ± IQR across seeds — the one-number leak comparison
  encoder_leak_aggregate_<tag>.json   the numbers

Across-seed alignment is by update index (all seeds use the same schedule), so the
series share a common length; seeds with a different length are truncated to the min.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# seaborn-deep hues, consistent with plot.py / the shared figure palette
ENC_COLOR = {"pixel": "#C44E52", "linproj": "#8172B3", "random": "#DD8452", "idm": "#4C72B0"}
FLOOR = 1e-12


def _stack(series: list[np.ndarray]) -> np.ndarray:
    """Stack per-seed arrays on a new leading axis, truncating axis-(-1) to the min
    common length so ragged runs still aggregate."""
    L = min(s.shape[-1] for s in series)
    return np.stack([s[..., :L] for s in series], axis=0)


def _band(ax, x, mat, color, label, logy=False):
    """mat: (seeds, T). Plot median + IQR band."""
    med = np.nanmedian(mat, axis=0)
    lo = np.nanpercentile(mat, 25, axis=0); hi = np.nanpercentile(mat, 75, axis=0)
    ax.plot(x[:len(med)], med, color=color, lw=2, label=label)
    ax.fill_between(x[:len(med)], lo, hi, color=color, alpha=0.18)
    if logy:
        ax.set_yscale("log")


def aggregate(res_dir: Path, fig_dir: Path, game: str, level: int, seeds: list[int]):
    res_dir = Path(res_dir); fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{game}_L{level + 1}"
    runs = []
    for s in seeds:
        p = res_dir / f"encoder_leak_series_{tag}_seed{s}.npz"
        if p.exists():
            runs.append(dict(np.load(p, allow_pickle=True)))
    if not runs:
        print(f"[aggregate] no seed npz found for {tag}"); return
    names = [str(x) for x in runs[0]["encoder_names"]]
    n_enc = len(names)
    print(f"[aggregate] {tag}: {len(runs)} seeds, encoders={names}")

    # geometry: (seeds, n_enc, T)
    l2u = _stack([r.get("mean_l2_unit", r["mean_l2"]) for r in runs])
    cos = _stack([r["mean_cos"] for r in runs])
    T = l2u.shape[-1]; x = np.arange(T)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2))
    for e, name in enumerate(names):
        c = ENC_COLOR.get(name, "#333")
        _band(axl, x, l2u[:, e], c, name); _band(axr, x, cos[:, e], c, name)
    axl.set_title("chord L2 (median ± IQR)", fontsize=10); axl.set_xlabel("update")
    axl.set_ylabel("mean unit-norm L2"); axl.grid(alpha=0.3); axl.legend(fontsize=8, frameon=False)
    axr.set_title("cosine (median ± IQR)", fontsize=10); axr.set_xlabel("update")
    axr.set_ylabel("mean cosine"); axr.grid(alpha=0.3); axr.legend(fontsize=8, frameon=False)
    fig.suptitle(f"exp_014_7 aggregate geometry  [{tag}, {len(runs)} seeds]", y=1.02)
    fig.tight_layout(); out = fig_dir / f"agg_geometry_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"[aggregate] wrote {out}")

    # probe leak: mean probe novelty per seed → (seeds, n_enc, T)
    is_probe = runs[0]["is_probe"]
    nov = _stack([r["nov"] for r in runs])                       # (seeds, n_enc, n_mon, T)
    probe_nov = nov[:, :, is_probe == 1, :].mean(axis=2)         # (seeds, n_enc, T)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for e, name in enumerate(names):
        _band(ax, np.arange(probe_nov.shape[-1]), np.maximum(probe_nov[:, e], FLOOR),
              ENC_COLOR.get(name, "#333"), name, logy=True)
    ax.set_xlabel("update"); ax.set_ylabel("mean probe novelty")
    ax.set_title(f"exp_014_7 — probe (held-out) novelty: leak = decay  [{tag}, "
                 f"{len(runs)} seeds]\nflat = no leak (idm); decaying = leak", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); out = fig_dir / f"agg_probe_leak_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"[aggregate] wrote {out}")

    # leak bar: final influence off-diagonal per encoder, across seeds
    leak_summary = {}
    have_infl = all(r["influence"].shape[1] > 0 for r in runs)
    if have_infl:
        n_mon = runs[0]["nov"].shape[1]
        d_idx = np.where(is_probe == 0)[0]; p_idx = np.where(is_probe == 1)[0]
        # leak = how much a never-distilled PROBE drops when a DRIVER is distilled.
        leak = np.array([[r["influence"][e, -1][np.ix_(d_idx, p_idx)].mean()
                          for e in range(n_enc)] for r in runs])  # (seeds, n_enc)
        self_ = np.array([[np.diag(r["influence"][e, -1]).mean() for e in range(n_enc)]
                          for r in runs])
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        med = np.nanmedian(leak, axis=0)
        lo = med - np.nanpercentile(leak, 25, axis=0); hi = np.nanpercentile(leak, 75, axis=0) - med
        ax.bar(range(n_enc), med, yerr=np.vstack([lo, hi]), capsize=5,
               color=[ENC_COLOR.get(n, "#333") for n in names], alpha=0.85)
        ax.set_xticks(range(n_enc)); ax.set_xticklabels(names)
        ax.set_ylabel("driver→probe leak (final)")
        ax.set_title(f"exp_014_7 — cross-state leak by encoder  [{tag}, {len(runs)} seeds]\n"
                     "novelty drop on a never-distilled probe when a driver is counted "
                     "(lower = less leak)", fontsize=10)
        ax.grid(alpha=0.3, axis="y"); ax.axhline(0, color="k", lw=0.6)
        fig.tight_layout(); out = fig_dir / f"agg_leak_bar_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"[aggregate] wrote {out}")
        leak_summary = {
            n: {"leak_offdiag_median": float(np.nanmedian(leak[:, e])),
                "leak_offdiag_iqr": [float(np.nanpercentile(leak[:, e], 25)),
                                     float(np.nanpercentile(leak[:, e], 75))],
                "self_diag_median": float(np.nanmedian(self_[:, e]))}
            for e, n in enumerate(names)}

    summary = {
        "tag": tag, "n_seeds": len(runs), "seeds_used": [int(s) for s in seeds],
        "encoders": names,
        "final_chord_l2_median": {n: float(np.nanmedian(l2u[:, e, -1])) for e, n in enumerate(names)},
        "final_cosine_median": {n: float(np.nanmedian(cos[:, e, -1])) for e, n in enumerate(names)},
        "final_probe_novelty_median": {n: float(np.nanmedian(probe_nov[:, e, -1])) for e, n in enumerate(names)},
        "leak": leak_summary,
    }
    out = res_dir / f"encoder_leak_aggregate_{tag}.json"
    out.write_text(json.dumps(summary, indent=2)); print(f"[aggregate] wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20"); ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    HERE = Path(__file__).resolve().parent
    aggregate(HERE / "results", HERE / "figures", a.game, a.level, a.seeds)
