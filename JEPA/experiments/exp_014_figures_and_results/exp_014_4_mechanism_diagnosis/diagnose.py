"""exp_014_4 — Mechanism diagnosis: why do methods work/fail on L1/L2/L3?

Reads exp_013 training metrics and produces four sub-figures:

(A) Novelty signal decay: std_RND vs leaky_RND on tu93 L1 (succeeds) and ls20 L2 (fails)
    → shows leaky maintains signal better, but signal alone doesn't explain L2 failure

(B) Policy entropy trajectory: std_RND vs leaky vs ICM on failing L2 runs
    → leaky RND preserves entropy better than std; ICM collapses entropy more but
      that collapse is PRODUCTIVE on tu93 L3 (policy converges to the reward path)

(C) External value signal: v_ext over training for all methods on L2
    → v_ext stays ~0 for ALL methods on ls20/re86/g50t L2 = agent NEVER sees the reward
      This is the key bottleneck: not saturation, but zero reward coverage

(D) ICM forward error on tu93 L3 (works) vs ls20 L2 (fails even with ICM)
    → on tu93 L3, forward_error stays high → sustained exploration toward hard transitions
      on ls20 L2, ICM also gives up (forward_err decays, still no reward)

    uv run python -m \
      JEPA.experiments.exp_014_figures_and_results.exp_014_4_mechanism_diagnosis.diagnose
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = (Path(__file__).resolve().parent.parent / "data")
FIG  = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {
    "std_rnd":  "#2e86de",
    "A_leaky":  "#8e44ad",
    "ICM":      "#e67e22",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_runs(folder: str, filt: str | None, game: str, level: int,
              max_seeds: int = 4) -> list[list[dict]]:
    runs_dir = DATA / folder / "runs"
    if not runs_dir.exists():
        return []
    runs = sorted(d for d in runs_dir.iterdir()
                  if f"_{game}_L{level+1}_" in d.name
                  and (filt is None or f"_{filt}_" in d.name))[:max_seeds]
    out = []
    for r in runs:
        mf = r / "metrics.jsonl"
        if mf.exists():
            out.append([json.loads(l) for l in mf.read_text().splitlines() if l])
    return out


def smooth(arr, w=10):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def extract(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def steps(rows: list[dict]) -> np.ndarray:
    return np.array([r["step"] for r in rows], dtype=float)


def _safe_interp(xi, yi, n):
    xi, yi = np.asarray(xi), np.asarray(yi)
    k = min(len(xi), len(yi))
    if k < 2:
        return None
    return np.interp(np.linspace(0, xi[k-1], n), xi[:k], yi[:k])


def plot_band(ax, x, ys, color, label, lw=2.0):
    """Plot median ± IQR band across multiple seeds."""
    ymat = np.full((len(ys), len(x)), np.nan)
    for i, y in enumerate(ys):
        n = min(len(y), len(x))
        ymat[i, :n] = y[:n]
    med = np.nanmedian(ymat, axis=0)
    lo  = np.nanpercentile(ymat, 25, axis=0)
    hi  = np.nanpercentile(ymat, 75, axis=0)
    ax.plot(x, med, color=color, lw=lw, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15)


def render_metric(ax, configs, metric_key, ylabel, title, xscale=1e3, extra=None):
    """Generic panel renderer — handles all the interpolation safely."""
    max_x_all = 0.0
    series = []  # (label, color, ys_interp, max_x)
    for row in configs:
        label, folder, filt, game, lvl = row[:5]
        color = row[5] if len(row) > 5 else "#333"
        max_seeds = row[6] if len(row) > 6 else 4
        run_data = load_runs(folder, filt, game, lvl, max_seeds=max_seeds)
        if not run_data:
            continue
        all_sig = [smooth(extract(r, metric_key)) for r in run_data]
        all_x   = [smooth(steps(r)[:len(s)]) for r, s in zip(run_data, all_sig)]
        if not any(len(x) > 1 for x in all_x):
            continue
        max_x = max(x[-1] for x in all_x if len(x) > 1)
        max_x_all = max(max_x_all, max_x)
        n = max(len(s) for s in all_sig)
        ys_interp = []
        for xi, yi in zip(all_x, all_sig):
            r = _safe_interp(xi, yi, n)
            if r is not None:
                ys_interp.append(r)
        if ys_interp:
            series.append((label, color, ys_interp, max_x))

    for label, color, ys_interp, max_x in series:
        n = max(len(y) for y in ys_interp)
        xgrid = np.linspace(0, max_x, n) / xscale
        plot_band(ax, xgrid, ys_interp, color, label[:28], lw=1.8)

    if extra:
        extra(ax)
    ax.set_xlabel(f"env steps ({'k' if xscale==1e3 else ''})")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, loc="best")
    ax.grid(alpha=0.3)


# ── figure ────────────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()

    # (A) Novelty signal ───────────────────────────────────────────────────────
    render_metric(axes[0], [
        ("std_RND tu93-L1 (5/8 solved)", "baseline_icm_rnd", "rnd", "tu93", 0, COLORS["std_rnd"]),
        ("leaky-A tu93-L1 (4/4 solved)", "leaky_rnd",  None, "tu93", 0, COLORS["A_leaky"]),
        ("std_RND ls20-L2 (0/8 NEVER)",  "baseline_icm_rnd", "rnd", "ls20", 1, "#5d9cec"),
        ("leaky-A ls20-L2 (0/2 NEVER)",  "leaky_rnd",  None, "ls20", 1, "#be90d4"),
    ], "intrinsic_reward_raw_mean", "intrinsic reward (raw mean)",
       "(A) Novelty signal: solved vs unsolved")

    # (B) Policy entropy ───────────────────────────────────────────────────────
    def ent_extra(ax):
        ax.axhline(np.log(4), color="#95a5a6", lw=1, ls=":", label="max entropy (4 act)")
    render_metric(axes[1], [
        ("std_RND ls20-L2",  "baseline_icm_rnd", "rnd", "ls20", 1, COLORS["std_rnd"]),
        ("leaky-A ls20-L2",  "leaky_rnd",  None, "ls20", 1, COLORS["A_leaky"]),
        ("ICM     ls20-L2",  "baseline_icm_rnd", "icm", "ls20", 1, COLORS["ICM"]),
        ("ICM     tu93-L3 (SOLVED)", "baseline_icm_rnd", "icm", "tu93", 2, "#f39c12"),
    ], "policy_entropy", "policy entropy (nats)",
       "(B) Policy entropy: leaky preserves exploration, ICM converges productively",
       extra=ent_extra)

    # (C) Extrinsic value ─────────────────────────────────────────────────────
    def vext_extra(ax):
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.annotate("Reward NEVER seen on L2 → v_ext stays at 0\n"
                    "Bottleneck: credit assignment, not saturation",
                    xy=(0.02, 0.05), xycoords="axes fraction",
                    fontsize=7.5, color="#c0392b",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    render_metric(axes[2], [
        ("std_RND tu93-L1 (solved)", "baseline_icm_rnd", "rnd", "tu93", 0, COLORS["std_rnd"], 2),
        ("leaky-A tu93-L1 (solved)", "leaky_rnd",  None, "tu93", 0, COLORS["A_leaky"],  2),
        ("ICM     tu93-L3 (solved)", "baseline_icm_rnd", "icm", "tu93", 2, COLORS["ICM"],    2),
        ("std_RND ls20-L2 (NEVER)",  "baseline_icm_rnd", "rnd", "ls20", 1, "#5d9cec",   2),
        ("ICM     ls20-L2 (NEVER)",  "baseline_icm_rnd", "icm", "ls20", 1, "#f0a500",   2),
        ("ICM     re86-L2 (NEVER)",  "baseline_icm_rnd", "icm", "re86", 1, "#c0392b",   2),
    ], "v_ext_mean", "extrinsic value estimate",
       "(C) v_ext stays at 0 on L2 — reward NEVER seen by any method",
       extra=vext_extra)

    # (D) ICM forward error ────────────────────────────────────────────────────
    def fw_extra(ax):
        ax.annotate("sustained →\nproductive curiosity\n(tu93-L3 solved)",
                    xy=(0.55, 0.82), xycoords="axes fraction",
                    fontsize=7.5, color="#e67e22")
        ax.annotate("decays →\ndead signal\n(reward never reached)",
                    xy=(0.55, 0.28), xycoords="axes fraction",
                    fontsize=7.5, color="#c0392b")
    render_metric(axes[3], [
        ("ICM tu93-L3 (8/8 solved)", "baseline_icm_rnd", "icm", "tu93", 2, "#e67e22"),
        ("ICM tu93-L1 (8/8 solved)", "baseline_icm_rnd", "icm", "tu93", 0, "#f39c12"),
        ("ICM ls20-L2 (0/8 NEVER)",  "baseline_icm_rnd", "icm", "ls20", 1, "#c0392b"),
        ("ICM re86-L2 (0/8 NEVER)",  "baseline_icm_rnd", "icm", "re86", 1, "#922b21"),
    ], "forward_error_mean", "ICM forward prediction error",
       "(D) ICM forward error: sustained on tu93-L3, decays on failing L2",
       extra=fw_extra)

    fig.suptitle(
        "Mechanism diagnosis: leaky RND maintains exploration entropy (+), "
        "but L2/L3 reward NEVER seen on ls20/re86/g50t\n"
        "Bottleneck = credit assignment over long sequences, not novelty saturation  "
        "| tu93 is special: L2 trivially easy (784 steps), L3 ICM-aligned",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    out = FIG / "mechanism_diagnosis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
