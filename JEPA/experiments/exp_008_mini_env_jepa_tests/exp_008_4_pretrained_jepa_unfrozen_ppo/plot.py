"""Assemble exp_008_4 results.

Six-curve overlapped success-rate plot:
    - 1rot / 2rot × {joint baseline, frozen JEPA + PPO (from 008_2),
                     unfrozen JEPA + PPO (this experiment)}

Plus two diagnostic figures:
    - mean_feature_cosine over PPO updates, all six runs overlapped
      (the one collapse metric every PPO trainer in the family logs).
    - feat_std / feat_pairwise_l2 / feat_effective_rank vs update,
      008_4 unfrozen runs only (the new runs are the only ones that
      log these per-update; baseline + frozen runs don't).

Usage:
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_4_pretrained_jepa_unfrozen_ppo.plot
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_4_pretrained_jepa_unfrozen_ppo.plot --no_fig
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.config import (
    PPO_RUNS_DIR as EXP_008_2_PPO_RUNS,
)

from .config import PPO_RUNS_DIR, RESULTS_DIR


# (env_tag, role) → run-glob context.
# `role` is one of "baseline" / "frozen" / "unfrozen".
RUN_KEYS = [
    ("1rot", "baseline"),
    ("1rot", "frozen"),
    ("1rot", "unfrozen"),
    ("2rot", "baseline"),
    ("2rot", "frozen"),
    ("2rot", "unfrozen"),
]


def _read_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    out = []
    for line in metrics_path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _curve(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in records:
        if key in r:
            xs.append(int(r.get("env_step", r.get("update", 0))))
            ys.append(float(r[key]))
    return xs, ys


def _steps_to_threshold(records: list[dict], threshold: float) -> int | None:
    for r in records:
        if r.get("eval_success_rate", 0.0) >= threshold:
            return int(r["env_step"])
    return None


def _final_metrics(records: list[dict]) -> dict:
    last_eval = None
    for r in records:
        if "eval_success_rate" in r:
            last_eval = r
    if last_eval is None:
        return {}
    return {
        "final_env_step": int(last_eval["env_step"]),
        "final_eval_success_rate": float(last_eval["eval_success_rate"]),
        "final_avg_steps_to_solve": float(last_eval.get("eval_avg_steps_to_solve", float("nan"))),
    }


def _latest(glob_pat: str, base: Path) -> Path | None:
    matches = sorted(base.glob(glob_pat))
    return matches[-1] if matches else None


def _resolve_runs(args) -> dict[tuple[str, str], Path | None]:
    runs: dict[tuple[str, str], Path | None] = {}

    if args.baseline_1rot_dir:
        runs[("1rot", "baseline")] = Path(args.baseline_1rot_dir)
    else:
        ext = Path("JEPA/experiments/exp_007_mini_env_cnn_ppo_baseline/runs")
        runs[("1rot", "baseline")] = _latest("exp_007_0_naive_*", ext)

    runs[("2rot", "baseline")] = _latest(
        "exp_008_2_joint_cnn_ppo__2rot_*", EXP_008_2_PPO_RUNS
    )
    runs[("1rot", "frozen")] = _latest(
        "exp_008_2_frozen_jepa_ppo__1rot_*", EXP_008_2_PPO_RUNS
    )
    runs[("2rot", "frozen")] = _latest(
        "exp_008_2_frozen_jepa_ppo__2rot_*", EXP_008_2_PPO_RUNS
    )
    runs[("1rot", "unfrozen")] = _latest(
        "exp_008_4_pretrained_jepa_unfrozen_ppo__1rot_*", PPO_RUNS_DIR
    )
    runs[("2rot", "unfrozen")] = _latest(
        "exp_008_4_pretrained_jepa_unfrozen_ppo__2rot_*", PPO_RUNS_DIR
    )
    return runs


# Style map for the six (env_tag, role) curves on the headline plot.
STYLE_MAP: dict[tuple[str, str], dict] = {
    ("1rot", "baseline"): {"color": "#444",    "linestyle": "--", "label": "1rot · joint CNN+PPO (baseline)"},
    ("1rot", "frozen"):   {"color": "#1f77b4", "linestyle": ":",  "label": "1rot · frozen JEPA + PPO"},
    ("1rot", "unfrozen"): {"color": "#1f77b4", "linestyle": "-",  "label": "1rot · unfrozen JEPA + PPO"},
    ("2rot", "baseline"): {"color": "#aa3333", "linestyle": "--", "label": "2rot · joint CNN+PPO (baseline)"},
    ("2rot", "frozen"):   {"color": "#d62728", "linestyle": ":",  "label": "2rot · frozen JEPA + PPO"},
    ("2rot", "unfrozen"): {"color": "#d62728", "linestyle": "-",  "label": "2rot · unfrozen JEPA + PPO"},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_1rot_dir", type=str, default=None,
                   help="override path to exp_007_0_naive 1rot run")
    p.add_argument("--no_fig", action="store_true",
                   help="skip matplotlib (text-only summary)")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    runs = _resolve_runs(args)
    print("[plot-008_4] resolved runs:")
    for k, v in runs.items():
        print(f"  {k}: {v}")

    summary: dict = {"runs": {}, "headline": {}}
    curves_eval: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
    curves_cos: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
    for key in RUN_KEYS:
        env_tag, role = key
        run_dir = runs.get(key)
        run_label = f"{env_tag}__{role}"
        if run_dir is None or not run_dir.exists():
            summary["runs"][run_label] = {"status": "missing"}
            continue
        recs = _read_metrics(run_dir)
        curves_eval[key] = _curve(recs, "eval_success_rate")
        curves_cos[key] = _curve(recs, "mean_feature_cosine")
        summary["runs"][run_label] = {
            "status": "ok",
            "run_dir": str(run_dir),
            "steps_to_90pct": _steps_to_threshold(recs, 0.9),
            "steps_to_99pct": _steps_to_threshold(recs, 0.99),
            **_final_metrics(recs),
        }

    # Headline numbers: per env, steps-to-90% for each treatment.
    for env_tag in ("1rot", "2rot"):
        row = {}
        for role in ("baseline", "frozen", "unfrozen"):
            row[f"{role}_steps_to_90pct"] = (
                summary["runs"].get(f"{env_tag}__{role}", {}).get("steps_to_90pct")
            )
        summary["headline"][env_tag] = row

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[plot-008_4] summary -> {summary_path}")
    print(json.dumps(summary["headline"], indent=2))

    if args.no_fig:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot-008_4] matplotlib not available; skipping figures")
        return

    # ── Headline: all six eval_success_rate curves overlapped ─────────
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for key, style in STYLE_MAP.items():
        xs_ys = curves_eval.get(key)
        if not xs_ys:
            continue
        xs, ys = xs_ys
        if not xs:
            continue
        ax.plot(xs, ys, **style)
    ax.set_xlabel("env_step")
    ax.set_ylabel("eval_success_rate")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.9, color="lightgray", linewidth=0.5)
    ax.set_title("exp_008_4 — pretrained JEPA + unfrozen PPO\n"
                 "vs frozen-JEPA-PPO (008_2) vs joint baseline")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "headline.png"
    fig.savefig(fig_path, dpi=120)
    print(f"[plot-008_4] eval-success figure -> {fig_path}")

    # ── mean_feature_cosine across all six runs ───────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for key, style in STYLE_MAP.items():
        xs_ys = curves_cos.get(key)
        if not xs_ys:
            continue
        xs, ys = xs_ys
        if not xs:
            continue
        ax.plot(xs, ys, **style)
    ax.set_xlabel("env_step")
    ax.set_ylabel("mean cos(h_t, h_{t+1}) over same-episode pairs")
    ax.set_title("exp_008_4 — feature-cosine collapse diagnostic, all six runs")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "feat_cosine.png"
    fig.savefig(fig_path, dpi=120)
    print(f"[plot-008_4] feat-cosine figure -> {fig_path}")

    # ── 008_4-only collapse panel (std / pairwise L2 / effective rank) ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panel_keys = [
        ("feat_std",            "feat_std  (0 ⇒ collapse)"),
        ("feat_pairwise_l2",    "feat_pairwise_l2  (0 ⇒ collapse)"),
        ("feat_effective_rank", "feat_effective_rank  (1 ⇒ rank-1)"),
    ]
    colour = {"1rot": "#1f77b4", "2rot": "#d62728"}
    for ax, (mkey, title) in zip(axes, panel_keys):
        for env_tag in ("1rot", "2rot"):
            run_dir = runs.get((env_tag, "unfrozen"))
            if run_dir is None or not run_dir.exists():
                continue
            recs = _read_metrics(run_dir)
            xs, ys = _curve(recs, mkey)
            if xs:
                ax.plot(xs, ys, color=colour[env_tag],
                        label=f"unfrozen · {env_tag}")
        ax.set_xlabel("env_step")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("exp_008_4 — encoder collapse-risk diagnostics under unfrozen PPO")
    fig.tight_layout()
    fig_path = RESULTS_DIR / "unfrozen_collapse.png"
    fig.savefig(fig_path, dpi=120)
    print(f"[plot-008_4] unfrozen-collapse figure -> {fig_path}")


if __name__ == "__main__":
    main()
