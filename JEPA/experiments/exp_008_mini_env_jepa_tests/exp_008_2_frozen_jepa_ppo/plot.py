"""Assemble the headline 2-panel comparison plot and write summary.json.

Reads metrics.jsonl from up to four PPO runs:

    1rot baseline (CNN+PPO, joint)         — exp_007_0_naive runs dir
    1rot treatment (frozen JEPA + PPO)     — our ppo_runs/, env_tag 1rot
    2rot baseline (CNN+PPO, joint)         — our ppo_runs/, env_tag 2rot
    2rot treatment (frozen JEPA + PPO)     — our ppo_runs/, env_tag 2rot

For each run, plots `eval_success_rate` vs `env_step`. Reports:
    - steps to 90% / 99% rolling success
    - final eval_success_rate
    - average solve length at end of training

Usage:
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.plot
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.plot \\
        --baseline_1rot_dir JEPA/experiments/exp_007_mini_env_cnn_ppo_baseline/runs/exp_007_0_naive_gaefix_<TS>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import EXP_DIR, JEPA_RUNS_DIR, PPO_RUNS_DIR, RESULTS_DIR


def _read_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    out = []
    for line in metrics_path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _eval_curve(records: list[dict]) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in records:
        if "eval_success_rate" in r:
            xs.append(int(r["env_step"]))
            ys.append(float(r["eval_success_rate"]))
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


def _resolve_runs(args) -> dict[str, Path | None]:
    """Map (env_tag, role) → run directory."""
    runs: dict[str, Path | None] = {}

    # 1rot baseline (exp_007_0_naive) — search the exp_007 runs/ dir if no override.
    if args.baseline_1rot_dir:
        runs[("1rot", "baseline")] = Path(args.baseline_1rot_dir)
    else:
        ext = Path("JEPA/experiments/exp_007_mini_env_cnn_ppo_baseline/runs")
        runs[("1rot", "baseline")] = _latest("exp_007_0_naive_*", ext)

    runs[("1rot", "frozen")] = _latest(
        "exp_008_2_frozen_jepa_ppo__1rot_*", PPO_RUNS_DIR
    )
    runs[("2rot", "baseline")] = _latest(
        "exp_008_2_joint_cnn_ppo__2rot_*", PPO_RUNS_DIR
    )
    runs[("2rot", "frozen")] = _latest(
        "exp_008_2_frozen_jepa_ppo__2rot_*", PPO_RUNS_DIR
    )
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_1rot_dir", type=str, default=None,
                   help="override path to the exp_007_0_naive 1rot run")
    p.add_argument("--no_fig", action="store_true",
                   help="skip matplotlib (text-only summary)")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    runs = _resolve_runs(args)
    print("[plot] resolved runs:")
    for k, v in runs.items():
        print(f"  {k}: {v}")

    summary: dict = {"runs": {}, "headline": {}}
    curves: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
    for key, run_dir in runs.items():
        env_tag, role = key
        if run_dir is None or not run_dir.exists():
            summary["runs"][f"{env_tag}__{role}"] = {"status": "missing"}
            continue
        recs = _read_metrics(run_dir)
        xs, ys = _eval_curve(recs)
        curves[key] = (xs, ys)
        summary["runs"][f"{env_tag}__{role}"] = {
            "status": "ok",
            "run_dir": str(run_dir),
            "n_eval_points": len(xs),
            "steps_to_90pct": _steps_to_threshold(recs, 0.9),
            "steps_to_99pct": _steps_to_threshold(recs, 0.99),
            **_final_metrics(recs),
        }

    # Headline: steps-to-90pct gap, per env.
    for env_tag in ("1rot", "2rot"):
        b = summary["runs"].get(f"{env_tag}__baseline", {})
        t = summary["runs"].get(f"{env_tag}__frozen", {})
        if b.get("status") == "ok" and t.get("status") == "ok":
            b_steps = b.get("steps_to_90pct")
            t_steps = t.get("steps_to_90pct")
            summary["headline"][env_tag] = {
                "baseline_steps_to_90pct": b_steps,
                "frozen_steps_to_90pct": t_steps,
                "speedup_steps": (b_steps - t_steps)
                                  if (b_steps is not None and t_steps is not None)
                                  else None,
            }

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[plot] summary -> {summary_path}")
    print(json.dumps(summary["headline"], indent=2))

    if args.no_fig:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping figure")
        return

    # ── Headline: all four eval_success_rate curves overlapped ─────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    style_map = {
        ("1rot", "baseline"): {"color": "#444",     "linestyle": "--", "label": "1rot · joint CNN+PPO (baseline)"},
        ("1rot", "frozen"):   {"color": "#1f77b4", "linestyle": "-",  "label": "1rot · frozen JEPA + PPO"},
        ("2rot", "baseline"): {"color": "#aa3333", "linestyle": "--", "label": "2rot · joint CNN+PPO (baseline)"},
        ("2rot", "frozen"):   {"color": "#d62728", "linestyle": "-",  "label": "2rot · frozen JEPA + PPO"},
    }
    for key, style in style_map.items():
        xs_ys = curves.get(key)
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
    ax.set_title("exp_008_2 — frozen-JEPA PPO vs joint baseline (eval success rate)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "headline.png"
    fig.savefig(fig_path, dpi=120)
    print(f"[plot] eval-success figure -> {fig_path}")

    # ── JEPA collapse diagnostics over epochs ──────────────────────────
    _plot_jepa_collapse(plt)


def _plot_jepa_collapse(plt) -> None:
    """Plot JEPA collapse metrics (cosine, std, pairwise L2, effective rank)
    for each available JEPA pretrain run."""
    runs_by_tag: dict[str, Path] = {}
    for env_tag in ("1rot", "2rot"):
        matches = sorted(JEPA_RUNS_DIR.glob(f"{env_tag}_*"))
        if matches:
            runs_by_tag[env_tag] = matches[-1]
    if not runs_by_tag:
        print("[plot] no JEPA runs found; skipping collapse figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    metrics = [
        ("feat_cosine_consecutive", "cos(h_t, h_{t+1})  (1 ⇒ collapse)"),
        ("feat_std",                "feat_std  (0 ⇒ collapse)"),
        ("feat_pairwise_l2",        "feat_pairwise_l2  (0 ⇒ collapse)"),
        ("feat_effective_rank",     "feat_effective_rank  (1 ⇒ rank-1)"),
    ]
    colour_for = {"1rot": "#1f77b4", "2rot": "#d62728"}
    for (key, title), ax in zip(metrics, axes.ravel()):
        for env_tag, run_dir in runs_by_tag.items():
            recs = _read_metrics(run_dir)
            xs = [r["epoch"] for r in recs if r.get("split") == "epoch_summary" and key in r]
            ys = [r[key] for r in recs if r.get("split") == "epoch_summary" and key in r]
            if xs:
                ax.plot(xs, ys, "-o", color=colour_for[env_tag],
                        label=f"JEPA · {env_tag}", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("exp_008_2 — JEPA pretraining: collapse diagnostics")
    fig.tight_layout()
    fig_path = RESULTS_DIR / "jepa_collapse.png"
    fig.savefig(fig_path, dpi=120)
    print(f"[plot] JEPA-collapse figure -> {fig_path}")


if __name__ == "__main__":
    main()
