"""Assemble exp_008_3 results.

Headline: per target env (hard1, hard2), one panel overlaying all seven
transfer curves for that env:
    jepa/ppo_early/ppo_final × {frozen, unfrozen}  +  scratch (unfrozen floor).

Secondary: collapse diagnostics (feat_std / feat_pairwise_l2 /
feat_effective_rank) vs env_step for the *unfrozen* runs only — they are the
only ones that log per-update diagnostics (the encoder moves).

summary.json records, per run: steps-to-90%/99%, final success rate, final
avg solve length.

Usage:
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.plot
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.plot --no_fig
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PPO_RUNS_DIR, RESULTS_DIR, freeze_tag

ENVS = ("hard1", "hard2")
# (source, freeze) cells present for every env, minus scratch+frozen.
CELLS = [
    ("jepa", True), ("jepa", False),
    ("ppo_early", True), ("ppo_early", False),
    ("ppo_final", True), ("ppo_final", False),
    ("scratch", False),
]

# Colour per source; linestyle per freeze treatment.
SRC_COLOR = {
    "jepa": "#1f77b4",
    "ppo_early": "#2ca02c",
    "ppo_final": "#d62728",
    "scratch": "#444444",
}
FREEZE_LS = {True: ":", False: "-"}


def _read_metrics(run_dir: Path) -> list[dict]:
    mp = run_dir / "metrics.jsonl"
    if not mp.exists():
        return []
    return [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]


def _curve(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in records:
        if key in r:
            xs.append(int(r.get("env_step", r.get("update", 0))))
            ys.append(float(r[key]))
    return xs, ys


def _steps_to_threshold(records: list[dict], thr: float) -> int | None:
    for r in records:
        if r.get("eval_success_rate", 0.0) >= thr:
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


def _latest(glob_pat: str) -> Path | None:
    matches = sorted(PPO_RUNS_DIR.glob(glob_pat))
    return matches[-1] if matches else None


def _run_label(source: str, freeze: bool) -> str:
    return f"{source}_{freeze_tag(freeze)}"


def _resolve(env: str, source: str, freeze: bool) -> Path | None:
    return _latest(f"008_3_transfer__{source}_{freeze_tag(freeze)}__{env}_*")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no_fig", action="store_true", help="text-only summary")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {"runs": {}, "headline": {}}
    # curves[env][(source, freeze)] = (xs, ys)
    curves: dict[str, dict] = {env: {} for env in ENVS}

    for env in ENVS:
        for source, freeze in CELLS:
            run_dir = _resolve(env, source, freeze)
            label = f"{env}__{_run_label(source, freeze)}"
            if run_dir is None or not run_dir.exists():
                summary["runs"][label] = {"status": "missing"}
                continue
            recs = _read_metrics(run_dir)
            curves[env][(source, freeze)] = recs
            summary["runs"][label] = {
                "status": "ok",
                "run_dir": str(run_dir),
                "steps_to_90pct": _steps_to_threshold(recs, 0.9),
                "steps_to_99pct": _steps_to_threshold(recs, 0.99),
                **_final_metrics(recs),
            }

    # Headline: steps-to-90% per cell, per env.
    for env in ENVS:
        summary["headline"][env] = {
            _run_label(s, f): summary["runs"].get(f"{env}__{_run_label(s, f)}", {}).get("steps_to_90pct")
            for s, f in CELLS
        }

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[plot-008_3] summary -> {summary_path}")
    print(json.dumps(summary["headline"], indent=2))

    if args.no_fig:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot-008_3] matplotlib not available; skipping figures")
        return

    # ── Headline: two panels (hard1, hard2), 7 eval curves each ───────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, env in zip(axes, ENVS):
        for source, freeze in CELLS:
            recs = curves[env].get((source, freeze))
            if not recs:
                continue
            xs, ys = _curve(recs, "eval_success_rate")
            if not xs:
                continue
            ax.plot(xs, ys, color=SRC_COLOR[source], linestyle=FREEZE_LS[freeze],
                    label=_run_label(source, freeze))
        ax.set_xlabel("env_step")
        ax.set_title(env)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.9, color="lightgray", linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("eval_success_rate")
    fig.suptitle("exp_008_3 — encoder transfer to a new env "
                 "(solid = unfrozen, dotted = frozen)")
    fig.tight_layout()
    fig_path = RESULTS_DIR / "headline.png"
    fig.savefig(fig_path, dpi=120)
    print(f"[plot-008_3] headline figure -> {fig_path}")

    # ── Unfrozen collapse diagnostics (3 metrics × 2 envs) ────────────────
    panel_keys = [
        ("feat_std", "feat_std  (0 ⇒ collapse)"),
        ("feat_pairwise_l2", "feat_pairwise_l2  (0 ⇒ collapse)"),
        ("feat_effective_rank", "feat_effective_rank  (1 ⇒ rank-1)"),
    ]
    fig, axes = plt.subplots(len(ENVS), 3, figsize=(14, 8), squeeze=False)
    unfrozen_cells = [(s, f) for (s, f) in CELLS if not f]
    for row, env in enumerate(ENVS):
        for col, (mkey, title) in enumerate(panel_keys):
            ax = axes[row][col]
            for source, freeze in unfrozen_cells:
                recs = curves[env].get((source, freeze))
                if not recs:
                    continue
                xs, ys = _curve(recs, mkey)
                if xs:
                    ax.plot(xs, ys, color=SRC_COLOR[source], label=source)
            ax.set_xlabel("env_step")
            if col == 0:
                ax.set_ylabel(env)
            ax.set_title(title, fontsize=9)
            ax.legend(loc="best", fontsize=7)
    fig.suptitle("exp_008_3 — encoder collapse-risk diagnostics (unfrozen runs)")
    fig.tight_layout()
    fig_path = RESULTS_DIR / "unfrozen_collapse.png"
    fig.savefig(fig_path, dpi=120)
    print(f"[plot-008_3] collapse figure -> {fig_path}")


if __name__ == "__main__":
    main()
