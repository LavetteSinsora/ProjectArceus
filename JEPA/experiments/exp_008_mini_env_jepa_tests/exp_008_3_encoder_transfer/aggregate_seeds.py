"""Aggregate the multi-seed exp_008_3 sweep into mean ± std error bars.

For each (env, cell), collects the latest run per seed (run dirs are tagged
``_s<seed>_``), then reports across seeds:
    - n_solved / n_seeds and mean ± std steps-to-99% (over solved seeds only)
    - mean ± std final eval_success_rate

Writes results/summary_multiseed.json and (unless --no_fig) a per-env figure
overlaying every seed's eval curve, coloured by source, solid=unfrozen /
dotted=frozen.

Usage:
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.aggregate_seeds
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.aggregate_seeds --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

from .config import PPO_RUNS_DIR, RESULTS_DIR, freeze_tag

ENVS = ("hard1", "hard2")
CELLS = [
    ("jepa", True), ("jepa", False),
    ("ppo_early", True), ("ppo_early", False),
    ("ppo_final", True), ("ppo_final", False),
    ("scratch", False),
]
SRC_COLOR = {"jepa": "#1f77b4", "ppo_early": "#2ca02c",
             "ppo_final": "#d62728", "scratch": "#444444"}
FREEZE_LS = {True: ":", False: "-"}


def _cell_name(source: str, freeze: bool) -> str:
    return f"{source}_{freeze_tag(freeze)}"


def _read_metrics(run_dir: Path) -> list[dict]:
    mp = run_dir / "metrics.jsonl"
    if not mp.exists():
        return []
    return [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]


def _latest_for_seed(env: str, source: str, freeze: bool, seed: int) -> Path | None:
    pat = f"008_3_transfer__{source}_{freeze_tag(freeze)}__{env}_s{seed}_*"
    matches = sorted(PPO_RUNS_DIR.glob(pat))
    return matches[-1] if matches else None


def _steps_to(records: list[dict], thr: float) -> int | None:
    for r in records:
        if r.get("eval_success_rate", 0.0) >= thr:
            return int(r["env_step"])
    return None


def _final_eval(records: list[dict]) -> float | None:
    fe = None
    for r in records:
        if "eval_success_rate" in r:
            fe = float(r["eval_success_rate"])
    return fe


def _curve(records: list[dict]) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in records:
        if "eval_success_rate" in r:
            xs.append(int(r["env_step"]))
            ys.append(float(r["eval_success_rate"]))
    return xs, ys


def _fmt(mean, std):
    if mean is None:
        return "        —"
    return f"{mean:>7.0f}±{std:<5.0f}" if std else f"{mean:>7.0f}      "


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--no_fig", action="store_true")
    args = p.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {"seeds": args.seeds, "cells": {}}
    curves: dict = {env: {} for env in ENVS}   # (cell)->list[(seed, xs, ys)]

    for env in ENVS:
        print(f"\n===== {env}  (seeds {args.seeds}) =====")
        print(f"{'cell':22} {'n_solved':>8} {'steps_to_99 (mean±std)':>24} {'final_eval (mean±std)':>22}")
        for source, freeze in CELLS:
            cell = _cell_name(source, freeze)
            s99s, finals, seed_curves = [], [], []
            for seed in args.seeds:
                d = _latest_for_seed(env, source, freeze, seed)
                if d is None:
                    continue
                recs = _read_metrics(d)
                s99 = _steps_to(recs, 0.99)
                if s99 is not None:
                    s99s.append(s99)
                fe = _final_eval(recs)
                if fe is not None:
                    finals.append(fe)
                xs, ys = _curve(recs)
                if xs:
                    seed_curves.append((seed, xs, ys))
            curves[env][(source, freeze)] = seed_curves
            n = len(seed_curves)
            s99_mean = statistics.mean(s99s) if s99s else None
            s99_std = statistics.pstdev(s99s) if len(s99s) > 1 else 0.0
            fe_mean = statistics.mean(finals) if finals else None
            fe_std = statistics.pstdev(finals) if len(finals) > 1 else 0.0
            print(f"{cell:22} {f'{len(s99s)}/{n}':>8} "
                  f"{_fmt(s99_mean, s99_std):>24} "
                  f"{(f'{fe_mean:.2f}±{fe_std:.2f}' if fe_mean is not None else '—'):>22}")
            summary["cells"][f"{env}__{cell}"] = {
                "n_runs": n, "n_solved": len(s99s),
                "steps_to_99_mean": s99_mean, "steps_to_99_std": s99_std,
                "steps_to_99_per_seed": s99s,
                "final_eval_mean": fe_mean, "final_eval_std": fe_std,
                "final_eval_per_seed": finals,
            }

    out = RESULTS_DIR / "summary_multiseed.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[aggregate] -> {out}")

    if args.no_fig:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aggregate] matplotlib unavailable; skipping figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, env in zip(axes, ENVS):
        seen = set()
        for source, freeze in CELLS:
            for (seed, xs, ys) in curves[env].get((source, freeze), []):
                lbl = _cell_name(source, freeze)
                ax.plot(xs, ys, color=SRC_COLOR[source], linestyle=FREEZE_LS[freeze],
                        alpha=0.5, linewidth=1.2,
                        label=lbl if lbl not in seen else None)
                seen.add(lbl)
        ax.set_xlabel("env_step")
        ax.set_title(env)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.9, color="lightgray", linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("eval_success_rate")
    fig.suptitle(f"exp_008_3 multi-seed (seeds {args.seeds}) — "
                 "each line = one seed (solid=unfrozen, dotted=frozen)")
    fig.tight_layout()
    fpath = RESULTS_DIR / "headline_multiseed.png"
    fig.savefig(fpath, dpi=120)
    print(f"[aggregate] figure -> {fpath}")


if __name__ == "__main__":
    main()
