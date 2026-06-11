"""exp_019 evaluator — Gates A and B.

Gate A: exact next-frame accuracy on held-out L4 transitions (never trained).
        Reported masked (UI rows 61-62 ignored, = what Go-Explore cell hashing
        uses) and unmasked. Baseline to beat: exp_006 full-frame U-Net = 0.0%.
Gate B: open-loop rollout of the known 100-action L4 solution inside the
        model; first divergence step (masked), i.e. how deep planning can see.

  python3 eval_wm.py runs/delta_v1 [--data data/ls20_L4_evalonly.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from model import DeltaWorldModel, FullFrameBaseline

HERE = Path(__file__).resolve().parent
MASK = np.ones((64, 64), bool)
MASK[61:63, :] = False  # ls20 UI timer rows


def load_model(run):
    ck = torch.load(HERE / run / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    m = DeltaWorldModel() if ck["arch"] == "delta" else FullFrameBaseline()
    m.load_state_dict(ck["model"])
    m.eval()
    m._mask_input = bool(ck.get("mask_input", False))
    return m, ck["step"]


def gate_a(model, data, batch=64):
    f, a, nf = data["frames"], data["actions"], data["next_frames"]
    if getattr(model, "_mask_input", False):
        f = f.copy(); f[:, 61:63, :] = 0
    n = len(f)
    exact_m = exact_u = 0
    wrong_cells = []
    changed_sizes = (f != nf).reshape(n, -1).sum(1)
    fail_sizes = []
    for i in range(0, n, batch):
        pred, _ = model.predict_batch(f[i:i + batch], a[i:i + batch])
        diff = pred != nf[i:i + batch]
        exact_u += int((~diff.reshape(len(pred), -1).any(1)).sum())
        dm = diff & MASK[None]
        ok = ~dm.reshape(len(pred), -1).any(1)
        exact_m += int(ok.sum())
        wrong_cells += list(dm.reshape(len(pred), -1).sum(1))
        fail_sizes += list(changed_sizes[i:i + batch][~ok])
    return {
        "n": n,
        "exact_masked": exact_m / n,
        "exact_unmasked": exact_u / n,
        "mean_wrong_cells_masked": float(np.mean(wrong_cells)),
        "failures_mean_changed_cells": float(np.mean(fail_sizes)) if fail_sizes else 0.0,
    }


def gate_b(model, level=3):
    """Open-loop rollout of cached L4 solution in the model vs real env."""
    sys.path.insert(0, str(HERE.parents[2]))
    sys.path.insert(0, str(HERE.parents[3] / "claude_automate"))
    sys.path.insert(0, str(HERE.parents[3]))
    from framework.env_api import make_arc_env
    from collect_data import load_solution
    sol = load_solution(level)
    env = make_arc_env("ls20-9607627b", level_index=level)
    obs = env.reset()
    model_frame = obs.copy()
    first_div = None
    per_step_exact = 0
    for t, act in enumerate(sol):
        real_next, term = env.step(act)
        pred, _ = model.predict_batch(model_frame[None], np.array([act]))
        pred = pred[0]
        # teacher-forced exactness (prediction from the *real* prev frame)
        tf_pred, _ = model.predict_batch(obs[None], np.array([act]))
        if ((tf_pred[0] == real_next) | ~MASK).all():
            per_step_exact += 1
        # open-loop: model evolves its own frame; record first mismatch
        if first_div is None and not ((pred == real_next) | ~MASK).all():
            first_div = t + 1
        model_frame = pred
        obs = real_next
        if term:
            break
    return {"solution_len": len(sol),
            "teacher_forced_exact": per_step_exact / len(sol),
            "open_loop_first_divergence": first_div if first_div is not None else -1,
            "real_steps_used": len(sol)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=str(HERE / "data/ls20_L4_evalonly.npz"))
    ap.add_argument("--gate-b", action="store_true")
    args = ap.parse_args()
    model, step = load_model(args.run)
    out = {"run": args.run, "train_step": step}
    out["gate_a_L4"] = gate_a(model, np.load(args.data))
    if args.gate_b:
        out["gate_b_L4"] = gate_b(model)
    print(json.dumps(out, indent=1))
    with open(HERE / args.run / "eval_log.jsonl", "a") as fh:
        fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
