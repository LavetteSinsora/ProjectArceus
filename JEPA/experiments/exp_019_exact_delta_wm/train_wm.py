"""exp_019 trainer — chunked (45 s sandbox calls), checkpoint-resume.

Each invocation: load ckpt if present, train for --seconds wall-clock, save
ckpt + append metrics to train_log.jsonl, exit. Idempotent and resumable.

  python3 train_wm.py --arch delta --run runs/delta_main --seconds 33
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import DeltaWorldModel, FullFrameBaseline

HERE = Path(__file__).resolve().parent


def load_train(files):
    fs, as_, ns, ts = [], [], [], []
    for f in files:
        d = np.load(f)
        fs.append(d["frames"]); as_.append(d["actions"])
        ns.append(d["next_frames"]); ts.append(d["terminals"])
    return (np.concatenate(fs), np.concatenate(as_),
            np.concatenate(ns), np.concatenate(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["delta", "full"], default="delta")
    ap.add_argument("--run", default="runs/delta_main")
    ap.add_argument("--seconds", type=float, default=33.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=30.0)
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of train data (scaling ablation)")
    ap.add_argument("--mask-input", action="store_true",
                    help="zero UI rows 61-62 in both frame and target: the "
                         "canonical model state is the UI-masked frame")
    args = ap.parse_args()

    run = HERE / args.run
    run.mkdir(parents=True, exist_ok=True)
    ckpt_path = run / "ckpt.pt"
    log_path = run / "train_log.jsonl"

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)

    files = args.data or [HERE / "data/ls20_L1.npz", HERE / "data/ls20_L2.npz",
                          HERE / "data/ls20_L3.npz"]
    F_, A_, N_, T_ = load_train(files)
    if args.mask_input:
        F_ = F_.copy(); N_ = N_.copy()
        F_[:, 61:63, :] = 0; N_[:, 61:63, :] = 0
    if args.frac < 1.0:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(F_))[: int(len(F_) * args.frac)]
        F_, A_, N_, T_ = F_[idx], A_[idx], N_[idx], T_[idx]
    n = len(F_)

    model = DeltaWorldModel() if args.arch == "delta" else FullFrameBaseline()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]
        for g in opt.param_groups:   # ckpt restores old lr; CLI wins
            g["lr"] = args.lr
        torch.manual_seed(step)  # vary batch order across chunks

    pos_w = torch.tensor(args.pos_weight)
    # NOTE: move-biased sampling (75% moved) was tried at step 1021->1307 and
    # REGRESSED exact-masked 64.2%->46.6% (no-op gate calibration broke).
    # Reverted to uniform sampling (RESEARCH_LOG 2026-06-10).
    t0 = time.time()
    losses = []
    while time.time() - t0 < args.seconds and step < args.max_steps:
        idx = torch.randint(0, n, (args.batch,))
        f = torch.from_numpy(F_[idx.numpy()]).long()
        nf = torch.from_numpy(N_[idx.numpy()]).long()
        a = torch.from_numpy(A_[idx.numpy()]).long()
        t = torch.from_numpy(T_[idx.numpy()].astype(np.float32))
        oh = F.one_hot(f, 16).permute(0, 3, 1, 2).float()
        changed = (f != nf)

        if args.arch == "delta":
            chg_logit, col_logit, term_logit, noop_logit = model(oh, a)
            # per-cell change loss only on transitions where something moves
            # (the global gate owns the no-op case)
            moved = changed.flatten(1).any(1)
            if moved.any():
                loss_chg = F.binary_cross_entropy_with_logits(
                    chg_logit[moved], changed[moved].float(), pos_weight=pos_w)
                loss_col = F.cross_entropy(
                    col_logit.permute(0, 2, 3, 1)[changed], nf[changed])
            else:
                loss_chg = loss_col = torch.zeros(())
            loss_noop = F.binary_cross_entropy_with_logits(
                noop_logit, (~moved).float())
            loss_term = F.binary_cross_entropy_with_logits(term_logit, t)
            loss = loss_chg + loss_col + loss_noop + 0.1 * loss_term
        else:
            logits, term_logit = model(oh, a)
            # change-weighted CE so the comparison is fair (same emphasis on
            # moving content as the delta model gets)
            ce = F.cross_entropy(logits, nf, reduction="none")
            w = 1.0 + (args.pos_weight - 1.0) * changed.float()
            loss_frame = (ce * w).sum() / w.sum()
            loss_term = F.binary_cross_entropy_with_logits(term_logit, t)
            loss = loss_frame + 0.1 * loss_term

        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss)); step += 1

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "arch": args.arch,
                "mask_input": args.mask_input}, ckpt_path)
    rec = {"step": step, "n_train": n, "chunk_batches": len(losses),
           "loss_mean": float(np.mean(losses)) if losses else None,
           "secs": round(time.time() - t0, 1)}
    with open(log_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
