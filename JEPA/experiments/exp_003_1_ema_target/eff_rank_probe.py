"""
Per-token effective rank probe.

For each checkpoint, populate a buffer of fresh transitions, then for the
stored h_target tensor of shape (N, 4, 128) compute:

  - eff_rank of the flattened (N, 512) matrix (theoretical max = min(N, 512))
  - eff_rank of each per-token slice (N, 128)        (max = min(N, 128))
  - top-10 singular values per slice
  - mean L2 norm of each token across samples
  - cosine similarity between token i, token j stacked across samples
    (i.e. the cossim between the row spaces of token i and token j)

Goal: tell whether the flattened eff_rank=24 we saw at 40K is "all four tokens
each occupying ~6 dimensions" or "one token doing 24 dims and the other three
near-dead", and whether the four tokens are redundant copies of one another.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_1_ema_target.config import Config
from JEPA.experiments.exp_003_1_ema_target.models import load_models_with_target
from JEPA.experiments.exp_003_1_ema_target.train import LatentBuffer, load_checkpoint
from JEPA.experiments.exp_003_1_ema_target.grad_probe import collect_buffer


def eff_rank(M: torch.Tensor) -> tuple:
    M = M.detach().float().cpu()
    s = torch.linalg.svdvals(M)
    s = s[s > 1e-12]
    p = s / s.sum()
    H = -(p * (p + 1e-30).log()).sum()
    return float(torch.exp(H).item()), s.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=[
        "step_005000.pt", "step_020000.pt", "step_040000.pt",
        "step_060000.pt", "step_080000.pt",
    ])
    ap.add_argument("--n-buffer", type=int, default=1024)
    args = ap.parse_args()

    cfg = Config()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device = {device}")

    ckpt_root = Path(__file__).parent / "checkpoints"
    out = []

    for name in args.checkpoints:
        p = ckpt_root / name
        if not p.exists():
            continue
        enc, tenc, pred, aemb, pol, _ = load_models_with_target(cfg, device)
        step = load_checkpoint(p, enc, tenc, pred, aemb, pol, device)
        enc.eval(); tenc.eval(); pred.eval()

        print(f"\n=== {name} (step {step}) ===")
        buf = collect_buffer(enc, tenc, aemb, pred, cfg, args.n_buffer, device)
        H = torch.from_numpy(buf._h_targets[:len(buf)]).float()  # (N, 4, 128)
        N = H.shape[0]

        # Flattened
        Hf = H.reshape(N, -1)
        er_flat, s_flat = eff_rank(Hf)
        print(f"flattened (N={N}, D={Hf.shape[1]}, max={min(N, Hf.shape[1])}): "
              f"eff_rank = {er_flat:.2f}")
        print(f"  top-10 singular values: "
              f"{['%.1f' % v for v in s_flat[:10]]}")

        # Per-token
        per_tok = {}
        for i in range(H.shape[1]):
            er, s = eff_rank(H[:, i, :])
            per_tok[i] = {
                "eff_rank":   er,
                "top10":      [round(v, 2) for v in s[:10]],
                "mean_norm":  float(H[:, i, :].norm(dim=-1).mean().item()),
                "norm_std":   float(H[:, i, :].norm(dim=-1).std().item()),
            }
            print(f"  token {i}: eff_rank = {er:6.2f} / {min(N, 128)}  "
                  f"mean ||h|| = {per_tok[i]['mean_norm']:.3f} ± "
                  f"{per_tok[i]['norm_std']:.3f}  "
                  f"top sv = {per_tok[i]['top10'][:3]}")

        # Pairwise row-space similarity between tokens (token_i vs token_j as
        # (N, 128) matrices) — measured as the cosine similarity between
        # vec(M_i) and vec(M_j), centered.
        print(f"  pairwise token row-stack cossim (centered):")
        rs_sim = {}
        for i in range(H.shape[1]):
            for j in range(i + 1, H.shape[1]):
                a = H[:, i, :] - H[:, i, :].mean(dim=0, keepdim=True)
                b = H[:, j, :] - H[:, j, :].mean(dim=0, keepdim=True)
                cs = float(F.cosine_similarity(
                    a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
                ).item())
                rs_sim[f"{i},{j}"] = cs
                print(f"    ({i},{j}): {cs:+.4f}")

        out.append({
            "ckpt": name,
            "step": step,
            "N": N,
            "flattened_eff_rank": er_flat,
            "flattened_top10_sv": [round(v, 2) for v in s_flat[:10]],
            "per_token": per_tok,
            "row_stack_cossim": rs_sim,
        })

    out_path = Path(__file__).parent / "results" / "eff_rank_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
