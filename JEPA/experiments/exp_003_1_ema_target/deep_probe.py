"""
Deep probe of Perceiver internals to localize the round0.cross gradient.

For each checkpoint, on a real batch of transitions:

  A. SVD of every Q/K/V/out projection in round0.cross, round0.self,
     round1.cross, round1.self.  Report effective rank of each weight matrix.

  B. Attention-pattern statistics for each cross-attention pass:
     - max attention weight per query (across batch and 4 query slots)
     - entropy of the attention distribution (lower = peakier)
     - eff rank of the (B*Lq) x 16 attention matrix

  C. Per-block residual contribution ratio:
     for each block in the perceiver, ||f(LN(x))|| / ||x||
     where x is the residual stream entering the block and f is the block's
     non-skip contribution.  A ratio near 0 = block is doing nothing.

  D. For sanity: at how many of the 16 patch positions does each cross-attn
     concentrate its mass?  Mean of (1 - entropy/log(16)) -- a "peakedness".
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_1_ema_target.config import Config
from JEPA.experiments.exp_003_1_ema_target.models import load_models_with_target
from JEPA.experiments.exp_003_1_ema_target.train import load_checkpoint
from JEPA.experiments.exp_003_1_ema_target.grad_probe import collect_buffer


def eff_rank(M: torch.Tensor) -> float:
    M = M.detach().float().cpu()
    s = torch.linalg.svdvals(M)
    s = s[s > 1e-12]
    p = s / s.sum()
    H = -(p * (p + 1e-30).log()).sum()
    return float(torch.exp(H).item())


def top_singular_ratios(M: torch.Tensor, k: int = 5) -> list:
    s = torch.linalg.svdvals(M.detach().float().cpu())
    total = float(s.sum().item())
    return [round(float(v.item()) / max(total, 1e-12), 3) for v in s[:k]]


# Manually re-implement cross-attn forward to grab attention weights and
# residual contribution at each stage.
def _cross_attn_internals(block, queries, context):
    """Replicate _CrossAttentionBlock.forward but return everything."""
    B, Lq, D = queries.shape
    Lc = context.shape[1]
    h = block.n_heads
    dh = block.d_head
    qn = block.norm_q(queries)
    kn = block.norm_kv(context)
    Q = block.q_proj(qn).view(B, Lq, h, dh).transpose(1, 2)
    K = block.k_proj(kn).view(B, Lc, h, dh).transpose(1, 2)
    V = block.v_proj(context).view(B, Lc, h, dh).transpose(1, 2)
    attn_w = F.softmax((Q @ K.transpose(-2, -1)) / block.scale, dim=-1)
    out = (attn_w @ V).transpose(1, 2).contiguous().view(B, Lq, D)
    cross_residual = block.out_proj(out)
    after_residual = queries + cross_residual
    ffn_in = block.ffn.norm(after_residual)
    ffn_out = block.ffn.net(ffn_in)
    final = after_residual + ffn_out
    return {
        "attn_w":         attn_w,           # (B, n_heads, Lq, Lc)
        "Q":              Q,
        "K":              K,
        "V":              V,
        "cross_residual": cross_residual,   # contribution of attn path
        "after_attn":     after_residual,
        "ffn_residual":   ffn_out,          # contribution of FFN path
        "final":          final,
        "queries_in":     queries,
    }


def _self_attn_internals(block, x):
    B, L, D = x.shape
    h = block.n_heads
    dh = block.d_head
    xn = block.norm(x)
    Q = block.q_proj(xn).view(B, L, h, dh).transpose(1, 2)
    K = block.k_proj(xn).view(B, L, h, dh).transpose(1, 2)
    V = block.v_proj(xn).view(B, L, h, dh).transpose(1, 2)
    attn_w = F.softmax((Q @ K.transpose(-2, -1)) / block.scale, dim=-1)
    out = (attn_w @ V).transpose(1, 2).contiguous().view(B, L, D)
    sa_residual = block.out_proj(out)
    after_residual = x + sa_residual
    ffn_in = block.ffn.norm(after_residual)
    ffn_out = block.ffn.net(ffn_in)
    final = after_residual + ffn_out
    return {
        "attn_w": attn_w, "sa_residual": sa_residual,
        "ffn_residual": ffn_out, "final": final, "x_in": x,
    }


def probe(ckpt_name: str, cfg: Config, device, n_buffer: int = 512) -> dict:
    ckpt_root = Path(__file__).parent / "checkpoints"
    p = ckpt_root / ckpt_name
    enc, tenc, pred, aemb, pol, _ = load_models_with_target(cfg, device)
    step = load_checkpoint(p, enc, tenc, pred, aemb, pol, device)
    enc.eval()

    print(f"\n=========================== {ckpt_name} (step {step}) ===========================")

    # ---- collect a real batch -------------------------------------------------
    buf = collect_buffer(enc, tenc, aemb, pred, cfg, n_buffer, device)
    # Use recurrent samples only — these are the ones with giant gradient.
    rec_idx = np.where(~buf._is_initial[:len(buf)])[0][:64]
    frames    = torch.from_numpy(buf._frames[rec_idx]).to(device)
    h_queries = torch.from_numpy(buf._h_queries[rec_idx]).to(device)

    # ---- (A) projection-matrix SVD --------------------------------------------
    proj_rows = []
    perc = enc.perceiver
    blocks = [
        ("round0.cross", perc.rounds[0].cross_attn, "cross"),
        ("round0.self",  perc.rounds[0].self_attn,  "self"),
        ("round1.cross", perc.rounds[1].cross_attn, "cross"),
        ("round1.self",  perc.rounds[1].self_attn,  "self"),
    ]
    for label, blk, _ in blocks:
        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            W = getattr(blk, proj_name).weight  # (d_model, d_model) = (128, 128)
            er = eff_rank(W)
            top5 = top_singular_ratios(W, 5)
            proj_rows.append({
                "block": label, "proj": proj_name,
                "shape": list(W.shape), "eff_rank": er,
                "max_eff_rank": min(W.shape),
                "top5_ratio": top5,
            })

    # ---- run the forward to grab attention internals --------------------------
    with torch.no_grad():
        sa_out = enc.encode_patches(frames)             # (B, 16, 128)
        # round 0 cross
        r0c = _cross_attn_internals(perc.rounds[0].cross_attn, h_queries, sa_out)
        # round 0 self
        r0s = _self_attn_internals(perc.rounds[0].self_attn, r0c["final"])
        # round 1 cross
        r1c = _cross_attn_internals(perc.rounds[1].cross_attn, r0s["final"], sa_out)
        # round 1 self
        r1s = _self_attn_internals(perc.rounds[1].self_attn,  r1c["final"])

    # ---- (B) attention statistics for the two cross-attention blocks ---------
    def attn_stats(attn_w, n_context):
        # attn_w: (B, H, Lq, Lc)
        # entropy across the context dim
        # peakedness = 1 - H_actual / log(Lc)  -> in [0, 1]; 1 = one-hot, 0 = uniform
        eps = 1e-30
        H = -(attn_w * (attn_w + eps).log()).sum(dim=-1)         # (B, H, Lq)
        peakedness = 1.0 - H / math.log(n_context)
        max_w = attn_w.amax(dim=-1)                                # (B, H, Lq)
        # which patch position each (sample, head, query) attends to most:
        argmax_pos = attn_w.argmax(dim=-1)                         # (B, H, Lq)
        # the average attention vector per (head, query) across batch,
        # i.e. how concentrated is the *batch-averaged* attention pattern?
        batch_avg_attn = attn_w.mean(dim=0)                        # (H, Lq, Lc)
        batch_avg_H    = -(batch_avg_attn * (batch_avg_attn + eps).log()).sum(dim=-1)
        # eff rank of the (B*H*Lq, Lc) flat attention matrix
        flat = attn_w.reshape(-1, n_context).cpu()
        er = eff_rank(flat)
        return {
            "mean_peakedness":     float(peakedness.mean().item()),
            "max_peakedness":      float(peakedness.max().item()),
            "mean_max_weight":     float(max_w.mean().item()),
            "argmax_consistency":  argmax_consistency(argmax_pos),
            "batch_avg_entropy":   float(batch_avg_H.mean().item()),
            "attn_matrix_eff_rank": er,
        }

    def argmax_consistency(argmax_pos):
        # For each (head, query slot), what fraction of batch samples attend
        # to the same patch position? 1.0 = all samples target the same patch.
        # argmax_pos: (B, H, Lq)
        B, H, Lq = argmax_pos.shape
        consistencies = []
        for h_i in range(H):
            for q_i in range(Lq):
                vals = argmax_pos[:, h_i, q_i].cpu().numpy()
                _, counts = np.unique(vals, return_counts=True)
                consistencies.append(float(counts.max() / B))
        return float(np.mean(consistencies))

    r0c_stats = attn_stats(r0c["attn_w"], 16)
    r1c_stats = attn_stats(r1c["attn_w"], 16)
    r0s_stats = attn_stats(r0s["attn_w"], 4)
    r1s_stats = attn_stats(r1s["attn_w"], 4)

    # ---- (C) residual contribution ratio --------------------------------------
    def ratio(residual, x_in):
        return float((residual.norm(dim=-1).mean()
                      / (x_in.norm(dim=-1).mean() + 1e-12)).item())
    residual_rows = [
        ("round0.cross attn",  ratio(r0c["cross_residual"], r0c["queries_in"])),
        ("round0.cross ffn",   ratio(r0c["ffn_residual"],   r0c["after_attn"])),
        ("round0.self  attn",  ratio(r0s["sa_residual"],    r0s["x_in"])),
        ("round0.self  ffn",   ratio(r0s["ffn_residual"],   r0s["x_in"] + r0s["sa_residual"])),
        ("round1.cross attn",  ratio(r1c["cross_residual"], r1c["queries_in"])),
        ("round1.cross ffn",   ratio(r1c["ffn_residual"],   r1c["after_attn"])),
        ("round1.self  attn",  ratio(r1s["sa_residual"],    r1s["x_in"])),
        ("round1.self  ffn",   ratio(r1s["ffn_residual"],   r1s["x_in"] + r1s["sa_residual"])),
    ]

    # ---- (D) input-side stats: how collapsed are the queries entering each cross? ----
    q_in_r0c = h_queries                  # input to round0.cross  -- 4 latents
    q_in_r1c = r0s["final"]               # input to round1.cross  -- 4 latents

    def query_collapse(q):
        # q: (B, 4, 128)
        # 1) within-state pairwise cossim
        within = []
        for i in range(q.shape[1]):
            for j in range(i+1, q.shape[1]):
                within.append(F.cosine_similarity(q[:, i, :], q[:, j, :], dim=-1).mean().item())
        # 2) cross-sample variance of each token
        same_idx_cs = []
        for i in range(q.shape[1]):
            v = q[:, i, :]
            v_n = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
            cs = v_n @ v_n.T
            mask = ~torch.eye(cs.shape[0], dtype=torch.bool, device=cs.device)
            same_idx_cs.append(float(cs[mask].mean().item()))
        return {
            "within_state_cossim":    float(np.mean(within)),
            "cross_state_same_idx_cs": float(np.mean(same_idx_cs)),
            "norm_mean":               float(q.norm(dim=-1).mean().item()),
            "norm_std":                float(q.norm(dim=-1).std().item()),
        }

    q_in_r0c_stats = query_collapse(q_in_r0c)
    q_in_r1c_stats = query_collapse(q_in_r1c)

    # ---- pretty print --------------------------------------------------------
    print(f"\n[A] PROJECTION WEIGHT EFFECTIVE RANK (max = 128 per matrix)")
    print(f"  {'block':14s} {'proj':9s} {'eff_rank':>9s}  top-5 σ ratios")
    for r in proj_rows:
        print(f"  {r['block']:14s} {r['proj']:9s} {r['eff_rank']:9.2f}  {r['top5_ratio']}")

    print(f"\n[B] CROSS-ATTENTION PATTERN STATISTICS")
    print(f"  metric                       round0.cross   round1.cross   round0.self   round1.self")
    for k in ("mean_peakedness", "mean_max_weight", "argmax_consistency",
              "attn_matrix_eff_rank"):
        v0c = r0c_stats[k]; v1c = r1c_stats[k]; v0s = r0s_stats[k]; v1s = r1s_stats[k]
        print(f"  {k:28s} {v0c:13.4f}  {v1c:13.4f}  {v0s:12.4f}  {v1s:12.4f}")
    print(f"  (peakedness=1 → one-hot attention; 0 → uniform)")
    print(f"  (argmax_consistency: fraction of batch samples that attend to the "
          f"same patch — 1.0 means batch-invariant attention)")

    print(f"\n[C] PER-BLOCK RESIDUAL CONTRIBUTION  ||f(LN(x))|| / ||x||")
    for name, val in residual_rows:
        print(f"  {name:25s}  {val:.4f}")
    print(f"  (low value → block adds almost nothing to its residual stream)")

    print(f"\n[D] QUERIES ENTERING EACH CROSS-ATTN")
    for name, st in [("round0.cross input", q_in_r0c_stats),
                     ("round1.cross input", q_in_r1c_stats)]:
        print(f"  {name}:")
        print(f"    within-state pairwise cossim (4 tokens): {st['within_state_cossim']:+.4f}")
        print(f"    cross-state same-token cossim:           {st['cross_state_same_idx_cs']:+.4f}")
        print(f"    ||q|| mean={st['norm_mean']:.3f}   std={st['norm_std']:.5f}")

    return {
        "step": step, "ckpt": ckpt_name,
        "proj_eff_ranks": proj_rows,
        "attn_stats": {
            "round0.cross": r0c_stats, "round1.cross": r1c_stats,
            "round0.self":  r0s_stats, "round1.self":  r1s_stats,
        },
        "residual_contribution_ratio": residual_rows,
        "queries_into_cross":  {
            "round0": q_in_r0c_stats, "round1": q_in_r1c_stats,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+",
                    default=["step_040000.pt", "step_080000.pt"])
    ap.add_argument("--n-buffer", type=int, default=512)
    ap.add_argument("--out", default="results/deep_probe.json")
    args = ap.parse_args()

    cfg = Config()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device = {device}")

    results = []
    for name in args.checkpoints:
        results.append(probe(name, cfg, device, n_buffer=args.n_buffer))

    out_path = Path(__file__).parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
