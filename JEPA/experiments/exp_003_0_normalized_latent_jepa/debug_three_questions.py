"""
Three focused analyses:

Q1. Why K_sim=0.78 at initialization?
    Trace: raw patches → color_embed → patch_proj → SA block 0 → SA block 1 → sa_norm
    → norm_kv → K-proj.  At each step, measure pairwise cos-sim of the 16 patch vectors.

Q2. Full step-by-step trace through the COMPLETE Perceiver at step 50k.
    Every sub-operation from placeholder → ... → final output_norm.

Q3. Print the actual attention weight matrices (per head, not head-averaged)
    for all 4 latent tokens at step 400k.  Let the numbers speak.

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.debug_three_questions
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_0_normalized_latent_jepa.config import Config
from JEPA.experiments.exp_003_0_normalized_latent_jepa.models import load_models
from JEPA.shared.env_wrapper import LS20Env

CKPT_DIR = Path(__file__).parent / "checkpoints"


def avg_pairwise_cossim(vecs: torch.Tensor) -> float:
    n = vecs.shape[0]
    normed = F.normalize(vecs.float(), dim=-1)
    return float(np.mean([normed[i].dot(normed[j]).item()
                          for i in range(n) for j in range(i + 1, n)]))


def all_pairs_cossim(vecs: torch.Tensor) -> list:
    n = vecs.shape[0]
    normed = F.normalize(vecs.float(), dim=-1)
    return [(i, j, round(normed[i].dot(normed[j]).item(), 5))
            for i in range(n) for j in range(i + 1, n)]


# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why K_sim=0.78 at init?
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def q1_trace_patch_similarity(encoder, frame_t, step_label):
    enc = encoder
    B = frame_t.shape[0]
    p = enc.patch_size
    grid_h, grid_w = enc.patch_grid_h, enc.patch_grid_w

    print(f"\n{'='*65}")
    print(f"Q1: Patch token similarity trace  [{step_label}]")
    print(f"    (16 patch tokens — avg pairwise cos-sim at each stage)")
    print(f"{'='*65}")

    # raw pixel patches → color embedding → flat
    emb = enc.color_embed(frame_t.long())                  # (1,64,64,4)
    emb = emb.view(B, grid_h, p, grid_w, p, -1)
    emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous()
    emb = emb.view(B, grid_h * grid_w, -1)                # (1,16,1024)
    sim_color = avg_pairwise_cossim(emb.squeeze(0))
    print(f"  After color_embed + flatten (1024-dim):  {sim_color:.5f}")

    # patch projection → (1,16,d)
    emb = enc.patch_proj(emb)
    sim_proj = avg_pairwise_cossim(emb.squeeze(0))
    print(f"  After patch_proj (128-dim):              {sim_proj:.5f}")

    # SA block 0
    emb = enc.sa_blocks[0](emb)
    sim_sa0 = avg_pairwise_cossim(emb.squeeze(0))
    print(f"  After SA block 0:                        {sim_sa0:.5f}")

    # SA block 1
    emb = enc.sa_blocks[1](emb)
    sim_sa1 = avg_pairwise_cossim(emb.squeeze(0))
    print(f"  After SA block 1:                        {sim_sa1:.5f}")

    # sa_norm
    sa_out = enc.sa_norm(emb)
    sim_sanorm = avg_pairwise_cossim(sa_out.squeeze(0))
    print(f"  After sa_norm (sa_out):                  {sim_sanorm:.5f}")

    # norm_kv (inside cross-attn)
    cross = enc.perceiver.rounds[0].cross_attn
    normed_kv = cross.norm_kv(sa_out.squeeze(0))
    sim_normkv = avg_pairwise_cossim(normed_kv)
    print(f"  After norm_kv (inside cross-attn):       {sim_normkv:.5f}")

    # K-projection
    K = cross.k_proj(normed_kv)
    sim_K = avg_pairwise_cossim(K)
    print(f"  After k_proj (K-vectors, 128-dim):       {sim_K:.5f}")

    # V-projection (no norm on V)
    V = cross.v_proj(sa_out.squeeze(0))
    sim_V = avg_pairwise_cossim(V)
    print(f"  After v_proj (V-vectors, 128-dim):       {sim_V:.5f}")

    # Also: SA block 0 self-attn weights at init — are they uniform?
    # Proxy: check variance of the attn logits in SA block 0
    sa0 = enc.sa_blocks[0]
    h = sa0.norm(enc.patch_proj(
        enc.color_embed(frame_t.long()).view(B, grid_h, p, grid_w, p, -1)
        .permute(0,1,3,2,4,5).contiguous().view(B,grid_h*grid_w,-1)
    ))
    Q_sa = sa0.q_proj(h).view(B, 16, sa0.n_heads, sa0.d_head).transpose(1, 2)
    K_sa = sa0.k_proj(h).view(B, 16, sa0.n_heads, sa0.d_head).transpose(1, 2)
    import math
    logits_sa = (Q_sa @ K_sa.transpose(-2, -1)) / math.sqrt(sa0.d_head)  # (1, H, 16, 16)
    logit_range_sa = (logits_sa.max(dim=-1).values - logits_sa.min(dim=-1).values).mean().item()
    print(f"\n  SA block 0 self-attn logit range (avg over tokens/heads): {logit_range_sa:.5f}")
    print(f"  (small → SA self-attn is near-uniform → all tokens avg together)")

    return sim_K


# ─────────────────────────────────────────────────────────────────────────────
# Q2: Full Perceiver trace at step 50k
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def q2_full_perceiver_trace(encoder, frame_t):
    enc = encoder
    cross0 = enc.perceiver.rounds[0].cross_attn
    self0  = enc.perceiver.rounds[0].self_attn
    cross1 = enc.perceiver.rounds[1].cross_attn
    self1  = enc.perceiver.rounds[1].self_attn

    print(f"\n{'='*65}")
    print(f"Q2: Full Perceiver trace at step 50k  (t=0, placeholder queries)")
    print(f"    (4 latent tokens — avg pairwise cos-sim at every sub-step)")
    print(f"{'='*65}")

    sa_out = enc.encode_patches(frame_t)       # (1, 16, d)
    sa_sq  = sa_out.squeeze(0)                 # (16, d)
    placeholders = enc.perceiver.placeholders  # (4, d)

    print(f"\n  --- INPUT ---")
    print(f"  Placeholder raw:                  {avg_pairwise_cossim(placeholders):.5f}")

    # ── Round 0 cross-attention ───────────────────────────────────────────────
    # Q-embed
    normed_q0 = cross0.norm_q(placeholders)
    Q0 = cross0.q_proj(normed_q0)
    print(f"\n  --- ROUND 0 CROSS-ATTN ---")
    print(f"  Q-embed (q_proj output):          {avg_pairwise_cossim(Q0):.5f}")

    # attn logits and weights
    n_heads, d_head = cross0.n_heads, cross0.d_head
    K0 = cross0.k_proj(cross0.norm_kv(sa_sq))    # (16, d)
    V0 = cross0.v_proj(sa_sq)                     # (16, d)
    Q0_h = Q0.view(4,  n_heads, d_head)
    K0_h = K0.view(16, n_heads, d_head)
    V0_h = V0.view(16, n_heads, d_head)

    logits0 = torch.einsum('ihd,jhd->ihj', Q0_h, K0_h) / (d_head**0.5)  # (4,H,16)
    attn0   = F.softmax(logits0, dim=-1)                                   # (4,H,16)

    attn0_flat = attn0.reshape(4, n_heads * 16)
    attn0_havg = attn0.mean(1)                      # (4, 16)
    print(f"  Attn dist sim (head-avg, R^16):   {avg_pairwise_cossim(attn0_havg):.5f}")
    print(f"  Attn dist sim (all heads, R^64):  {avg_pairwise_cossim(attn0_flat):.5f}")

    agg0 = torch.einsum('ihj,jhd->ihd', attn0, V0_h).reshape(4, n_heads * d_head)
    print(f"  Weighted-V aggregation:           {avg_pairwise_cossim(agg0):.5f}")

    out0 = cross0.out_proj(agg0)
    print(f"  After out_proj (no residual):     {avg_pairwise_cossim(out0):.5f}")

    after_res0 = placeholders + out0
    print(f"  After residual:                   {avg_pairwise_cossim(after_res0):.5f}")

    after_ffn0 = cross0.ffn(after_res0)
    print(f"  After FFN  [= full block output]: {avg_pairwise_cossim(after_ffn0):.5f}")
    h0 = after_ffn0  # output of full cross-attn block 0

    # ── Round 0 self-attention among latents ──────────────────────────────────
    print(f"\n  --- ROUND 0 SELF-ATTN AMONG LATENTS ---")
    normed_s0 = self0.norm(h0)
    Q_s0 = self0.q_proj(normed_s0).view(4, n_heads, d_head)
    K_s0 = self0.k_proj(normed_s0).view(4, n_heads, d_head)
    V_s0 = self0.v_proj(normed_s0).view(4, n_heads, d_head)

    print(f"  Q-embed of latents:               {avg_pairwise_cossim(Q_s0.reshape(4,-1)):.5f}")

    logits_s0 = torch.einsum('ihd,jhd->ihj', Q_s0, K_s0) / (d_head**0.5)  # (4,H,4)
    attn_s0   = F.softmax(logits_s0, dim=-1)                                 # (4,H,4)
    attn_s0_havg = attn_s0.mean(1)   # (4, 4)
    print(f"\n  Self-attn weight matrix (head-avg), rows=query, cols=key:")
    for i in range(4):
        row = "  ".join(f"{v:.4f}" for v in attn_s0_havg[i].tolist())
        print(f"    Latent {i}: [{row}]")

    # TV distance between self-attn distributions
    def tv(p, q): return 0.5 * (p - q).abs().sum().item()
    tv_s0 = np.mean([tv(attn_s0_havg[i], attn_s0_havg[j])
                     for i in range(4) for j in range(i+1,4)])
    print(f"  Attn TV distance (head-avg):      {tv_s0:.5f}")

    agg_s0 = torch.einsum('ihj,jhd->ihd', attn_s0, V_s0).reshape(4, n_heads * d_head)
    print(f"  Weighted-V aggregation:           {avg_pairwise_cossim(agg_s0):.5f}")
    out_s0 = self0.out_proj(agg_s0)
    print(f"  After out_proj (no residual):     {avg_pairwise_cossim(out_s0):.5f}")
    after_res_s0 = h0 + out_s0
    print(f"  After residual:                   {avg_pairwise_cossim(after_res_s0):.5f}")
    after_ffn_s0 = self0.ffn(after_res_s0)
    print(f"  After FFN  [= full block output]: {avg_pairwise_cossim(after_ffn_s0):.5f}")
    h1 = after_ffn_s0

    # ── Round 1 cross-attention ───────────────────────────────────────────────
    print(f"\n  --- ROUND 1 CROSS-ATTN ---")
    normed_q1 = cross1.norm_q(h1)
    Q1 = cross1.q_proj(normed_q1)
    print(f"  Q-embed of latents:               {avg_pairwise_cossim(Q1):.5f}")

    K1 = cross1.k_proj(cross1.norm_kv(sa_sq))
    V1 = cross1.v_proj(sa_sq)
    Q1_h = Q1.view(4,  n_heads, d_head)
    K1_h = K1.view(16, n_heads, d_head)
    V1_h = V1.view(16, n_heads, d_head)

    logits1 = torch.einsum('ihd,jhd->ihj', Q1_h, K1_h) / (d_head**0.5)
    attn1   = F.softmax(logits1, dim=-1)
    attn1_havg = attn1.mean(1)
    attn1_flat = attn1.reshape(4, n_heads * 16)
    print(f"  Attn dist sim (head-avg, R^16):   {avg_pairwise_cossim(attn1_havg):.5f}")
    print(f"  Attn dist sim (all heads, R^64):  {avg_pairwise_cossim(attn1_flat):.5f}")

    agg1 = torch.einsum('ihj,jhd->ihd', attn1, V1_h).reshape(4, n_heads * d_head)
    print(f"  Weighted-V aggregation:           {avg_pairwise_cossim(agg1):.5f}")
    out1 = cross1.out_proj(agg1)
    print(f"  After out_proj (no residual):     {avg_pairwise_cossim(out1):.5f}")
    after_res1 = h1 + out1
    print(f"  After residual:                   {avg_pairwise_cossim(after_res1):.5f}")
    after_ffn1 = cross1.ffn(after_res1)
    print(f"  After FFN  [= full block output]: {avg_pairwise_cossim(after_ffn1):.5f}")
    h2 = after_ffn1

    # ── Round 1 self-attention ────────────────────────────────────────────────
    print(f"\n  --- ROUND 1 SELF-ATTN AMONG LATENTS ---")
    normed_s1 = self1.norm(h2)
    Q_s1 = self1.q_proj(normed_s1).view(4, n_heads, d_head)
    K_s1 = self1.k_proj(normed_s1).view(4, n_heads, d_head)
    V_s1 = self1.v_proj(normed_s1).view(4, n_heads, d_head)

    print(f"  Q-embed of latents:               {avg_pairwise_cossim(Q_s1.reshape(4,-1)):.5f}")

    logits_s1 = torch.einsum('ihd,jhd->ihj', Q_s1, K_s1) / (d_head**0.5)
    attn_s1   = F.softmax(logits_s1, dim=-1)
    attn_s1_havg = attn_s1.mean(1)
    print(f"\n  Self-attn weight matrix (head-avg), rows=query, cols=key:")
    for i in range(4):
        row = "  ".join(f"{v:.4f}" for v in attn_s1_havg[i].tolist())
        print(f"    Latent {i}: [{row}]")

    agg_s1 = torch.einsum('ihj,jhd->ihd', attn_s1, V_s1).reshape(4, n_heads * d_head)
    print(f"  Weighted-V aggregation:           {avg_pairwise_cossim(agg_s1):.5f}")
    out_s1 = self1.out_proj(agg_s1)
    print(f"  After out_proj (no residual):     {avg_pairwise_cossim(out_s1):.5f}")
    after_res_s1 = h2 + out_s1
    print(f"  After residual:                   {avg_pairwise_cossim(after_res_s1):.5f}")
    after_ffn_s1 = self1.ffn(after_res_s1)
    print(f"  After FFN  [= full block output]: {avg_pairwise_cossim(after_ffn_s1):.5f}")
    h3 = after_ffn_s1

    # ── Output norm ───────────────────────────────────────────────────────────
    h_final = enc.perceiver.output_norm(h3)
    print(f"\n  --- OUTPUT NORM ---")
    print(f"  After output_norm (final latents):{avg_pairwise_cossim(h_final):.5f}")
    print(f"\n  All pairwise at final output:")
    for i, j, v in all_pairs_cossim(h_final):
        print(f"    cos(L{i}, L{j}) = {v}")


# ─────────────────────────────────────────────────────────────────────────────
# Q3: Actual per-head attention weight matrices at step 400k
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def q3_per_head_attention(encoder, frame_t):
    enc = encoder
    cross0 = enc.perceiver.rounds[0].cross_attn
    n_heads, d_head = cross0.n_heads, cross0.d_head

    sa_out = enc.encode_patches(frame_t).squeeze(0)   # (16, d)
    placeholders = enc.perceiver.placeholders          # (4, d)

    Q0 = cross0.q_proj(cross0.norm_q(placeholders))   # (4, d)
    K0 = cross0.k_proj(cross0.norm_kv(sa_out))        # (16, d)

    Q0_h = Q0.view(4,  n_heads, d_head)
    K0_h = K0.view(16, n_heads, d_head)

    logits = torch.einsum('ihd,jhd->ihj', Q0_h, K0_h) / (d_head**0.5)  # (4,H,16)
    attn   = F.softmax(logits, dim=-1)                                    # (4,H,16)

    print(f"\n{'='*65}")
    print(f"Q3: Per-head attention weights at step 400k (t=0, placeholder queries)")
    print(f"    Cross-attn round 0  |  {n_heads} heads  |  4 latents × 16 patches")
    print(f"{'='*65}")

    def tv(p, q): return 0.5 * (p.float() - q.float()).abs().sum().item()

    for h in range(n_heads):
        print(f"\n  ── Head {h} ──────────────────────────────────────────────")
        w = attn[:, h, :]   # (4, 16)
        # Print table: rows=latents, cols=patches
        header = "         " + "".join(f"P{p:2d} " for p in range(16))
        print(f"  {header}")
        for i in range(4):
            row = "  ".join(f"{v:.3f}" for v in w[i].tolist())
            print(f"  Lat {i}:  {row}")

        # Pairwise TV between distributions
        print(f"\n  Pairwise TV distance (head {h}):")
        for i in range(4):
            for j in range(i+1, 4):
                print(f"    TV(L{i}, L{j}) = {tv(w[i], w[j]):.5f}")

        # Pairwise cos-sim between distributions
        print(f"\n  Pairwise cos-sim of attn distributions (head {h}):")
        for i, j, v in all_pairs_cossim(w):
            print(f"    cos(L{i}, L{j}) = {v}")

        # Max-attended patch per latent
        print(f"\n  Top-2 patches per latent (head {h}):")
        for i in range(4):
            top2_idx = w[i].topk(2).indices.tolist()
            top2_val = w[i].topk(2).values.tolist()
            print(f"    Lat {i}: P{top2_idx[0]}({top2_val[0]:.3f})  P{top2_idx[1]}({top2_val[1]:.3f})")

    # Summary: head-averaged
    attn_havg = attn.mean(1)  # (4, 16)
    print(f"\n  ── Head-averaged (4 × 16) ───────────────────────────────")
    header = "         " + "".join(f"P{p:2d} " for p in range(16))
    print(f"  {header}")
    for i in range(4):
        row = "  ".join(f"{v:.3f}" for v in attn_havg[i].tolist())
        print(f"  Lat {i}:  {row}")
    print(f"\n  Pairwise TV (head-avg): {np.mean([tv(attn_havg[i], attn_havg[j]) for i in range(4) for j in range(i+1,4)]):.5f}")
    print(f"  Pairwise cos-sim (head-avg): {avg_pairwise_cossim(attn_havg):.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg    = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(_repo_root / "environment_files"))
    env     = LS20Env(arc.make(cfg.game_id))
    frame_t = torch.from_numpy(env.reset()).unsqueeze(0).to(device)

    encoder, *_ = load_models(cfg, device)

    # ── Q1: init trace (step 600) ─────────────────────────────────────────────
    ckpt_init = torch.load(CKPT_DIR / "step_000600_final.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt_init["encoder"])
    encoder.eval()
    q1_trace_patch_similarity(encoder, frame_t, "step 600 (≈ init)")

    # ── Q1: also at step 50k for comparison ──────────────────────────────────
    ckpt_50k = torch.load(CKPT_DIR / "step_050000.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt_50k["encoder"])
    encoder.eval()
    q1_trace_patch_similarity(encoder, frame_t, "step 50k")

    # ── Q2: full Perceiver trace at step 50k ─────────────────────────────────
    # encoder already loaded with step 50k
    q2_full_perceiver_trace(encoder, frame_t)

    # ── Q3: per-head attention at step 400k ──────────────────────────────────
    ckpt_400k = torch.load(CKPT_DIR / "step_400000.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt_400k["encoder"])
    encoder.eval()
    q3_per_head_attention(encoder, frame_t)


if __name__ == "__main__":
    main()
