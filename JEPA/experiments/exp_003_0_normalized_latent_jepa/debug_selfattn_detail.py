"""
Deep-dive into self-attention round 0 at step 50k.

Questions:
  1. TV distance — what is it, what does 0.081 mean here?
  2. K-vector similarity among the 4 latents in self-attn (same source as Q).
  3. Raw logit matrix per head (before softmax) — are the raw scores similar
     across different query tokens attending to the same key?
  4. Per-head attention weight matrices (after softmax) — is every head
     uniformly attending, or do individual heads specialize?
  5. Per-head V-aggregation similarity — does each head collapse independently?

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.debug_selfattn_detail
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

CKPT_50K = Path(__file__).parent / "checkpoints" / "step_050000.pt"


def avg_pairwise_cossim(vecs: torch.Tensor) -> float:
    n = vecs.shape[0]
    normed = F.normalize(vecs.float(), dim=-1)
    return float(np.mean([normed[i].dot(normed[j]).item()
                          for i in range(n) for j in range(i + 1, n)]))


def pairwise_cossim_matrix(vecs: torch.Tensor) -> torch.Tensor:
    normed = F.normalize(vecs.float(), dim=-1)
    return normed @ normed.T


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total Variation distance between two probability distributions."""
    return 0.5 * (p.float() - q.float()).abs().sum().item()


@torch.no_grad()
def main():
    cfg    = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(_repo_root / "environment_files"))
    env     = LS20Env(arc.make(cfg.game_id))
    frame_t = torch.from_numpy(env.reset()).unsqueeze(0).to(device)

    encoder, *_ = load_models(cfg, device)
    ckpt = torch.load(CKPT_50K, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    # ── Reconstruct h (the 4 latent tokens entering self-attn round 0) ────────
    # This is the output of cross-attn round 0 FFN (the 0.589 point)
    cross0 = encoder.perceiver.rounds[0].cross_attn
    sa0    = encoder.perceiver.rounds[0].self_attn
    n_heads = sa0.n_heads
    d_head  = sa0.d_head

    placeholders = encoder.perceiver.placeholders          # (4, d)
    sa_out       = encoder.encode_patches(frame_t).squeeze(0)  # (16, d)

    # Recompute cross-attn round 0 output (h = the 0.589 point)
    normed_q = cross0.norm_q(placeholders)
    Q0 = cross0.q_proj(normed_q)
    K0 = cross0.k_proj(cross0.norm_kv(sa_out))
    V0 = cross0.v_proj(sa_out)
    Q0_h = Q0.view(4, n_heads, d_head)
    K0_h = K0.view(16, n_heads, d_head)
    V0_h = V0.view(16, n_heads, d_head)
    logits0  = torch.einsum('ihd,jhd->ihj', Q0_h, K0_h) / (d_head**0.5)
    attn0    = F.softmax(logits0, dim=-1)
    agg0     = torch.einsum('ihj,jhd->ihd', attn0, V0_h).reshape(4, n_heads*d_head)
    out0     = cross0.out_proj(agg0)
    h_cross0 = cross0.ffn(placeholders + out0)     # (4, d) — the 0.589 input to self-attn

    print(f"Input to self-attn R0 (h_cross0) cos-sim: {avg_pairwise_cossim(h_cross0):.5f}")
    print(f"(should be ≈ 0.589)\n")

    # ── 1. TV DISTANCE EXPLANATION ────────────────────────────────────────────
    print("=" * 65)
    print("1. WHAT IS TV DISTANCE?")
    print("=" * 65)
    print("""
  TV(P, Q) = 0.5 × Σᵢ |Pᵢ − Qᵢ|   for probability distributions P, Q.

  Interpretation:
    TV = 0.0  → P and Q are identical
    TV = 1.0  → P and Q have completely disjoint support
                (e.g. P=[1,0,0,0], Q=[0,0,0,1])
    TV = 0.5  → P and Q share half their probability mass

  For a 4-token uniform distribution (each weight = 0.25):
    If one token's weight shifts by δ (from 0.25 to 0.25+δ),
    the remaining 3 must redistribute −δ/3 each.
    TV ≈ δ + 3×(δ/3)/2 = δ.
    So TV=0.081 corresponds to a maximum weight deviation of ~0.08
    from the uniform baseline of 0.25 — i.e. weights in [0.17, 0.33].

  The self-attn weight matrix:
    L0: [0.205  0.280  0.227  0.288]   (max deviation from 0.25 ≈ 0.047)
    L1: [0.284  0.221  0.270  0.225]   (max deviation ≈ 0.034)
    L2: [0.225  0.264  0.245  0.266]   (max deviation ≈ 0.019)
    L3: [0.297  0.215  0.268  0.219]   (max deviation ≈ 0.047)

  TV=0.081 (head-averaged) confirms: these ARE near-uniform.
  But "head-averaged" means we averaged the 4×4 matrices across heads first.
  Individual heads may look very different — see section 4.
""")

    # ── 2. K-VECTOR SIMILARITY IN SELF-ATTN ──────────────────────────────────
    print("=" * 65)
    print("2. K-VECTOR SIMILARITY AMONG 4 LATENTS IN SELF-ATTN R0")
    print("   (Q and K both come from the same 4 latent tokens h_cross0)")
    print("=" * 65)

    h_normed = sa0.norm(h_cross0)                          # (4, d)
    Q_sa = sa0.q_proj(h_normed)                            # (4, d)
    K_sa = sa0.k_proj(h_normed)                            # (4, d)
    V_sa = sa0.v_proj(h_normed)                            # (4, d)

    print(f"\n  h_cross0 (input) cos-sim:         {avg_pairwise_cossim(h_cross0):.5f}")
    print(f"  After norm (norm_q/norm_k same):  {avg_pairwise_cossim(h_normed):.5f}")
    print(f"  Q-embed (q_proj) cos-sim:         {avg_pairwise_cossim(Q_sa):.5f}   ← the 0.476")
    print(f"  K-embed (k_proj) cos-sim:         {avg_pairwise_cossim(K_sa):.5f}")
    print(f"  V-embed (v_proj) cos-sim:         {avg_pairwise_cossim(V_sa):.5f}")

    # Per-head K-sim
    K_h = K_sa.view(4, n_heads, d_head)
    Q_h = Q_sa.view(4, n_heads, d_head)
    V_h = V_sa.view(4, n_heads, d_head)
    print(f"\n  Per-head K-embed cos-sim:")
    for h in range(n_heads):
        print(f"    Head {h}: {avg_pairwise_cossim(K_h[:, h, :]):.5f}")
    print(f"\n  Per-head Q-embed cos-sim:")
    for h in range(n_heads):
        print(f"    Head {h}: {avg_pairwise_cossim(Q_h[:, h, :]):.5f}")
    print(f"\n  Per-head V-embed cos-sim:")
    for h in range(n_heads):
        print(f"    Head {h}: {avg_pairwise_cossim(V_h[:, h, :]):.5f}")

    # ── 3. RAW LOGIT MATRICES PER HEAD (BEFORE SOFTMAX) ──────────────────────
    print("\n" + "=" * 65)
    print("3. RAW ATTENTION LOGIT MATRICES PER HEAD (before softmax)")
    print("   Rows = query token, Cols = key token")
    print("   Q_i · K_j / sqrt(d_head)  for i,j ∈ {L0,L1,L2,L3}")
    print("=" * 65)

    logits_sa = torch.einsum('ihd,jhd->ihj', Q_h, K_h) / (d_head**0.5)  # (4,H,4)

    for h in range(n_heads):
        L = logits_sa[:, h, :]   # (4, 4)
        print(f"\n  Head {h} — raw logits (rows=query, cols=key):")
        header = "              L0        L1        L2        L3"
        print(f"  {header}")
        for i in range(4):
            row = "  ".join(f"{v:+.5f}" for v in L[i].tolist())
            print(f"    Query L{i}: {row}")
        logit_range = (L.max(dim=-1).values - L.min(dim=-1).values)
        print(f"  Logit range per query (max−min over 4 keys):")
        for i in range(4):
            print(f"    Query L{i}: {logit_range[i].item():.5f}")
        print(f"  Avg logit range: {logit_range.mean().item():.5f}")

        # Are the raw logits for the SAME KEY similar across different queries?
        print(f"\n  Column-wise similarity (same key, different queries):")
        print(f"  (if col j is similar across rows → all queries agree on key j's score)")
        for j in range(4):
            col = L[:, j]   # logit of key j for each of the 4 queries
            print(f"    Key L{j}: logits = [{', '.join(f'{v:+.4f}' for v in col.tolist())}]  "
                  f"range={col.max()-col.min():.4f}  std={col.std():.4f}")

    # ── 4. PER-HEAD ATTENTION WEIGHT MATRICES (AFTER SOFTMAX) ────────────────
    print("\n" + "=" * 65)
    print("4. PER-HEAD ATTENTION WEIGHT MATRICES (after softmax)")
    print("   Rows = query token, Cols = key token")
    print("=" * 65)

    attn_sa = F.softmax(logits_sa, dim=-1)   # (4, H, 4)

    for h in range(n_heads):
        W = attn_sa[:, h, :]   # (4, 4)
        print(f"\n  Head {h}:")
        header = "             L0      L1      L2      L3"
        print(f"  {header}")
        for i in range(4):
            row = "  ".join(f"{v:.5f}" for v in W[i].tolist())
            print(f"    Query L{i}: {row}")

        # TV and cos-sim between rows
        print(f"  Pairwise TV between rows (query tokens):")
        for i in range(4):
            for j in range(i+1, 4):
                tv = tv_distance(W[i], W[j])
                cs = F.cosine_similarity(W[i].unsqueeze(0), W[j].unsqueeze(0)).item()
                print(f"    TV(L{i},L{j})={tv:.5f}   cos(L{i},L{j})={cs:.5f}")

        entropy = -(W * (W + 1e-9).log()).sum(dim=-1)
        max_ent = np.log(4)
        print(f"  Row entropy (max={max_ent:.3f} = uniform over 4):")
        for i in range(4):
            print(f"    Query L{i}: H={entropy[i].item():.5f}  ({100*entropy[i].item()/max_ent:.1f}% of max)")

    # ── 5. PER-HEAD V-AGGREGATION AND OUTPUT SIMILARITY ──────────────────────
    print("\n" + "=" * 65)
    print("5. PER-HEAD V-AGGREGATION SIMILARITY")
    print("   Does each head independently collapse, or only some?")
    print("=" * 65)

    for h in range(n_heads):
        agg_h = (attn_sa[:, h, :] @ V_h[:, h, :])  # (4, d_head)
        cs = avg_pairwise_cossim(agg_h)
        print(f"  Head {h} V-aggregation cos-sim: {cs:.5f}")

    # Concatenated across heads
    agg_all = torch.einsum('ihj,jhd->ihd', attn_sa, V_h).reshape(4, n_heads*d_head)
    print(f"\n  Concatenated (all heads) V-aggregation cos-sim: {avg_pairwise_cossim(agg_all):.5f}")
    print(f"  (should match 0.977 from the full trace)")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY OF KEY NUMBERS")
    print("=" * 65)
    print(f"  Input latents cos-sim:         {avg_pairwise_cossim(h_cross0):.5f}  (diverse)")
    print(f"  Q-embed cos-sim (full):        {avg_pairwise_cossim(Q_sa):.5f}")
    print(f"  K-embed cos-sim (full):        {avg_pairwise_cossim(K_sa):.5f}  ← KEY NUMBER")
    print(f"  V-embed cos-sim (full):        {avg_pairwise_cossim(V_sa):.5f}")
    print(f"  V-aggregation cos-sim:         {avg_pairwise_cossim(agg_all):.5f}  ← COLLAPSED")


if __name__ == "__main__":
    main()
