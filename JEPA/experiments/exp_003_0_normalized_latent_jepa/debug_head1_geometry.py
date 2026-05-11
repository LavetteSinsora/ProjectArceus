"""
Why does Head 1 produce near-uniform attention despite diverse Q and K vectors?

For Head 1 specifically, we decompose:
  Q_i = Q_mean + ΔQ_i       (shared part + unique deviation)
  K_j = K_mean + ΔK_j

The dot product expands as:
  Q_i · K_j = Q_mean·K_mean + Q_mean·ΔK_j + ΔQ_i·K_mean + ΔQ_i·ΔK_j
               ─────────────  ─────────────  ─────────────  ──────────
                 constant        varies by j    varies by i   interaction

If ΔQ_i · ΔK_j ≈ 0 and |Q_mean·ΔK_j| is small,
then Q_i · K_j ≈ constant for all (i, j) → uniform softmax regardless of Q.

We compare Head 1 (collapses) against Head 0 (does not collapse).

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.debug_head1_geometry
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

CKPT = Path(__file__).parent / "checkpoints" / "step_050000.pt"


def analyse_head(Q_h: torch.Tensor, K_h: torch.Tensor, head_idx: int, scale: float):
    """
    Q_h: (4, d_head)   K_h: (4, d_head)   — one head's vectors for 4 latent tokens
    """
    n = Q_h.shape[0]

    print(f"\n{'─'*60}")
    print(f"  HEAD {head_idx}")
    print(f"{'─'*60}")

    # ── Norms ─────────────────────────────────────────────────────────────────
    print(f"\n  Vector norms:")
    for i in range(n):
        print(f"    Q_{i}: ‖{Q_h[i].norm().item():.4f}‖   K_{i}: ‖{K_h[i].norm().item():.4f}‖")

    # ── Shared vs differential decomposition ─────────────────────────────────
    Q_mean = Q_h.mean(dim=0)      # (d_head,)
    K_mean = K_h.mean(dim=0)

    dQ = Q_h - Q_mean             # (4, d_head) — zero-mean deviations
    dK = K_h - K_mean

    print(f"\n  ‖Q_mean‖ = {Q_mean.norm().item():.4f}")
    print(f"  ‖K_mean‖ = {K_mean.norm().item():.4f}")
    print(f"  ‖ΔQ_i‖ per token:  " + "  ".join(f"Δ{i}={dQ[i].norm().item():.4f}" for i in range(n)))
    print(f"  ‖ΔK_j‖ per token:  " + "  ".join(f"Δ{j}={dK[j].norm().item():.4f}" for j in range(n)))

    ratio_q = dQ.norm(dim=-1).mean().item() / (Q_mean.norm().item() + 1e-8)
    ratio_k = dK.norm(dim=-1).mean().item() / (K_mean.norm().item() + 1e-8)
    print(f"\n  ‖ΔQ‖/‖Q_mean‖ (avg) = {ratio_q:.4f}  ← how much of Q is unique vs shared")
    print(f"  ‖ΔK‖/‖K_mean‖ (avg) = {ratio_k:.4f}  ← how much of K is unique vs shared")

    # ── Four-term expansion of Q_i · K_j ──────────────────────────────────────
    print(f"\n  Full logit matrix Q_i·K_j / scale  (raw):")
    full_logits = (Q_h @ K_h.T) / scale
    for i in range(n):
        print(f"    Q_{i}: " + "  ".join(f"{v:+.5f}" for v in full_logits[i].tolist()))

    # Term 1: Q_mean · K_mean — scalar, same for all (i,j)
    t1 = (Q_mean @ K_mean).item() / scale
    print(f"\n  ── Decomposition of Q_i·K_j/scale ──")
    print(f"  Term 1: Q_mean·K_mean/scale = {t1:+.5f}  (constant — same for every i,j)")

    # Term 2: Q_mean · ΔK_j — varies only by j (same for all queries)
    t2 = (Q_mean @ dK.T).float() / scale  # (4,)
    print(f"  Term 2: Q_mean·ΔK_j/scale   = " + "  ".join(f"j{j}:{t2[j].item():+.5f}" for j in range(n)))
    print(f"           (varies by key j, but SAME across all query tokens i)")
    print(f"           range = {(t2.max()-t2.min()).item():.5f}")

    # Term 3: ΔQ_i · K_mean — varies only by i (same logit offset per query)
    t3 = (dQ @ K_mean).float() / scale  # (4,)
    print(f"  Term 3: ΔQ_i·K_mean/scale   = " + "  ".join(f"i{i}:{t3[i].item():+.5f}" for i in range(n)))
    print(f"           (varies by query i, but SAME offset for all keys j)")
    print(f"           → shifts ALL of query i's logits by the same amount → no effect on softmax")

    # Term 4: ΔQ_i · ΔK_j — the only term that differs BOTH by i and by j
    t4 = (dQ @ dK.T).float() / scale    # (4, 4)
    print(f"  Term 4: ΔQ_i·ΔK_j/scale    (the interaction term — only this term")
    print(f"           can make different query tokens attend to different keys):")
    for i in range(n):
        print(f"    i={i}: " + "  ".join(f"j{j}:{t4[i,j].item():+.5f}" for j in range(n)))
    print(f"  Term 4 range within each row (= spread added to softmax input by unique Q):")
    for i in range(n):
        print(f"    Row {i}: range={t4[i].max()-t4[i].min():.5f}  std={t4[i].std():.5f}")
    t4_mean_range = float(np.mean([(t4[i].max()-t4[i].min()).item() for i in range(n)]))
    print(f"  Term 4 avg row range = {t4_mean_range:.5f}")

    # Summary: what fraction of total logit range comes from term 4 vs term 2?
    t2_range = (t2.max() - t2.min()).item()
    print(f"\n  Term 2 range (Q_mean·ΔK, same for all queries) = {t2_range:.5f}")
    print(f"  Term 4 avg range (ΔQ·ΔK, unique per query)      = {t4_mean_range:.5f}")
    if t2_range > 1e-6:
        print(f"  Ratio T4/T2 = {t4_mean_range/t2_range:.4f}  ← <1 means shared signal dominates")

    # ── Angle between ΔQ and ΔK subspaces ────────────────────────────────────
    print(f"\n  Pairwise cos-sim between ΔQ_i and ΔK_j (direction alignment):")
    for i in range(n):
        for j in range(n):
            nq = F.normalize(dQ[i].float().unsqueeze(0), dim=-1)
            nk = F.normalize(dK[j].float().unsqueeze(0), dim=-1)
            cs = (nq @ nk.T).item()
            print(f"    cos(ΔQ_{i}, ΔK_{j}) = {cs:+.4f}")

    # ── Per-query logit range (how peaked is the distribution?) ───────────────
    print(f"\n  Per-query logit range (max−min over all 4 keys):")
    for i in range(n):
        lr = (full_logits[i].max() - full_logits[i].min()).item()
        print(f"    Query {i}: {lr:.5f}")

    # ── Softmax entropy ───────────────────────────────────────────────────────
    attn = F.softmax(full_logits, dim=-1)
    print(f"\n  Softmax attention (rows=query):")
    for i in range(n):
        row = "  ".join(f"{v:.4f}" for v in attn[i].tolist())
        ent = -(attn[i] * (attn[i]+1e-9).log()).sum().item()
        print(f"    Q_{i}: [{row}]   H={ent:.4f} ({100*ent/np.log(n):.1f}% of max)")


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
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    # Reconstruct h entering self-attn round 0 (the 0.589 cos-sim point)
    cross0 = encoder.perceiver.rounds[0].cross_attn
    sa0    = encoder.perceiver.rounds[0].self_attn
    n_heads, d_head = sa0.n_heads, sa0.d_head
    scale = d_head ** 0.5

    placeholders = encoder.perceiver.placeholders
    sa_out       = encoder.encode_patches(frame_t).squeeze(0)

    Q0 = cross0.q_proj(cross0.norm_q(placeholders))
    K0 = cross0.k_proj(cross0.norm_kv(sa_out))
    V0 = cross0.v_proj(sa_out)
    Q0_h = Q0.view(4, n_heads, d_head)
    K0_h = K0.view(16, n_heads, d_head)
    V0_h = V0.view(16, n_heads, d_head)
    logits0 = torch.einsum('ihd,jhd->ihj', Q0_h, K0_h) / scale
    attn0   = F.softmax(logits0, dim=-1)
    agg0    = torch.einsum('ihj,jhd->ihd', attn0, V0_h).reshape(4, n_heads*d_head)
    out0    = cross0.out_proj(agg0)
    h       = cross0.ffn(placeholders + out0)   # (4, d) — input to self-attn R0

    h_normed = sa0.norm(h)
    Q_sa = sa0.q_proj(h_normed).view(4, n_heads, d_head)  # (4, H, dh)
    K_sa = sa0.k_proj(h_normed).view(4, n_heads, d_head)

    print("=" * 60)
    print("Geometry of self-attn R0 at step 50k: why does Head 1")
    print("produce uniform attention despite diverse Q and K vectors?")
    print("=" * 60)
    print(f"\nInput latent cos-sim: {float(np.mean([F.cosine_similarity(h[i].unsqueeze(0), h[j].unsqueeze(0)).item() for i in range(4) for j in range(i+1,4)])):.5f}")

    for h_idx in [0, 1]:   # head 0 = informative, head 1 = collapses
        analyse_head(Q_sa[:, h_idx, :], K_sa[:, h_idx, :], h_idx, scale)


if __name__ == "__main__":
    main()
