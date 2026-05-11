"""
Fine-grained decomposition of cross-attention round 0 at t=0 (placeholder queries).

For each checkpoint, using the fixed initial game frame, we decompose every
sub-operation inside _CrossAttentionBlock to find exactly WHERE the four latent
tokens converge to similar representations.

Sub-stages tracked:
  (0) placeholder raw vectors
  (1) Q-embeddings  = q_proj(norm_q(placeholder))
  (2) attention distributions  = softmax(Q·K^T / sqrt(d_head))  per head
  (3) weighted-V aggregation   = sum_j(attn_j * V_j)             per head, pre-concat
  (4) after out_proj           = W_out @ concat(heads)            no residual
  (5) after residual           = placeholder + out_proj_output
  (6) after FFN                = full _CrossAttentionBlock output

For (2) we measure pairwise cos-sim of the 16-dim attention distribution vector
(treating it as a vector in R^16) and also the Total-Variation distance between
the attention distributions of different latent pairs.

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.debug_crossattn_decompose
"""

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_0_normalized_latent_jepa.config import Config
from JEPA.experiments.exp_003_0_normalized_latent_jepa.models import load_models
from JEPA.shared.env_wrapper import LS20Env

CKPT_DIR = Path(__file__).parent / "checkpoints"
OUT_DIR  = Path(__file__).parent / "results" / "crossattn_decompose"


# ── helpers ───────────────────────────────────────────────────────────────────

def avg_pairwise_cossim(vecs: torch.Tensor) -> float:
    """vecs: (N, d) → mean of all unique pairwise cos-sim."""
    n = vecs.shape[0]
    normed = F.normalize(vecs.float(), dim=-1)
    return float(np.mean([normed[i].dot(normed[j]).item()
                          for i in range(n) for j in range(i+1, n)]))


def avg_pairwise_tv(dists: torch.Tensor) -> float:
    """dists: (N, K) probability distributions → mean pairwise Total Variation."""
    n = dists.shape[0]
    d = dists.float()
    vals = [0.5 * (d[i] - d[j]).abs().sum().item()
            for i in range(n) for j in range(i+1, n)]
    return float(np.mean(vals))


@torch.no_grad()
def decompose_cross_attn(encoder, frame_t: torch.Tensor) -> dict:
    """
    Decompose _CrossAttentionBlock round 0 at t=0 (placeholder queries)
    into every sub-stage and compute pairwise cos-sim between the 4 latents.
    """
    cross = encoder.perceiver.rounds[0].cross_attn
    n_heads, d_head = cross.n_heads, cross.d_head
    B = frame_t.shape[0]

    # ── Context (SA output) ───────────────────────────────────────────────────
    sa_out = encoder.encode_patches(frame_t)       # (1, 16, d)
    sa_out_sq = sa_out.squeeze(0)                  # (16, d)

    # ── Queries (placeholders) ────────────────────────────────────────────────
    placeholders = encoder.perceiver.placeholders  # (4, d)

    # Stage 0: raw placeholders
    s0_cossim = avg_pairwise_cossim(placeholders)

    # Stage 1: Q-embeddings
    normed_q = cross.norm_q(placeholders)          # (4, d)
    Q_full   = cross.q_proj(normed_q)              # (4, d)
    s1_cossim = avg_pairwise_cossim(Q_full)

    # ── K / V ─────────────────────────────────────────────────────────────────
    normed_kv = cross.norm_kv(sa_out_sq)           # (16, d)
    K_full    = cross.k_proj(normed_kv)             # (16, d)
    V_full    = cross.v_proj(sa_out_sq)             # (16, d)  (no norm on V)

    # Reshape into heads
    Q_h = Q_full.view(4,  n_heads, d_head)          # (4,  H, dh)
    K_h = K_full.view(16, n_heads, d_head)          # (16, H, dh)
    V_h = V_full.view(16, n_heads, d_head)          # (16, H, dh)

    scale = d_head ** 0.5

    # Stage 2: attention distributions (per head, then average)
    # attn_w shape: (4, H, 16)
    attn_logits = torch.einsum('ihd,jhd->ihj', Q_h, K_h) / scale  # (4, H, 16)
    attn_w = F.softmax(attn_logits, dim=-1)                         # (4, H, 16)

    # Per-latent: flatten over heads → (4, H*16) for cos-sim
    attn_flat = attn_w.reshape(4, n_heads * 16)
    s2_cossim = avg_pairwise_cossim(attn_flat)
    s2_tv     = avg_pairwise_tv(attn_flat)

    # Also compute head-averaged attention: (4, 16)
    attn_avg_heads = attn_w.mean(dim=1)   # (4, 16)
    s2_cossim_headavg = avg_pairwise_cossim(attn_avg_heads)
    s2_tv_headavg     = avg_pairwise_tv(attn_avg_heads)

    # Stage 3: weighted-V aggregation per head, before concatenation
    # agg shape: (4, H, dh)
    agg = torch.einsum('ihj,jhd->ihd', attn_w, V_h)   # (4, H, dh)
    # Flatten over heads for cos-sim: (4, H*dh) = (4, d)
    agg_flat = agg.reshape(4, n_heads * d_head)
    s3_cossim = avg_pairwise_cossim(agg_flat)

    # Stage 4: after out_proj (no residual, no FFN)
    out_proj_out = cross.out_proj(agg_flat)             # (4, d)
    s4_cossim = avg_pairwise_cossim(out_proj_out)

    # Stage 5: after residual  (placeholder + out_proj_out)
    after_residual = placeholders + out_proj_out        # (4, d)
    s5_cossim = avg_pairwise_cossim(after_residual)

    # Stage 6: after FFN (full _CrossAttentionBlock output including FFN residual)
    after_ffn = cross.ffn(after_residual)               # (4, d)
    s6_cossim = avg_pairwise_cossim(after_ffn)

    # Extra: how much does the out_proj output contribute vs the residual (placeholder)?
    out_proj_norm    = out_proj_out.norm(dim=-1).mean().item()
    placeholder_norm = placeholders.norm(dim=-1).mean().item()

    return {
        "s0_placeholder_raw":    s0_cossim,
        "s1_Q_embed":            s1_cossim,
        "s2_attn_dist_flat":     s2_cossim,
        "s2_attn_dist_tv":       s2_tv,
        "s2_attn_dist_headavg":  s2_cossim_headavg,
        "s2_tv_headavg":         s2_tv_headavg,
        "s3_V_aggregation":      s3_cossim,
        "s4_after_outproj":      s4_cossim,
        "s5_after_residual":     s5_cossim,
        "s6_after_ffn":          s6_cossim,
        "out_proj_norm":         out_proj_norm,
        "placeholder_norm":      placeholder_norm,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    cfg    = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Fixed game frame
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(_repo_root / "environment_files"))
    env     = LS20Env(arc.make(cfg.game_id))
    frame_t = torch.from_numpy(env.reset()).unsqueeze(0).to(device)
    print("Fixed game frame loaded.")

    encoder, *_ = load_models(cfg, device)
    ckpts = sorted(CKPT_DIR.glob("step_*.pt"))

    records = []
    for path in ckpts:
        m = re.match(r"step_(\d+)(?:_\w+)?\.pt$", path.name)
        if not m:
            continue
        step = int(m.group(1))
        ckpt = torch.load(path, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()

        stats = decompose_cross_attn(encoder, frame_t)
        records.append((step, stats))

        print(
            f"step={step:7d} | "
            f"s0={stats['s0_placeholder_raw']:.3f} "
            f"s1(Q)={stats['s1_Q_embed']:.3f} "
            f"s2(attn)={stats['s2_attn_dist_flat']:.3f} tv={stats['s2_attn_dist_tv']:.3f} "
            f"s3(Vagg)={stats['s3_V_aggregation']:.3f} "
            f"s4(oproj)={stats['s4_after_outproj']:.3f} "
            f"s5(res)={stats['s5_after_residual']:.3f} "
            f"s6(ffn)={stats['s6_after_ffn']:.3f} | "
            f"‖oproj‖={stats['out_proj_norm']:.3f} ‖q‖={stats['placeholder_norm']:.3f}"
        )

    steps = [r[0] for r in records]
    keys  = list(records[0][1].keys())
    series = {k: [r[1][k] for r in records] for k in keys}

    # ── Print final checkpoint detail ─────────────────────────────────────────
    last_step, last = records[-1]
    print(f"\n{'='*65}")
    print(f"Step {last_step} — sub-stage cosine similarity (t=0, placeholder queries):")
    print(f"{'='*65}")
    for k in keys:
        print(f"  {k:35s}: {last[k]:.5f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: sub-stage cos-sim across training ───────────────────────────
    cossim_keys = [k for k in keys if k.startswith("s") and "tv" not in k and "norm" not in k]
    colors = ["navy", "steelblue", "darkorange", "forestgreen",
              "firebrick", "purple", "hotpink"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for k, c in zip(cossim_keys, colors):
        label = k.replace("_", " ")
        ax.plot(steps, series[k], label=label, color=c, lw=1.6, marker=".", markersize=2)
    ax.axhline(1.0, color="gray", lw=0.8, linestyle="--")
    ax.axhline(0.0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Avg pairwise cos-sim (4 latent tokens)")
    ax.set_title("Exp-003: Where inside cross-attention round 0 do the 4 tokens collapse?\n"
                 "t=0 (placeholder queries), fixed real game frame")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.15, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "substage_cossim.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'substage_cossim.png'}")

    # ── Figure 2: attention distribution divergence (TV) ─────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax = axes[0]
    ax.plot(steps, series["s2_attn_dist_headavg"], color="darkorange", lw=1.5,
            label="Attn dist cos-sim (head-averaged, R^16)")
    ax.plot(steps, series["s2_attn_dist_flat"],    color="orange", lw=1.5, linestyle="--",
            label="Attn dist cos-sim (all heads flattened, R^64)")
    ax.axhline(1.0, color="red",  lw=0.8, linestyle=":")
    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.set_ylabel("Cos-sim between attn distributions")
    ax.set_title("Attention distribution similarity between 4 latent tokens\n"
                 "(cos-sim ≈ 1 → all latents attend to same patches)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, series["s2_tv_headavg"], color="firebrick", lw=1.5,
            label="Total Variation distance (head-avg, max=1)")
    ax.plot(steps, series["s2_attn_dist_tv"], color="salmon", lw=1.5, linestyle="--",
            label="TV distance (all heads flattened)")
    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Avg pairwise TV distance")
    ax.set_title("Attention distribution divergence between 4 latent tokens\n"
                 "(TV=0 → identical distributions; TV=1 → disjoint support)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 0.55)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "attn_distribution_divergence.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'attn_distribution_divergence.png'}")

    # ── Figure 3: norm of out_proj vs placeholder (residual vs update) ────────
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(steps, series["out_proj_norm"],    color="purple",    lw=1.5, label="‖out_proj output‖ (update signal)")
    ax.plot(steps, series["placeholder_norm"], color="steelblue", lw=1.5, label="‖placeholder‖ (residual)")
    ratio = [o/p for o, p in zip(series["out_proj_norm"], series["placeholder_norm"])]
    ax2 = ax.twinx()
    ax2.plot(steps, ratio, color="firebrick", lw=1.5, linestyle="--", label="ratio (update/residual)")
    ax2.set_ylabel("Ratio update/residual", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax.set_xlabel("Training step")
    ax.set_ylabel("L2 norm (mean over 4 latents)")
    ax.set_title("Residual vs update magnitude in cross-attention\n"
                 "If ‖out_proj‖ ≪ ‖placeholder‖, the residual dominates → tokens stay similar to input")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "residual_vs_update_norm.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'residual_vs_update_norm.png'}")

    # ── Figure 4: the critical comparison — Q vs attn vs Vagg vs outproj ─────
    fig, ax = plt.subplots(figsize=(13, 5))
    highlight = {
        "s1_Q_embed":         ("steelblue",   "Q-embed (after q_proj)"),
        "s2_attn_dist_headavg": ("darkorange", "Attn distribution (head-avg, R^16)"),
        "s3_V_aggregation":   ("forestgreen", "Weighted-V aggregation (pre-outproj)"),
        "s4_after_outproj":   ("firebrick",   "After out_proj (no residual)"),
        "s5_after_residual":  ("purple",      "After residual"),
        "s6_after_ffn":       ("hotpink",     "After FFN"),
    }
    for k, (c, lbl) in highlight.items():
        ax.plot(steps, series[k], color=c, lw=1.8, label=lbl, marker=".", markersize=2)
    ax.axhline(1.0, color="gray", lw=0.8, linestyle="--")
    ax.axhline(0.0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Avg pairwise cos-sim")
    ax.set_title("Exp-003: Collapse traced sub-operation by sub-operation\n"
                 "Cross-attn round 0, t=0 (placeholder queries), fixed game frame")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.15, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "collapse_trace.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'collapse_trace.png'}")

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
