"""
Debug: Why does cross-attention produce uniform attention?

Two hypotheses:
  H1: Patch K-vectors are all similar → Q·Kᵢ ≈ Q·Kⱼ for any Q → uniform softmax
  H2: Placeholder Q-embeddings are all similar → identical attention patterns per latent

For each checkpoint we compute:
  - avg pairwise cos-sim of the 16 patch K-vectors  (H1: are keys distinguishable?)
  - avg pairwise cos-sim of the 4 placeholder Q-vectors  (H2: are queries distinguishable?)
  - avg pairwise cos-sim of the placeholder raw vectors  (baseline: are the parameters diverse?)

All computed for cross-attn round 0 using a fixed real game frame.

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.debug_qk_collapse
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
OUT_DIR  = Path(__file__).parent / "results" / "qk_collapse_debug"


# ── Helpers ───────────────────────────────────────────────────────────────────

def avg_pairwise_cossim(vecs: torch.Tensor) -> float:
    """vecs: (N, d) → mean of all unique pairwise cosine similarities."""
    n = vecs.shape[0]
    normed = F.normalize(vecs.float(), dim=-1)
    sims = [normed[i].dot(normed[j]).item() for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(sims))


def per_head_pairwise_cossim(proj_out: torch.Tensor, n_heads: int) -> float:
    """
    proj_out: (N, d_model) — output of a linear projection, not yet split into heads.
    Reshape into (N, n_heads, d_head), compute avg pairwise cos-sim per head,
    then return the mean across heads.
    """
    N, D = proj_out.shape
    d_head = D // n_heads
    # (N, n_heads, d_head)
    per_head = proj_out.view(N, n_heads, d_head)
    head_sims = []
    for h in range(n_heads):
        head_sims.append(avg_pairwise_cossim(per_head[:, h, :]))
    return float(np.mean(head_sims))


@torch.no_grad()
def extract_qk_stats(encoder, frame_t: torch.Tensor, device: torch.device) -> dict:
    """
    Run one frame through the encoder and extract Q/K statistics
    for the cross-attention in round 0.

    Returns dict with:
      placeholder_raw_cossim:   cos-sim between raw placeholder parameters
      placeholder_normed_cossim: cos-sim between norm_q(placeholder)
      Q_embed_cossim:           cos-sim between q_proj(norm_q(placeholder))  [full d_model]
      Q_embed_perhead_cossim:   cos-sim between Q-vectors split per head
      K_patch_cossim:           cos-sim between k_proj(norm_kv(sa_out))  [full d_model]
      K_patch_perhead_cossim:   cos-sim between K-vectors split per head
      attn_logit_var:           variance of Q·Kᵀ logits (per latent, then avg)
      attn_logit_range:         max-min of Q·Kᵀ logits (per latent, then avg)
    """
    enc = encoder
    cross = enc.perceiver.rounds[0].cross_attn
    n_heads = cross.n_heads
    d_head  = cross.d_head

    # ── Patch K-vectors ───────────────────────────────────────────────────────
    sa_out = enc.encode_patches(frame_t)  # (1, 16, d_model)
    sa_out_sq = sa_out.squeeze(0)         # (16, d_model)

    normed_kv = cross.norm_kv(sa_out_sq)     # (16, d_model)
    K_full    = cross.k_proj(normed_kv)       # (16, d_model)

    k_cossim         = avg_pairwise_cossim(K_full)
    k_perhead_cossim = per_head_pairwise_cossim(K_full, n_heads)

    # ── Placeholder Q-vectors ─────────────────────────────────────────────────
    placeholders = enc.perceiver.placeholders  # (4, d_model)

    raw_cossim    = avg_pairwise_cossim(placeholders)
    normed_q      = cross.norm_q(placeholders)                 # (4, d_model)
    normed_cossim = avg_pairwise_cossim(normed_q)
    Q_full        = cross.q_proj(normed_q)                     # (4, d_model)

    q_cossim         = avg_pairwise_cossim(Q_full)
    q_perhead_cossim = per_head_pairwise_cossim(Q_full, n_heads)

    # ── Attention logit stats (per latent: Q_i · Kⱼ for j=0..15) ─────────────
    # Use per-head reshaping to match actual attention computation
    Q_heads = Q_full.view(4, n_heads, d_head)   # (4, n_heads, d_head)
    K_heads = K_full.view(16, n_heads, d_head)  # (16, n_heads, d_head)

    logit_ranges = []
    logit_vars   = []
    for i in range(4):
        for h in range(n_heads):
            q_vec = Q_heads[i, h]           # (d_head,)
            k_mat = K_heads[:, h, :]        # (16, d_head)
            logits = (k_mat @ q_vec) / (d_head ** 0.5)   # (16,)
            logit_ranges.append((logits.max() - logits.min()).item())
            logit_vars.append(logits.var().item())

    return {
        "placeholder_raw_cossim":    raw_cossim,
        "placeholder_normed_cossim": normed_cossim,
        "Q_embed_cossim":            q_cossim,
        "Q_embed_perhead_cossim":    q_perhead_cossim,
        "K_patch_cossim":            k_cossim,
        "K_patch_perhead_cossim":    k_perhead_cossim,
        "attn_logit_var":            float(np.mean(logit_vars)),
        "attn_logit_range":          float(np.mean(logit_ranges)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg    = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ── Get a fixed real game frame ───────────────────────────────────────────
    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    raw_env  = arc.make(cfg.game_id)
    env      = LS20Env(raw_env)
    frame_np = env.reset()
    frame_t  = torch.from_numpy(frame_np).unsqueeze(0).to(device)
    print(f"Using fixed game frame, shape={frame_np.shape}")

    # ── Iterate over all checkpoints ──────────────────────────────────────────
    ckpts = sorted(CKPT_DIR.glob("step_*.pt"))
    encoder, *_ = load_models(cfg, device)

    records = []  # list of (step, stats_dict)

    for path in ckpts:
        m = re.match(r"step_(\d+)(?:_\w+)?\.pt$", path.name)
        if not m:
            continue
        step = int(m.group(1))

        ckpt = torch.load(path, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()

        stats = extract_qk_stats(encoder, frame_t, device)
        records.append((step, stats))
        print(
            f"  step={step:7d} | "
            f"K_sim={stats['K_patch_cossim']:.4f} "
            f"K_sim_hd={stats['K_patch_perhead_cossim']:.4f} | "
            f"Q_sim={stats['Q_embed_cossim']:.4f} "
            f"Q_sim_hd={stats['Q_embed_perhead_cossim']:.4f} | "
            f"raw_sim={stats['placeholder_raw_cossim']:.4f} | "
            f"logit_range={stats['attn_logit_range']:.4f} "
            f"logit_var={stats['attn_logit_var']:.6f}"
        )

    steps  = [r[0] for r in records]
    keys   = list(records[0][1].keys())
    series = {k: [r[1][k] for r in records] for k in keys}

    # ── Print final checkpoint summary ────────────────────────────────────────
    last_step, last_stats = records[-1]
    print(f"\n{'='*65}")
    print(f"Most recent checkpoint (step {last_step}) summary:")
    print(f"{'='*65}")
    print(f"  Placeholder raw cos-sim (param space):        {last_stats['placeholder_raw_cossim']:.4f}")
    print(f"  Placeholder after norm_q cos-sim:             {last_stats['placeholder_normed_cossim']:.4f}")
    print(f"  Q-embedding cos-sim (full d_model):           {last_stats['Q_embed_cossim']:.4f}")
    print(f"  Q-embedding cos-sim (per-head avg):           {last_stats['Q_embed_perhead_cossim']:.4f}")
    print(f"  K-patch cos-sim (full d_model):               {last_stats['K_patch_cossim']:.4f}")
    print(f"  K-patch cos-sim (per-head avg):               {last_stats['K_patch_perhead_cossim']:.4f}")
    print(f"  Avg attn logit range (max-min across 16 pats):{last_stats['attn_logit_range']:.4f}")
    print(f"  Avg attn logit var:                           {last_stats['attn_logit_var']:.6f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: K-patch similarity and Q-embed similarity over training
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.plot(steps, series["K_patch_cossim"],        label="K-patch cos-sim (full d_model)", color="firebrick",   lw=1.5)
    ax.plot(steps, series["K_patch_perhead_cossim"], label="K-patch cos-sim (per-head avg)", color="salmon",      lw=1.5, linestyle="--")
    ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.set_ylabel("Avg pairwise cos-sim")
    ax.set_title("H1: Are patch K-vectors distinguishable?\n"
                 "(if all K similar → uniform Q·Kᵀ → uniform attention regardless of Q)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, series["placeholder_raw_cossim"],    label="Placeholder raw (param space)",   color="steelblue",   lw=1.5)
    ax.plot(steps, series["placeholder_normed_cossim"], label="Placeholder after norm_q",         color="royalblue",   lw=1.5, linestyle="--")
    ax.plot(steps, series["Q_embed_cossim"],            label="Q-embed cos-sim (full d_model)",   color="darkorange",  lw=1.5)
    ax.plot(steps, series["Q_embed_perhead_cossim"],    label="Q-embed cos-sim (per-head avg)",   color="orange",      lw=1.5, linestyle="--")
    ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Avg pairwise cos-sim")
    ax.set_title("H2: Are placeholder Q-embeddings distinguishable?\n"
                 "(if Q-embeds are similar → all latents produce same attention pattern)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Exp-003: Diagnosing uniform cross-attention\nCross-attn round 0, fixed game frame",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "qk_cossim_over_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'qk_cossim_over_training.png'}")

    # Figure 2: Attention logit range/variance — how much do logits vary across patches?
    fig, ax = plt.subplots(figsize=(11, 4))
    ax2 = ax.twinx()
    ax.plot(steps,  series["attn_logit_range"], label="Logit range (max-min)",   color="purple",  lw=1.5)
    ax2.plot(steps, series["attn_logit_var"],   label="Logit variance",           color="teal",    lw=1.5, linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Logit range (max − min over 16 patches)", color="purple")
    ax2.set_ylabel("Logit variance", color="teal")
    ax.tick_params(axis="y", labelcolor="purple")
    ax2.tick_params(axis="y", labelcolor="teal")
    ax.set_title("Attention logit spread across 16 patches (cross-attn round 0)\n"
                 "Small range → near-uniform softmax → averaging behavior")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "attn_logit_spread.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'attn_logit_spread.png'}")

    # Figure 3: Snapshot at final checkpoint — K-vector and Q-vector cos-sim matrices
    encoder_ckpt, *_ = load_models(cfg, device)
    last_ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
    encoder_ckpt.load_state_dict(last_ckpt["encoder"])
    encoder_ckpt.eval()

    with torch.no_grad():
        cross  = encoder_ckpt.perceiver.rounds[0].cross_attn
        sa_out = encoder_ckpt.encode_patches(frame_t).squeeze(0)  # (16, d)
        K_full = cross.k_proj(cross.norm_kv(sa_out))              # (16, d)
        placeholders = encoder_ckpt.perceiver.placeholders         # (4, d)
        Q_full = cross.q_proj(cross.norm_q(placeholders))         # (4, d)

    def cossim_matrix(vecs):
        normed = F.normalize(vecs.float(), dim=-1)
        return (normed @ normed.T).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    K_mat = cossim_matrix(K_full)
    im0 = axes[0].imshow(K_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[0].set_title(f"K-patch cos-sim matrix (step {last_step})\n16 patches × 16 patches, full d_model")
    axes[0].set_xlabel("Patch index")
    axes[0].set_ylabel("Patch index")
    plt.colorbar(im0, ax=axes[0], fraction=0.04)
    mean_off = K_mat[~np.eye(16, dtype=bool)].mean()
    axes[0].set_xlabel(f"Patch index\nmean off-diag cos-sim = {mean_off:.4f}")

    Q_mat = cossim_matrix(Q_full)
    im1 = axes[1].imshow(Q_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[1].set_title(f"Placeholder Q-embed cos-sim matrix (step {last_step})\n4 latents × 4 latents, full d_model")
    axes[1].set_xlabel("Latent index")
    axes[1].set_ylabel("Latent index")
    plt.colorbar(im1, ax=axes[1], fraction=0.06)
    for ii in range(4):
        for jj in range(4):
            axes[1].text(jj, ii, f"{Q_mat[ii,jj]:.2f}", ha="center", va="center", fontsize=9)
    mean_q_off = Q_mat[~np.eye(4, dtype=bool)].mean()
    axes[1].set_xlabel(f"Latent index\nmean off-diag cos-sim = {mean_q_off:.4f}")

    plt.suptitle("Final checkpoint snapshot: K-patch and Q-placeholder similarity matrices\n"
                 "(cross-attn round 0, fixed real game frame)", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "qk_cossim_matrices_final.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'qk_cossim_matrices_final.png'}")

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
