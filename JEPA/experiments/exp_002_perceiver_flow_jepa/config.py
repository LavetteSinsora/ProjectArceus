"""
Experiment 002 — Perceiver-JEPA with Flow Matching Predictor.
Single source of truth for all hyperparameters.
All values are immutable (frozen=True).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ── Model dimensions ──────────────────────────────────────────────────────
    d_model: int = 128
    d_color: int = 4           # color embedding dim (16 colors → 4-dim vector)
    n_actions: int = 4
    d_action: int = 32         # action embedding dim in predictor
    patch_size: int = 16       # 16×16 pixel patches → 4×4 grid = 16 patches

    # ── Encoder: self-attention blocks (applied to 16 patch tokens) ───────────
    n_sa_blocks: int = 2
    n_sa_heads: int = 4
    sa_ffn_dim: int = 512      # = 4×d_model

    # ── Perceiver resampler ───────────────────────────────────────────────────
    n_latents: int = 4         # number of latent state vectors
    n_placeholders: int = 4    # placeholder queries at episode start
    n_perceiver_rounds: int = 2   # rounds through the SHARED weight-tied block
    n_perceiver_heads: int = 4    # heads for both cross-attn and self-attn
    perceiver_ffn_dim: int = 512  # = 4×d_model

    # ── 2D RoPE ───────────────────────────────────────────────────────────────
    rope_theta: float = 10000.0
    patch_grid_h: int = 4      # 4×4 grid of 16×16-px patches
    patch_grid_w: int = 4

    # ── Flow matching predictor ───────────────────────────────────────────────
    n_ode_steps: int = 3           # Euler steps at rollout
    predictor_hidden: int = 512    # = 4×d_model; 1 hidden layer per MLP
    time_emb_dim: int = 128        # sinusoidal embedding dim
    time_proj_dim: int = 512       # linear projection dim (= 4×d_model)

    # ── Policy (MLP) ──────────────────────────────────────────────────────────
    policy_hidden: int = 512   # input=4×d_model=512, hidden=512

    # ── Replay buffer ─────────────────────────────────────────────────────────
    buffer_size: int = 50_000
    min_buffer_size: int = 512
    batch_size: int = 64
    recency_fraction: float = 0.2
    recent_buffer_size: int = 10_000

    # ── Training schedule ─────────────────────────────────────────────────────
    update_freq: int = 5           # JEPA update every N env steps
    policy_update_freq: int = 64   # policy update every N env steps
    warmup_steps: int = 1_000      # encoder-only training before policy starts
    max_steps: int = 500_000       # total env steps

    # ── Optimisation ─────────────────────────────────────────────────────────
    encoder_lr: float = 1e-4
    predictor_lr: float = 1e-4
    policy_lr: float = 1e-4
    encoder_wd: float = 0.01      # AdamW weight decay for encoder
    predictor_wd: float = 0.01    # AdamW weight decay for predictor
    grad_clip_model: float = 5.0
    grad_clip_policy: float = 1.0

    # ── Policy REINFORCE ──────────────────────────────────────────────────────
    policy_entropy_lambda: float = 0.10
    policy_baseline_alpha: float = 0.99   # EMA factor for running-mean baseline

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42

    # ── Environment ───────────────────────────────────────────────────────────
    game_id: str = "ls20-9607627b"
