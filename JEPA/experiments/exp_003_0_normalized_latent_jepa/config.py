"""
Experiment 003 — Normalized Latent JEPA.

Key changes from exp_002:
  1. Perceiver output LayerNorm — prevents recurrent norm explosion
  2. Separate per-round Perceiver weights — eliminates gradient accumulation
  3. LatentBuffer — stores (frame_t, h_query, action, h_target) from recurrent rollout
  4. Stop gradient on h_{t+1} target — prevents collapse attractor
  5. Per-component optimizer LR — Perceiver at lower rate
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ── Model dimensions ──────────────────────────────────────────────────────
    d_model: int = 128
    d_color: int = 4
    n_actions: int = 4
    d_action: int = 32
    patch_size: int = 16

    # ── Encoder: self-attention blocks ────────────────────────────────────────
    n_sa_blocks: int = 2
    n_sa_heads: int = 4
    sa_ffn_dim: int = 512

    # ── Perceiver resampler ───────────────────────────────────────────────────
    n_latents: int = 4
    n_placeholders: int = 4
    n_perceiver_rounds: int = 2       # separate weights per round (not weight-tied)
    n_perceiver_heads: int = 4
    perceiver_ffn_dim: int = 512

    # ── 2D RoPE ───────────────────────────────────────────────────────────────
    rope_theta: float = 10000.0
    patch_grid_h: int = 4
    patch_grid_w: int = 4

    # ── Flow matching predictor ───────────────────────────────────────────────
    n_ode_steps: int = 3
    predictor_hidden: int = 512
    time_emb_dim: int = 128
    time_proj_dim: int = 512

    # ── Policy ───────────────────────────────────────────────────────────────
    policy_hidden: int = 512

    # ── Latent replay buffer ──────────────────────────────────────────────────
    buffer_size: int = 50_000
    min_buffer_size: int = 512
    batch_size: int = 64
    recency_fraction: float = 0.2
    recent_buffer_size: int = 10_000

    # ── Training schedule ─────────────────────────────────────────────────────
    update_freq: int = 5
    policy_update_freq: int = 64
    warmup_steps: int = 1_000
    max_steps: int = 500_000

    # ── Optimisation — per-component LR ──────────────────────────────────────
    sa_lr: float = 1e-4               # patch embed + SA blocks
    perceiver_lr: float = 5e-5        # Perceiver (lower: 2× gradient accumulation)
    predictor_lr: float = 1e-4
    policy_lr: float = 1e-4
    encoder_wd: float = 0.01
    predictor_wd: float = 0.01
    grad_clip_model: float = 5.0
    grad_clip_policy: float = 1.0

    # ── Policy REINFORCE ──────────────────────────────────────────────────────
    policy_entropy_lambda: float = 0.10
    policy_baseline_alpha: float = 0.99

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42

    # ── Environment ───────────────────────────────────────────────────────────
    game_id: str = "ls20-9607627b"
