from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # ── Model ──────────────────────────────────────────────────────────────────
    d_model: int = 128          # transformer / reasoning token dim
    d_color: int = 4            # per-pixel color embedding dim (16 colors → 4-dim)
    n_heads: int = 4            # attention heads (32-dim per head)
    n_blocks: int = 2           # transformer blocks in encoder
    ffn_dim: int = 512          # FFN hidden dim (4× d_model)
    d_action: int = 32          # action embedding dim
    patch_size: int = 16        # pixels per patch side (16×16 → 16 patches total)
    n_actions: int = 4          # ACTION1–4 for LS20 (no click actions)

    # ── EMA ────────────────────────────────────────────────────────────────────
    ema_start: float = 0.996    # initial EMA momentum (I-JEPA default)
    ema_end: float = 0.9999     # final EMA momentum (cosine schedule)

    # ── Loss ───────────────────────────────────────────────────────────────────
    change_weight_max: float = 3.0    # max patch-change loss weight (range [1, 3])
    variance_reg_lambda: float = 0.01 # collapse-prevention variance regularization

    # ── Replay buffer ──────────────────────────────────────────────────────────
    buffer_size: int = 50_000         # total capacity (uint8 frames, ~200 MB)
    min_buffer_size: int = 512        # minimum before JEPA training starts
    batch_size: int = 64
    recency_fraction: float = 0.2     # fraction of batch from recent transitions
    recent_buffer_size: int = 10_000  # size of "recent" window for oversampling

    # ── Training schedule ──────────────────────────────────────────────────────
    update_freq: int = 5              # JEPA gradient step every N env steps
    policy_update_freq: int = 64      # policy update every N env steps (on-policy batch)
    warmup_steps: int = 1_000         # random actions + JEPA-only warmup
    max_steps: int = 50_000

    # ── Optimizers ─────────────────────────────────────────────────────────────
    jepa_lr: float = 1e-4             # AdamW (reduced from 3e-4 to prevent embedding-scale drift)
    jepa_weight_decay: float = 0.01   # L2 regularisation: prevents encoder parameter growth
    policy_lr: float = 1e-4           # Adam for policy
    policy_entropy_lambda: float = 0.10  # raised from 0.02 — keeps policy exploratory long-term
    policy_attn_gain_init: float = 4.0   # per-dim gain on cross-attn output; >1 to defeat
                                          # the residual washout induced by encoder L2-norm
    grad_clip_jepa: float = 5.0
    grad_clip_policy: float = 1.0

    # ── Environment ────────────────────────────────────────────────────────────
    game_id: str = "ls20-9607627b"

    # ── Logging ────────────────────────────────────────────────────────────────
    log_freq: int = 100
    checkpoint_path: str = "jepa_checkpoint.pt"
