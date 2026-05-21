"""Dreamer V3 XS configuration — single fixed hyperparameter set per the paper.

Sub-experiments inherit and override only what they need (typically just the
reward-shaping mode and a couple of flags).  Frozen dataclass to catch typos.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigBase:
    # ── Environment ──────────────────────────────────────────────────────────
    game_id: str = "ls20-9607627b"
    n_actions: int = 4
    obs_channels: int = 1
    obs_size: int = 64
    stop_levels: int = 1

    # ── World-model architecture (nano — sized for Apple M3 Pro MPS) ─────────
    # Paper XS reference for comparison:
    #   deter=512, n_groups=32, n_classes=32, hidden_units=512, cnn_depth=32,
    #   embed_dim=1024, twohot_bins=255.
    # Nano roughly quarters per-step compute by halving every width and
    # halving the categorical resolution.
    deter: int = 256
    n_groups: int = 16
    n_classes: int = 16
    hidden_units: int = 256
    cnn_depth: int = 24
    embed_dim: int = 512
    unimix: float = 0.01

    # ── Twohot reward / value head ───────────────────────────────────────────
    twohot_bins: int = 127
    twohot_low: float = -20.0
    twohot_high: float = 20.0

    # ── Loss coefficients ────────────────────────────────────────────────────
    beta_pred: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    free_nats: float = 1.0

    # ── Actor / critic / imagination ─────────────────────────────────────────
    imag_horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    entropy_eta: float = 3e-4
    critic_ema_decay: float = 0.98
    return_scale_decay: float = 0.99

    # ── Plan2Explore ─────────────────────────────────────────────────────────
    p2e_n_heads: int = 8
    p2e_hidden_units: int = 128
    p2e_ensemble_lr: float = 1e-4
    use_p2e_ensemble: bool = True            # train the ensemble (cheap, ~always on)
    use_p2e_actor: bool = True               # train a separate exploration actor π_e
    p2e_intrinsic_weight: float = 0.0        # added to extrinsic reward in actor training
    p2e_acting_steps: int = 100_000          # use π_e for the first N env steps, then π_t

    # ── Optimisation ─────────────────────────────────────────────────────────
    wm_lr: float = 1e-4
    wm_adam_eps: float = 1e-8
    wm_grad_clip: float = 1000.0
    actor_lr: float = 3e-5
    actor_adam_eps: float = 1e-5
    actor_grad_clip: float = 100.0
    critic_lr: float = 3e-5
    critic_adam_eps: float = 1e-5
    critic_grad_clip: float = 100.0

    # ── Training loop ────────────────────────────────────────────────────────
    batch_size: int = 16
    batch_length: int = 32                   # paper uses 64; halved for MPS budget
    replay_capacity: int = 250_000
    train_ratio: int = 32                    # transitions trained per env step (paper: 512). MPS-tuned.
    prefill_steps: int = 5_000               # random-policy prefill
    max_env_steps: int = 500_000
    log_every: int = 500                     # gradient steps
    ckpt_every: int = 5_000                  # gradient steps
    eval_every: int = 25_000                 # env steps
    eval_episodes: int = 30
    seed: int = 0

    # ── Reward shaping (sub-exps override) ───────────────────────────────────
    reward_mode: str = "sparse"              # 'sparse' | 'step_penalty' | 'curiosity' | 'p2e_only'
    step_penalty: float = 0.0                # only used when reward_mode == 'step_penalty'

    # ── Misc ─────────────────────────────────────────────────────────────────
    # "auto" selects cuda > mps > cpu at runtime.  Override to force a device.
    device: str = "auto"
    run_name: str = "run"
