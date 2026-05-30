"""Config for exp_007_3_jepa_sg.

CNN encoder is decoupled from PPO: features are stop-gradient'd before
policy/value heads, and the encoder is trained instead by a JEPA-style
predictor loss MSE(predictor(h_t, a_t), sg(h_{t+1})).

Same reward setting as exp_007_0_naive (terminal only) so the only
variable under study is the encoder's training signal.
"""

from dataclasses import dataclass

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    exp_name: str = "exp_007_3_jepa_sg"
    reward_mode: str = "terminal_only"

    # ── JEPA-specific ───────────────────────────────────────────────
    d_action: int = 32              # action embedding dim
    predictor_hidden: int = 256     # MLP hidden width
    encoder_lr: float = 3e-4        # opt_enc: encoder + predictor
    policy_value_lr: float = 3e-4   # opt_pp:  policy + value heads
    jepa_epochs: int = 2            # mirror PPO epochs over the same minibatches
    jepa_grad_clip: float = 0.5
    # The base config's `learning_rate` field is unused in this variant.

    # PPO2-style clipped value loss. Off by default for this variant:
    # vf_clip=0.2 hurt mean success in the exp_007_0 A/B (0.51 → 0.15) under
    # the sparse terminal-only reward, so we run JEPA-sg without it.
    vf_clip_eps: float | None = None

    # Override base: checkpoint cadence (updates, not episodes).
    save_every: int = 50
