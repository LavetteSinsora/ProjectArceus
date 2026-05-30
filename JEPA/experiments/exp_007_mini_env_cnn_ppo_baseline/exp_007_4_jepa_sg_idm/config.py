"""Config for exp_007_4_jepa_sg_idm.

Same as exp_007_3_jepa_sg (CNN encoder stop-grad'd from PPO; trained by
JEPA predictor loss) but adds an auxiliary Inverse Dynamics Model (IDM)
loss that explicitly punishes h_t ≈ h_{t+1} collapse:

    L_encoder = L_JEPA + λ_idm · L_IDM
    L_JEPA    = MSE(predictor(h_t, a_t), sg(h_{t+1}))
    L_IDM     = CE(idm(h_t, h_{t+1}), a_t)        ← gradients into BOTH endpoints
"""

from dataclasses import dataclass

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    exp_name: str = "exp_007_4_jepa_sg_idm"
    reward_mode: str = "terminal_only"

    # ── JEPA predictor (same as exp_007_3) ──────────────────────────
    d_action: int = 32
    predictor_hidden: int = 256
    encoder_lr: float = 3e-4
    policy_value_lr: float = 3e-4
    jepa_epochs: int = 2
    jepa_grad_clip: float = 0.5

    # ── Inverse Dynamics Model ──────────────────────────────────────
    idm_hidden: int = 256
    idm_loss_weight: float = 1.0       # λ_idm in L_encoder

    # PPO value-clip OFF by default (vf_clip=0.2 hurt exp_007_0 success
    # rate in A/B: 0.51 → 0.15 under sparse terminal reward).
    vf_clip_eps: float | None = None

    # Checkpoint cadence (updates, not episodes).
    save_every: int = 50
