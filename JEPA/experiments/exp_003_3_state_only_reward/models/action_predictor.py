"""
Exp-003-3 Action Predictor.

System card §2.5. Input: (h_t, h_{t+1}) — both (B, n_latents, d_model).
Output: logits over n_actions discrete actions.

This module is the explicit anti-collapse mechanism for exp_003_3. Gradient
flows from L_action = CE(logits, a_t) back through *both* h_t and h_{t+1}
into the encoder (no detach on either side). If the encoder collapses so
that h_t ≈ h_{t+1}, the action predictor cannot beat chance and the CE
loss stays at ln(n_actions) ≈ 1.386 nats — pushing the encoder away from
the collapse attractor.

No internal masking is applied at the action logits. Masking is only used
when computing diagnostic action-pred entropy on the rollout (where the
available-actions set is known per step).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionPredictor(nn.Module):
    """(h_t, h_{t+1}) → logits over n_actions discrete actions."""

    def __init__(
        self,
        n_latents: int = 4,
        d_model: int = 128,
        hidden: int = 512,
        n_actions: int = 4,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model
        self.n_actions = n_actions
        in_dim = 2 * n_latents * d_model  # 2 * 4 * 128 = 1024
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, h_t: torch.Tensor, h_tp1: torch.Tensor) -> torch.Tensor:
        """
        h_t:   (B, n_latents, d_model)
        h_tp1: (B, n_latents, d_model)
        returns: (B, n_actions) logits
        """
        z = torch.cat([h_t.flatten(1), h_tp1.flatten(1)], dim=-1)  # (B, 2·L·D)
        return self.net(z)
