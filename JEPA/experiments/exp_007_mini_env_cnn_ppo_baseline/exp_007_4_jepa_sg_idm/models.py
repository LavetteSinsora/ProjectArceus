"""Auxiliary heads trained alongside the encoder.

ActionConditionedPredictor : h_t, a_t        -> hat_h_{t+1}    (JEPA target)
InverseDynamicsModel       : h_t, h_{t+1}    -> logits over a  (anti-collapse)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _orth(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class ActionConditionedPredictor(nn.Module):
    def __init__(self, d_feat: int = 256, n_actions: int = 4,
                 d_action: int = 32, hidden: int = 256):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, d_action)
        nn.init.normal_(self.action_embed.weight, mean=0.0, std=0.1)
        self.fc1 = _orth(nn.Linear(d_feat + d_action, hidden), gain=2 ** 0.5)
        self.fc2 = _orth(nn.Linear(hidden, d_feat), gain=1.0)
        self.act = nn.GELU()

    def forward(self, h_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        a = self.action_embed(a_t)
        x = torch.cat([h_t, a], dim=-1)
        return self.fc2(self.act(self.fc1(x)))


class InverseDynamicsModel(nn.Module):
    """Predict a_t given (h_t, h_{t+1}). Gradients flow back into both
    endpoints, which explicitly punishes the h_t ≈ h_{t+1} attractor.
    """

    def __init__(self, d_feat: int = 256, n_actions: int = 4, hidden: int = 256):
        super().__init__()
        self.fc1 = _orth(nn.Linear(2 * d_feat, hidden), gain=2 ** 0.5)
        self.fc2 = _orth(nn.Linear(hidden, n_actions), gain=1.0)
        self.act = nn.GELU()
        self.n_actions = n_actions

    def forward(self, h_t: torch.Tensor, h_tp1: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_t, h_tp1], dim=-1)
        return self.fc2(self.act(self.fc1(x)))
