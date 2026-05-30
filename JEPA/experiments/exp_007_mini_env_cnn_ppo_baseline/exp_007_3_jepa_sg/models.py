"""Action-conditioned predictor used to train the encoder via JEPA loss.

Input:  h_t (B, d_feat), a_t (B,) long
Output: predicted h_{t+1} (B, d_feat)
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
