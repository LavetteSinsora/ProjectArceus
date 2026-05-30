"""CNN encoder + actor-critic heads (fresh, self-contained for exp_007).

Input:  one-hot encoded 32x32 mini-LS20 frame, shape (B, 16, 32, 32) float32.
Output: (policy_logits (B, 4), value (B,)).

Initialisation:
    conv + trunk Linear       gain = sqrt(2)
    value_head                gain = 1.0
    policy_head               gain = 0.01     (near-uniform initial policy)

No BatchNorm, no pooling. Strided convs only. ~1.1M params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


N_COLORS = 16
FRAME_SIZE = 32
N_ACTIONS = 4
TRUNK_DIM = 256


def _orth(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def one_hot_frame(frame_uint8: torch.Tensor) -> torch.Tensor:
    """Convert (..., 32, 32) uint8/int frame -> (..., 16, 32, 32) float32.

    Accepts batch shapes (B, 32, 32) or (B, T, 32, 32).
    """
    x = frame_uint8.long()
    oh = F.one_hot(x, num_classes=N_COLORS).float()
    # Move colour channel to position -3.
    return oh.movedim(-1, -3)


class CNNEncoder(nn.Module):
    """One-hot 32x32 frame -> 256-d feature vector."""

    def __init__(self, n_colors: int = N_COLORS, trunk_dim: int = TRUNK_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            _orth(nn.Conv2d(n_colors, 32, kernel_size=3, stride=1, padding=1), 2 ** 0.5),
            nn.ReLU(inplace=True),
            _orth(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 2 ** 0.5),
            nn.ReLU(inplace=True),
            _orth(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), 2 ** 0.5),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_colors, FRAME_SIZE, FRAME_SIZE)
            flat = self.conv(dummy).flatten(1).shape[1]
        self.fc = _orth(nn.Linear(flat, trunk_dim), 2 ** 0.5)
        self.trunk_dim = trunk_dim

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(self.conv(x_onehot).flatten(1)))


class ActorCritic(nn.Module):
    """Shared CNN encoder feeding split policy-logit and value heads."""

    def __init__(self, n_actions: int = N_ACTIONS, n_colors: int = N_COLORS,
                 trunk_dim: int = TRUNK_DIM):
        super().__init__()
        self.encoder = CNNEncoder(n_colors=n_colors, trunk_dim=trunk_dim)
        self.policy_head = _orth(nn.Linear(trunk_dim, n_actions), gain=0.01)
        self.value_head = _orth(nn.Linear(trunk_dim, 1), gain=1.0)
        self.n_actions = n_actions

    def features(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        """obs_uint8: (B, 32, 32) palette indices -> (B, trunk_dim) features."""
        return self.encoder(one_hot_frame(obs_uint8))

    def forward(self, obs_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.features(obs_uint8)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1), feat

    @torch.no_grad()
    def act(self, obs_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action. Returns (action, log_prob, value, feature)."""
        logits, value, feat = self.forward(obs_uint8)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, feat

    def evaluate(self, obs_uint8: torch.Tensor, actions: torch.Tensor):
        """For PPO updates. Returns (log_prob, entropy, value, feature)."""
        logits, value, feat = self.forward(obs_uint8)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value, feat
