"""CNN encoder + actor-critic heads + JEPA modules for the *real* LS20 game.

This is a 64x64 generalisation of exp_007's self-contained CNN+PPO model.
Differences vs exp_007/shared/model.py:

  * Input is a (B, 64, 64) palette-index frame -> one-hot (B, 16, 64, 64).
  * One extra stride-2 conv so the flattened spatial map is 8x8 (same as the
    32x32 model), keeping the trunk Linear ~the same size.
  * Adds an `ActionConditionedPredictor` (forward JEPA head) and an
    `InverseDynamicsModel` (IDM head) used by the JEPA variants (exp_010_1,
    exp_010_2). The plain CNN+PPO baseline (exp_010_0) ignores them.

Initialisation follows the exp_007 recipe:
    conv + trunk Linear       gain = sqrt(2)
    value_head                gain = 1.0
    policy_head               gain = 0.01   (near-uniform initial policy)
No BatchNorm, no pooling. Strided convs only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_COLORS = 16
FRAME_SIZE = 64
N_ACTIONS = 4
TRUNK_DIM = 256


def _orth(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def one_hot_frame(frame_uint8: torch.Tensor, n_colors: int = N_COLORS) -> torch.Tensor:
    """(..., H, W) uint8/int frame -> (..., n_colors, H, W) float32.

    Palette indices are categorical, so we one-hot rather than feed raw ints.
    Accepts batch shapes (B, H, W) or (B, T, H, W).
    """
    x = frame_uint8.long().clamp_(0, n_colors - 1)
    oh = F.one_hot(x, num_classes=n_colors).float()
    return oh.movedim(-1, -3)


class CNNEncoder(nn.Module):
    """One-hot 64x64 frame -> trunk_dim feature vector.

    Four strided convs downsample 64 -> 8 (stride 1,2,2,2) before a flatten +
    trunk Linear. The conv stack auto-sizes the trunk Linear from a dummy pass,
    so the same class also works for the 32x32 mini-env should we want it.
    """

    def __init__(self, n_colors: int = N_COLORS, frame_size: int = FRAME_SIZE,
                 trunk_dim: int = TRUNK_DIM):
        super().__init__()
        self.n_colors = n_colors
        self.frame_size = frame_size
        self.conv = nn.Sequential(
            _orth(nn.Conv2d(n_colors, 32, kernel_size=3, stride=1, padding=1), 2 ** 0.5),
            nn.ReLU(inplace=True),
            _orth(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 2 ** 0.5),
            nn.ReLU(inplace=True),
            _orth(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), 2 ** 0.5),
            nn.ReLU(inplace=True),
            _orth(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), 2 ** 0.5),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_colors, frame_size, frame_size)
            flat = self.conv(dummy).flatten(1).shape[1]
        self.fc = _orth(nn.Linear(flat, trunk_dim), 2 ** 0.5)
        self.trunk_dim = trunk_dim

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(self.conv(x_onehot).flatten(1)))


class ActorCritic(nn.Module):
    """Shared CNN encoder feeding split policy-logit and value heads."""

    def __init__(self, n_actions: int = N_ACTIONS, n_colors: int = N_COLORS,
                 frame_size: int = FRAME_SIZE, trunk_dim: int = TRUNK_DIM):
        super().__init__()
        self.encoder = CNNEncoder(n_colors=n_colors, frame_size=frame_size,
                                  trunk_dim=trunk_dim)
        self.policy_head = _orth(nn.Linear(trunk_dim, n_actions), gain=0.01)
        self.value_head = _orth(nn.Linear(trunk_dim, 1), gain=1.0)
        self.n_actions = n_actions
        self.n_colors = n_colors

    def features(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        """obs_uint8: (B, H, W) palette indices -> (B, trunk_dim) features."""
        return self.encoder(one_hot_frame(obs_uint8, self.n_colors))

    def forward(self, obs_uint8: torch.Tensor):
        feat = self.features(obs_uint8)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1), feat

    @torch.no_grad()
    def act(self, obs_uint8: torch.Tensor):
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


# ── JEPA modules (used by exp_010_1 / exp_010_2 only) ───────────────────────

class ActionConditionedPredictor(nn.Module):
    """Forward JEPA head: (h_t, a_t) -> h_hat_{t+1} in trunk space.

    A small MLP over the concatenation of the trunk feature and a learned
    action embedding. Trained to match the (stop-gradient) encoding of the
    next frame.
    """

    def __init__(self, trunk_dim: int = TRUNK_DIM, n_actions: int = N_ACTIONS,
                 action_emb_dim: int = 32, hidden: int = 512):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, action_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(trunk_dim + action_emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, trunk_dim),
        )

    def forward(self, h_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        a = self.action_emb(a_t.long())
        return self.net(torch.cat([h_t, a], dim=-1))


class InverseDynamicsModel(nn.Module):
    """IDM head: (h_t, h_{t+1}) -> action logits. Regularises the encoder so
    its features retain action-relevant information (cf. exp_007_4)."""

    def __init__(self, trunk_dim: int = TRUNK_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * trunk_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, h_t: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_t, h_next], dim=-1))
