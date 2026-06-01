"""Actor-critic with TWO value heads for RND dual-stream PPO.

Identical to exp_010's `ActorCritic` except for a second value head: RND tracks
extrinsic and intrinsic returns with separate discounts, so it needs separate
value estimates V_E and V_I sharing one CNN encoder. The encoder, policy head,
and initialisation are byte-for-byte the exp_010 / "7_0" recipe (reused by
import so the two experiments can never drift apart).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import (
    CNNEncoder, one_hot_frame, _orth,
    N_COLORS, FRAME_SIZE, N_ACTIONS, TRUNK_DIM,
)


class ActorCriticRND(nn.Module):
    """Shared CNN encoder feeding a policy head and two value heads.

    forward(obs) -> (logits, value_ext, value_int, feature)
    """

    def __init__(self, n_actions: int = N_ACTIONS, n_colors: int = N_COLORS,
                 frame_size: int = FRAME_SIZE, trunk_dim: int = TRUNK_DIM):
        super().__init__()
        self.encoder = CNNEncoder(n_colors=n_colors, frame_size=frame_size,
                                  trunk_dim=trunk_dim)
        self.policy_head = _orth(nn.Linear(trunk_dim, n_actions), gain=0.01)
        self.value_head_ext = _orth(nn.Linear(trunk_dim, 1), gain=1.0)
        self.value_head_int = _orth(nn.Linear(trunk_dim, 1), gain=1.0)
        self.n_actions = n_actions
        self.n_colors = n_colors

    def features(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        """obs_uint8: (B, H, W) palette indices -> (B, trunk_dim) features."""
        return self.encoder(one_hot_frame(obs_uint8, self.n_colors))

    def forward(self, obs_uint8: torch.Tensor):
        feat = self.features(obs_uint8)
        return (self.policy_head(feat),
                self.value_head_ext(feat).squeeze(-1),
                self.value_head_int(feat).squeeze(-1),
                feat)

    @torch.no_grad()
    def act(self, obs_uint8: torch.Tensor):
        """Sample an action. Returns (action, log_prob, v_ext, v_int, feature)."""
        logits, v_ext, v_int, feat = self.forward(obs_uint8)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), v_ext, v_int, feat

    def evaluate(self, obs_uint8: torch.Tensor, actions: torch.Tensor):
        """For PPO updates. Returns (log_prob, entropy, v_ext, v_int, feature)."""
        logits, v_ext, v_int, feat = self.forward(obs_uint8)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), v_ext, v_int, feat
