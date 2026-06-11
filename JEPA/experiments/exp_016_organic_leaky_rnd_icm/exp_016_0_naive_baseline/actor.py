"""Actor for exp_016_0 — policy-only (NO value head). REINFORCE. See SYSTEM_CARD §2.

A CNNEncoder (exp_010, the 7_0 recipe) feeding a single policy head. The encoder is
trained ONLY by the policy gradient; it is separate from the tracker's IDM encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import (
    CNNEncoder, one_hot_frame, _orth,
)


def mask_frames(frames_uint8: torch.Tensor, rows: tuple) -> torch.Tensor:
    """Zero the timer/UI rows (inclusive) so novelty/policy see the TRUE board.
    frames_uint8: (..., H, W) int/uint8."""
    x = frames_uint8.clone()
    x[..., int(rows[0]):int(rows[1]) + 1, :] = 0
    return x


class Actor(nn.Module):
    def __init__(self, n_actions: int, n_colors: int = 16, frame_size: int = 64,
                 trunk_dim: int = 256, mask_rows: tuple = (60, 63),
                 value_head: bool = False):
        super().__init__()
        self.encoder = CNNEncoder(n_colors=n_colors, frame_size=frame_size,
                                  trunk_dim=trunk_dim)
        self.policy = _orth(nn.Linear(trunk_dim, n_actions), 0.01)  # near-uniform start
        # Optional value head V(s) sharing the encoder — the STATE-DEPENDENT baseline.
        # advantage = return − V(s) removes the per-state/episode-position structure a
        # constant baseline can't, so an uninformative batch yields ~0 advantage.
        self.value = _orth(nn.Linear(trunk_dim, 1), 1.0) if value_head else None
        self.n_colors = n_colors
        self.mask_rows = mask_rows

    def _encode(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        oh = one_hot_frame(mask_frames(obs_uint8, self.mask_rows), self.n_colors)
        return self.encoder(oh)

    def logits(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        return self.policy(self._encode(obs_uint8))

    @torch.no_grad()
    def act(self, obs_uint8: torch.Tensor):
        """obs: (N,H,W) uint8 → (action, log_prob, entropy) each (N,)."""
        dist = Categorical(logits=self.logits(obs_uint8))
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy()

    def evaluate(self, obs_uint8: torch.Tensor, actions: torch.Tensor):
        """With grad: (log_prob, entropy, probs, value). value is None if no value head."""
        h = self._encode(obs_uint8)
        dist = Categorical(logits=self.policy(h))
        v = self.value(h).squeeze(-1) if self.value is not None else None
        return dist.log_prob(actions), dist.entropy(), dist.probs, v
