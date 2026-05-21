"""Reward / continue / value heads for Dreamer V3.

All heads take the RSSM feature `s_t = [h_t || z_t.flatten()]` and emit a
distribution.  Reward and value use 255-bin twohot-symlog; continue uses
Bernoulli.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .distributions import BernoulliDist, TwohotSymlogDist


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class RewardHead(nn.Module):
    """Predicts r̂_t as a TwohotSymlogDist over K=255 bins on [-20, 20]."""

    def __init__(self, feat_dim: int, hidden: int = 512, K: int = 255, low: float = -20.0, high: float = 20.0):
        super().__init__()
        self.net = _mlp(feat_dim, hidden, K, layers=2)
        self.low = low
        self.high = high

    def forward(self, feat: torch.Tensor) -> TwohotSymlogDist:
        return TwohotSymlogDist(self.net(feat), low=self.low, high=self.high)


class ContinueHead(nn.Module):
    """Predicts ĉ_t ∈ [0, 1] as a Bernoulli."""

    def __init__(self, feat_dim: int, hidden: int = 512):
        super().__init__()
        self.net = _mlp(feat_dim, hidden, 1, layers=2)

    def forward(self, feat: torch.Tensor) -> BernoulliDist:
        return BernoulliDist(self.net(feat))


class ValueHead(nn.Module):
    """Critic v_ψ(s_t) as a TwohotSymlogDist (same shape as RewardHead)."""

    def __init__(self, cfg):
        super().__init__()
        feat_dim = cfg.deter + cfg.n_groups * cfg.n_classes
        self.net = _mlp(feat_dim, cfg.hidden_units, cfg.twohot_bins, layers=2)
        self.low = cfg.twohot_low
        self.high = cfg.twohot_high

    def forward(self, feat: torch.Tensor) -> TwohotSymlogDist:
        return TwohotSymlogDist(self.net(feat), low=self.low, high=self.high)
