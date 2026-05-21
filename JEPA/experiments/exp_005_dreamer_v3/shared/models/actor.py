"""Actor — UnimixCategorical over n_actions, MLP from RSSM feature."""

from __future__ import annotations

import torch
import torch.nn as nn

from .distributions import UnimixCategorical


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class Actor(nn.Module):
    """π_θ(a_t | h_t, z_t) — UnimixCategorical over discrete actions."""

    def __init__(self, cfg):
        super().__init__()
        feat_dim = cfg.deter + cfg.n_groups * cfg.n_classes
        self.net = _mlp(feat_dim, cfg.hidden_units, cfg.n_actions, layers=2)
        self.unimix = cfg.unimix
        self.n_actions = cfg.n_actions

    def _feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.reshape(*z.shape[:-2], -1)
        return torch.cat([h, z_flat], dim=-1)

    def distribution(self, h: torch.Tensor, z: torch.Tensor) -> UnimixCategorical:
        logits = self.net(self._feat(h, z))
        return UnimixCategorical(logits, mix=self.unimix)

    def act(self, h: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Return one-hot action (B, n_actions) — sample (training) or argmax (eval)."""
        dist = self.distribution(h, z)
        return dist.mode() if deterministic else dist.sample()
