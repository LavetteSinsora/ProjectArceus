"""ConvDecoder: (h_t, z_t) → 64×64×1 pixel reconstruction.

Outputs are the *mean* of a SymlogMSE distribution: we predict a tensor in
symlog space whose symexp(·) is the expected pixel value.  The trainer computes
log_prob by passing the raw target pixels — see `SymlogMSEDist`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _norm_act(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.GroupNorm(num_groups=1, num_channels=channels),
        nn.SiLU(),
    )


class ConvDecoder(nn.Module):
    """(B, feat_dim) → (B, 1, 64, 64) in symlog space."""

    def __init__(self, feat_dim: int, depth: int = 32, out_channels: int = 1):
        super().__init__()
        c1 = depth * 8      # 256
        c2 = depth * 4      # 128
        c3 = depth * 2      # 64
        c4 = depth          # 32
        self.minres = 4
        self.c1 = c1
        self.proj = nn.Linear(feat_dim, c1 * self.minres * self.minres)
        self.net = nn.Sequential(
            _norm_act(c1),
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),  # 4→8
            _norm_act(c2),
            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),  # 8→16
            _norm_act(c3),
            nn.ConvTranspose2d(c3, c4, kernel_size=4, stride=2, padding=1),  # 16→32
            _norm_act(c4),
            nn.ConvTranspose2d(c4, out_channels, kernel_size=4, stride=2, padding=1),  # 32→64
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, feat_dim) — concat(h_t, z_t.flatten())
        Returns:
            (B, out_channels, 64, 64) in symlog space; pass to SymlogMSEDist.
        """
        h = self.proj(feat).reshape(-1, self.c1, self.minres, self.minres)
        return self.net(h)
