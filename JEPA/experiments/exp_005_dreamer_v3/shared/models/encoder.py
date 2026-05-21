"""ConvEncoder: 64×64×1 → 4×4×D → MLP → embed_dim.

Mirrors the Dreamer V3 image encoder for 64×64 pixel observations.  Four
stride-2 conv stages take 64→32→16→8→4 with channel doubling, then flatten +
linear projection to the embed dim consumed by the RSSM posterior.

LS20 frames are uint8 colour indices in [0, 15]; the trainer normalises to
[-0.5, 0.5] (via `/15.0 - 0.5`) before this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _norm_act(channels: int) -> nn.Sequential:
    """LayerNorm + SiLU after each conv stage (DV3 uses RMSNorm; LayerNorm is
    a near-equivalent that is built-in to PyTorch)."""
    return nn.Sequential(
        nn.GroupNorm(num_groups=1, num_channels=channels),
        nn.SiLU(),
    )


class ConvEncoder(nn.Module):
    """64×64 → embed_dim."""

    def __init__(self, in_channels: int = 1, depth: int = 32, embed_dim: int = 1024):
        super().__init__()
        c1 = depth          # 32
        c2 = depth * 2      # 64
        c3 = depth * 4      # 128
        c4 = depth * 8      # 256
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),  # 64→32
            _norm_act(c1),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),           # 32→16
            _norm_act(c2),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),           # 16→8
            _norm_act(c3),
            nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1),           # 8→4
            _norm_act(c4),
        )
        self.out_channels = c4
        self.minres = 4
        flat_dim = c4 * self.minres * self.minres
        self.proj = nn.Linear(flat_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, 64, 64) — already centred to roughly [-0.5, 0.5].
        Returns:
            (B, embed_dim) image embedding.
        """
        h = self.net(x)
        h = h.flatten(start_dim=1)
        return self.proj(h)
