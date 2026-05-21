"""Critic EMA (decay 0.98) — separate from JEPA's encoder EMA (different semantics)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class CriticEMA:
    """Maintain an EMA copy of a critic for the DV3 critic-regularisation term.

    Usage:
        critic_ema = CriticEMA(critic, decay=0.98)
        # in training loop:
        critic_ema.update(critic)
        target_dist = critic_ema(feat)
    """

    def __init__(self, critic: nn.Module, decay: float = 0.98):
        self.decay = decay
        self.module = copy.deepcopy(critic)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        for p_t, p_s in zip(self.module.parameters(), source.parameters()):
            p_t.mul_(self.decay).add_(p_s.detach(), alpha=1.0 - self.decay)
        for b_t, b_s in zip(self.module.buffers(), source.buffers()):
            b_t.copy_(b_s)

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            return self.module(*args, **kwargs)

    def to(self, device):
        self.module.to(device)
        return self

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
