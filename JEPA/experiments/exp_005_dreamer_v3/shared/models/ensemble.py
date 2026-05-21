"""Plan2Explore one-step dynamics ensemble.

K=8 MLP heads, each predicting next-step latent ẑ_{t+1} from (h_t, z_t, a_t).
They share the same input features but are initialised independently so that
their predictions diverge on out-of-distribution states.  Intrinsic reward is
the per-dim variance across the K heads, averaged over latent dims.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class DynamicsEnsemble(nn.Module):
    """K small MLP heads → predict next stochastic latent (flattened logits or means).

    To keep things simple and stable we predict the *flat z_{t+1}* vector
    (n_groups * n_classes) in continuous space and treat the prediction as the
    head's "mean".  This is the standard P2E formulation — we use the
    cross-head variance as the disagreement signal.
    """

    def __init__(self, deter: int, n_groups: int, n_classes: int, n_actions: int, hidden: int = 256, K: int = 8):
        super().__init__()
        self.K = K
        in_dim = deter + n_groups * n_classes + n_actions
        out_dim = n_groups * n_classes
        self.heads = nn.ModuleList([_mlp(in_dim, hidden, out_dim, layers=2) for _ in range(K)])

    def _feat(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        z_flat = z.reshape(*z.shape[:-2], -1)
        return torch.cat([h, z_flat, a], dim=-1)

    def forward(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Returns predictions stacked as (K, ..., n_groups*n_classes)."""
        feat = self._feat(h, z, a)
        return torch.stack([head(feat) for head in self.heads], dim=0)

    def disagreement(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Intrinsic reward = mean over latent-dim of the K-variance.

        Returns a scalar reward of shape (...,) matching the batch dims of (h, z, a).
        """
        preds = self.forward(h, z, a)                # (K, ..., D)
        var = preds.var(dim=0, unbiased=False)       # (..., D)
        return var.mean(dim=-1)                      # (...,)

    def train_loss(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor, z_next_target: torch.Tensor) -> torch.Tensor:
        """MSE loss training each head to predict the *posterior* next-z (one-hot vector).

        Args:
            h, z, a:       (B, T, ...) RSSM features and one-hot action at time t.
            z_next_target: (B, T, n_groups, n_classes) — posterior one-hot at time t+1, detached.
        """
        z_next_flat = z_next_target.reshape(*z_next_target.shape[:-2], -1).detach()
        preds = self.forward(h, z, a)                                       # (K, B, T, D)
        loss = (preds - z_next_flat.unsqueeze(0)).pow(2).mean()
        return loss
