"""Plan2Explore-style ENSEMBLE DISAGREEMENT novelty (exp_013_4).

The genuinely non-redundant signal (vs ICM/RND error, which both → 0 with visitation
and are mutually correlated): an ENSEMBLE of forward models predicts φ(s') from
(φ(s), a); the intrinsic reward is their **disagreement** (variance across members).

  * disagreement → 0 only where the ensemble is CONFIDENT = (s,a) well-sampled AND
    learnable; stays high where EPISTEMICALLY uncertain = genuinely unexplored.
  * aleatoric noise → members regress to the conditional mean → AGREE → low reward
    (no white-noise/TV trap that single-error curiosity falls for).
  * count-free; needs NO stationary ruler / running count.

φ = a FROZEN RANDOM encoder (stationary by construction) → sidesteps the ICM-φ
controllability problem we measured (held-out inv_acc ≈ chance on LS20). Disagreement
about predicting a fixed random projection of s' is still a valid epistemic-novelty
signal: members differ in UNSAMPLED regions regardless of φ's semantics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import (
    CNNEncoder, one_hot_frame, _orth, N_COLORS, FRAME_SIZE, N_ACTIONS, TRUNK_DIM,
)


class FrozenPhi(nn.Module):
    """Fixed random CNN encoder: frame → φ (trunk_dim). Never trained."""

    def __init__(self, n_colors: int = N_COLORS, frame_size: int = FRAME_SIZE,
                 trunk_dim: int = TRUNK_DIM):
        super().__init__()
        self.n_colors = n_colors
        self.trunk_dim = trunk_dim                 # so it can stand in for ICMModule
        self.encoder = CNNEncoder(n_colors=n_colors, frame_size=frame_size, trunk_dim=trunk_dim)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        return self.encoder(one_hot_frame(obs_uint8, self.n_colors))

    @torch.no_grad()
    def encode(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        """Alias so a FrozenPhi can be passed wherever an ICMModule.encode is expected."""
        return self.forward(obs_uint8)


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        _orth(nn.Linear(d_in, hidden), 2 ** 0.5), nn.ReLU(inplace=True),
        _orth(nn.Linear(hidden, hidden), 2 ** 0.5), nn.ReLU(inplace=True),
        _orth(nn.Linear(hidden, d_out), 2 ** 0.5),
    )


class ForwardEnsemble(nn.Module):
    """K forward models f_k([φ(s); onehot(a)]) → φ̂(s'). Reward = Var_k over members.

    Members differ only by random init (orthogonal init is re-sampled per member) +
    SGD order — sufficient to make them disagree in under-sampled regions while
    converging where data is plentiful (the standard disagreement recipe)."""

    def __init__(self, k: int = 5, dim: int = TRUNK_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 256):
        super().__init__()
        self.k = k
        self.n_actions = n_actions
        self.members = nn.ModuleList([_mlp(dim + n_actions, hidden, dim) for _ in range(k)])

    def _inp(self, phi: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat([phi, F.one_hot(a.long(), self.n_actions).float()], dim=-1)

    @torch.no_grad()
    def disagreement(self, phi: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """(B, dim), (B,) → (B,) mean-over-feature variance across the K members."""
        x = self._inp(phi, a)
        preds = torch.stack([m(x) for m in self.members], dim=0)   # (K, B, dim)
        return preds.var(dim=0, unbiased=False).mean(dim=-1)

    def loss(self, phi: torch.Tensor, a: torch.Tensor, phi_next: torch.Tensor) -> torch.Tensor:
        """Mean over members of MSE(f_k(φ(s),a), sg φ(s')). φ is frozen, so the target
        is a fixed random embedding."""
        x = self._inp(phi, a)
        tgt = phi_next.detach()
        return sum((m(x) - tgt).pow(2).mean() for m in self.members) / self.k
