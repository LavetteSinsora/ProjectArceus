"""RND novelty in the ICM inverse-dynamics feature space φ, with a LEAKY predictor.

The novelty/count engine for exp_013_1 (RND+ICM / "OCC"), per SYSTEM_CARD §4.3–4.4:

  * φ = the ICM inverse-dynamics encoder (controllable features), frozen after warm-up.
    We operate on φ-VECTORS (dim = trunk_dim), NOT raw frames — pixel-space RND was
    measured to have no count resolution + a 99.9% generalisation leak (probes/).
  * target T(φ): fixed random MLP (never trained). predictor P(φ): trainable MLP.
  * raw novelty(s') = ½ · mean_j ( P(φ(s'))_j − T(φ(s'))_j )².
  * LEAK: after each predictor optimiser step, shrink P toward its INIT weights:
        θ_P ← (1 − μ)·θ_P + μ·θ_P^init        (μ = leak)
    This is the decoupled (AdamW-style) realisation of an L2-to-init pull — decoupled
    from Adam so the forget rate is exactly μ per update, not distorted by Adam's
    per-parameter scaling. It turns RND's one-way error ratchet into a *visitation-
    rate* signal that never permanently saturates (the exploration-stall fix).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import _orth


class _RNDMLP(nn.Module):
    """Plain orthogonally-initialised MLP over a feature vector."""

    def __init__(self, dim: int, hidden: int, out: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = dim
        for _ in range(max(1, n_layers - 1)):
            layers += [_orth(nn.Linear(d, hidden), 2 ** 0.5), nn.ReLU(inplace=True)]
            d = hidden
        layers += [_orth(nn.Linear(d, out), 2 ** 0.5)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPhi(nn.Module):
    """Frozen random target + trainable, leaky predictor over φ-features."""

    def __init__(self, dim: int = 256, hidden: int = 256, out: int = 256,
                 leak: float = 0.01):
        super().__init__()
        # Predictor is deeper than the target (paper convention): it *can* fit the
        # random target on visited φ's, but only there.
        self.target = _RNDMLP(dim, hidden, out, n_layers=2)
        self.predictor = _RNDMLP(dim, hidden, out, n_layers=3)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()
        self.leak = float(leak)
        # Snapshot the predictor's init weights — the leak shrinks back toward THESE.
        self._pred_init = [p.detach().clone() for p in self.predictor.parameters()]

    @torch.no_grad()
    def novelty(self, phi: torch.Tensor) -> torch.Tensor:
        """φ: (B, dim) → (B,) raw (un-normalised) novelty = ½·mean_j (P−T)²."""
        return 0.5 * (self.predictor(phi) - self.target(phi)).pow(2).mean(dim=-1)

    def distill_loss(self, phi: torch.Tensor) -> torch.Tensor:
        """MSE(P(φ), sg(T(φ))) over a φ minibatch. Grad flows ONLY through P
        (target is frozen and detached)."""
        with torch.no_grad():
            t = self.target(phi)
        return (self.predictor(phi) - t).pow(2).mean(dim=-1).mean()

    @torch.no_grad()
    def apply_leak(self) -> None:
        """θ_P ← (1−μ)·θ_P + μ·θ_P^init. Call AFTER optimizer.step()."""
        if self.leak <= 0.0:
            return
        for p, p0 in zip(self.predictor.parameters(), self._pred_init):
            # p0 was snapshotted at init (CPU); move to p's device once-per-call
            # (params are tiny) so the leak works regardless of .to(device).
            p.mul_(1.0 - self.leak).add_(p0.to(p.device), alpha=self.leak)
