"""Dreamer V3 distributions: 1% unimix categorical with straight-through samples,
twohot-symlog scalar distribution, symlog-MSE reconstruction distribution.

Each class exposes a tiny `.sample()`, `.log_prob(x)`, `.mode()`, and (where useful)
`.mean()` interface so the training loop and heads can treat them uniformly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .functional import (
    make_twohot_bins,
    symexp,
    symlog,
    twohot_decode,
    twohot_encode,
)


# ── Unimix categorical (1% uniform + 99% softmax) ─────────────────────────────

class UnimixCategorical:
    """Categorical distribution with 1% uniform mixing.

    p = (1 - mix) * softmax(logits) + mix * 1/n

    Used everywhere DV3 has a discrete distribution (z_t groups, actor over n_actions).
    `.sample()` returns one-hot with straight-through gradient.
    """

    def __init__(self, logits: torch.Tensor, mix: float = 0.01):
        # logits: (..., n_classes)
        self.mix = mix
        self._logits = logits
        n = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)
        self.probs = (1.0 - mix) * probs + mix / n
        # Re-derive logits from the unimixed probs so log_softmax is consistent.
        self.log_probs = torch.log(self.probs.clamp(min=1e-10))

    def sample(self) -> torch.Tensor:
        """Straight-through one-hot sample (..., n_classes)."""
        # Sample category index from the unimixed distribution.
        flat = self.probs.reshape(-1, self.probs.shape[-1])
        idx = torch.multinomial(flat, num_samples=1).squeeze(-1)
        idx = idx.reshape(*self.probs.shape[:-1])
        one_hot = F.one_hot(idx, num_classes=self.probs.shape[-1]).to(self.probs.dtype)
        # Straight-through: sample on forward, soft probs on backward.
        return one_hot + (self.probs - self.probs.detach())

    def mode(self) -> torch.Tensor:
        idx = self.probs.argmax(dim=-1)
        one_hot = F.one_hot(idx, num_classes=self.probs.shape[-1]).to(self.probs.dtype)
        return one_hot + (self.probs - self.probs.detach())

    def log_prob(self, one_hot: torch.Tensor) -> torch.Tensor:
        """log p(category) for a one-hot value (..., n_classes) → (...,)."""
        return (one_hot * self.log_probs).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return -(self.probs * self.log_probs).sum(dim=-1)

    @property
    def logits(self) -> torch.Tensor:
        return self._logits


# ── TwohotSymlogDist (reward head + critic) ──────────────────────────────────

class TwohotSymlogDist:
    """Distribution over a scalar via twohot-encoded symlog targets.

    The model outputs `logits` of shape (..., K=255); we softmax over bins and
    treat the result as a categorical over symlog-spaced bin centres.  The
    distribution's "mean" (used as a point estimate) is symexp(E[bin]).
    """

    def __init__(self, logits: torch.Tensor, low: float = -20.0, high: float = 20.0):
        self.logits = logits
        self.low = low
        self.high = high
        self.K = logits.shape[-1]
        # Bins live on the same device as logits.
        self.bins = make_twohot_bins(low, high, self.K, device=logits.device, dtype=logits.dtype)
        self.log_probs = F.log_softmax(logits, dim=-1)
        self.probs = self.log_probs.exp()

    def mean(self) -> torch.Tensor:
        """Point estimate in raw (un-symlogged) space."""
        return symexp(twohot_decode(self.probs, self.bins))

    def mode(self) -> torch.Tensor:
        # Same as mean for a unimodal distribution; we use the expectation.
        return self.mean()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of soft twohot(symlog(x)) target against self.log_probs.

        Returns log p(x) of shape (...,) — i.e. the negative cross-entropy.
        Used by the trainer as `-log_prob(x).mean()` for the reward-head loss.
        """
        target_logits = twohot_encode(symlog(x), self.bins)   # soft labels
        return (target_logits * self.log_probs).sum(dim=-1)


# ── SymlogMSE (decoder pixel reconstruction) ──────────────────────────────────

class SymlogMSEDist:
    """Continuous distribution with mean = symexp(net_output).

    log_prob(x) is defined as -0.5 * (net_output - symlog(x))^2 summed over
    the trailing dims (image dims) — i.e. a Gaussian likelihood in symlog space
    with unit variance, summed over pixels.  Per DV3 §3 the decoder uses
    `symlog_mse` for pixel reconstruction.
    """

    def __init__(self, net_output: torch.Tensor, event_dims: int = 3):
        # net_output: e.g. (B, C, H, W) in symlog space.
        self.net_output = net_output
        self.event_dims = event_dims

    def mean(self) -> torch.Tensor:
        return symexp(self.net_output)

    def mode(self) -> torch.Tensor:
        return self.mean()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        target = symlog(x)
        # Gaussian NLL up to constants: -0.5 * (out - target)^2 summed over event dims.
        diff2 = (self.net_output - target).pow(2)
        # Sum over the last `event_dims` dimensions.
        for _ in range(self.event_dims):
            diff2 = diff2.sum(dim=-1)
        return -0.5 * diff2


# ── BernoulliDist (continue head) ─────────────────────────────────────────────

class BernoulliDist:
    """Bernoulli over a single logit (continue probability)."""

    def __init__(self, logits: torch.Tensor):
        # logits: (..., 1) or (...,) — we squeeze trailing 1 if present.
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        self.logits = logits

    def mean(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def mode(self) -> torch.Tensor:
        return (self.logits > 0).float()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # x in {0, 1} (or [0, 1] for soft targets).
        return -F.binary_cross_entropy_with_logits(self.logits, x.float(), reduction='none')
