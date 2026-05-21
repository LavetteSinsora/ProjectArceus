"""Stateless math primitives for Dreamer V3.

Spec references (Hafner et al. 2023, arXiv:2301.04104):
  symlog/symexp   — App. A
  twohot          — Eq. (9), App. C
  lambda-returns  — Eq. (5)
  KL with free bits + balance — Eq. (4)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── symlog / symexp ───────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """symlog(x) = sign(x) * ln(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """symexp(x) = sign(x) * (exp(|x|) - 1).  Inverse of symlog."""
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


# ── twohot encoding ───────────────────────────────────────────────────────────

def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Soft-label encode scalar `x` (already symlog-transformed by caller if desired)
    onto a fixed bucket grid `bins` (1-D, monotonically increasing, K bins).

    For each x:
      find the two adjacent bins b_lo, b_hi enclosing x.
      put weight (b_hi - x) / (b_hi - b_lo) on b_lo
      put weight (x - b_lo) / (b_hi - b_lo) on b_hi

    Args:
        x:    (...,)            float
        bins: (K,)               float, monotonically increasing
    Returns:
        soft labels of shape (..., K) summing to 1 over the last dim.
    """
    K = bins.numel()
    x = x.clamp(min=bins[0].item(), max=bins[-1].item())
    # Index of the largest bin <= x.
    diff = x.unsqueeze(-1) - bins                # (..., K)
    # last bin where diff >= 0
    idx_lo = (diff >= 0).sum(dim=-1) - 1         # (...,)
    idx_lo = idx_lo.clamp(min=0, max=K - 2)
    idx_hi = idx_lo + 1

    b_lo = bins[idx_lo]
    b_hi = bins[idx_hi]
    w_hi = (x - b_lo) / (b_hi - b_lo).clamp(min=1e-8)
    w_lo = 1.0 - w_hi

    out = torch.zeros(*x.shape, K, device=x.device, dtype=x.dtype)
    out.scatter_(-1, idx_lo.unsqueeze(-1), w_lo.unsqueeze(-1))
    out.scatter_(-1, idx_hi.unsqueeze(-1), w_hi.unsqueeze(-1))
    return out


def twohot_decode(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expectation under a categorical over `bins`.  Inverse of twohot_encode
    in the average-case sense (E_p[bins])."""
    return (probs * bins).sum(dim=-1)


def make_twohot_bins(low: float, high: float, K: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Uniform grid of K bins over [low, high].  Caller is responsible for
    symlog-ing the target before encoding (and symexp-ing the decoded value)."""
    return torch.linspace(low, high, K, device=device, dtype=dtype)


# ── λ-returns ─────────────────────────────────────────────────────────────────

def lambda_returns(
    rewards: torch.Tensor,      # (H, B)
    values: torch.Tensor,       # (H+1, B)   — bootstrap value at end
    continues: torch.Tensor,    # (H, B)     — γ * ĉ_t, already gamma-multiplied OR raw ĉ_t with gamma supplied
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute the bootstrapped λ-returns of Dreamer V3 (Eq. 5).

    R^λ_t = r_t + γ ĉ_t [ (1 - λ) v_{t+1} + λ R^λ_{t+1} ],   R^λ_H = v_H

    Inputs are (T, B) where T is the imagination horizon.  `continues` is the
    per-step continue probability ĉ_t ∈ [0, 1] (NOT pre-multiplied by γ).

    Returns:
        (H, B) tensor of λ-returns aligned with `rewards`.
    """
    H = rewards.shape[0]
    assert values.shape[0] == H + 1, "values must be length H+1 (bootstrap at end)"
    discount = gamma * continues                                   # (H, B)
    out = [None] * H
    next_R = values[-1]                                            # v_H
    for t in reversed(range(H)):
        next_R = rewards[t] + discount[t] * ((1 - lam) * values[t + 1] + lam * next_R)
        out[t] = next_R
    return torch.stack(out, dim=0)


# ── Percentile return scaling (Dreamer V3 §3, Per95 - Per5, EMA-smoothed) ─────

class PercentileReturnScale:
    """EMA-smoothed S = Per(R, 95) - Per(R, 5).

    Per-update: compute new S_now from the current batch of λ-returns,
                EMA-blend it with the running estimate.
    """

    def __init__(self, decay: float = 0.99, low: float = 0.05, high: float = 0.95):
        self.decay = decay
        self.low = low
        self.high = high
        self._S: float = 1.0  # init to 1 so max(1, S) = 1 on first call

    def update(self, returns: torch.Tensor) -> float:
        with torch.no_grad():
            flat = returns.detach().flatten().float()
            lo = torch.quantile(flat, self.low).item()
            hi = torch.quantile(flat, self.high).item()
            s_now = max(hi - lo, 0.0)
        self._S = self.decay * self._S + (1.0 - self.decay) * s_now
        return self._S

    @property
    def scale(self) -> float:
        return max(1.0, self._S)


# ── KL helpers (with free bits + KL balance) ──────────────────────────────────

def categorical_kl(post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """KL[ Cat(post) || Cat(prior) ] over the LAST dim.

    Shapes: (..., n_classes) → (...,)
    """
    log_post = F.log_softmax(post_logits, dim=-1)
    log_prior = F.log_softmax(prior_logits, dim=-1)
    p = log_post.exp()
    return (p * (log_post - log_prior)).sum(dim=-1)


def kl_balance_loss(
    post_logits: torch.Tensor,   # (..., n_groups, n_classes)
    prior_logits: torch.Tensor,  # (..., n_groups, n_classes)
    balance: float = 0.8,
    free_nats: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dreamer V3 KL with stop-gradient balance + free bits.

    Returns (L_dyn, L_rep) where:
      L_dyn = KL[ sg(post) || prior   ]   (trains prior toward posterior)
      L_rep = KL[ post     || sg(prior)]  (trains posterior toward prior)

    Each KL is summed over the n_groups categoricals and then floored at
    free_nats (per-group sum) per the paper's "1 nat per dim" rule.
    """
    # Per the official codebase, free-bits is applied per-categorical (per group),
    # but the convention reported in DV3 paper §3 is "free bits 1 nat per KL term".
    # We follow the paper: clip the *summed* KL at free_nats.
    kl_dyn = categorical_kl(post_logits.detach(), prior_logits).sum(dim=-1)  # (...,)
    kl_rep = categorical_kl(post_logits, prior_logits.detach()).sum(dim=-1)  # (...,)

    L_dyn = torch.clamp(kl_dyn, min=free_nats).mean()
    L_rep = torch.clamp(kl_rep, min=free_nats).mean()
    return L_dyn, L_rep
