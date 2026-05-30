"""Collapse-risk diagnostics for the JEPA encoder.

All inputs are a single feature batch h of shape (B, D). Returns scalar
floats (Python floats), safe to drop into the metrics.jsonl record.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def feature_std(h: torch.Tensor) -> float:
    """Mean over feature dims of std(h_i) across the batch.

    Collapses to 0 if the encoder produces a constant.
    """
    if h.shape[0] < 2:
        return float("nan")
    return float(h.std(dim=0).mean().item())


@torch.no_grad()
def feature_pairwise_l2(h: torch.Tensor, n_pairs: int = 1024) -> float:
    """Mean L2 distance between randomly sampled pairs of rows in h.

    Collapses to 0 if all features map to the same point.
    """
    B = h.shape[0]
    if B < 2:
        return float("nan")
    k = min(n_pairs, B * (B - 1) // 2)
    i = torch.randint(0, B, (k,), device=h.device)
    j = torch.randint(0, B, (k,), device=h.device)
    mask = (i != j)
    if not bool(mask.any()):
        return float("nan")
    i, j = i[mask], j[mask]
    d = (h[i] - h[j]).norm(dim=-1)
    return float(d.mean().item())


@torch.no_grad()
def feature_effective_rank(h: torch.Tensor) -> float:
    """Entropy-based effective rank of the feature matrix.

    exp(- sum_i p_i log p_i) where p_i = sigma_i^2 / sum_j sigma_j^2 and
    sigma_i are the singular values of h. Collapses to 1 under rank-1
    representations; equals D when sigma is uniform.
    """
    if h.shape[0] < 2:
        return float("nan")
    # Center to ignore the mean direction (cheap effective rank).
    x = h - h.mean(dim=0, keepdim=True)
    # SVD on CPU is more robust on MPS than torch.linalg.svd on device.
    try:
        s = torch.linalg.svdvals(x.detach().to("cpu").float())
    except Exception:
        return float("nan")
    if s.numel() == 0:
        return float("nan")
    p = s.pow(2)
    total = p.sum()
    if float(total) <= 0:
        return float("nan")
    p = p / total
    # Add eps for numerical stability of log.
    eps = 1e-12
    entropy = -(p * (p + eps).log()).sum()
    return float(entropy.exp().item())


@torch.no_grad()
def all_diagnostics(h: torch.Tensor) -> dict[str, float]:
    return {
        "feat_std": feature_std(h),
        "feat_pairwise_l2": feature_pairwise_l2(h),
        "feat_effective_rank": feature_effective_rank(h),
    }
