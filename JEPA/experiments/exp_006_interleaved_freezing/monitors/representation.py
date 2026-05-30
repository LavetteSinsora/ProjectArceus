"""
Representation-health probes (metrics.md §1).

All functions are pure / stateless. Computations are intentionally cheap so
they can be called every JEPA update without affecting throughput.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def placeholder_pairwise_cossim(placeholders: torch.Tensor) -> float:
    """
    Mean pairwise cosine similarity between the n_placeholder vectors.

    placeholders: (n_placeholders, d_model) nn.Parameter
    Returns scalar float.
    """
    P = placeholders.detach()
    n = P.shape[0]
    if n < 2:
        return float("nan")
    P_n = F.normalize(P, dim=-1)
    G = P_n @ P_n.T
    iu = torch.triu_indices(n, n, offset=1)
    return float(G[iu[0], iu[1]].mean().item())


@torch.no_grad()
def placeholder_drift_from_init(
    current: torch.Tensor, init: torch.Tensor
) -> list:
    """
    Per-token cosine similarity between each placeholder's current value and
    its value at step 0.

    Returns list of floats of length n_placeholders (caller can report mean
    separately if desired).
    """
    cur = current.detach().to(init.device)
    sims = F.cosine_similarity(cur, init, dim=-1)
    return [float(v.item()) for v in sims]


@torch.no_grad()
def latent_pairwise_cossim(H: torch.Tensor) -> float:
    """
    Mean pairwise cosine similarity between the L latent tokens.

    H: (..., L, D) — averaged over leading batch dims.
    """
    if H.dim() == 2:
        H = H.unsqueeze(0)
    H_n = F.normalize(H, dim=-1)
    L = H.shape[-2]
    if L < 2:
        return float("nan")
    G = H_n @ H_n.transpose(-1, -2)              # (..., L, L)
    iu = torch.triu_indices(L, L, offset=1, device=H.device)
    pairs = G[..., iu[0], iu[1]]                  # (..., n_pairs)
    return float(pairs.mean().item())


@torch.no_grad()
def latent_pairwise_l2(H: torch.Tensor) -> float:
    """Mean pairwise L2 distance between the L latent tokens. Kept distinct
    from cossim per metrics.md §1.2 soundness note."""
    if H.dim() == 2:
        H = H.unsqueeze(0)
    B, L, _ = H.shape
    diff = H.unsqueeze(-3) - H.unsqueeze(-2)      # (B, L, L, D)
    dists = diff.norm(dim=-1)
    iu = torch.triu_indices(L, L, offset=1, device=H.device)
    return float(dists[..., iu[0], iu[1]].mean().item())


@torch.no_grad()
def latent_norms(H: torch.Tensor) -> list:
    """Per-latent L2 norm, averaged over the batch. Returns list of length L."""
    if H.dim() == 2:
        H = H.unsqueeze(0)
    return [float(H[:, i, :].norm(dim=-1).mean().item()) for i in range(H.shape[1])]


def effective_rank(matrix: torch.Tensor) -> float:
    """
    Entropy-based effective rank: exp(H(p)) where p is the normalized
    singular-value distribution. Returns 1.0 if matrix has degenerated to a
    single direction, up to min(L, D) for a full-rank matrix.

    matrix: (L, D).
    """
    try:
        sv = torch.linalg.svdvals(matrix.float().cpu())
        sv = sv / (sv.sum() + 1e-8)
        return float(torch.exp(-(sv * (sv + 1e-8).log()).sum()).item())
    except Exception:
        return float("nan")


@torch.no_grad()
def ht_htp1_cossim(h_t: torch.Tensor, h_tp1: torch.Tensor) -> float:
    """
    Average cosine similarity between successive latents (token-wise).

    h_t, h_tp1: (..., L, D). Token-i to token-i cosine averaged over L and
    leading batch dims.
    """
    return float(F.cosine_similarity(h_t, h_tp1, dim=-1).mean().item())
