"""
Attention probes (metrics.md §2 and §3.1).

The encoder caches its post-softmax attention into `module._debug_attn` only
when `module.training` is False (see exp_003_0/models/encoder.py:101, 159).
For training-time probes we therefore briefly flip the encoder into eval()
mode, run a small forward, read the cached tensors, and restore training()
mode. No forward-signature changes are required.
"""

from __future__ import annotations

import contextlib
from typing import Iterable

import torch


# ── Pairwise JSD over K discrete distributions of length V ───────────────────

def pairwise_jsd(P: torch.Tensor) -> torch.Tensor:
    """
    Mean pairwise Jensen–Shannon divergence over K rows of P.

    P: (K, V) — each row a probability distribution.
    Returns: scalar tensor in [0, ln 2].
    """
    P = P.clamp_min(1e-12)
    M = 0.5 * (P.unsqueeze(0) + P.unsqueeze(1))   # (K, K, V)
    kl = (P.unsqueeze(1) * (P.unsqueeze(1).log() - M.log())).sum(-1)  # (K, K)
    jsd = 0.5 * (kl + kl.transpose(0, 1))
    K = P.shape[0]
    if K < 2:
        return torch.tensor(0.0, device=P.device)
    iu = torch.triu_indices(K, K, offset=1, device=P.device)
    return jsd[iu[0], iu[1]].mean()


# ── Cache readers (assume the module just ran a forward in eval()) ───────────

def read_sa_attn(block) -> torch.Tensor | None:
    """Read cached (B, n_heads, 16, 16) post-softmax attn from a SA block."""
    a = getattr(block, "_debug_attn", None)
    return a if a is not None else None


def read_latent_self_attn(round_block) -> torch.Tensor | None:
    """Read cached (B, n_heads, 4, 4) attn from _SelfAttentionAmongLatents."""
    a = getattr(round_block.self_attn, "_debug_attn", None)
    return a if a is not None else None


# ── Probe orchestrator: flip eval(), run forward, restore train() ────────────

@contextlib.contextmanager
def _temporarily_eval(modules: Iterable[torch.nn.Module]):
    prev = [(m, m.training) for m in modules]
    for m, _ in prev:
        m.eval()
    try:
        yield
    finally:
        for m, was_train in prev:
            if was_train:
                m.train()


@torch.no_grad()
def probe_attention(encoder, frames: torch.Tensor, queries: torch.Tensor) -> dict:
    """
    Run the encoder once in eval-mode and return captured attention tensors.

    frames:  (B, 64, 64) uint8
    queries: (B, n_latents, d_model)

    Returns dict with keys:
        "sa_blocks":   list[Tensor (B, n_heads, 16, 16)] per SA block
        "perc_self":   list[Tensor (B, n_heads, 4, 4)] per Perceiver round
    """
    with _temporarily_eval([encoder]):
        encoder(frames, queries)
    return {
        "sa_blocks": [read_sa_attn(b) for b in encoder.sa_blocks],
        "perc_self": [read_latent_self_attn(r) for r in encoder.perceiver.rounds],
    }


# ── Aggregate metrics from captured tensors ──────────────────────────────────

def patch_sa_row_jsd(sa_attns: list) -> float:
    """
    Metric 2.1. Within-step pairwise JSD between the 16 attention rows.

    sa_attns: list of (B, n_heads, 16, 16) tensors (one per SA block).
    Averaged over (batch, head, block).
    """
    vals = []
    for A in sa_attns:
        if A is None:
            continue
        B, H, _, _ = A.shape
        for b in range(B):
            for h in range(H):
                vals.append(pairwise_jsd(A[b, h]).item())
    return float(sum(vals) / len(vals)) if vals else float("nan")


def latent_self_attn_row_jsd_per_round(perc_self_attns: list) -> list:
    """
    Metric 3.1. Pairwise JSD between the 4 query-rows of latent self-attn,
    per Perceiver round. Averaged over (batch, head). Returns one float per
    round; rounds with no captured attention return nan.
    """
    out = []
    for A in perc_self_attns:
        if A is None:
            out.append(float("nan"))
            continue
        B, H, _, _ = A.shape
        vals = []
        for b in range(B):
            for h in range(H):
                vals.append(pairwise_jsd(A[b, h]).item())
        out.append(float(sum(vals) / len(vals)) if vals else float("nan"))
    return out


def patch_sa_temporal_jsd(per_step_attns: list) -> float:
    """
    Metric 2.2. Temporal pairwise JSD of A_t[h, i, :] across t for each
    patch i and head h, then averaged.

    per_step_attns: list of length T_eval; each element is a list of per-block
    tensors of shape (1, n_heads, 16, 16).
    """
    # Reshape into (n_blocks, T, n_heads, 16, 16)
    if not per_step_attns:
        return float("nan")
    blocks = list(zip(*per_step_attns))   # n_blocks tuples of length T
    vals = []
    for block_t in blocks:                # each: tuple of T tensors
        if any(a is None for a in block_t):
            continue
        # Stack over time → (T, n_heads, 16, 16) (assuming batch=1)
        stack = torch.stack([a.squeeze(0) for a in block_t], dim=0)  # (T, H, 16, 16)
        T, H, P, _ = stack.shape
        for h in range(H):
            for i in range(P):
                rows = stack[:, h, i, :]   # (T, 16)
                if T >= 2:
                    vals.append(pairwise_jsd(rows).item())
    return float(sum(vals) / len(vals)) if vals else float("nan")
