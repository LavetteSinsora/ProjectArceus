"""
Predictor diagnostics (metrics.md §4).

State predictor: ODE-step cossim, first-vs-final cossim, velocity norm.
Action predictor: entropy, per-action-class CE.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# ── State predictor ──────────────────────────────────────────────────────────

@torch.no_grad()
def ode_step_cossim(predictor, h_t: torch.Tensor, action_emb: torch.Tensor) -> float:
    """
    Metric 4.1. Cossim between consecutive ODE step outputs in
    predict_with_trajectory. Uses a single row (h_t[:1]) to amortize cost.
    """
    if h_t.shape[0] == 0:
        return float("nan")
    _, traj = predictor.predict_with_trajectory(h_t[:1], action_emb[:1])
    sims = []
    for i in range(len(traj) - 1):
        sims.append(F.cosine_similarity(
            traj[i].view(-1).unsqueeze(0),
            traj[i + 1].view(-1).unsqueeze(0),
        ).item())
    return float(np.mean(sims)) if sims else float("nan")


@torch.no_grad()
def ode_first_vs_final_cossim(
    predictor, h_t: torch.Tensor, action_emb: torch.Tensor
) -> float:
    """
    Metric 4.1b. Cossim between x_{1/N} (after the first Euler step) and x_1
    (the final prediction). Token-wise cosine, mean over the 4 latents, then
    mean over batch.
    """
    if h_t.shape[0] == 0:
        return float("nan")
    _, traj = predictor.predict_with_trajectory(h_t, action_emb)
    if len(traj) < 2:
        return float("nan")
    x_first = traj[1]                                                 # (B, L, D)
    x_final = traj[-1]
    sims = F.cosine_similarity(x_first, x_final, dim=-1)              # (B, L)
    return float(sims.mean().item())


@torch.no_grad()
def predictor_velocity_norm(
    predictor, h_t: torch.Tensor, action_emb: torch.Tensor
) -> float:
    """
    Metric 4.1c. Mean L2 norm of (x_1 - x_0) from the ODE trajectory.
    A collapse to ~0 means the predictor has saturated to "no change".
    """
    if h_t.shape[0] == 0:
        return float("nan")
    _, traj = predictor.predict_with_trajectory(h_t, action_emb)
    delta = traj[-1] - traj[0]                                        # (B, L, D)
    return float(delta.flatten(1).norm(dim=-1).mean().item())


# ── Action predictor ─────────────────────────────────────────────────────────

def action_pred_entropy_from_logits(logits: torch.Tensor) -> float:
    """
    Metric 4.3. Entropy of softmax(logits), in nats, averaged over the batch.
    Caller is responsible for masking unavailable actions before passing in.
    """
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return float(-(p * log_p).sum(-1).mean().item())


def action_pred_ce_per_class(
    ce_per_sample: torch.Tensor,
    actions: torch.Tensor,
    n_actions: int,
) -> dict:
    """
    Metric 6.1 (per-action-class CE row).

    ce_per_sample: (B,) — F.cross_entropy(..., reduction="none")
    actions: (B,) long — ground-truth actions
    Returns {action_idx: (mean_ce_or_nan, count)} for action_idx in 0..n_actions-1.
    """
    out: dict = {}
    for a in range(n_actions):
        mask = actions == a
        n = int(mask.sum().item())
        if n == 0:
            out[a] = (float("nan"), 0)
        else:
            out[a] = (float(ce_per_sample[mask].mean().item()), n)
    return out


# ── Misc constants ───────────────────────────────────────────────────────────

def entropy_max(n_actions: int) -> float:
    """H_max = ln(n_actions) — displayed next to entropy curves."""
    return float(math.log(max(n_actions, 1)))
