"""
Gradient diagnostics (metrics.md §7).

Three families of quantities:
  - grad_norm(params)              — L2 norm of accumulated .grad (used post-backward).
  - source decomposition           — per-source × per-sub-block gradient norms,
                                     measured via 3 separate backward passes on a
                                     fresh batch (no optimizer.step). Gated by
                                     cfg.grad_decomp_freq because of cost.
  - update-to-weight ratios        — snapshot per-group flat params pre-step,
                                     compute ‖θ_new − θ_old‖ / ‖θ_old‖ post-step.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


# ── Basic helpers ────────────────────────────────────────────────────────────

def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm of accumulated .grad over the given param iterable."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def _flat_grad_l2(grads: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.detach().pow(2).sum().item()
    return total ** 0.5


def _flat_grad_vector(grads: Sequence[torch.Tensor | None]) -> torch.Tensor:
    """Concatenate non-None grads into a 1-D vector on the device of the first non-None grad."""
    flat = [g.detach().reshape(-1) for g in grads if g is not None]
    if not flat:
        return torch.zeros(0)
    return torch.cat(flat, dim=0)


def grad_cosine(g_a: Sequence[torch.Tensor | None],
                g_b: Sequence[torch.Tensor | None]) -> float:
    """Cosine similarity between two parallel grad sequences (same param order)."""
    a, b = [], []
    for ga, gb in zip(g_a, g_b):
        if ga is None or gb is None:
            continue
        a.append(ga.detach().reshape(-1))
        b.append(gb.detach().reshape(-1))
    if not a:
        return float("nan")
    va, vb = torch.cat(a), torch.cat(b)
    if va.norm() < 1e-12 or vb.norm() < 1e-12:
        return float("nan")
    return float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())


def grad_sum(g_a: Sequence[torch.Tensor | None],
             g_b: Sequence[torch.Tensor | None]) -> List[torch.Tensor | None]:
    """Element-wise (per-parameter) sum of two grad sequences."""
    out: List[torch.Tensor | None] = []
    for ga, gb in zip(g_a, g_b):
        if ga is None and gb is None:
            out.append(None)
        elif ga is None:
            out.append(gb.detach().clone() if gb is not None else None)
        elif gb is None:
            out.append(ga.detach().clone())
        else:
            out.append(ga.detach() + gb.detach())
    return out


# ── Sub-block parameter partition ────────────────────────────────────────────

def build_sub_block_params(encoder, state_predictor, action_predictor,
                           action_embed, policy) -> Dict[str, List[torch.nn.Parameter]]:
    """
    Return a dict mapping sub-block name → list of parameters. Used both as
    the partition for `[total]` gnorms after the training backward, and as
    the slicing key for source-decomposition autograd.grad calls.
    """
    sub: Dict[str, List[torch.nn.Parameter]] = {}
    sub["patch_sa"] = (
        list(encoder.color_embed.parameters())
        + list(encoder.patch_proj.parameters())
        + list(encoder.sa_blocks.parameters())
        + list(encoder.sa_norm.parameters())
    )
    sub["perc_cross_r0"] = list(encoder.perceiver.rounds[0].cross_attn.parameters())
    sub["perc_self_r0"]  = list(encoder.perceiver.rounds[0].self_attn.parameters())
    if len(encoder.perceiver.rounds) > 1:
        sub["perc_cross_r1"] = list(encoder.perceiver.rounds[1].cross_attn.parameters())
        sub["perc_self_r1"]  = list(encoder.perceiver.rounds[1].self_attn.parameters())
    sub["perc_other"] = (
        [encoder.perceiver.placeholders]
        + list(encoder.perceiver.output_norm.parameters())
    )
    for i, mlp in enumerate(state_predictor.mlps):
        sub[f"state_pred_mlp_{i}"] = list(mlp.parameters())
    sub["state_pred_time_emb"] = list(state_predictor.time_embed.parameters())
    sub["action_embed"] = list(action_embed.parameters())
    sub["action_pred"]  = list(action_predictor.parameters())
    sub["policy"]       = list(policy.parameters())
    return sub


# ── Source decomposition (gated by grad_decomp_freq) ─────────────────────────

UPSTREAM_KEYS = (
    "patch_sa",
    "perc_cross_r0", "perc_self_r0",
    "perc_cross_r1", "perc_self_r1",
    "perc_other",
)


def compute_source_decomposition(
    encoder, state_predictor, action_predictor, action_embed,
    sub_blocks: Dict[str, List[torch.nn.Parameter]],
    frames, h_queries, next_frames, actions,
) -> dict:
    """
    Compute per-sub-block × per-source gradient norms and cross-source cosines.

    Runs a fresh forward + 3 separate autograd.grad passes — no optimizer step.

    Returns a flat dict with keys:
        gnorm_<sub>_from_state
        gnorm_<sub>_from_action_via_Ht
        gnorm_<sub>_from_action_via_Htp1
        gcossim_state_vs_action[<sub>]
        gcossim_action_Ht_vs_Htp1[<sub>]
    for each <sub> in UPSTREAM_KEYS.
    """
    upstream_params: List[torch.nn.Parameter] = []
    sub_slices: Dict[str, slice] = {}
    cursor = 0
    for k in UPSTREAM_KEYS:
        ps = sub_blocks.get(k, [])
        sub_slices[k] = slice(cursor, cursor + len(ps))
        upstream_params.extend(ps)
        cursor += len(ps)

    # Fresh forward (must build graph — autograd.grad needs it)
    h_t_m, _, _   = encoder(frames, h_queries.detach())
    h_tp1_m, _, _ = encoder(next_frames, h_t_m.detach())
    a_emb_m = action_embed(actions)

    L_state_m, _ = state_predictor.compute_loss(h_t_m, h_tp1_m.detach(), a_emb_m)
    # Isolate Ht-only and Htp1-only action paths via opposite-side detach.
    L_action_via_Ht   = F.cross_entropy(
        action_predictor(h_t_m, h_tp1_m.detach()), actions
    )
    L_action_via_Htp1 = F.cross_entropy(
        action_predictor(h_t_m.detach(), h_tp1_m), actions
    )

    g_state = torch.autograd.grad(
        L_state_m, upstream_params, retain_graph=True, allow_unused=True,
    )
    g_via_Ht = torch.autograd.grad(
        L_action_via_Ht, upstream_params, retain_graph=True, allow_unused=True,
    )
    g_via_Htp1 = torch.autograd.grad(
        L_action_via_Htp1, upstream_params, retain_graph=False, allow_unused=True,
    )

    out: dict = {}
    for k in UPSTREAM_KEYS:
        sl = sub_slices[k]
        gs   = g_state[sl]
        ght  = g_via_Ht[sl]
        ghp1 = g_via_Htp1[sl]
        out[f"gnorm_{k}_from_state"]            = _flat_grad_l2(gs)
        out[f"gnorm_{k}_from_action_via_Ht"]    = _flat_grad_l2(ght)
        out[f"gnorm_{k}_from_action_via_Htp1"]  = _flat_grad_l2(ghp1)
        g_action_total = grad_sum(ght, ghp1)
        out[f"gcossim_state_vs_action_{k}"]      = grad_cosine(gs, g_action_total)
        out[f"gcossim_action_Ht_vs_Htp1_{k}"]    = grad_cosine(ght, ghp1)
    return out


# ── Update-to-weight ratios ──────────────────────────────────────────────────

class UWRSnapshot:
    """
    Snapshot pre-step flat-parameter vectors per sub-block; compute the L2
    update ratio per group after optimizer.step().
    """

    def __init__(self, sub_blocks: Dict[str, List[torch.nn.Parameter]]):
        self._sub_blocks = sub_blocks
        self._pre: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def snapshot(self) -> None:
        self._pre.clear()
        for k, params in self._sub_blocks.items():
            if not params:
                continue
            self._pre[k] = torch.cat([p.detach().reshape(-1) for p in params]).clone()

    @torch.no_grad()
    def ratios(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, pre in self._pre.items():
            params = self._sub_blocks[k]
            post = torch.cat([p.detach().reshape(-1) for p in params])
            denom = pre.norm().item()
            if denom < 1e-12:
                out[k] = float("nan")
            else:
                out[k] = float((post - pre).norm().item() / denom)
        return out
