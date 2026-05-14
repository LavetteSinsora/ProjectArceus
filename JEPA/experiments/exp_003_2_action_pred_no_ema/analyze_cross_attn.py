"""
Cross-attention diagnostic — why do different latent queries produce nearly
identical attention rows over patch keys?

Tests four hypotheses on a single debug episode:
  H1 — Q-projection collapse: q_proj maps the 4 distinct h_{t-1} to nearly-
       identical query vectors.
  H2 — Q lives orthogonal to K-span: Q vectors are distinct in input space but
       their component along col-span(K) is identical.
  H3 — Softmax saturation: raw logits differ across queries but softmax
       flattens row-to-row differences.
  H4 — Patch-embedding (key-source) collapse: sa_out patches are nearly the
       same, leaving K vectors only weakly distinguished.

For each environment step we compute, per Perceiver round, per attention head:

  patch_pairwise_cos        — cossim between the 16 sa_out patch embeddings (H4)
  q_pairwise_cos / _l2      — cossim + L2 between the 4 post-q_proj queries (H1)
  q_norms                   — norm per query vector (magnitude collapse check)
  k_pairwise_cos / _l2      — cossim + L2 between the 16 post-k_proj keys
  raw_logit_row_pairwise_cos — cossim between the 4 raw-logit rows of QK^T/√d (H3 numerator)
  softmax_row_pairwise_jsd  — Jensen–Shannon divergence between the 4 attention
                              distributions (H3 denominator + final readout)
  q_on_kspan_pairwise_cos   — cossim between the 4 Qs projected onto col-span(K) (H2)
  q_orth_kspan_norm_frac    — fraction of ‖Q‖² lying outside col-span(K)
                              (1.0 ⇒ Q is entirely in the K-null direction ⇒ H2 confirmed)

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_2_action_pred_no_ema.analyze_cross_attn \\
        --checkpoint JEPA/experiments/exp_003_2_action_pred_no_ema/checkpoints/step_015000.pt \\
        --max-steps 40
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_2_action_pred_no_ema.config import Config
from JEPA.experiments.exp_003_2_action_pred_no_ema.models import load_models
from JEPA.experiments.exp_003_2_action_pred_no_ema.reward_shaping import is_end_of_life
from JEPA.shared.env_wrapper import LS20Env


# ── Pairwise helpers ─────────────────────────────────────────────────────────

def _pairwise_cos(v: torch.Tensor) -> float:
    """Mean upper-triangle cosine similarity between rows of v (K × D)."""
    K = v.shape[0]
    if K < 2:
        return float("nan")
    vn = F.normalize(v, dim=-1)
    G = vn @ vn.T
    iu = torch.triu_indices(K, K, offset=1)
    return float(G[iu[0], iu[1]].mean().item())


def _pairwise_l2(v: torch.Tensor) -> float:
    K = v.shape[0]
    if K < 2:
        return float("nan")
    diff = v.unsqueeze(0) - v.unsqueeze(1)
    iu = torch.triu_indices(K, K, offset=1)
    return float(diff.norm(dim=-1)[iu[0], iu[1]].mean().item())


def _pairwise_jsd(P: torch.Tensor) -> float:
    """Mean pairwise JSD over K rows of P (K × V), each a distribution."""
    P = P.clamp_min(1e-12)
    M = 0.5 * (P.unsqueeze(0) + P.unsqueeze(1))
    kl = (P.unsqueeze(1) * (P.unsqueeze(1).log() - M.log())).sum(-1)
    jsd = 0.5 * (kl + kl.transpose(0, 1))
    K = P.shape[0]
    if K < 2:
        return float("nan")
    iu = torch.triu_indices(K, K, offset=1)
    return float(jsd[iu[0], iu[1]].mean().item())


def _project_onto_rowspan(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Project each row of Q (size N × d) onto the column-span of K^T,
    i.e. the row-span of K (also size M × d).
    Returns the in-span component of Q (same shape as Q).

    Uses pinv(K) so the projector handles rank deficiency.
    P_K = K^T (K K^T)^+ K   (projection onto row-span of K, in d-dim space)
    Q_in = Q @ P_K
    """
    KKt = K @ K.T                              # (M, M)
    KKt_pinv = torch.linalg.pinv(KKt.float())  # (M, M)
    P_K = K.T @ KKt_pinv @ K                   # (d, d) projector
    return Q @ P_K


def _orth_frac(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Per-row fraction of ‖Q[i]‖² that lies OUTSIDE row-span(K).
    Returns shape (N,). 1.0 means Q[i] is orthogonal to K's row-span.
    """
    Q_in = _project_onto_rowspan(Q, K)
    num = (Q - Q_in).pow(2).sum(-1)
    den = Q.pow(2).sum(-1).clamp_min(1e-12)
    return num / den


# ── Per-step analysis ────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_step(encoder, queries: torch.Tensor, frame_t: torch.Tensor) -> dict:
    """
    Capture per-round, per-head diagnostics for one cross-attention call.

    queries: (1, L=4, d_model) — h_{t-1} (or placeholders at t=0)
    frame_t: (1, 64, 64) uint8

    Returns a nested dict keyed by round → head_idx → metric_name.
    """
    sa_out = encoder.encode_patches(frame_t)                # (1, 16, d)
    h = queries
    out_per_round: list = []

    for r_idx, round_block in enumerate(encoder.perceiver.rounds):
        ca = round_block.cross_attn

        q_in = ca.norm_q(h)                                 # (1, 4, d)
        k_in = ca.norm_kv(sa_out)                           # (1, 16, d)

        Q = ca.q_proj(q_in)                                 # (1, 4, d)
        K = ca.k_proj(k_in)                                 # (1, 16, d)
        V = ca.v_proj(sa_out)                               # (1, 16, d)

        n_heads = ca.n_heads
        d_head = ca.d_head
        # Reshape to per-head views: (heads, K_or_4, d_head)
        Q_h = Q.view(1, -1, n_heads, d_head).permute(0, 2, 1, 3).squeeze(0)   # (h, 4, d_head)
        K_h = K.view(1, -1, n_heads, d_head).permute(0, 2, 1, 3).squeeze(0)   # (h, 16, d_head)

        # Patch-embedding (post-SA) pairwise cossim — single scalar per step
        sa_patch_cos = _pairwise_cos(sa_out.squeeze(0))      # (16, d)

        scale = math.sqrt(d_head)
        logits = (Q_h @ K_h.transpose(-1, -2)) / scale       # (h, 4, 16)
        attn   = F.softmax(logits, dim=-1)                   # (h, 4, 16)

        per_head: list = []
        for h_idx in range(n_heads):
            Q_i = Q_h[h_idx]                                  # (4, d_head)
            K_i = K_h[h_idx]                                  # (16, d_head)
            logits_i = logits[h_idx]                          # (4, 16) raw
            attn_i   = attn[h_idx]                            # (4, 16) softmax

            Q_in = _project_onto_rowspan(Q_i, K_i)            # (4, d_head)
            orth = _orth_frac(Q_i, K_i)                       # (4,)

            per_head.append({
                "q_pairwise_cos": _pairwise_cos(Q_i),
                "q_pairwise_l2":  _pairwise_l2(Q_i),
                "q_norms":        [float(n_.item()) for n_ in Q_i.norm(dim=-1)],
                "k_pairwise_cos": _pairwise_cos(K_i),
                "k_pairwise_l2":  _pairwise_l2(K_i),
                "raw_logit_row_pairwise_cos": _pairwise_cos(logits_i),
                "softmax_row_pairwise_jsd":   _pairwise_jsd(attn_i),
                "q_on_kspan_pairwise_cos":    _pairwise_cos(Q_in),
                "q_orth_kspan_norm_frac":     [float(o.item()) for o in orth],
            })

        out_per_round.append({
            "patch_post_sa_pairwise_cos": sa_patch_cos,
            "h_input_pairwise_cos": _pairwise_cos(h.squeeze(0)),  # cossim of the round's input queries
            "per_head": per_head,
        })

        # Advance the recurrent stream to the next round (full forward replicating Round)
        # Cross-attention output:
        attn_w_full = F.softmax(
            (Q.view(1, h.shape[1], n_heads, d_head).transpose(1, 2) @
             K.view(1, 16, n_heads, d_head).transpose(1, 2).transpose(-1, -2)) / scale,
            dim=-1,
        )
        V_full = V.view(1, 16, n_heads, d_head).transpose(1, 2)    # (1, h, 16, d_head)
        out_attn = (attn_w_full @ V_full).transpose(1, 2).contiguous().view(1, h.shape[1], -1)
        h = h + ca.out_proj(out_attn)
        h = ca.ffn(h)
        # Then latent self-attention to produce input for next round
        h = round_block.self_attn(h)

    return {"rounds": out_per_round}


# ── Aggregation across timesteps ─────────────────────────────────────────────

def _mean_safe(xs):
    xs = [x for x in xs if x is not None and (isinstance(x, (int, float)) and math.isfinite(x))]
    return float(np.mean(xs)) if xs else float("nan")


def summarize(per_step: list, n_rounds: int, n_heads: int) -> dict:
    summary = {"n_steps": len(per_step), "rounds": []}
    for r in range(n_rounds):
        round_summary = {
            "patch_post_sa_pairwise_cos": _mean_safe(
                [s["rounds"][r]["patch_post_sa_pairwise_cos"] for s in per_step]
            ),
            "h_input_pairwise_cos": _mean_safe(
                [s["rounds"][r]["h_input_pairwise_cos"] for s in per_step]
            ),
            "per_head": [],
        }
        for h in range(n_heads):
            keys = [
                "q_pairwise_cos", "q_pairwise_l2",
                "k_pairwise_cos", "k_pairwise_l2",
                "raw_logit_row_pairwise_cos",
                "softmax_row_pairwise_jsd",
                "q_on_kspan_pairwise_cos",
            ]
            per_head_summary = {
                k: _mean_safe([s["rounds"][r]["per_head"][h][k] for s in per_step])
                for k in keys
            }
            # Average q_norm + orth_frac across the 4 latents and steps
            q_norm_avg = _mean_safe([
                v for s in per_step
                for v in s["rounds"][r]["per_head"][h]["q_norms"]
            ])
            orth_avg = _mean_safe([
                v for s in per_step
                for v in s["rounds"][r]["per_head"][h]["q_orth_kspan_norm_frac"]
            ])
            per_head_summary["q_norm_mean"] = q_norm_avg
            per_head_summary["q_orth_kspan_norm_frac_mean"] = orth_avg
            round_summary["per_head"].append(per_head_summary)
        summary["rounds"].append(round_summary)
    return summary


# ── Pretty-print decision-tree readout ───────────────────────────────────────

def report(summary: dict) -> None:
    print("\n" + "=" * 80)
    print(f"Cross-attention diagnostic — {summary['n_steps']} steps averaged")
    print("=" * 80)
    for r_idx, r in enumerate(summary["rounds"]):
        print(f"\n── Round {r_idx} " + "─" * 60)
        print(f"  Round input queries pairwise cossim (h_{{t-1}}):  {r['h_input_pairwise_cos']:+.4f}")
        print(f"  Post-SA patch embeddings pairwise cossim:       {r['patch_post_sa_pairwise_cos']:+.4f}")
        print(f"  (H4 — patch-embedding collapse if ≥ ~0.95)")
        print(f"  {'head':<6}{'q_cos':>8}{'q_l2':>8}{'q_norm':>8}"
              f"{'k_cos':>8}{'k_l2':>9}"
              f"{'logitR_cos':>12}{'JSD':>9}"
              f"{'q|Kspan_cos':>14}{'orth_frac':>11}")
        for h_idx, h in enumerate(r["per_head"]):
            print(
                f"  h{h_idx:<5}"
                f"{h['q_pairwise_cos']:+8.3f}"
                f"{h['q_pairwise_l2']:8.3f}"
                f"{h['q_norm_mean']:8.3f}"
                f"{h['k_pairwise_cos']:+8.3f}"
                f"{h['k_pairwise_l2']:9.3f}"
                f"{h['raw_logit_row_pairwise_cos']:+12.3f}"
                f"{h['softmax_row_pairwise_jsd']:9.4f}"
                f"{h['q_on_kspan_pairwise_cos']:+14.3f}"
                f"{h['q_orth_kspan_norm_frac_mean']:11.3f}"
            )

    print("\n" + "=" * 80)
    print("Decision-tree key:")
    print("  q_cos ≈ 1               ⇒ H1 (Q-projection collapse)")
    print("  q_cos low AND q|Kspan_cos ≈ 1 ⇒ H2 (Q distinct, but distinctness orthogonal to K-span)")
    print("  logitR_cos low AND JSD ≈ 0 ⇒ H3 (softmax-saturation flattens differences)")
    print("  patch_post_sa_pairwise_cos ≈ 1 ⇒ H4 (patch-embedding collapse)")
    print("=" * 80 + "\n")


# ── Driver ───────────────────────────────────────────────────────────────────

def run(checkpoint_path: str, max_steps: int = 40, out_json: str | None = None):
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[analyze] device={device}  checkpoint={checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, dict):
        valid = {f.name for f in dataclasses.fields(Config)}
        cfg = Config(**{k: v for k, v in cfg_raw.items() if k in valid})
    else:
        cfg = cfg_raw

    encoder, state_predictor, action_predictor, action_embed, policy, _ = \
        load_models(cfg, device)
    encoder.load_state_dict(ckpt["encoder"])
    policy.load_state_dict(ckpt["policy"])
    # action_predictor / state_predictor / action_embed not used here but loaded for completeness
    if "state_predictor" in ckpt:
        state_predictor.load_state_dict(ckpt["state_predictor"])
    if "action_predictor" in ckpt:
        action_predictor.load_state_dict(ckpt["action_predictor"])
    if "action_embed" in ckpt:
        action_embed.load_state_dict(ckpt["action_embed"])
    encoder.eval(); policy.eval()
    print(f"[analyze] loaded step={ckpt.get('step', 0)}")

    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)
    frame_np = env.reset()

    h_t: torch.Tensor | None = None
    per_step: list = []

    for t in range(max_steps):
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
        if h_t is None:
            queries = encoder.perceiver.get_initial_queries(1, device)
        else:
            queries = h_t.detach()

        step_data = analyze_step(encoder, queries, frame_t)
        per_step.append(step_data)

        with torch.no_grad():
            h_current, _, _ = encoder(frame_t, queries)
            action_idx, _, _ = policy.act(h_current.squeeze(0), env.available_actions)
        next_np, is_terminal = env.step(action_idx)
        h_t = h_current
        if is_end_of_life(frame_np, next_np, is_terminal):
            print(f"[analyze] life ended at step {t+1}")
            break
        frame_np = next_np

    summary = summarize(
        per_step,
        n_rounds=cfg.n_perceiver_rounds,
        n_heads=cfg.n_perceiver_heads,
    )
    report(summary)

    if out_json:
        with open(out_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[analyze] wrote summary → {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--out-json",  default=None)
    args = parser.parse_args()
    run(args.checkpoint, max_steps=args.max_steps, out_json=args.out_json)


if __name__ == "__main__":
    main()
