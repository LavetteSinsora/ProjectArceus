"""
analyze_residual_imbalance.py — Localize why the reasoning token h_t is stale.

Hypothesis: ||h|| ≈ √128, ||attn_out|| ≈ O(1) (because z is L2-normalized per
patch), so the cross-attention update is washed out by the residual in
    h_new = LayerNorm(h + attn_out).

At every step of a rollout this script captures the per-step quantities listed
in the plan and writes one CSV per checkpoint to results/residual_imbalance/.
A stdout summary table aggregates over all steps for each checkpoint.

Usage:
    cd "Code Repo" && uv run python -m \
        JEPA.experiments.exp_001_vit_jepa_baseline.analyze_residual_imbalance \
        --episodes 3

Optional flags:
    --checkpoints step_005000.pt,step_100000.pt,...   override sweep
    --attn-gain 4.0                                   inject a learnable-style
                                                      gain on attn_out at infer-
                                                      time (Phase B / C check)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent  # exp_001 → experiments → JEPA → Code Repo
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_001_vit_jepa_baseline import config as _exp_config_mod
from JEPA.experiments.exp_001_vit_jepa_baseline.config import Config
from JEPA.experiments.exp_001_vit_jepa_baseline.models.encoder import Encoder
from JEPA.experiments.exp_001_vit_jepa_baseline.models.policy import PolicyNetwork
from JEPA.shared.env_wrapper import LS20Env

# Old checkpoints (pre-refactor) were pickled with a top-level `config` module;
# alias the package config module so torch.load(weights_only=False) can resolve
# `config.Config` at unpickle time.
sys.modules.setdefault("config", _exp_config_mod)


_DEFAULT_CHECKPOINTS = [
    "step_005000.pt",
    "step_100000.pt",
    "step_500000.pt",
    "step_1000000.pt",
    "step_2000000.pt",
    "step_5055000.pt",
]


# ──────────────────────────────────────────────────────────────────────────
# Geometric helpers
# ──────────────────────────────────────────────────────────────────────────

def _pairwise_cos_mean(X: torch.Tensor) -> float:
    """Mean off-diagonal pairwise cosine similarity over the 16 patch dim."""
    Xn = F.normalize(X, dim=-1)
    C = Xn @ Xn.T          # (N, N)
    N = C.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=C.device)
    return C[mask].mean().item()


def _row_jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence between two prob vectors (natural log)."""
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum() + (q * (q / m).log()).sum()).item()


def _entropy(p: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum().item()


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    return (p * (p / q).log()).sum().item()


# ──────────────────────────────────────────────────────────────────────────
# Instrumented forward — mirrors policy._cross_attn_update + action_head,
# returning every intermediate we want to log.
# ──────────────────────────────────────────────────────────────────────────

def _instrumented_step(
    policy: PolicyNetwork,
    h: torch.Tensor,
    z: torch.Tensor,
    gain_override: float = 1.0,
):
    """Mirrors policy._cross_attn_update but exposes every intermediate.

    The effective gain on attn_out is `policy.attn_gain * gain_override`. So
    gain_override=1.0 reproduces the policy's native behavior (which is 1.0 for
    legacy checkpoints loaded via load_state_dict_with_legacy_gain). Setting it
    to anything else is a hypothesis test, not a faithful replay.
    """
    Q = policy.q_proj(h).unsqueeze(0)           # (1, d)
    K = policy.k_proj(z)                         # (16, d)
    V = policy.v_proj(z)                         # (16, d)

    raw_logits = (Q @ K.T) * policy.scale        # (1, 16)
    attn_w = F.softmax(raw_logits, dim=-1)       # (1, 16)
    pooled = attn_w @ V                          # (1, d)
    attn_out = policy.out_proj(pooled.squeeze(0))  # (d,)

    update = policy.attn_gain * attn_out * gain_override
    h_mid = policy.norm1(h + update)
    ffn_out = policy.ffn(h_mid)
    h_new = policy.norm2(h_mid + ffn_out)

    logits = policy.action_head(h_new)           # (n_actions,)
    probs = F.softmax(logits, dim=-1)

    return {
        "Q": Q.squeeze(0),
        "K": K,
        "V": V,
        "attn_w": attn_w.squeeze(0),
        "attn_out": attn_out,
        "update": update,
        "ffn_out": ffn_out,
        "h_in": h,
        "h_mid": h_mid,
        "h_new": h_new,
        "logits": logits,
        "probs": probs,
    }


# ──────────────────────────────────────────────────────────────────────────
# Single-episode rollout with full per-step logging
# ──────────────────────────────────────────────────────────────────────────

def rollout_episode(
    env: LS20Env,
    encoder: Encoder,
    policy: PolicyNetwork,
    device: torch.device,
    attn_gain: float,
    ep_idx: int,
) -> list[dict]:
    frame_np = env.reset()
    h = policy.initial_state().to(device)

    rows: list[dict] = []
    prev = None

    while True:
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
        with torch.no_grad():
            z = encoder(frame_t).squeeze(0)                  # (16, d)
            cap = _instrumented_step(policy, h, z, attn_gain)

        # Sample action from probs (matches policy.act behavior, no mask here
        # — we want pure geometry diagnostics).
        action_idx = int(torch.distributions.Categorical(cap["probs"]).sample().item())

        h_norm        = cap["h_in"].norm().item()
        attn_out_norm = cap["attn_out"].norm().item()
        update_norm   = cap["update"].norm().item()
        ffn_norm      = cap["ffn_out"].norm().item()
        h_new_norm    = cap["h_new"].norm().item()

        # Pairwise geometry across 16 patches
        V_pw = _pairwise_cos_mean(cap["V"])
        K_pw = _pairwise_cos_mean(cap["K"])
        z_pw = _pairwise_cos_mean(z)

        # Attention diagnostics
        attn_w = cap["attn_w"]                                # (16,)
        attn_H = _entropy(attn_w)

        # Comparison vs previous step
        if prev is None:
            cos_h_prev    = float("nan")
            l2_h_prev     = float("nan")
            cos_q_prev    = float("nan")
            attn_jsd_prev = float("nan")
            logits_diff   = float("nan")
            kl_pi_prev    = float("nan")
        else:
            # Compare OUTPUT-to-OUTPUT across consecutive cross-attn calls. (Comparing
            # h_in[t] to h_new[t-1] is degenerate — they're identical by construction.)
            cos_h_prev = F.cosine_similarity(
                cap["h_new"].unsqueeze(0), prev["h_new"].unsqueeze(0)
            ).item()
            l2_h_prev = (cap["h_new"] - prev["h_new"]).norm().item()
            cos_q_prev = F.cosine_similarity(
                cap["Q"].unsqueeze(0), prev["Q"].unsqueeze(0)
            ).item()
            attn_jsd_prev = _row_jsd(cap["attn_w"], prev["attn_w"])
            logits_diff = (cap["logits"] - prev["logits"]).abs().max().item()
            kl_pi_prev = _kl(cap["probs"], prev["probs"])

        rows.append({
            "episode": ep_idx,
            "step": len(rows),
            "h_norm": h_norm,
            "attn_out_norm": attn_out_norm,
            "update_norm": update_norm,
            "ffn_norm": ffn_norm,
            "h_new_norm": h_new_norm,
            "ratio_update_h": update_norm / max(h_norm, 1e-9),
            "cos_h_prev": cos_h_prev,
            "l2_h_prev": l2_h_prev,
            "cos_q_prev": cos_q_prev,
            "V_pw_cos": V_pw,
            "K_pw_cos": K_pw,
            "z_pw_cos": z_pw,
            "attn_entropy": attn_H,
            "attn_jsd_prev": attn_jsd_prev,
            "logits_max_diff": logits_diff,
            "kl_pi_prev": kl_pi_prev,
            "action": action_idx,
        })

        prev = cap  # keep tensors for next-step deltas
        h = cap["h_new"].detach()

        next_np, is_terminal = env.step(action_idx)
        if is_terminal:
            break
        frame_np = next_np

    return rows


# ──────────────────────────────────────────────────────────────────────────
# Per-checkpoint orchestration
# ──────────────────────────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_raw = ckpt.get("config", {})
    cfg = Config(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw

    encoder = Encoder(
        cfg.d_model, cfg.d_color, cfg.n_heads, cfg.n_blocks, cfg.ffn_dim, cfg.patch_size
    ).to(device)
    policy = PolicyNetwork(cfg.d_model, cfg.n_actions).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    # Legacy checkpoints lack `attn_gain`; the helper forces it to 1.0 to
    # reproduce pre-fix behavior. New checkpoints carry the learned gain.
    policy.load_state_dict_with_legacy_gain(ckpt["policy"])
    encoder.eval()
    policy.eval()
    return cfg, encoder, policy


def run_checkpoint(
    ckpt_path: Path,
    out_dir: Path,
    episodes: int,
    attn_gain: float,
    device: torch.device,
) -> dict:
    cfg, encoder, policy = _load_checkpoint(ckpt_path, device)

    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    env = LS20Env(arc.make(cfg.game_id))

    all_rows: list[dict] = []
    for ep in range(episodes):
        rows = rollout_episode(env, encoder, policy, device, attn_gain, ep)
        all_rows.extend(rows)
        print(f"  ep {ep+1}/{episodes}: {len(rows)} steps")

    # CSV dump
    csv_path = out_dir / f"{ckpt_path.stem}.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

    # Aggregate (skip first step of each episode for delta metrics — they're NaN)
    def _mean(key: str, drop_first: bool) -> float:
        if not all_rows:
            return float("nan")
        vals = [r[key] for r in all_rows if not (drop_first and r["step"] == 0)]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    # ratio_update_h is meaningless at step 0 (h is initialized to zero).
    return {
        "checkpoint": ckpt_path.name,
        "n_steps": len(all_rows),
        "h_norm": _mean("h_norm", True),
        "attn_out_norm": _mean("attn_out_norm", False),
        "ratio_update_h": _mean("ratio_update_h", True),
        "cos_h_prev": _mean("cos_h_prev", True),
        "l2_h_prev": _mean("l2_h_prev", True),
        "cos_q_prev": _mean("cos_q_prev", True),
        "V_pw_cos": _mean("V_pw_cos", False),
        "K_pw_cos": _mean("K_pw_cos", False),
        "z_pw_cos": _mean("z_pw_cos", False),
        "attn_entropy": _mean("attn_entropy", False),
        "attn_jsd_prev": _mean("attn_jsd_prev", True),
        "logits_max_diff": _mean("logits_max_diff", True),
        "kl_pi_prev": _mean("kl_pi_prev", True),
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--attn-gain", type=float, default=1.0,
                   help="Multiplier on attn_out before residual add. 1.0 = native model.")
    p.add_argument("--checkpoints", type=str, default=None,
                   help="Comma-separated list of checkpoint filenames in checkpoints/")
    p.add_argument("--out", type=str, default=None,
                   help="Output directory (default: results/residual_imbalance[/gain_X])")
    args = p.parse_args()

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_names = (
        args.checkpoints.split(",") if args.checkpoints else _DEFAULT_CHECKPOINTS
    )
    ckpt_paths = []
    for n in ckpt_names:
        path = ckpt_dir / n.strip()
        if not path.exists():
            print(f"  [skip] missing: {path.name}")
            continue
        ckpt_paths.append(path)

    if not ckpt_paths:
        print("[analyze] No checkpoints to evaluate.")
        return

    if args.out:
        out_dir = Path(args.out)
    else:
        base = Path(__file__).parent / "results" / "residual_imbalance"
        if args.attn_gain != 1.0:
            out_dir = base / f"gain_{args.attn_gain:g}"
        else:
            out_dir = base
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] device={device}  episodes/ckpt={args.episodes}  "
          f"attn_gain={args.attn_gain}  out={out_dir}")
    print(f"[analyze] checkpoints: {[p.name for p in ckpt_paths]}\n")

    summaries: list[dict] = []
    for ckpt in ckpt_paths:
        print(f"[ckpt] {ckpt.name}")
        s = run_checkpoint(ckpt, out_dir, args.episodes, args.attn_gain, device)
        summaries.append(s)
        print(f"  → {s['n_steps']} steps, "
              f"ratio_update_h={s['ratio_update_h']:.3f}, "
              f"cos_h_prev={s['cos_h_prev']:.3f}, "
              f"V_pw_cos={s['V_pw_cos']:.3f}\n")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "═" * 110)
    print(f"SUMMARY (episodes={args.episodes}, attn_gain={args.attn_gain})")
    print("═" * 110)
    hdr = (
        f"{'ckpt':<20}{'steps':>7}{'‖h‖':>8}{'‖Δ‖':>8}"
        f"{'Δ/h':>7}{'cos hₜ':>9}{'cos Qₜ':>9}"
        f"{'V pw':>8}{'K pw':>8}{'z pw':>8}"
        f"{'H(att)':>9}{'JSDₜ':>8}{'KL πₜ':>9}{'|Δlog|∞':>10}"
    )
    print(hdr)
    print("-" * 110)
    for s in summaries:
        print(
            f"{s['checkpoint']:<20}{s['n_steps']:>7d}"
            f"{s['h_norm']:>8.2f}{s['attn_out_norm']:>8.2f}"
            f"{s['ratio_update_h']:>7.3f}"
            f"{s['cos_h_prev']:>9.3f}{s['cos_q_prev']:>9.3f}"
            f"{s['V_pw_cos']:>8.3f}{s['K_pw_cos']:>8.3f}{s['z_pw_cos']:>8.3f}"
            f"{s['attn_entropy']:>9.3f}{s['attn_jsd_prev']:>8.4f}"
            f"{s['kl_pi_prev']:>9.4f}{s['logits_max_diff']:>10.4f}"
        )

    # CSV of summaries
    sum_csv = out_dir / "_summary.csv"
    with sum_csv.open("w", newline="") as f:
        if summaries:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
    print(f"\n[analyze] Summary CSV → {sum_csv}")

    # ── Hypothesis verdict ────────────────────────────────────────────────
    print("\n" + "═" * 110)
    print("HYPOTHESIS CHECK (residual-magnitude washout)")
    print("═" * 110)
    if not summaries:
        print("  (no data)")
        return
    bad_ratio  = all(s["ratio_update_h"]   < 0.20 for s in summaries)
    stale_h    = all(s["cos_h_prev"]       > 0.90 for s in summaries)
    V_alive    = all(s["V_pw_cos"]         < 0.95 for s in summaries)
    attn_alive = all(s["attn_jsd_prev"]    > 0.01 for s in summaries)
    print(f"  ‖update‖/‖h‖ < 0.20 across all ckpts:  {bad_ratio}")
    print(f"  cos(hₜ, hₜ₋₁) > 0.90 across all ckpts: {stale_h}")
    print(f"  V not collapsed (V_pw_cos<0.95):       {V_alive}")
    print(f"  Attention does vary (JSD>0.01):        {attn_alive}")
    if bad_ratio and stale_h and V_alive and attn_alive:
        print("  → Residual-magnitude washout CONFIRMED. Proceed to Phase B.")
    elif not V_alive:
        print("  → V-collapse pathway. Different fix needed (re-examine encoder).")
    else:
        print("  → Mixed signal; inspect per-checkpoint CSVs before deciding.")


if __name__ == "__main__":
    main()
