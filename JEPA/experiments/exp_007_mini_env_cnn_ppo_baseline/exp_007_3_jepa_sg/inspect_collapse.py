"""Inspect a collapsed JEPA encoder checkpoint to pin down the mechanism.

Implements the four diagnostics from the investigation plan:

  1. Pre-ReLU vs post-ReLU trunk activation variance + dead-unit count.
  2. Conv weight (||W||_F) vs bias (||b||_2) per-layer, plus per-channel
     output variance on a real batch, comparing collapsed vs fresh-init.
  3. Zero-input trunk vs real-input trunk (input-invariance check).
  4. Effective rank of the trunk output on a large batch.

Usage:
    uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.inspect_collapse
    uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.inspect_collapse \\
        --checkpoint runs/exp_007_3_jepa_sg_20260524_145529/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.model import (
    ActorCritic, one_hot_frame,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.vec_env import VecMiniEnv
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.diagnostics import (
    feature_std, feature_pairwise_l2, feature_effective_rank,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CKPT = (
    REPO_ROOT
    / "JEPA/experiments/exp_007_mini_env_cnn_ppo_baseline/runs"
    / "exp_007_3_jepa_sg_novfclip_gaefixed_20260525_170716/checkpoints/final.pt"
)
DEFAULT_LEVEL = REPO_ROOT / "mini_env/configs/level_01/simple_1_rotation.json"


def collect_batch(level_path: Path, n_envs: int, n_steps: int,
                  seed: int) -> torch.Tensor:
    """Roll a uniform-random policy through VecMiniEnv to gather distinct obs.

    Returns a (B, 32, 32) uint8 tensor with B = n_envs * (n_steps + 1).
    """
    env = VecMiniEnv(str(level_path), n_envs=n_envs, seed=seed)
    rng = np.random.default_rng(seed)
    obs = env.reset_all()
    frames = [obs.copy()]
    for _ in range(n_steps):
        actions = rng.integers(0, 4, size=n_envs).astype(np.int64)
        obs, _r, _d, _i = env.step(actions)
        frames.append(obs.copy())
    arr = np.concatenate(frames, axis=0)
    return torch.from_numpy(arr)  # (B, 32, 32) uint8


def encoder_pre_relu(encoder, x_onehot: torch.Tensor) -> torch.Tensor:
    """Trunk activations *before* the final ReLU. Mirrors CNNEncoder.forward
    minus the outer torch.relu wrap (`shared/model.py:65`)."""
    return encoder.fc(encoder.conv(x_onehot).flatten(1))


@torch.no_grad()
def diagnose(model: ActorCritic, obs: torch.Tensor, label: str) -> None:
    print(f"\n══════════════════════════════════════════")
    print(f"  {label}")
    print(f"══════════════════════════════════════════")

    enc = model.encoder
    x = one_hot_frame(obs)  # (B, 16, 32, 32)
    B = x.shape[0]

    # ── 1. pre-ReLU vs post-ReLU trunk ────────────────────────────────
    pre = encoder_pre_relu(enc, x)               # (B, 256)
    post = torch.relu(pre)                       # (B, 256) — matches enc.forward()
    pre_std = float(pre.std(dim=0).mean().item())
    post_std = float(post.std(dim=0).mean().item())
    # Dead unit = pre-activation max across batch ≤ 0 → ReLU outputs 0 for all
    # inputs at that unit.
    dead_units = int((pre.max(dim=0).values <= 0).sum().item())
    print(f"[1] Pre/post-ReLU trunk activations (B={B}, D=256):")
    print(f"    pre-ReLU  std (across batch, mean over units): {pre_std:.6f}")
    print(f"    post-ReLU std (across batch, mean over units): {post_std:.6f}")
    print(f"    dead ReLU units (pre.max <= 0):                {dead_units}/256")
    if pre_std > 0:
        ratio = post_std / pre_std
        print(f"    post/pre std ratio:                            {ratio:.4f}")

    # ── 2. weight vs bias per conv layer ──────────────────────────────
    print(f"\n[2] Conv weight vs bias magnitudes (and per-channel output std):")
    convs = []
    for m in enc.conv:
        if isinstance(m, torch.nn.Conv2d):
            convs.append(m)
    feat = x
    for i, conv in enumerate(convs):
        w_norm = float(conv.weight.norm().item())
        b_norm = float(conv.bias.norm().item())
        # Run input through this conv (pre-ReLU at this layer).
        out = conv(feat)
        # Per-channel std across (B, H, W) — collapses to 0 if conv outputs a
        # constant tensor (only bias survives).
        per_ch_std = float(out.std(dim=(0, 2, 3)).mean().item())
        per_ch_mean_abs = float(out.mean(dim=(0, 2, 3)).abs().mean().item())
        print(f"    conv{i}:  ||W||_F = {w_norm:7.4f}   ||b||_2 = {b_norm:7.4f}   "
              f"out per-ch std = {per_ch_std:.5f}   |mean| = {per_ch_mean_abs:.5f}")
        # Advance to next conv input: apply ReLU like the real Sequential does.
        feat = F.relu(out)
    # Final Linear
    w_norm = float(enc.fc.weight.norm().item())
    b_norm = float(enc.fc.bias.norm().item())
    print(f"    fc:     ||W||_F = {w_norm:7.4f}   ||b||_2 = {b_norm:7.4f}")

    # ── 3. zero-input vs real-input trunk output ──────────────────────
    print(f"\n[3] Zero-input vs real-input trunk output:")
    zero_obs = torch.zeros(min(8, B), 32, 32, dtype=obs.dtype, device=obs.device)
    h_zero = enc(one_hot_frame(zero_obs))        # (8, 256)
    h_real = enc(x[:8])                          # (8, 256)
    # Note: zero observation = palette index 0 everywhere → one-hot fires
    # channel 0; not literally the zero tensor in input space. Useful baseline
    # nonetheless because it's a single image, so post-ReLU output must repeat.
    zero_const = float(h_zero.std(dim=0).mean().item())  # should be 0 — single
    diff = (h_real.mean(dim=0) - h_zero.mean(dim=0)).norm()
    print(f"    h_zero (palette-0 frame) std across copies:    {zero_const:.6f}")
    print(f"    ||mean(h_real) - mean(h_zero)||_2:             {float(diff):.6f}")
    print(f"    mean ||h_real||_2:                             {float(h_real.norm(dim=-1).mean()):.6f}")
    print(f"    mean ||h_zero||_2:                             {float(h_zero.norm(dim=-1).mean()):.6f}")
    # Real-vs-real pairwise across the full batch
    h_full = enc(x)
    pair_l2 = feature_pairwise_l2(h_full)
    print(f"    pairwise L2 between real-input trunks:         {pair_l2:.6f}")

    # ── 4. effective rank + variance summary ──────────────────────────
    print(f"\n[4] Trunk feature statistics across the full batch (B={B}):")
    print(f"    feat_std:             {feature_std(h_full):.6f}")
    print(f"    feat_pairwise_l2:     {feature_pairwise_l2(h_full):.6f}")
    print(f"    feat_effective_rank:  {feature_effective_rank(h_full):.4f}  (1=rank-1, 256=uniform)")
    print(f"    fraction of trunk units with std≈0 across batch: "
          f"{float((h_full.std(dim=0) < 1e-6).float().mean()):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--level", type=str, default=str(DEFAULT_LEVEL))
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=31,
                    help="random-policy rollout length; B = n_envs * (n_steps + 1)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  update={ckpt.get('update')}   env_step={ckpt.get('env_step')}")

    collapsed = ActorCritic()
    collapsed.load_state_dict(ckpt["model_state_dict"])
    collapsed.eval()

    fresh = ActorCritic()
    fresh.eval()

    print(f"\nCollecting batch: n_envs={args.n_envs} steps={args.n_steps} "
          f"→ B={args.n_envs * (args.n_steps + 1)} obs")
    obs = collect_batch(Path(args.level), args.n_envs, args.n_steps, args.seed)

    diagnose(fresh, obs, "FRESH-INIT ENCODER (baseline)")
    diagnose(collapsed, obs, f"COLLAPSED ENCODER ({ckpt_path.parent.parent.name})")

    print("\n══════════════════════════════════════════")
    print("  Interpretation guide")
    print("══════════════════════════════════════════")
    print("  Lever 1 (dying-ReLU trunk):    high pre-ReLU std, low post-ReLU std,")
    print("                                 many dead units.")
    print("  Lever 2 (bias-dominated conv): conv ||W||_F shrunk vs fresh-init,")
    print("                                 per-channel output std → 0, |mean| > 0.")
    print("  Headline collapse signal:      ||mean(h_real)-mean(h_zero)|| ≈ 0,")
    print("                                 effective rank ≈ 1, pairwise L2 ≈ 0.")


if __name__ == "__main__":
    main()
