"""
inspect_policy.py — Diagnostic rollout of a trained JEPA policy checkpoint.

Runs the trained policy in OFFLINE mode and reports:
  - Full action probability distribution at every step (not just the sampled action)
  - Per-episode entropy trace and mean
  - Patch exploration heatmap (which of 16 patches changed vs initial frame)
  - Fine-grid exploration (64-cell 8×8 grid, player position via pixel diff)
  - Qualitative diagnosis: is the policy near-uniform?

Usage:
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.inspect_policy
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.inspect_policy --episodes 10

The action indices map to GameAction.ACTION1-4 (exact direction depends on the game).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent  # exp_001 → experiments → JEPA → Code Repo
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_001_vit_jepa_baseline.config import Config
from JEPA.experiments.exp_001_vit_jepa_baseline.models.encoder import Encoder
from JEPA.experiments.exp_001_vit_jepa_baseline.models.policy import PolicyNetwork
from JEPA.shared.env_wrapper import LS20Env

_STEP_COUNTER_ROWS = slice(61, 63)   # rows 61-62 always change; mask for exploration tracking


def _action_probs(policy: PolicyNetwork, h: torch.Tensor, z: torch.Tensor):
    """Return (probs array, entropy float, h_new, action_idx) in one forward pass."""
    h_new = policy._cross_attn_update(h, z)
    logits = policy.action_head(h_new)
    probs  = F.softmax(logits, dim=-1)
    dist   = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    H      = dist.entropy().item()
    return probs.detach().cpu().numpy(), H, h_new.detach(), action


def _patch_diff_16(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """(64,64)→(64,64) diff → (16,) mean abs change per 16×16 patch."""
    diff = np.abs(frame_b.astype(np.float32) - frame_a.astype(np.float32))
    diff[_STEP_COUNTER_ROWS, :] = 0     # mask step-counter rows
    return diff.reshape(4, 16, 4, 16).mean(axis=(1, 3)).flatten()


def _fine_grid_cell(frame_a: np.ndarray, frame_b: np.ndarray):
    """
    Return the 8×8-cell index (0-63) with the most pixel change, ignoring counter rows.
    The 64×64 frame → 8×8 grid of 8×8 cells.  Returns None if no change.
    """
    diff = np.abs(frame_b.astype(np.float32) - frame_a.astype(np.float32))
    diff[_STEP_COUNTER_ROWS, :] = 0
    cell_diff = diff.reshape(8, 8, 8, 8).sum(axis=(1, 3))   # (8, 8)
    if cell_diff.max() < 2.0:
        return None
    r, c = np.unravel_index(cell_diff.argmax(), cell_diff.shape)
    return int(r * 8 + c)


def _patch_heatmap(visited: set, title: str) -> str:
    """ASCII 4×4 heatmap of visited 16×16 patches."""
    lines = [f"  {title}"]
    for r in range(4):
        row = "    "
        for c in range(4):
            row += "X " if (r, c) in visited else ". "
        lines.append(row)
    return "\n".join(lines)


def _fine_grid_heatmap(counts: np.ndarray) -> str:
    """ASCII 8×8 heatmap with visit-count shading."""
    mx = counts.max()
    if mx == 0:
        return "  (no movement detected)"
    symbols = " ░▒▓█"
    lines = ["  Fine grid (8×8, each cell = 8×8 px):"]
    for r in range(8):
        row = "    "
        for c in range(8):
            v = counts[r, c]
            idx = min(4, int(v / mx * 4 + 0.5)) if mx > 0 else 0
            row += symbols[idx]
        lines.append(row)
    return "\n".join(lines)


def run_episode(env: LS20Env, encoder, policy, device):
    """Run one full episode. Returns a stats dict."""
    frame_np     = env.reset()
    h            = policy.initial_state().to(device)
    initial_frame = frame_np.copy()
    prev_frame    = frame_np.copy()

    probs_history = []
    entropies     = []
    visited_patches: set = set()
    fine_counts   = np.zeros((8, 8), dtype=np.int32)

    while True:
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
        with torch.no_grad():
            z_t = encoder(frame_t).squeeze(0)
            probs, H, h, action_idx = _action_probs(policy, h, z_t)

        probs_history.append(probs)
        entropies.append(H)

        next_np, is_terminal = env.step(action_idx)

        # Patch-level exploration: which patches changed vs initial frame
        patch_d = _patch_diff_16(initial_frame, next_np)
        for i, d in enumerate(patch_d):
            if d > 2.0:
                visited_patches.add((i // 4, i % 4))

        # Fine-grid: where did the player move this step?
        cell = _fine_grid_cell(prev_frame, next_np)
        if cell is not None:
            fine_counts[cell // 8, cell % 8] += 1

        prev_frame = next_np.copy()
        frame_np   = next_np
        if is_terminal:
            break

    mean_probs = np.stack(probs_history).mean(axis=0)
    return {
        "steps"           : len(probs_history),
        "completed"       : env.level_completed,
        "mean_probs"      : mean_probs,
        "mean_H"          : float(np.mean(entropies)),
        "min_H"           : float(np.min(entropies)),
        "visited_patches" : visited_patches,
        "pct_patches"     : len(visited_patches) / 16.0,
        "fine_counts"     : fine_counts,
        "pct_fine_cells"  : (fine_counts > 0).sum() / 64.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect policy from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (default: latest in JEPA/checkpoints/)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    args = parser.parse_args()

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    # ── Checkpoint ─────────────────────────────────────────────────────────
    ckpt_dir = Path(__file__).parent / "checkpoints"
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
        if not checkpoints:
            print("[inspect] No checkpoints found in checkpoints/")
            return
        ckpt_path = checkpoints[-1]

    print(f"[inspect] Loading: {ckpt_path.name}  device={device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[inspect] Checkpoint step: {ckpt.get('step', '?')}")
    cfg_raw = ckpt.get("config", {})
    cfg = Config(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw

    # ── Models ─────────────────────────────────────────────────────────────
    encoder = Encoder(
        cfg.d_model, cfg.d_color, cfg.n_heads, cfg.n_blocks, cfg.ffn_dim, cfg.patch_size
    ).to(device)
    policy = PolicyNetwork(cfg.d_model, cfg.n_actions).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    policy.load_state_dict(ckpt["policy"])
    encoder.eval(); policy.eval()

    # ── Environment ─────────────────────────────────────────────────────────
    from arc_agi import Arcade, OperationMode
    repo_root = _repo_root
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root / "environment_files"),
    )
    env = LS20Env(arc.make(cfg.game_id))

    # ── Rollout ─────────────────────────────────────────────────────────────
    all_probs      = []
    all_H          = []
    all_pct_patches = []
    all_pct_fine   = []
    combined_fine  = np.zeros((8, 8), dtype=np.int32)

    print(f"\n[inspect] Running {args.episodes} episodes...\n")
    print("  Action indices: 0=ACTION1  1=ACTION2  2=ACTION3  3=ACTION4")
    print("─" * 65)

    for ep in range(args.episodes):
        stats = run_episode(env, encoder, policy, device)
        all_probs.append(stats["mean_probs"])
        all_H.append(stats["mean_H"])
        all_pct_patches.append(stats["pct_patches"])
        all_pct_fine.append(stats["pct_fine_cells"])
        combined_fine += stats["fine_counts"]

        p = stats["mean_probs"]
        print(f"\nEp {ep+1:2d}  steps={stats['steps']:3d}  "
              f"completed={'YES ✓' if stats['completed'] else 'no'}")
        print(f"  Probs: A0={p[0]:.3f}  A1={p[1]:.3f}  A2={p[2]:.3f}  A3={p[3]:.3f}  "
              f"max-min={p.max()-p.min():.3f}")
        print(f"  H mean={stats['mean_H']:.4f}  H min={stats['min_H']:.4f}  "
              f"(uniform H={1.386:.4f})")
        print(f"  Patch exploration: {len(stats['visited_patches'])}/16 "
              f"({stats['pct_patches']:.0%})  "
              f"Fine-cell exploration: {stats['pct_fine_cells']:.0%} of 64 cells")
        print(_patch_heatmap(stats["visited_patches"], "16-patch grid (X=visited):"))

    # ── Summary ─────────────────────────────────────────────────────────────
    mean_probs_all = np.stack(all_probs).mean(axis=0)
    diff = mean_probs_all.max() - mean_probs_all.min()
    print("\n" + "═" * 65)
    print("SUMMARY OVER ALL EPISODES")
    print("═" * 65)
    p = mean_probs_all
    print(f"  Mean probs:  A0={p[0]:.4f}  A1={p[1]:.4f}  A2={p[2]:.4f}  A3={p[3]:.4f}")
    print(f"  Max-min gap: {diff:.4f}  (near-uniform if < 0.02)")
    print(f"  Mean H(π):   {np.mean(all_H):.4f}  (uniform = 1.3863)")
    print(f"  Mean patches explored / ep: {np.mean(all_pct_patches):.0%}")
    print(f"  Mean fine cells / ep:       {np.mean(all_pct_fine):.0%}")
    print()
    print(_fine_grid_heatmap(combined_fine))
    print()

    # ── Diagnosis ────────────────────────────────────────────────────────────
    if diff < 0.02:
        print("  DIAGNOSIS: NEAR-UNIFORM policy (max-min < 0.02)")
        print("  The policy assigns equal probability to all 4 actions at every state.")
        print("  This means REINFORCE gradient ≈ 0: the reward signal carries no")
        print("  consistent advantage to push any action above the others.")
        print()
        print("  Root cause candidates:")
        print("  A) Intrinsic reward too uniform — predictor has learned all visited")
        print("     transitions, so reward ≈ constant regardless of action.")
        print("     Fix: add a non-decaying signal (e.g. count-based exploration,")
        print("     step-counter penalty, or goal-proximity if detectable).")
        print("  B) Entropy lambda (cfg.policy_entropy_lambda=0.10) is too high,")
        print("     pulling the policy toward uniform distribution faster than")
        print("     REINFORCE can push it toward better actions.")
        print("     Fix: reduce lambda to 0.01–0.03.")
        print("  C) Policy batch size (64 steps) too small for the variance of the")
        print("     REINFORCE gradient with near-zero advantages.")
    elif diff < 0.05:
        print(f"  DIAGNOSIS: Weak preference (max-min = {diff:.3f})")
        print(f"  Policy slightly prefers A{p.argmax()} over A{p.argmin()}.")
        print("  May improve with more training steps.")
    else:
        print(f"  DIAGNOSIS: Clear preference (max-min = {diff:.3f})")
        print(f"  Policy consistently prefers A{p.argmax()} (p={p.max():.3f}) over "
              f"A{p.argmin()} (p={p.min():.3f}).")


if __name__ == "__main__":
    main()
