"""
run_online.py — Run the trained JEPA policy against the live ARC-AGI API.

Each run creates a server-side scorecard with a replay URL you can view in a browser.
The episode plays out step-by-step via the remote API.

Prerequisites:
    Add your API key to the project .env file:
        echo "ARC_API_KEY=your_key_here" >> .env
    Or set the environment variable directly:
        export ARC_API_KEY=your_key_here

Usage:
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.run_online
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.run_online --episodes 3

Output:
    Prints the scorecard replay URL after each run.
    The replay shows the exact pixel-level playthrough on the ARC-AGI website.
"""

import argparse
import os
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


def _load_dotenv():
    """Load ARC_API_KEY from .env file in the repo root if not already set."""
    if os.environ.get("ARC_API_KEY"):
        return
    env_path = _repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("ARC_API_KEY="):
            os.environ["ARC_API_KEY"] = line.split("=", 1)[1].strip()
            return


def _action_probs(policy: PolicyNetwork, h: torch.Tensor, z: torch.Tensor):
    """Single forward pass returning (probs, entropy, h_new, action_idx)."""
    h_new  = policy._cross_attn_update(h, z)
    logits = policy.action_head(h_new)
    probs  = F.softmax(logits, dim=-1)
    dist   = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    H      = dist.entropy().item()
    return probs.detach().cpu().numpy(), H, h_new.detach(), action


def run_online_episode(arc, cfg, encoder, policy, device, scorecard_id, ep_num):
    """Run one episode against the remote API under the given scorecard."""
    from arcengine import GameState

    raw_env = arc.make(cfg.game_id, scorecard_id=scorecard_id)
    env     = LS20Env(raw_env)

    frame_np  = env.reset()
    h         = policy.initial_state().to(device)
    step      = 0
    probs_all = []
    entropies = []

    while True:
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
        with torch.no_grad():
            z_t = encoder(frame_t).squeeze(0)
            probs, H, h, action_idx = _action_probs(policy, h, z_t)

        probs_all.append(probs)
        entropies.append(H)
        step += 1

        next_np, is_terminal = env.step(action_idx)
        frame_np = next_np

        if is_terminal:
            break

    mean_p = np.stack(probs_all).mean(axis=0)
    print(f"  Ep {ep_num}: steps={step}  completed={'YES ✓' if env.level_completed else 'no'}  "
          f"A0={mean_p[0]:.3f} A1={mean_p[1]:.3f} A2={mean_p[2]:.3f} A3={mean_p[3]:.3f}  "
          f"H={np.mean(entropies):.4f}")
    return env.level_completed


def main():
    parser = argparse.ArgumentParser(description="Run JEPA policy in ARC-AGI online mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (default: latest in JEPA/checkpoints/)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to play (default: 1)")
    parser.add_argument("--tags", type=str, default="jepa-policy",
                        help="Tags for the scorecard (default: jepa-policy)")
    args = parser.parse_args()

    _load_dotenv()
    if not os.environ.get("ARC_API_KEY"):
        print("[run_online] ERROR: ARC_API_KEY not set.")
        print("  Add it to the repo .env file:  ARC_API_KEY=your_key_here")
        print("  Or set it directly:            export ARC_API_KEY=your_key_here")
        sys.exit(1)

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
            print("[run_online] No checkpoints found in checkpoints/")
            sys.exit(1)
        ckpt_path = checkpoints[-1]

    print(f"[run_online] Checkpoint: {ckpt_path.name}  device={device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[run_online] Checkpoint step: {ckpt.get('step', '?')}")
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

    # ── ARC-AGI API ─────────────────────────────────────────────────────────
    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        arc_api_key=os.environ["ARC_API_KEY"],
        operation_mode=OperationMode.ONLINE,
    )

    tags = [t.strip() for t in args.tags.split(",")]
    scorecard_id = arc.open_scorecard(tags=tags)
    replay_url   = f"https://three.arcprize.org/scorecards/{scorecard_id}"
    print(f"\n[run_online] Scorecard opened: {scorecard_id}")
    print(f"[run_online] Replay URL: {replay_url}\n")

    completions = 0
    for ep in range(1, args.episodes + 1):
        done = run_online_episode(arc, cfg, encoder, policy, device, scorecard_id, ep)
        if done:
            completions += 1

    result = arc.close_scorecard(scorecard_id)
    print(f"\n[run_online] Scorecard closed.")
    print(f"  Final score:    {result.score:.2f}%")
    print(f"  Levels completed: {result.total_levels_completed}")
    print(f"  Completions in this run: {completions}/{args.episodes}")
    print(f"\n  View replay: {replay_url}")


if __name__ == "__main__":
    main()
