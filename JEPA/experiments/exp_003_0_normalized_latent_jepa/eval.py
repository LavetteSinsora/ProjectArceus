"""
Exp-003 evaluation: runs N episodes with the trained policy and reports completion rate.

Usage:
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.eval
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.eval --checkpoint checkpoints/step_050000.pt --episodes 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_0_normalized_latent_jepa.config import Config
from JEPA.experiments.exp_003_0_normalized_latent_jepa.models import load_models
from JEPA.experiments.exp_003_0_normalized_latent_jepa.reward_shaping import is_end_of_life
from JEPA.shared.env_wrapper import LS20Env


def eval_policy(cfg: Config, checkpoint_path: str, n_episodes: int, device: torch.device):
    encoder, predictor, action_embed, policy, _ = load_models(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    policy.load_state_dict(ckpt["policy"])
    step = ckpt.get("step", 0)
    print(f"[eval] Loaded checkpoint at step {step}")
    encoder.eval(); policy.eval()

    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)

    completions = 0
    for ep in range(n_episodes):
        frame_np = env.reset()
        h_t = None
        ep_steps = 0

        while True:
            frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
            with torch.no_grad():
                queries = encoder.perceiver.get_initial_queries(1, device) if h_t is None else h_t
                h_current, _, _ = encoder(frame_t, queries)
                action_idx, _, _ = policy.act(h_current.squeeze(0), env.available_actions)

            next_np, is_terminal = env.step(action_idx)
            life_end = is_end_of_life(frame_np, next_np, is_terminal)
            h_t = h_current
            ep_steps += 1

            if life_end:
                if env.level_completed:
                    completions += 1
                break
            frame_np = next_np

        print(f"  Episode {ep+1}/{n_episodes}: steps={ep_steps}  completed={env.level_completed}")

    rate = completions / n_episodes
    print(f"\n[eval] Completion rate: {completions}/{n_episodes} = {rate:.1%}")
    return rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config()

    ckpt_dir = Path(__file__).parent / "checkpoints"
    if args.checkpoint is None:
        checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
        if not checkpoints:
            print("[eval] No checkpoints found")
            return
        checkpoint_path = str(checkpoints[-1])
    else:
        checkpoint_path = args.checkpoint

    eval_policy(cfg, checkpoint_path, args.episodes, device)


if __name__ == "__main__":
    main()
