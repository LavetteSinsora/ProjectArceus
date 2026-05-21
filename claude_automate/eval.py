"""Evaluation entrypoint — load a checkpoint and report completion rate.

    cd "Code Repo"
    uv run python claude_automate/eval.py --checkpoint <run_dir>/best.pt
    uv run python claude_automate/eval.py --checkpoint <ckpt> --episodes 50 --stochastic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from claude_automate.framework.config import Config
from claude_automate.framework.env_api import make_arc_env
from claude_automate.framework.networks import ActorCritic
from claude_automate.framework.ppo import collect_episodes
from claude_automate.framework.rewards import RewardComputer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of argmax")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt.get("config", {}))
    device = torch.device("cpu")

    env = make_arc_env(cfg.game_id, cfg.level_index)
    model = ActorCritic(n_actions=env.n_actions, n_colors=cfg.n_colors,
                        hidden_dim=cfg.hidden_dim,
                        frame_size=cfg.frame_size).to(device)
    model.load_state_dict(ckpt["model"])

    reward_computer = RewardComputer(
        cfg, masked_rows=getattr(env, "_MASKED_ROWS", None))

    roll = collect_episodes(env, model, reward_computer, cfg, device,
                            args.episodes, greedy=not args.stochastic)
    comp = [e.completed for e in roll.episodes]
    lens = [e.length for e in roll.episodes]
    mode = "stochastic" if args.stochastic else "greedy"
    print(f"checkpoint: {args.checkpoint}")
    print(f"mode: {mode}  episodes: {args.episodes}")
    print(f"completion rate: {np.mean(comp):.0%}  ({sum(comp)}/{len(comp)})")
    print(f"episode length: min={min(lens)} max={max(lens)} "
          f"mean={np.mean(lens):.1f}")


if __name__ == "__main__":
    main()
