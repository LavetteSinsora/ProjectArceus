"""
exp_004_0 evaluation — per-env greedy/sampling rollout.

Usage:
    uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.eval \\
        --checkpoint <path> --env ls20 --episodes 50
    uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.eval \\
        --checkpoint <path> --env tu93 --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_004_0_ls20_tu93.config import Config
from JEPA.experiments.exp_004_0_ls20_tu93.models import load_models
from JEPA.experiments.exp_004_0_ls20_tu93.reward_shaping import is_end_of_life
from JEPA.shared.env_wrapper import make_env


def eval_episodes(env_name: str, env, encoder, policy, n_episodes: int,
                  device, sample: bool = True) -> dict:
    """Roll `n_episodes` and report aggregate metrics for the env."""
    lengths = []
    completions = []
    for ep in range(n_episodes):
        frame_np = env.reset()
        h_t = None
        steps = 0
        completed = 0
        while True:
            frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
            with torch.no_grad():
                queries = encoder.perceiver.get_initial_queries(1, device) if h_t is None else h_t.detach()
                h_current, _, _ = encoder(frame_t, queries)
                action_idx, _, _ = policy.act(h_current.squeeze(0), env.available_actions)
            next_np, is_terminal = env.step(action_idx)
            steps += 1
            if is_end_of_life(env_name, frame_np, next_np, is_terminal):
                completed = int(env.level_completed)
                break
            h_t = h_current
            frame_np = next_np
        lengths.append(steps)
        completions.append(completed)
        if (ep + 1) % 10 == 0:
            print(f"  [{env_name}] ep {ep+1}/{n_episodes}: len={steps} completed={completed}")
    return {
        "env": env_name,
        "n_episodes": n_episodes,
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "length_min": int(min(lengths)),
        "length_max": int(max(lengths)),
        "completion_rate": float(np.mean(completions)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", choices=["ls20", "tu93", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[eval] device={device}  checkpoint={args.checkpoint}")
    encoder, state_predictor, action_predictor, action_embeds, policies, baselines = \
        load_models(cfg, device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    state_predictor.load_state_dict(ckpt["state_predictor"])
    action_predictor.load_state_dict(ckpt["action_predictor"])
    for k, sd in ckpt["action_embeds"].items():
        if k in action_embeds:
            action_embeds[k].load_state_dict(sd)
    for k, sd in ckpt["policies"].items():
        if k in policies:
            policies[k].load_state_dict(sd)
    print(f"[eval] loaded from step {ckpt.get('step', '?')}")

    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )

    envs_to_eval = ["ls20", "tu93"] if args.env == "both" else [args.env]
    for env_name in envs_to_eval:
        gid_map = dict(zip(cfg.env_names, cfg.game_ids))
        gid = gid_map[env_name]
        raw = arc.make(gid)
        env = make_env(raw, gid)
        encoder.eval(); policies[env_name].eval()
        result = eval_episodes(env_name, env, encoder, policies[env_name],
                               args.episodes, device)
        encoder.train(); policies[env_name].train()
        print(f"\n[eval:{env_name}] " + ", ".join(f"{k}={v}" for k, v in result.items()))


if __name__ == "__main__":
    main()
