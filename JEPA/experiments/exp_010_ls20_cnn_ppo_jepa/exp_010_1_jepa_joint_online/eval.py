"""Evaluate an exp_010_1 checkpoint (same as exp_010_0; encoder differs only in
how it was trained, not in architecture)."""

import argparse
import json

import torch

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.evaluator import evaluate
from .config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = Config(**ck["config"]) if isinstance(ck["config"], dict) else ck["config"]
    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    eval_envs = VecLS20Env(cfg.env_name, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=cfg.seed + 999)
    print(json.dumps(evaluate(model, eval_envs, device, args.episodes, args.greedy), indent=2))


if __name__ == "__main__":
    main()
