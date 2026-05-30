"""Standalone eval — load checkpoint and run N episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .device import pick_device
from .model import ActorCritic
from .metrics import run_eval_episodes


def evaluate(checkpoint_path: str, level_path: str | None = None,
             n_episodes: int = 200, sample: bool = True) -> dict:
    device = pick_device()
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    if level_path is None:
        level_path = cfg["level_path"]

    model = ActorCritic().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    metrics = run_eval_episodes(level_path, model, device=device,
                                  n_episodes=n_episodes, sample=sample)
    metrics["checkpoint"] = checkpoint_path
    metrics["sample"] = sample
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--level", default=None)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--greedy", action="store_true", help="argmax instead of sample")
    args = p.parse_args()

    metrics = evaluate(args.checkpoint, args.level, args.episodes, sample=not args.greedy)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
