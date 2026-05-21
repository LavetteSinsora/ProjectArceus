"""Eval sub-experiment C on a checkpoint."""

from __future__ import annotations

import argparse

import torch

from JEPA.experiments.exp_005_dreamer_v3.exp_005_2_curiosity.config import Config
from JEPA.experiments.exp_005_dreamer_v3.shared.evaluator import evaluate
from JEPA.experiments.exp_005_dreamer_v3.shared.models import load_models
from JEPA.experiments.exp_005_dreamer_v3.shared.trainer import _build_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    wm, actor, *_ = load_models(cfg, device)
    blob = torch.load(args.checkpoint, map_location=device)
    wm.load_state_dict(blob["wm"]); actor.load_state_dict(blob["actor"])
    print(evaluate(wm, actor, lambda: _build_env(cfg), n_episodes=args.episodes, device=device))


if __name__ == "__main__":
    main()
