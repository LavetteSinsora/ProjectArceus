"""Eval sub-experiment D — note we eval the EXPLORATION actor (π_e), not π_t.

P2E-only never trains a task actor; π_e is what we deploy.
"""

from __future__ import annotations

import argparse

import torch

from JEPA.experiments.exp_005_dreamer_v3.exp_005_3_plan2explore_only.config import Config
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
    wm, actor, critic, critic_ema, actor_p2e, *_ = load_models(cfg, device)
    blob = torch.load(args.checkpoint, map_location=device)
    wm.load_state_dict(blob["wm"])
    actor_p2e.load_state_dict(blob["actor_p2e"])
    # Use π_e for eval.
    print(evaluate(wm, actor_p2e, lambda: _build_env(cfg), n_episodes=args.episodes, device=device))


if __name__ == "__main__":
    main()
