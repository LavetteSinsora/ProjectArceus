"""Read-only ablation driver for exp_013_1 (does NOT modify method/harness code).

Imports the shipped Config + train() and only OVERRIDES config fields the System Card
already exposes as knobs (c_entropy, leak, max_env_steps, seed). Used to test two
hypotheses cheaply (<=120k env steps each):

  * --c-entropy 0.05  : does staying closer to uniform (higher entropy coef) prevent the
    entropy collapse seen in the censored seeds and recover a first reward?
  * --leak 0.0        : does the leak matter for FIRST-reward, or only for not-stalling?
    (vanilla RND control.)

Usage:
  uv run python JEPA/experiments/exp_013_headline_experiment/probes/ablation_driver.py \
      --seed 0 --max-env-steps 120000 --c-entropy 0.05
"""
from __future__ import annotations

import argparse
import dataclasses

from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.config import Config
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-env-steps", type=int, default=120000)
    ap.add_argument("--c-entropy", type=float, default=None)
    ap.add_argument("--leak", type=float, default=None)
    ap.add_argument("--tag", default="abl")
    args = ap.parse_args()

    cfg = Config(game=args.game, level_index=args.level, seed=args.seed)
    cfg.max_env_steps = args.max_env_steps
    if args.c_entropy is not None:
        cfg.c_entropy = args.c_entropy
    if args.leak is not None:
        cfg.leak = args.leak
    # tag the exp_dir's run name via game-agnostic field: append to exp_dir? No —
    # keep runs separable by leaving a marker in config; exp_name drives the folder.
    print(f"[ablation] {args.tag}: c_entropy={cfg.c_entropy} leak={cfg.leak} "
          f"seed={cfg.seed} cap={cfg.max_env_steps}")
    res = train(cfg)
    print(f"[ablation] RESULT {args.tag}: solved={res['solved']} "
          f"frr={res['env_steps_to_first_reward']} freeze={res['phi_freeze_step']}")


if __name__ == "__main__":
    main()
