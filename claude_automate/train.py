"""Training entrypoint — PPO + SimHash exploration on an ARC-AGI level.

    cd "Code Repo"
    uv run python claude_automate/train.py
    uv run python claude_automate/train.py --total-env-steps 200000
    uv run python claude_automate/train.py --resume <checkpoint.pt>

Writes metrics + checkpoints under claude_automate/experiments/run_<timestamp>/.
Nothing outside claude_automate/ is modified.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from claude_automate.framework.config import Config
from claude_automate.framework.env_api import make_arc_env, make_mini_env
from claude_automate.framework.networks import ActorCritic
from claude_automate.framework.ppo import PPO, collect_episodes
from claude_automate.framework.rewards import RewardComputer

_EXP_DIR = Path(__file__).resolve().parent / "experiments"


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(env, model, reward_computer, cfg, device, n_episodes):
    """Greedy rollout; returns (completion_rate, mean_length)."""
    roll = collect_episodes(env, model, reward_computer, cfg, device,
                            n_episodes, greedy=True)
    comp = np.mean([e.completed for e in roll.episodes])
    length = np.mean([e.length for e in roll.episodes])
    return float(comp), float(length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-env-steps", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--level", type=int, default=None,
                    help="0-indexed level to start each episode on")
    ap.add_argument("--mini-env", type=str, default=None,
                    help="Path to a mini_env level JSON. If set, uses the pure-numpy "
                         "MiniLS20Env instead of the arcengine ARC env.")
    args = ap.parse_args()

    cfg = Config()
    if args.total_env_steps is not None:
        cfg.total_env_steps = args.total_env_steps
    if args.level is not None:
        cfg.level_index = args.level
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)

    if args.mini_env:
        env = make_mini_env(args.mini_env)
        cfg.game_id = f"mini:{Path(args.mini_env).stem}"
    else:
        env = make_arc_env(cfg.game_id, cfg.level_index)
    masked_rows = getattr(env, "_MASKED_ROWS", None)
    reward_computer = RewardComputer(cfg, masked_rows=masked_rows)

    model = ActorCritic(n_actions=env.n_actions, n_colors=cfg.n_colors,
                        hidden_dim=cfg.hidden_dim,
                        frame_size=cfg.frame_size).to(device)
    ppo = PPO(model, cfg, device)

    start_update = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ppo.optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("update", 0)
        print(f"[train] resumed from {args.resume} at update {start_update}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _EXP_DIR / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    metrics_path = run_dir / "metrics.jsonl"
    print(f"[train] device={device}  run_dir={run_dir}")
    print(f"[train] env={cfg.game_id}  level_index={cfg.level_index}  "
          f"n_actions={env.n_actions}")

    env_steps = 0
    update = start_update
    best_completion = 0.0
    t0 = time.time()

    while env_steps < cfg.total_env_steps:
        roll = collect_episodes(env, model, reward_computer, cfg, device,
                                cfg.rollout_episodes, greedy=False)
        env_steps += len(roll)
        update += 1
        stats = ppo.update(roll)

        eps = roll.episodes
        train_comp = float(np.mean([e.completed for e in eps]))
        mean_len = float(np.mean([e.length for e in eps]))
        mean_ret = float(np.mean([e.raw_return for e in eps]))
        mean_nov = float(np.mean([e.novelty_sum for e in eps]))

        record = {
            "update": update, "env_steps": env_steps,
            "train_completion": train_comp, "mean_ep_len": mean_len,
            "mean_return": mean_ret, "mean_novelty": mean_nov,
            "n_distinct_states": reward_computer.global_counter.n_distinct,
            "elapsed_s": round(time.time() - t0, 1),
            **{k: round(v, 4) for k, v in stats.items()},
        }

        if update % cfg.eval_every_updates == 0:
            comp, elen = evaluate(env, model, reward_computer, cfg, device,
                                  cfg.eval_episodes)
            record["eval_completion"] = comp
            record["eval_ep_len"] = elen
            if comp >= best_completion:
                best_completion = comp
                torch.save({"model": model.state_dict(),
                            "optimizer": ppo.optimizer.state_dict(),
                            "config": cfg.to_dict(), "update": update,
                            "eval_completion": comp},
                           run_dir / "best.pt")
            print(f"[eval] update={update} env_steps={env_steps} "
                  f"completion={comp:.0%} ep_len={elen:.0f} "
                  f"best={best_completion:.0%}")

        if update % cfg.checkpoint_every_updates == 0:
            torch.save({"model": model.state_dict(),
                        "optimizer": ppo.optimizer.state_dict(),
                        "config": cfg.to_dict(), "update": update},
                       run_dir / f"step_{env_steps}.pt")

        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        print(f"upd {update:4d} | steps {env_steps:7d} | "
              f"train_comp {train_comp:.0%} | ret {mean_ret:6.2f} | "
              f"len {mean_len:5.1f} | nov {mean_nov:5.2f} | "
              f"states {reward_computer.global_counter.n_distinct:5d} | "
              f"ent {stats['entropy']:.3f} | kl {stats['approx_kl']:.4f}")

    print(f"[train] done. best eval completion = {best_completion:.0%}")


if __name__ == "__main__":
    main()
