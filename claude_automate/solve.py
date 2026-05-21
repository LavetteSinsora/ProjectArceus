"""solve.py — Go-Explore search + policy distillation entrypoint.

The framework's tool for deep-exploration levels (where PPO+novelty detaches).
Runs Go-Explore to find a completing trajectory, distills it into the
ActorCritic policy, and evaluates.

    cd "Code Repo"
    uv run python claude_automate/solve.py --level 1                 # LS20 L2
    uv run python claude_automate/solve.py --level 0                 # LS20 L1
    uv run python claude_automate/solve.py --level 1 --max-env-steps 800000

Writes the solution trajectory, the distilled checkpoint, and a summary under
claude_automate/experiments/solve_<timestamp>/. Nothing outside
claude_automate/ is modified.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from claude_automate.framework.config import Config
from claude_automate.framework.distill import distill_trajectory_recurrent
from claude_automate.framework.env_api import (
    frame_to_tensor, make_arc_env, make_mini_env,
)
from claude_automate.framework.go_explore import (
    GoExplore, SearchResult, collect_trajectory_frames,
)
from claude_automate.framework.networks import RecurrentActorCritic

_EXP_DIR = Path(__file__).resolve().parent / "experiments"


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def greedy_eval(env, model, cfg, device, n_episodes: int):
    """Greedily unroll the recurrent policy; return (completion_rate, lengths)."""
    model.eval()
    completions, lengths = 0, []
    for _ in range(n_episodes):
        frame = env.reset()
        h = model.initial_state().to(device)
        steps = 0
        for _ in range(cfg.max_episode_steps):
            obs = frame_to_tensor(frame, cfg.n_colors).unsqueeze(0).to(device)
            logits, _, h = model.step(obs, h)
            frame, terminal = env.step(int(logits.argmax(-1).item()))
            steps += 1
            if env.level_completed:
                completions += 1
                break
            if terminal:
                break
        lengths.append(steps)
    return completions / n_episodes, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", type=str, default="ls20-9607627b")
    ap.add_argument("--level", type=int, default=1,
                    help="0-indexed level (0 = level 1)")
    ap.add_argument("--max-env-steps", type=int, default=600_000)
    ap.add_argument("--explore-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--distill-epochs", type=int, default=4000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--solution", type=str, default=None,
                    help="path to a solution.json — skip search, distill it")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="write artifacts here instead of an auto-named dir")
    ap.add_argument("--mini-env", type=str, default=None,
                    help="Path to a mini_env level JSON. If set, uses the pure-numpy "
                         "MiniLS20Env instead of the arcengine ARC env.")
    args = ap.parse_args()

    cfg = Config()
    cfg.game_id = args.game_id
    cfg.level_index = args.level
    device = pick_device(args.device)

    # One env instance, reused throughout (reset() is a full deterministic
    # reset, so Go-Explore can return to any cell via reset + replay).
    if args.mini_env:
        env = make_mini_env(args.mini_env)
        cfg.game_id = f"mini:{Path(args.mini_env).stem}"
    else:
        env = make_arc_env(cfg.game_id, cfg.level_index)
    masked_rows = getattr(env, "_MASKED_ROWS", None)
    n_actions = env.n_actions

    # Meaningful run name: solve_<game>_L<human-level>_<timestamp>
    # (level is 1-indexed in the name; level_index 2 -> "L3").
    game_short = cfg.game_id.split("-")[0].replace(":", "_")
    if args.out_dir:
        run_dir = Path(args.out_dir)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _EXP_DIR / f"solve_{game_short}_L{cfg.level_index + 1}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[solve] game={cfg.game_id} level_index={cfg.level_index} "
          f"device={device} run_dir={run_dir}")

    # ── 1. Obtain a completing trajectory ────────────────────────────────────
    if args.solution:
        # Skip the search — re-use a previously found trajectory.
        prev = json.loads(Path(args.solution).read_text())
        solution = prev["trajectory"]
        result = SearchResult(solution=solution, archive_size=-1,
                              env_steps=prev.get("search_env_steps", -1),
                              iterations=-1, elapsed_s=0.0)
        print(f"[solve] loaded solution ({len(solution)} actions) "
              f"from {args.solution}")
    else:
        explorer = GoExplore(env, masked_rows=masked_rows,
                             explore_steps=args.explore_steps, seed=args.seed)
        result = explorer.search(max_env_steps=args.max_env_steps)
        if result.solution is None:
            print("[solve] FAILED — no completing trajectory found.")
            (run_dir / "summary.json").write_text(json.dumps({
                "solved": False, "archive_size": result.archive_size,
                "env_steps": result.env_steps,
            }, indent=2))
            sys.exit(1)

    (run_dir / "solution.json").write_text(json.dumps({
        "trajectory": result.solution,
        "length": len(result.solution),
        "archive_size": result.archive_size,
        "search_env_steps": result.env_steps,
    }, indent=2))

    # ── 2. Distill the trajectory into a policy ──────────────────────────────
    frames, actions = collect_trajectory_frames(env, result.solution)
    model = RecurrentActorCritic(n_actions=n_actions, n_colors=cfg.n_colors,
                                 hidden_dim=cfg.hidden_dim,
                                 frame_size=cfg.frame_size).to(device)
    distill_stats = distill_trajectory_recurrent(
        model, frames, actions, cfg, device, epochs=args.distill_epochs)

    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "solution_length": len(result.solution)},
               run_dir / "policy.pt")

    # ── 3. Evaluate the distilled policy ─────────────────────────────────────
    comp_rate, lengths = greedy_eval(env, model, cfg, device,
                                     args.eval_episodes)

    summary = {
        "solved": True,
        "game_id": cfg.game_id, "level_index": cfg.level_index,
        "solution_length": len(result.solution),
        "search_env_steps": result.env_steps,
        "search_iterations": result.iterations,
        "search_elapsed_s": round(result.elapsed_s, 1),
        "archive_size": result.archive_size,
        "distill": distill_stats,
        "eval_episodes": args.eval_episodes,
        "eval_completion_rate": comp_rate,
        "eval_len_min": int(min(lengths)),
        "eval_len_max": int(max(lengths)),
        "eval_len_mean": round(float(np.mean(lengths)), 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"[solve] SOLVED  level_index={cfg.level_index}")
    print(f"  search: {result.env_steps} env steps, "
          f"{result.iterations} iters, {result.elapsed_s:.0f}s, "
          f"archive {result.archive_size} cells")
    print(f"  solution: {len(result.solution)} actions")
    print(f"  distilled policy eval: {comp_rate:.0%} completion "
          f"over {args.eval_episodes} greedy episodes "
          f"(len {summary['eval_len_min']}-{summary['eval_len_max']})")
    print(f"  artifacts: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
