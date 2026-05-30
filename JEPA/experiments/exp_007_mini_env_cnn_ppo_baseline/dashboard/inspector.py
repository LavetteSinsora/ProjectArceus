"""Load an exp_007 checkpoint, run an episode, and return per-step data
for the dashboard. CPU-only so it can run alongside MPS training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mini_env.env import MiniLS20Env

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.model import ActorCritic


# ARC-AGI 16-color palette (RGB).
ARC_COLORS_RGB = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (61, 153, 112), (255, 255, 255),
    (0, 31, 63), (1, 255, 112), (133, 20, 75), (1, 75, 101),
]


def _frame_to_rgb_list(frame_uint8: np.ndarray) -> list[list[list[int]]]:
    """(32,32) palette indices → (32, 32, 3) RGB nested list."""
    h, w = frame_uint8.shape
    out = [[ARC_COLORS_RGB[int(frame_uint8[r, c])] for c in range(w)] for r in range(h)]
    return [[list(rgb) for rgb in row] for row in out]


def load_checkpoint(checkpoint_path: str) -> tuple[ActorCritic, dict]:
    device = torch.device("cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ActorCritic().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck.get("config", {})


def run_episode(checkpoint_path: str, level_path: str | None = None,
                seed: int = 0, sample: bool = True, max_steps: int = 100) -> dict:
    """Run one full episode and return structured per-step data.

    Returns dict with:
        config:     run config
        level:      level path
        sample:     whether actions were sampled or argmax
        n_steps:    number of steps taken
        won:        final terminal status
        goal_rot:   goal rotation
        steps:      list of per-step dicts, each containing:
            t:                  step index (0..n_steps)
            frame:              (32, 32, 3) RGB list
            player_c, player_r: player position before step
            player_rotation:    player rotation before step
            step_counter:       remaining steps before this step
            denial_frames:      remaining denial flash frames
            logits:             [4] policy logits
            probs:              [4] softmaxed probabilities
            value:              scalar V estimate
            feature_norm:       L2 norm of trunk feature h_t
            action:             0..3 (or None if final)
            action_name:        "up"/"down"/"left"/"right" or null
            reward:             reward earned by this action (terminal-only basis)
            wall_hit:           bool
            won:                bool after this action
            done:               bool after this action
    """
    model, cfg = load_checkpoint(checkpoint_path)
    if level_path is None:
        level_path = cfg.get("level_path", "mini_env/configs/level_01/simple_1_rotation.json")
    env = MiniLS20Env(level_path)
    torch.manual_seed(seed)
    np.random.seed(seed)

    action_names = ["up", "down", "left", "right"]
    steps_data = []
    obs = env.reset()
    won_at_end = False

    for t in range(max_steps):
        # Snapshot state BEFORE action.
        pc, pr, rot, sc, dn = (env.player_c, env.player_r, env.player_rotation,
                                env.step_counter, env.denial_frames)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs[None, ...]).to("cpu")
            logits, value, feat = model.forward(obs_t)
        logits_np = logits.cpu().numpy()[0].tolist()
        probs_np = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        v = float(value.cpu().item())
        fnorm = float(torch.linalg.norm(feat).item())

        # Compose pre-step record.
        rec = {
            "t": t,
            "frame": _frame_to_rgb_list(obs),
            "player_c": int(pc),
            "player_r": int(pr),
            "player_rotation": int(rot),
            "step_counter": int(sc),
            "denial_frames": int(dn),
            "goal_rotation_matched": bool(rot == env.goal_rotation),
            "logits": [float(x) for x in logits_np],
            "probs": [float(x) for x in probs_np],
            "value": v,
            "feature_norm": fnorm,
        }

        if sample:
            dist = torch.distributions.Categorical(logits=logits)
            a = int(dist.sample().item())
        else:
            a = int(np.argmax(logits_np))

        next_obs, done = env.step(a)
        post_pos = (env.player_c, env.player_r)
        won = bool(env.won)
        wall_hit = (post_pos == (pc, pr)) and not won
        reward = 1.0 if won else 0.0

        rec.update({
            "action": a,
            "action_name": action_names[a],
            "reward": reward,
            "wall_hit": wall_hit,
            "won": won,
            "done": done,
        })
        steps_data.append(rec)

        obs = next_obs
        if done:
            won_at_end = won
            break

    # Append a terminal-state snapshot (no action) so the UI can show the
    # final frame the agent ended up in.
    with torch.no_grad():
        obs_t = torch.from_numpy(obs[None, ...]).to("cpu")
        logits, value, feat = model.forward(obs_t)
    terminal_rec = {
        "t": len(steps_data),
        "frame": _frame_to_rgb_list(obs),
        "player_c": int(env.player_c),
        "player_r": int(env.player_r),
        "player_rotation": int(env.player_rotation),
        "step_counter": int(env.step_counter),
        "denial_frames": int(env.denial_frames),
        "goal_rotation_matched": bool(env.player_rotation == env.goal_rotation),
        "logits": [float(x) for x in logits.cpu().numpy()[0]],
        "probs": [float(x) for x in torch.softmax(logits, -1).cpu().numpy()[0]],
        "value": float(value.cpu().item()),
        "feature_norm": float(torch.linalg.norm(feat).item()),
        "action": None,
        "action_name": None,
        "reward": 0.0,
        "wall_hit": False,
        "won": bool(env.won),
        "done": True,
        "terminal": True,
    }
    steps_data.append(terminal_rec)

    return {
        "checkpoint": checkpoint_path,
        "level": level_path,
        "sample": sample,
        "seed": seed,
        "n_steps": len(steps_data) - 1,
        "won": bool(won_at_end),
        "goal_rotation": int(env.goal_rotation),
        "grid_cols": int(env.cols),
        "grid_play_rows": int(env.play_rows),
        "step_limit": int(env.config.step_limit),
        "palette_rgb": ARC_COLORS_RGB,
        "steps": steps_data,
    }


def run_many(checkpoint_path: str, n: int = 8, sample: bool = True,
             level_path: str | None = None) -> dict:
    """Run multiple episodes and return summary + first episode's trace."""
    episodes = []
    for i in range(n):
        ep = run_episode(checkpoint_path, level_path=level_path, seed=i, sample=sample)
        episodes.append({
            "seed": i,
            "won": ep["won"],
            "n_steps": ep["n_steps"],
            "final_rotation_match": ep["steps"][-1]["goal_rotation_matched"],
        })
    won = sum(e["won"] for e in episodes)
    solved_steps = [e["n_steps"] for e in episodes if e["won"]]
    return {
        "n_episodes": n,
        "success_count": won,
        "success_rate": won / n,
        "min_steps_to_solve": min(solved_steps) if solved_steps else None,
        "avg_steps_to_solve": sum(solved_steps) / len(solved_steps) if solved_steps else None,
        "episodes": episodes,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    data = run_episode(args.checkpoint, seed=args.seed, sample=not args.greedy)
    print(json.dumps({k: v for k, v in data.items() if k != "steps"}, indent=2))
    print(f"--- {len(data['steps'])} step records suppressed ---")
