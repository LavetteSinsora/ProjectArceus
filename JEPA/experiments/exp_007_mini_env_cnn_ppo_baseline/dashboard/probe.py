"""State-probe inspector: configure an arbitrary env state and query the
policy's distribution and value at that state.

No training, no session — the model is loaded read-only and cached by
checkpoint path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mini_env.env import MiniLS20Env, CELL_WALL, CELL_GOAL, CELL_CROSS

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.model import ActorCritic
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.dashboard.inspector import (
    _frame_to_rgb_list, ARC_COLORS_RGB,
)


ACTION_NAMES = ["up", "down", "left", "right"]
VALID_ROTATIONS = (0, 90, 180, 270)

_MODEL_CACHE: dict[str, tuple[ActorCritic, dict]] = {}


def get_or_load(checkpoint_path: str) -> tuple[ActorCritic, dict]:
    if checkpoint_path in _MODEL_CACHE:
        return _MODEL_CACHE[checkpoint_path]
    device = torch.device("cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ActorCritic().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    cfg = ck.get("config", {}) or {}
    _MODEL_CACHE[checkpoint_path] = (model, cfg)
    return model, cfg


def get_level_info(level_path: str) -> dict:
    """Return everything the UI needs to draw the grid + place the agent."""
    env = MiniLS20Env(level_path)
    cells: list[dict] = []
    for r in range(env.play_rows):
        for c in range(env.cols):
            cell_type = int(env.grid[r, c])
            kind = {0: "empty", CELL_WALL: "wall", CELL_GOAL: "goal", CELL_CROSS: "cross"}[cell_type]
            cells.append({"c": c, "r": r, "kind": kind})
    return {
        "level_path": level_path,
        "cols": env.cols,
        "play_rows": env.play_rows,
        "tile_px": env.config.tile_px,
        "step_limit": env.config.step_limit,
        "goal_rotation": int(env.goal_rotation),
        "goal_gated": bool(env.config.goal_gated),
        "show_match_cue": bool(env.config.show_match_cue),
        "player_start": {
            "c": int(env.config.player_start.col),
            "r": int(env.config.player_start.row),
            "rotation": int(env.config.player_rotation),
        },
        "goal_cell": {"c": int(env.config.goal.col), "r": int(env.config.goal.row)},
        "cross_cell": {"c": int(env.config.cross.col), "r": int(env.config.cross.row)},
        "cells": cells,
        "palette_rgb": ARC_COLORS_RGB,
    }


def _configure_env(
    level_path: str, player_c: int, player_r: int, player_rotation: int,
    step_counter: int, denial_frames: int,
) -> MiniLS20Env:
    if player_rotation not in VALID_ROTATIONS:
        raise ValueError(f"player_rotation must be in {VALID_ROTATIONS}; got {player_rotation}")
    env = MiniLS20Env(level_path)
    if not (0 <= player_c < env.cols and 0 <= player_r < env.play_rows):
        raise ValueError(f"player position ({player_c},{player_r}) out of bounds")
    if int(env.grid[player_r, player_c]) == CELL_WALL:
        raise ValueError(f"cannot place player on wall at ({player_c},{player_r})")
    step_counter = int(max(0, min(env.config.step_limit, step_counter)))
    denial_frames = int(max(0, min(99, denial_frames)))
    env.player_c = int(player_c)
    env.player_r = int(player_r)
    env.player_rotation = int(player_rotation)
    env.step_counter = step_counter
    env.denial_frames = denial_frames
    env.won = False  # configured states are non-terminal by definition
    env._sync_state()
    return env


def _eval_model_on_env(model: ActorCritic, env: MiniLS20Env) -> dict:
    frame = env._render()
    obs_t = torch.from_numpy(frame[None, ...]).to("cpu")
    with torch.no_grad():
        logits, value, feat = model.forward(obs_t)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
    return {
        "frame": _frame_to_rgb_list(frame),
        "frame_indices": frame.astype(np.uint8).tolist(),
        "state": {
            "player_c": int(env.player_c),
            "player_r": int(env.player_r),
            "player_rotation": int(env.player_rotation),
            "step_counter": int(env.step_counter),
            "denial_frames": int(env.denial_frames),
            "rotation_matched": bool(env.player_rotation == env.goal_rotation),
            "on_cross": bool(int(env.grid[env.player_r, env.player_c]) == CELL_CROSS),
        },
        "logits": [float(x) for x in logits.cpu().numpy()[0]],
        "probs": probs,
        "value": float(value.item()),
        "feature_norm": float(torch.linalg.norm(feat).item()),
        "action_names": ACTION_NAMES,
    }


def eval_state(
    checkpoint_path: str, level_path: str | None,
    player_c: int, player_r: int, player_rotation: int,
    step_counter: int, denial_frames: int,
) -> dict:
    model, cfg = get_or_load(checkpoint_path)
    if level_path is None:
        level_path = cfg.get("level_path", "mini_env/configs/level_01/simple_1_rotation.json")
    env = _configure_env(level_path, player_c, player_r, player_rotation,
                          step_counter, denial_frames)
    out = _eval_model_on_env(model, env)
    out["level_path"] = level_path
    return out


def step_at(
    checkpoint_path: str, level_path: str | None,
    player_c: int, player_r: int, player_rotation: int,
    step_counter: int, denial_frames: int,
    action: int,
) -> dict:
    """Apply env.step(action) from a configured state and return next-state eval.

    Also reports the transition info (won/done/wall_hit/rotation_change) so the
    UI can label what just happened.
    """
    if action not in (0, 1, 2, 3):
        raise ValueError(f"action must be 0..3; got {action}")
    model, cfg = get_or_load(checkpoint_path)
    if level_path is None:
        level_path = cfg.get("level_path", "mini_env/configs/level_01/simple_1_rotation.json")
    env = _configure_env(level_path, player_c, player_r, player_rotation,
                          step_counter, denial_frames)
    pre_pos = (env.player_c, env.player_r)
    pre_rot = env.player_rotation
    _, done = env.step(action)
    won = bool(env.won)
    wall_hit = (env.player_c, env.player_r) == pre_pos and not won
    rotation_changed = env.player_rotation != pre_rot
    transition = {
        "action": int(action),
        "action_name": ACTION_NAMES[action],
        "wall_hit": wall_hit,
        "rotation_changed": rotation_changed,
        "won": won,
        "done": bool(done),
    }
    # If terminal, we still want to show the frame, but π/V on a terminal state
    # is meaningless to the agent — return them anyway so the UI can grey them out.
    out = _eval_model_on_env(model, env)
    out["transition"] = transition
    out["level_path"] = level_path
    out["done"] = bool(done)
    return out


def reset_state(checkpoint_path: str, level_path: str | None) -> dict:
    """Return the eval at the level's initial state."""
    model, cfg = get_or_load(checkpoint_path)
    if level_path is None:
        level_path = cfg.get("level_path", "mini_env/configs/level_01/simple_1_rotation.json")
    env = MiniLS20Env(level_path)
    out = _eval_model_on_env(model, env)
    out["level_path"] = level_path
    return out
