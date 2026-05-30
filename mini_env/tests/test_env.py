"""Unit tests for MiniLS20Env — one assertion per rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mini_env.env import MiniLS20Env


_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# Action indices (the env exposes 0..3 internally; available_actions = [1..4]).
A_UP, A_DOWN, A_LEFT, A_RIGHT = 0, 1, 2, 3


@pytest.fixture
def level_01_env():
    return MiniLS20Env(str(_CONFIGS / "level_01" / "simple_1_rotation.json"))


@pytest.fixture
def env_with_wall_north_of_player(tmp_path):
    """Custom level: player at (3, 3), wall directly above at (3, 2)."""
    cfg = {
        "name": "test_wall",
        "grid_cells": [8, 8],
        "tile_px": 4,
        "step_limit": 20,
        "palette": {"bg": 3, "wall": 4, "player": 9, "goal_frame": 5,
                    "cross": 5, "preview_bg": 0, "highlight": 14,
                    "energy": 6, "denial_flash": 0},
        "walls": [[3, 2]],
        "player_start": {"cell": [3, 3], "rotation": 0},
        "goal":         {"cell": [6, 1], "rotation": 0},
        "cross":        {"cell": [1, 1]},
        "goal_gated":   True,
        "show_match_cue": True,
    }
    p = tmp_path / "wall.json"
    p.write_text(json.dumps(cfg))
    return MiniLS20Env(str(p))


def test_reset_returns_32x32_uint8(level_01_env):
    f = level_01_env.reset()
    assert isinstance(f, np.ndarray)
    assert f.shape == (32, 32)
    assert f.dtype == np.uint8


def test_n_actions_is_4(level_01_env):
    assert level_01_env.n_actions == 4
    assert level_01_env.available_actions == [1, 2, 3, 4]


def test_wall_blocks_movement(env_with_wall_north_of_player):
    env = env_with_wall_north_of_player
    env.reset()
    c0, r0 = env.player_c, env.player_r
    frame, term = env.step(A_UP)  # try to move into wall
    assert env.player_c == c0 and env.player_r == r0
    assert term is False


def test_goal_blocks_when_rotation_mismatches(level_01_env):
    # Level 01: goal rotation = 0, player_start rotation = 270 (mismatch).
    env = level_01_env
    env.reset()
    # Teleport player adjacent to goal in a controlled way: bypass rules via
    # multiple steps. Simpler: hand-set player adjacent to goal cell (6, 1),
    # so that taking RIGHT from (5, 1) hits the goal cell with rotation 270.
    env.player_c, env.player_r = 5, 1
    env.player_rotation = 270   # still mismatch with goal rotation 0
    _, term = env.step(A_RIGHT)
    assert term is False
    assert env.won is False
    assert env.player_c == 5 and env.player_r == 1


def test_goal_blocks_sets_denial_flash_for_5_steps(level_01_env):
    env = level_01_env
    env.reset()
    env.player_c, env.player_r = 5, 1
    env.player_rotation = 270
    env.step(A_RIGHT)
    assert env.denial_frames == 5


def test_cross_cycles_rotation_by_90(level_01_env):
    env = level_01_env
    env.reset()
    # Cross is at (3, 3). Stand player at (3, 4) facing into it.
    env.player_c, env.player_r = 3, 4
    rot0 = env.player_rotation
    env.step(A_UP)
    assert env.player_rotation == (rot0 + 90) % 360
    assert (env.player_c, env.player_r) == (3, 3)


def test_4_cross_touches_returns_to_original_rotation(level_01_env, tmp_path):
    """4×90° = 360°, so rotation should match the starting rotation again.
    We simulate by repeatedly snapping the player next to the cross and stepping in.
    """
    env = level_01_env
    env.reset()
    rot0 = env.player_rotation
    for _ in range(4):
        env.player_c, env.player_r = 3, 4
        env.step(A_UP)  # land on cross at (3, 3) and rotate +90
    assert env.player_rotation == rot0


def test_goal_wins_when_rotation_matches(level_01_env):
    env = level_01_env
    env.reset()
    env.player_c, env.player_r = 5, 1
    env.player_rotation = env.goal_rotation   # match
    _, term = env.step(A_RIGHT)
    assert term is True
    assert env.won is True
    assert env.level_completed is True
    assert (env.player_c, env.player_r) == (6, 1)


def test_step_counter_terminates_episode(level_01_env):
    env = level_01_env
    env.reset()
    env.step_counter = 1  # next step decrements to 0
    _, term = env.step(A_DOWN)
    assert term is True
    assert env.won is False


def test_out_of_bounds_blocks_movement(level_01_env):
    env = level_01_env
    env.reset()
    # Player at (1, 5). Try moving LEFT twice — second move goes out of bounds.
    env.player_c, env.player_r = 0, 5
    _, term = env.step(A_LEFT)
    assert term is False
    assert env.player_c == 0 and env.player_r == 5


def test_frame_diff_zeros_ui_rows(level_01_env):
    env = level_01_env
    f0 = env.reset()
    # Make a frame that differs everywhere; verify UI rows are masked.
    f1 = np.full_like(f0, 15)
    d = env.frame_diff(f0, f1)
    assert d.shape == (32, 32)
    assert d.dtype == np.float32
    assert np.all(d[28:32, :] == 0)
    # The playfield rows should have non-zero differences.
    assert d[0:28, :].sum() > 0


def test_patch_weights_returns_16_values(level_01_env):
    env = level_01_env
    f0 = env.reset()
    _, _ = env.step(A_DOWN)  # any move that changes the frame
    f1 = env._render()
    pw = env.patch_weights(f0, f1)
    assert pw.shape == (16,)
    assert pw.dtype == np.float32
    assert pw.min() >= 0.0 and pw.max() <= 1.0


def test_match_cue_highlight_appears_when_rotation_matches(level_01_env):
    env = level_01_env
    f_mismatch = env.reset()
    pal = env.config.palette
    n_highlight_mismatch = int(np.sum(f_mismatch == pal["highlight"]))

    # Now flip rotation to match the goal; re-render directly.
    env.player_rotation = env.goal_rotation
    env._sync_state()
    f_match = env._render()
    n_highlight_match = int(np.sum(f_match == pal["highlight"]))

    # Highlight pixels appear only when rotation matches.
    assert n_highlight_mismatch == 0
    assert n_highlight_match > 0
