"""Environment construction and frame preprocessing.

Wraps the repo-shared `JEPA.shared.env_wrapper` (imported, never modified).
Exposes a minimal, game-agnostic surface the trainer relies on:

    env = make_arc_env("ls20-9607627b")
    frame = env.reset()                  # (64, 64) uint8
    frame, terminal = env.step(action)   # action in [0, n_actions)
    env.n_actions, env.level_completed, env.frame_diff(f0, f1)

`frame_to_tensor` turns a uint8 colour-index frame into a one-hot
`(n_colors, H, W)` float tensor — the encoder input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from JEPA.shared.env_wrapper import make_env  # noqa: E402


class LevelStartWrapper:
    """Make every `reset()` start the episode at a chosen 0-indexed level.

    Generic over ARC games: it only uses the engine-level `set_level` API, not
    any LS20-specific knowledge. After resetting (which always lands on level
    0) it jumps the underlying game to `level_index` and renders that level's
    initial frame with a no-op action (`ACTION5` is outside LS20's 4-action
    set, so it advances no state and costs no energy — a pure render).

    All other calls (`step`, `level_completed`, `frame_diff`, `_MASKED_ROWS`,
    `n_actions`, …) delegate unchanged to the wrapped env.
    """

    def __init__(self, base_env, level_index: int):
        self._base = base_env
        self._level_index = level_index
        # Render no-op = the first GameAction *beyond* this game's action set,
        # so it advances no game state. 4-action games → ACTION5;
        # 5-action games (re86, g50t) → ACTION6.
        self._render_action = f"ACTION{base_env.n_actions + 1}"

    def reset(self):
        self._base.reset()
        if self._level_index > 0:
            from arcengine import ActionInput, GameAction
            game = self._base._env._game
            game.set_level(self._level_index)
            raw = game.perform_action(
                ActionInput(id=getattr(GameAction, self._render_action)),
                raw=True,
            )
            self._base._latest_raw = raw
        return self._base._extract(self._base._latest_raw)

    def __getattr__(self, name):
        # Reached only for attributes not defined on the wrapper itself.
        return getattr(self._base, name)


def make_arc_env(game_id: str, level_index: int = 0):
    """Build an ARC-AGI environment wrapper for `game_id` in OFFLINE mode.

    `level_index` (0-indexed) selects which level each episode starts on;
    0 is the default first level. Non-zero values return a LevelStartWrapper.
    """
    from arc_agi import Arcade, OperationMode

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_REPO_ROOT / "environment_files"),
    )
    raw = arc.make(game_id)
    env = make_env(raw, game_id)
    if level_index > 0:
        return LevelStartWrapper(env, level_index)
    return env


def make_mini_env(level_json_path):
    """Build a MiniLS20Env from a JSON level file.

    Plug-compatible with BaseArcEnv: exposes reset/step/n_actions/
    level_completed/frame_diff/patch_weights so existing trainers work unchanged.
    """
    from mini_env.env import MiniLS20Env
    return MiniLS20Env(level_json_path)


def frame_to_tensor(frame: np.ndarray, n_colors: int = 16) -> torch.Tensor:
    """(H, W) uint8 colour-index frame → (n_colors, H, W) one-hot float32."""
    f = torch.as_tensor(np.asarray(frame), dtype=torch.long).clamp_(0, n_colors - 1)
    onehot = torch.zeros(n_colors, *f.shape, dtype=torch.float32)
    onehot.scatter_(0, f.unsqueeze(0), 1.0)
    return onehot


def frames_to_batch(frames, n_colors: int = 16) -> torch.Tensor:
    """List of (H, W) frames → (B, n_colors, H, W) float32 batch tensor."""
    return torch.stack([frame_to_tensor(f, n_colors) for f in frames], dim=0)
