"""
Environment wrappers for ARC-AGI-3 games.

All wrappers share a common interface via BaseArcEnv so the training loop
can swap environments without any other code changes.

Public interface (what the training loop calls):
    env.reset()                    → (64, 64) uint8 ndarray
    env.step(action_idx: int)      → ((64, 64) uint8 ndarray, bool is_terminal)
    env.available_actions          → list[int]  (1-indexed GameAction values)
    env.level_completed            → bool
    env.n_actions                  → int

Mask-aware diff utilities (reward computation + exploration tracking):
    env.frame_diff(f0, f1)         → (64, 64) float32  — UI rows zeroed
    env.patch_weights(f0, f1)      → (16,) float32 weights in [0, 1]
    env.detect_moved_cell(f0, f1)  → int | None  (0–63 cell index in 8×8 grid)

Adding a new game:
    1. Subclass BaseArcEnv, set _ACTIONS / _MASKED_ROWS / _STOP_LEVELS.
    2. Add one entry to _REGISTRY.
    That's it — make_env() handles the rest.

Game summary:
    ls20  4 actions  move         rows 61–62 = step counter
    tu93  4 actions  move         row  63    = step-count bar (changes every step)
    re86  5 actions  move+switch  row  63    = step-count bar
    g50t  5 actions  move+undo    row  63    = timer strip (scrolls left)
"""

from __future__ import annotations

import numpy as np
from arcengine import GameAction, GameState
from JEPA.shared.simplified_ls20 import SimplifiedLS20Env, LevelDef, LEVEL_1  # noqa: F401


class BaseArcEnv:
    """
    Base class for all ARC-AGI-3 game wrappers.

    Subclasses must set:
        _ACTIONS     — ordered list[GameAction] the game supports
        _STOP_LEVELS — stop the episode when levels_completed >= this (default 1)
        _MASKED_ROWS — slice of always-changing UI rows to zero in diffs, or None
    """

    _ACTIONS: list[GameAction] = []
    _STOP_LEVELS: int = 1
    _MASKED_ROWS: slice | None = None

    def __init__(self, arc_env) -> None:
        self._env = arc_env
        self._latest_raw = None
        self.n_actions: int = len(self._ACTIONS)

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset the game; return initial (64, 64) uint8 frame."""
        raw = self._env.reset()
        self._latest_raw = raw
        return self._extract(raw)

    def step(self, action_idx: int) -> tuple[np.ndarray, bool]:
        """
        Execute action_idx (0-indexed into _ACTIONS).
        Returns (frame_np, is_terminal).
        """
        action = self._ACTIONS[action_idx]
        raw = self._env.step(action)
        self._latest_raw = raw
        return self._extract(raw), self._is_terminal(raw)

    @property
    def available_actions(self) -> list[int]:
        """1-indexed ints for currently legal actions."""
        if self._latest_raw is None:
            return [a.value for a in self._ACTIONS]
        avail = getattr(self._latest_raw, "available_actions", None)
        return avail if avail else [a.value for a in self._ACTIONS]

    @property
    def level_completed(self) -> bool:
        """True once the agent has cleared _STOP_LEVELS levels."""
        if self._latest_raw is None:
            return False
        return getattr(self._latest_raw, "levels_completed", 0) >= self._STOP_LEVELS

    @property
    def won(self) -> bool:
        """True if the game reached WIN state (all levels done)."""
        return (
            self._latest_raw is not None
            and self._latest_raw.state is GameState.WIN
        )

    # ── Mask-aware diff utilities ─────────────────────────────────────────────

    def frame_diff(self, f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
        """
        Pixel-wise absolute difference with UI rows zeroed.
        Returns (64, 64) float32.
        """
        d = np.abs(f1.astype(np.float32) - f0.astype(np.float32))
        if self._MASKED_ROWS is not None:
            d[self._MASKED_ROWS, :] = 0.0
        return d

    def patch_weights(self, f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
        """
        (64,64) frames → (16,) patch change weights in [0, 1].
        Uses masked diff so UI rows don't inflate the normaliser.
        """
        pw = self.frame_diff(f0, f1).reshape(4, 16, 4, 16).mean(axis=(1, 3)).flatten()
        m = float(pw.max())
        return (pw / m) if m > 1e-8 else np.zeros(16, dtype=np.float32)

    def detect_moved_cell(self, f0: np.ndarray, f1: np.ndarray) -> int | None:
        """
        Return the 8×8 fine-grid cell index (0–63) with the most masked change.
        Returns None if no significant change detected (threshold < 2.0).
        """
        cell_diff = self.frame_diff(f0, f1).reshape(8, 8, 8, 8).sum(axis=(1, 3))
        if cell_diff.max() < 2.0:
            return None
        r, c = np.unravel_index(cell_diff.argmax(), cell_diff.shape)
        return int(r * 8 + c)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract(raw) -> np.ndarray:
        """FrameDataRaw → (64, 64) uint8 colour-index frame."""
        return np.array(raw.frame, dtype=np.uint8)[-1]

    def _is_terminal(self, raw) -> bool:
        if raw.state in (GameState.WIN, GameState.GAME_OVER):
            return True
        return getattr(raw, "levels_completed", 0) >= self._STOP_LEVELS


# ── Game-specific wrappers ────────────────────────────────────────────────────

class LS20Env(BaseArcEnv):
    """
    LS20 — locked-door puzzle.
    4 actions: ACTION1–4 = directional movement.
    Rows 61–62: step-counter display that changes every step.
    """
    _ACTIONS = [
        GameAction.ACTION1, GameAction.ACTION2,
        GameAction.ACTION3, GameAction.ACTION4,
    ]
    _MASKED_ROWS = slice(61, 63)  # rows 61 and 62


class Tu93Env(BaseArcEnv):
    """
    TU93 — graph-maze navigation with moving obstacles.
    4 actions: ACTION1–4 = directional movement along maze tracks.
    9 levels; single-level training stops at _STOP_LEVELS=1.
    Row 63: colour-6 step-count bar, shrinks every step (always changes).
    """
    _ACTIONS = [
        GameAction.ACTION1, GameAction.ACTION2,
        GameAction.ACTION3, GameAction.ACTION4,
    ]
    _MASKED_ROWS = slice(63, 64)  # row 63 only


class Re86Env(BaseArcEnv):
    """
    RE86 — cross/plus-piece sliding puzzle.
    5 actions: ACTION1–4 = move selected piece, ACTION5 = switch active piece.
    8 levels; single-level training stops at _STOP_LEVELS=1.
    Row 63: step-count bar that decrements every action.
    """
    _ACTIONS = [
        GameAction.ACTION1, GameAction.ACTION2,
        GameAction.ACTION3, GameAction.ACTION4,
        GameAction.ACTION5,
    ]
    _MASKED_ROWS = slice(63, 64)  # row 63 only


class G50tEnv(BaseArcEnv):
    """
    G50T — Sokoban-style push puzzle with undo and timer.
    5 actions: ACTION1–4 = move player, ACTION5 = undo last move.
    7 levels; single-level training stops at _STOP_LEVELS=1.
    Row 63: colour-9 timer strip, scrolls 1 px left every 2 steps.
    """
    _ACTIONS = [
        GameAction.ACTION1, GameAction.ACTION2,
        GameAction.ACTION3, GameAction.ACTION4,
        GameAction.ACTION5,
    ]
    _MASKED_ROWS = slice(63, 64)  # row 63 only


# ── Registry and factory ──────────────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseArcEnv]] = {
    "ls20": LS20Env,
    "tu93": Tu93Env,
    "re86": Re86Env,
    "g50t": G50tEnv,
}

# Standalone envs that don't need an underlying arcengine game object.
# make_env() instantiates these directly without arc_env.
_STANDALONE_REGISTRY: dict[str, type] = {
    "ls20s": SimplifiedLS20Env,
}


def make_env(arc_env, game_id: str) -> BaseArcEnv:
    """
    Instantiate the correct wrapper for game_id.

    Accepts full IDs ('ls20-9607627b') or short IDs ('ls20').
    For standalone game_ids (e.g. 'ls20s') arc_env is ignored.
    Raises ValueError if the game has no registered wrapper.
    """
    base_id = game_id.split("-")[0]

    standalone_cls = _STANDALONE_REGISTRY.get(base_id)
    if standalone_cls is not None:
        return standalone_cls()

    cls = _REGISTRY.get(base_id)
    if cls is None:
        raise ValueError(
            f"No wrapper registered for {base_id!r}. "
            f"Registered games: {sorted(_REGISTRY | _STANDALONE_REGISTRY)}"
        )
    return cls(arc_env)


def make_env_auto(game_id: str, environments_dir: str | None = None):
    """
    One-call factory that handles both standalone and arcengine-backed games.

    Standalone games (e.g. 'ls20s') are instantiated directly — no arcengine
    Arcade session is created, and environments_dir is ignored.

    Arcengine-backed games (e.g. 'ls20-9607627b') open an OFFLINE Arcade
    session pointed at environments_dir before loading the game bundle.

    Usage in training scripts:
        env = make_env_auto(cfg.game_id, str(repo_root / "environment_files"))
    """
    base_id = game_id.split("-")[0]

    standalone_cls = _STANDALONE_REGISTRY.get(base_id)
    if standalone_cls is not None:
        return standalone_cls()

    # arcengine-backed — lazy import keeps it out of standalone code paths
    from arc_agi import Arcade, OperationMode  # noqa: PLC0415
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=environments_dir,
    )
    cls = _REGISTRY.get(base_id)
    if cls is None:
        raise ValueError(
            f"No wrapper registered for {base_id!r}. "
            f"Registered games: {sorted(_REGISTRY | _STANDALONE_REGISTRY)}"
        )
    return cls(arc.make(game_id))
