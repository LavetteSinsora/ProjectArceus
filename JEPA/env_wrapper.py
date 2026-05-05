import numpy as np
from arcengine import GameAction, GameState

# LS20 only uses ACTION1–4 (confirmed via env.action_space check)
_LS20_ACTIONS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
]


class LS20Env:
    """
    Thin wrapper around arcengine's LocalEnvironmentWrapper for LS20.

    Responsibilities:
    - Extracts (64, 64) uint8 frames from FrameDataRaw
    - Detects terminal states: GAME_OVER, WIN, or Level 1 completed
    - Exposes a clean reset() / step(action_idx) interface
    - Returns available actions as a list of ints (1-indexed, matching GameAction.value)

    We stop at Level 1 (levels_completed >= 1) for single-level world model training.
    """

    def __init__(self, arc_env):
        self._env = arc_env
        self._latest_raw = None
        self.n_actions = len(_LS20_ACTIONS)

    def reset(self) -> np.ndarray:
        """Reset the environment; returns initial (64, 64) uint8 frame."""
        raw = self._env.reset()
        self._latest_raw = raw
        return self._extract(raw)

    def step(self, action_idx: int):
        """
        Execute action_idx (0-indexed) in the environment.
        Returns (frame_np, is_terminal):
          frame_np: (64, 64) uint8 with color values 0–15
          is_terminal: bool — True on WIN, GAME_OVER, or after Level 1 completes
        """
        action = _LS20_ACTIONS[action_idx]
        raw = self._env.step(action)
        self._latest_raw = raw
        return self._extract(raw), self._is_terminal(raw)

    @property
    def available_actions(self) -> list:
        """List of 1-indexed ints for actions available in the current state."""
        if self._latest_raw is None:
            return [1, 2, 3, 4]
        avail = getattr(self._latest_raw, "available_actions", None)
        return avail if avail else [1, 2, 3, 4]

    @property
    def level_completed(self) -> bool:
        """True once the agent has finished Level 1."""
        if self._latest_raw is None:
            return False
        return getattr(self._latest_raw, "levels_completed", 0) >= 1

    @property
    def won(self) -> bool:
        """True if the game is in WIN state (all levels done)."""
        return self._latest_raw is not None and self._latest_raw.state is GameState.WIN

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract(raw) -> np.ndarray:
        """FrameDataRaw → (64, 64) uint8, color index 0–15."""
        return np.array(raw.frame, dtype=np.uint8)[-1]

    @staticmethod
    def _is_terminal(raw) -> bool:
        if raw.state in (GameState.WIN, GameState.GAME_OVER):
            return True
        # Stop after Level 1 for single-level training
        return getattr(raw, "levels_completed", 0) >= 1
