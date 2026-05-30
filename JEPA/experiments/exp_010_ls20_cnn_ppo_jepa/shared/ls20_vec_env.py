"""Synchronous vectorised wrapper around N *real* LS20 environments.

Unlike exp_007's `VecMiniEnv` (which wraps the pure-numpy MiniLS20Env and can
read out player position / rotation for shaping), the real ARC-AGI-3 LS20 game
is observed only through 64x64 colour-index frames + a terminal `level_completed`
flag. So the only reward available here is **terminal-only**:

    r_t = +1  if the step cleared the level (env.level_completed at terminal)
    r_t =  0  otherwise (including running out of the game's step budget, and
              our own max_episode_steps truncation)

That is exactly the exp_007_0_naive ("7_0") reward mode, ported to real LS20.

`step(actions)` returns:
    obs      (N, 64, 64) uint8  — next observations (post-reset if the env ended)
    rewards  (N,)        float32 — terminal-only success reward
    dones    (N,)        bool    — true on terminal OR truncation (also resets)
    infos    list[dict]          — per-env step info; carries `episode_summary`
                                    on the step an episode finished
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from JEPA.shared.env_wrapper import make_env, full_game_id


FRAME = 64


@dataclass
class EpisodeStats:
    steps: int = 0
    success: bool = False
    truncated: bool = False
    return_unshaped: float = 0.0


class VecLS20Env:
    """N synchronous real-LS20 instances sharing one offline Arcade."""

    FRAME = FRAME  # 64; consumed by shared.rollout to size the obs buffers

    def __init__(self, env_name: str = "ls20", n_envs: int = 8,
                 max_episode_steps: int = 200, seed: int = 0,
                 environments_dir: str | None = None):
        from arc_agi import Arcade, OperationMode

        self.env_name = env_name
        self.n_envs = n_envs
        self.max_episode_steps = max_episode_steps
        self.gid = full_game_id(env_name)

        if environments_dir is None:
            # repo_root/environment_files — this file is 5 levels under repo root:
            # shared -> exp_010_.. -> experiments -> JEPA -> Code Repo
            from pathlib import Path
            environments_dir = str(
                Path(__file__).resolve().parents[4] / "environment_files"
            )

        self._arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=environments_dir,
        )
        self.envs = [make_env(self._arc.make(self.gid), self.gid) for _ in range(n_envs)]
        self.n_actions = self.envs[0].n_actions

        self.ep_stats: list[EpisodeStats] = [EpisodeStats() for _ in range(n_envs)]
        self._last_obs = np.zeros((n_envs, FRAME, FRAME), dtype=np.uint8)
        self._completed: list[list[EpisodeStats]] = [[] for _ in range(n_envs)]
        self.reset_all()

    # ── observation utilities ────────────────────────────────────────────

    def reset_all(self) -> np.ndarray:
        for i, e in enumerate(self.envs):
            self._last_obs[i] = e.reset()
            self.ep_stats[i] = EpisodeStats()
        return self._last_obs.copy()

    def current_obs(self) -> np.ndarray:
        return self._last_obs.copy()

    # ── step ─────────────────────────────────────────────────────────────

    def step(self, actions: np.ndarray):
        N = self.n_envs
        next_obs = np.empty((N, FRAME, FRAME), dtype=np.uint8)
        rewards = np.zeros(N, dtype=np.float32)
        dones = np.zeros(N, dtype=bool)
        infos: list[dict] = []

        for i in range(N):
            e = self.envs[i]
            a = int(actions[i])
            frame, is_terminal = e.step(a)
            stats = self.ep_stats[i]
            stats.steps += 1

            success = bool(e.level_completed)
            truncated = (not is_terminal) and (stats.steps >= self.max_episode_steps)
            done = bool(is_terminal) or truncated

            if success:
                rewards[i] = 1.0
                stats.return_unshaped += 1.0

            info: dict = {"success": success, "truncated": truncated}

            if done:
                stats.success = success
                stats.truncated = truncated and not is_terminal
                info["episode_summary"] = {
                    "steps": stats.steps,
                    "success": stats.success,
                    "truncated": stats.truncated,
                    "return_unshaped": stats.return_unshaped,
                }
                self._completed[i].append(stats)
                frame = e.reset()
                self.ep_stats[i] = EpisodeStats()

            dones[i] = done
            next_obs[i] = frame
            infos.append(info)

        self._last_obs = next_obs
        return next_obs.copy(), rewards, dones, infos

    # ── episode summary drain ────────────────────────────────────────────

    def drain_completed_episodes(self) -> list[EpisodeStats]:
        out: list[EpisodeStats] = []
        for i in range(self.n_envs):
            out.extend(self._completed[i])
            self._completed[i] = []
        return out
