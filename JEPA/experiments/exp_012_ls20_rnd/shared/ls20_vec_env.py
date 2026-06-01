"""Multi-level synchronous vec env for LS20, extending exp_010's VecLS20Env.

exp_010's VecLS20Env hardcodes `level_completed` / `is_terminal` to the
wrapper's `_STOP_LEVELS = 1`, so an episode ends the moment Level 1 is cleared.
To train an agent *through* L1 into L2 (etc.), this subclass:

  * sets `_STOP_LEVELS = stop_levels` on each underlying wrapper instance, so the
    game keeps running (auto-advancing to the next level) until `stop_levels`
    levels are cleared (or the game's own step budget / our truncation hits);
  * gives an **incremental +1 reward each time a new level is cleared** (so the
    agent gets credit for L1 *and* L2). `success` = reached `stop_levels`.

`stop_levels=1` + incremental reward is identical to the exp_010 terminal-only
L1 setup (clearing L1 gives the single +1 and ends the episode), so the L1
sub-experiment is unaffected by using this class.
"""

from __future__ import annotations

import numpy as np

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import (
    VecLS20Env, EpisodeStats, FRAME,
)


def _levels_done(env) -> int:
    raw = getattr(env, "_latest_raw", None)
    return int(getattr(raw, "levels_completed", 0)) if raw is not None else 0


class MultiLevelVecLS20Env(VecLS20Env):
    def __init__(self, env_name: str = "ls20", n_envs: int = 8,
                 max_episode_steps: int = 200, seed: int = 0,
                 stop_levels: int = 1, environments_dir: str | None = None):
        self.stop_levels = stop_levels
        super().__init__(env_name, n_envs, max_episode_steps, seed, environments_dir)
        for e in self.envs:
            e._STOP_LEVELS = stop_levels      # instance override of the class default
        self._prev_levels = [_levels_done(e) for e in self.envs]

    def reset_all(self) -> np.ndarray:
        obs = super().reset_all()
        if hasattr(self, "_prev_levels"):     # not yet set during super().__init__
            self._prev_levels = [_levels_done(e) for e in self.envs]
        return obs

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

            cur = _levels_done(e)
            new_levels = max(0, cur - self._prev_levels[i])
            self._prev_levels[i] = cur
            if new_levels > 0:
                rewards[i] = float(new_levels)
                stats.return_unshaped += float(new_levels)

            success = cur >= self.stop_levels          # == e.level_completed
            truncated = (not is_terminal) and (stats.steps >= self.max_episode_steps)
            done = bool(is_terminal) or truncated

            info: dict = {"success": success, "truncated": truncated, "levels_completed": cur}

            if done:
                stats.success = success
                stats.truncated = truncated and not is_terminal
                info["episode_summary"] = {
                    "steps": stats.steps,
                    "success": stats.success,
                    "truncated": stats.truncated,
                    "return_unshaped": stats.return_unshaped,
                    "levels_completed": cur,
                }
                self._completed[i].append(stats)
                frame = e.reset()
                self._prev_levels[i] = _levels_done(e)
                self.ep_stats[i] = EpisodeStats()

            dones[i] = done
            next_obs[i] = frame
            infos.append(info)

        self._last_obs = next_obs
        return next_obs.copy(), rewards, dones, infos
