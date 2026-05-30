"""Synchronous vectorised wrapper around MiniLS20Env.

Tracks per-env state needed for reward shaping and metrics:
  - previous (player_c, player_r) and player_rotation (for wall-hit + match
    transition detection)
  - per-episode wall-hit count, coverage set, step count, won-at-end flag

`step(actions)` returns:
    obs:        (N, 32, 32) uint8 — next observations (post-reset if done)
    rewards:    (N,) float32       — caller computes shaping in rewards.py
    dones:      (N,) bool          — true on terminal/truncation, also resets
    infos: list of dicts with keys used by reward + metric machinery
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mini_env.env import MiniLS20Env


@dataclass
class EpisodeStats:
    steps: int = 0
    wall_hits: int = 0
    visited: set = field(default_factory=set)
    rotation_matched_at_end: bool = False
    won: bool = False
    return_unshaped: float = 0.0
    return_shaped: float = 0.0


class VecMiniEnv:
    """N synchronous MiniLS20Env instances."""

    def __init__(self, level_path: str, n_envs: int, seed: int = 0):
        self.n_envs = n_envs
        self.envs: list[MiniLS20Env] = []
        # Snapshot of non-wall cell count and total play-area for coverage.
        for i in range(n_envs):
            np.random.seed(seed + i)  # mini_env is deterministic per config, no RNG usage
            self.envs.append(MiniLS20Env(level_path))

        # Count of non-wall non-goal-non-cross *walkable empty* cells.
        # Coverage denominator: all play-area cells except walls.
        sample = self.envs[0]
        self.n_walkable = int((sample.grid != 1).sum())
        self.goal_rotation = sample.goal_rotation

        # Tracking (per env).
        self.prev_pos = [(e.player_c, e.player_r) for e in self.envs]
        self.prev_rot = [e.player_rotation for e in self.envs]
        self.ep_stats: list[EpisodeStats] = [EpisodeStats() for _ in range(n_envs)]
        for i, e in enumerate(self.envs):
            self.ep_stats[i].visited.add((e.player_c, e.player_r))

        # Per-env trail of *completed* episode stats, drained by the trainer.
        self._completed: list[list[EpisodeStats]] = [[] for _ in range(n_envs)]

    # ── observation utilities ────────────────────────────────────────────

    def reset_all(self) -> np.ndarray:
        """Reset every env. Returns (N, 32, 32) uint8."""
        obs_list = []
        for i, e in enumerate(self.envs):
            obs = e.reset()
            obs_list.append(obs)
            self.prev_pos[i] = (e.player_c, e.player_r)
            self.prev_rot[i] = e.player_rotation
            self.ep_stats[i] = EpisodeStats()
            self.ep_stats[i].visited.add((e.player_c, e.player_r))
        return np.stack(obs_list, axis=0)

    def current_obs(self) -> np.ndarray:
        """Return current rendered observation from each env (no step)."""
        return np.stack([e._render() for e in self.envs], axis=0)

    # ── step ─────────────────────────────────────────────────────────────

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """actions: (N,) int in {0,1,2,3}. Returns (obs, raw_rewards, dones, infos).

        `raw_rewards` is +1 on a winning terminal step, 0 otherwise — shaping
        is applied separately by `shared.rewards`. `infos` carries the fields
        needed for shaping (wall_hit, prev_rot, post_rot, won) plus end-of-
        episode summary stats (drained on done).
        """
        N = self.n_envs
        next_obs = np.empty((N, 32, 32), dtype=np.uint8)
        rewards = np.zeros(N, dtype=np.float32)
        dones = np.zeros(N, dtype=bool)
        infos: list[dict] = []

        for i in range(N):
            e = self.envs[i]
            a = int(actions[i])
            pre_pos = (e.player_c, e.player_r)
            pre_rot = e.player_rotation
            frame, done = e.step(a)

            post_pos = (e.player_c, e.player_r)
            post_rot = e.player_rotation
            won = bool(e.won)
            # Wall hit: position unchanged AND we did not just win (winning
            # also "succeeds" in moving, so it is never a wall hit).
            wall_hit = (post_pos == pre_pos) and not won

            # Track episode stats.
            stats = self.ep_stats[i]
            stats.steps += 1
            if wall_hit:
                stats.wall_hits += 1
            stats.visited.add(post_pos)
            if won:
                rewards[i] = 1.0
                stats.return_unshaped += 1.0

            dones[i] = bool(done)

            info = {
                "wall_hit": wall_hit,
                "prev_rot": pre_rot,
                "post_rot": post_rot,
                "won": won,
                "goal_rot": self.goal_rotation,
            }

            if done:
                # Finalise episode summary.
                stats.won = won
                stats.rotation_matched_at_end = (post_rot == self.goal_rotation)
                info["episode_summary"] = {
                    "steps": stats.steps,
                    "wall_hits": stats.wall_hits,
                    "coverage": len(stats.visited) / max(1, self.n_walkable),
                    "rotation_matched_at_end": stats.rotation_matched_at_end,
                    "won": stats.won,
                    "return_unshaped": stats.return_unshaped,
                    # return_shaped is filled in by the trainer once shaping
                    # has been added — see step_with_shaping below.
                }
                # Snapshot before reset so the trainer can drain it.
                self._completed[i].append(stats)
                # Reset.
                frame = e.reset()
                self.prev_pos[i] = (e.player_c, e.player_r)
                self.prev_rot[i] = e.player_rotation
                new_stats = EpisodeStats()
                new_stats.visited.add((e.player_c, e.player_r))
                self.ep_stats[i] = new_stats
            else:
                self.prev_pos[i] = post_pos
                self.prev_rot[i] = post_rot

            next_obs[i] = frame
            infos.append(info)

        return next_obs, rewards, dones, infos

    # ── episode summary drain ────────────────────────────────────────────

    def drain_completed_episodes(self) -> list[EpisodeStats]:
        """Return and clear the per-env list of episodes that finished since
        last drain. Order is roughly chronological across envs."""
        out: list[EpisodeStats] = []
        for i in range(self.n_envs):
            out.extend(self._completed[i])
            self._completed[i] = []
        return out
