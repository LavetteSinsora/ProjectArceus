"""Generalizable reward composition.

The reward is assembled ONLY from signals that exist in every ARC-AGI game.
No term references LS20 geometry, the goal location, or a target direction.

    r = -w_step                                       time penalty
        - w_stuck            if masked frame unchanged  (wasted action)
        + w_novel        / sqrt(global_count)           global novelty
        + w_novel_episodic / sqrt(episodic_count)       episodic novelty
        + w_complete         on the level-completion transition

Two count-based novelty terms (Tang et al. "#Exploration" + NGU-style
episodic memory), both game-agnostic:

  GLOBAL novelty   — counts persist for the whole run. Answers "have I *ever*
      seen this screen?". Drives discovery of genuinely new territory, but
      dries up once the reachable region is covered.
  EPISODIC novelty — counts reset every episode. Answers "have I seen this
      screen *this episode*?". Never dries up: every episode is rewarded
      afresh for covering ground, so the exploration gradient stands instead
      of vanishing — this is what stops the policy freezing in a
      survive-and-revisit local optimum.

`RewardComputer` owns both counters. Call `reset_episode()` at the start of
every episode (the rollout collector does this). It returns a per-step
breakdown so training metrics can attribute reward to each term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from claude_automate.framework.exploration import make_counter


@dataclass
class RewardBreakdown:
    total: float
    step: float
    stuck: float
    novelty_global: float
    novelty_episodic: float
    completion: float
    frame_changed: bool
    global_count: int

    @property
    def novelty(self) -> float:
        """Combined novelty bonus (global + episodic)."""
        return self.novelty_global + self.novelty_episodic


class RewardComputer:
    """Composes the generalizable reward for one transition.

    The global counter persists across episodes; the episodic counter is
    rebuilt by `reset_episode()` so its visit counts are per-episode.
    """

    def __init__(self, cfg, masked_rows=None):
        self.cfg = cfg
        self._masked_rows = masked_rows
        self.global_counter = self._new_counter()
        self.episodic_counter = self._new_counter()

    def _new_counter(self):
        return make_counter(
            self.cfg.count_mode, hash_bits=self.cfg.hash_bits,
            frame_size=self.cfg.frame_size, seed=self.cfg.hash_seed,
            masked_rows=self._masked_rows,
        )

    def reset_episode(self) -> None:
        """Clear episodic visit counts — call at the start of each episode."""
        self.episodic_counter = self._new_counter()

    def compute(self, env, prev_frame: np.ndarray, cur_frame: np.ndarray,
                level_completed: bool) -> RewardBreakdown:
        cfg = self.cfg

        # Did anything observable change? (UI rows already masked by env.)
        diff = env.frame_diff(prev_frame, cur_frame)
        frame_changed = bool(diff.sum() > 1e-6)

        step_pen = -cfg.w_step
        stuck_pen = 0.0 if frame_changed else -cfg.w_stuck

        g_count = self.global_counter.visit(cur_frame)
        e_count = self.episodic_counter.visit(cur_frame)
        nov_global = min(cfg.w_novel * self.global_counter.novelty(g_count),
                         cfg.novelty_clip)
        nov_epis = min(cfg.w_novel_episodic
                       * self.episodic_counter.novelty(e_count),
                       cfg.novelty_clip)

        completion = cfg.w_complete if level_completed else 0.0

        total = step_pen + stuck_pen + nov_global + nov_epis + completion
        return RewardBreakdown(
            total=total, step=step_pen, stuck=stuck_pen,
            novelty_global=nov_global, novelty_episodic=nov_epis,
            completion=completion, frame_changed=frame_changed,
            global_count=g_count,
        )
