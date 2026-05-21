"""Sparse goal reward: r=1 on level_completed step, 0 otherwise."""

from __future__ import annotations


def make_reward_fn(cfg):
    """Build the reward function used by `shared/trainer.py`."""

    def reward_fn(prev_frame, frame, env, info):
        r = 1.0 if env.level_completed else 0.0
        return r, None, {}

    return reward_fn
