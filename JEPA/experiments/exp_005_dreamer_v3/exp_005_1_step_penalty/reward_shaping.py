"""Reward = 1 on level_completed else 0, MINUS step_penalty every step."""

from __future__ import annotations


def make_reward_fn(cfg):
    sp = cfg.step_penalty

    def reward_fn(prev_frame, frame, env, info):
        r = (1.0 if env.level_completed else 0.0) - sp
        return r, None, {}

    return reward_fn
