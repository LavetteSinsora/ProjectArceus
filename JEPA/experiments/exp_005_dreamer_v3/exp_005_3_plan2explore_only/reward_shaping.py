"""Zero extrinsic reward — P2E intrinsic only."""

from __future__ import annotations


def make_reward_fn(cfg):
    def reward_fn(prev_frame, frame, env, info):
        return 0.0, None, {}

    return reward_fn
