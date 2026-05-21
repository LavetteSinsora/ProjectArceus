"""Deterministic evaluator — runs the actor with argmax on a fresh env.

Reports the level-completion rate and average solve length over N rollouts.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def _obs_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    # frame: (H, W) uint8 → (1, 1, H, W) float in [-0.5, 0.5]
    return torch.from_numpy(frame).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 15.0 - 0.5


@torch.no_grad()
def evaluate(
    wm,
    actor,
    make_env: Callable,
    n_episodes: int,
    device: torch.device,
    max_steps: int = 200,
) -> dict:
    """Run `n_episodes` greedy rollouts. Returns a dict of summary stats."""
    completes = 0
    lengths: list[int] = []
    returns: list[float] = []
    for _ in range(n_episodes):
        env = make_env()
        frame = env.reset()
        state = wm.rssm.initial_state(1, device)
        last_a = torch.zeros(1, wm.cfg.n_actions, device=device)
        ep_return = 0.0
        for t in range(max_steps):
            x_emb = wm.encoder(_obs_to_tensor(frame, device))
            post, _ = wm.rssm.obs_step(state.h, state.z, last_a, x_emb)
            state = post
            a_onehot = actor.act(state.h, state.z, deterministic=True)
            a_idx = int(a_onehot.argmax(dim=-1).item())
            last_a = a_onehot
            frame, done = env.step(a_idx)
            ep_return += 1.0 if env.level_completed else 0.0
            if done:
                break
        if env.level_completed:
            completes += 1
        lengths.append(t + 1)
        returns.append(ep_return)
    return {
        "completion_rate": completes / n_episodes,
        "avg_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "avg_return": float(np.mean(returns)),
    }
