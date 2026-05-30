"""Rollout collection for the overfit test.

`collect_transitions(vec_env, action_fn, n_target)` runs the given vectorised
env, calling `action_fn(obs_uint8_np)` each step to get a batch of actions,
and returns a dict of flat tensors {obs, actions, next_obs, dones}.

Transitions where `done=True` are still returned (the caller can filter), but
in practice the `valid` mask = `~done` is what gets used during scoring,
matching how exp_007_4's training loop excludes (s_terminal, a, s_reset) from
the JEPA loss (see exp_007_4/train.py:198-200).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.vec_env import VecMiniEnv


def collect_transitions(
    vec_env: VecMiniEnv,
    action_fn: Callable[[np.ndarray], np.ndarray],
    n_target: int,
) -> dict[str, torch.Tensor]:
    """Collect at least `n_target` *valid* (non-done-crossing) transitions.

    Args:
        vec_env: already-reset VecMiniEnv.
        action_fn: callable taking obs (N, 32, 32) uint8 numpy -> actions
            (N,) int64 numpy in {0, 1, 2, 3}.
        n_target: minimum number of valid transitions to gather.

    Returns:
        dict with CPU tensors:
            obs       (M, 32, 32) uint8
            actions   (M,)        int64
            next_obs  (M, 32, 32) uint8
            dones     (M,)        bool      (True ⇒ next_obs is a reset frame)
        where M is the smallest multiple of n_envs s.t. (~dones).sum() >= n_target.
    """
    N = vec_env.n_envs
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    nxt_buf: list[np.ndarray] = []
    don_buf: list[np.ndarray] = []

    obs = vec_env.current_obs()  # (N, 32, 32) uint8
    n_valid = 0
    while n_valid < n_target:
        actions = action_fn(obs)
        next_obs, _rewards, dones, _infos = vec_env.step(actions)

        obs_buf.append(obs)
        act_buf.append(actions.astype(np.int64))
        nxt_buf.append(next_obs)
        don_buf.append(dones.astype(bool))

        n_valid += int((~dones).sum())
        # VecMiniEnv already drains completed-episode stats internally; we don't
        # need them here. Reset its drain list so it doesn't grow unbounded.
        vec_env.drain_completed_episodes()

        obs = next_obs  # next iter starts from the (possibly-reset) frame

    out_obs = torch.from_numpy(np.concatenate(obs_buf, axis=0))         # (M,32,32) uint8
    out_act = torch.from_numpy(np.concatenate(act_buf, axis=0))         # (M,)
    out_nxt = torch.from_numpy(np.concatenate(nxt_buf, axis=0))         # (M,32,32) uint8
    out_don = torch.from_numpy(np.concatenate(don_buf, axis=0))         # (M,)
    return {"obs": out_obs, "actions": out_act, "next_obs": out_nxt, "dones": out_don}


def trained_action_fn(model, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    """Stochastic sampling from the loaded ActorCritic policy."""
    @torch.no_grad()
    def _fn(obs_np: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(obs_np).to(device)
        action, _logp, _v, _feat = model.act(obs)
        return action.cpu().numpy().astype(np.int64)
    return _fn


def random_action_fn(n_actions: int, n_envs: int, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """Uniform-random action sampler."""
    rng = np.random.default_rng(seed)

    def _fn(_obs_np: np.ndarray) -> np.ndarray:
        return rng.integers(0, n_actions, size=n_envs, dtype=np.int64)
    return _fn
