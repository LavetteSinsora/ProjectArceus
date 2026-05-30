"""Rollout buffer + GAE for real-LS20 PPO.

A 64x64-generalised port of exp_007/shared/rollout.py. The only structural
change is that the frame size is configurable (`F`) instead of hardcoded 32,
and `next_obs` is also stored so the JEPA variants can form (s_t, a_t, s_{t+1})
transition pairs from the same buffer the PPO update consumes.

Layout (T, N, ...):
    obs        uint8   (T, N, F, F)
    next_obs   uint8   (T, N, F, F)         # s_{t+1}; for JEPA transition pairs
    actions    int64   (T, N)
    log_probs  float32 (T, N)
    values     float32 (T, N)
    rewards    float32 (T, N)
    dones      bool    (T, N)               # True iff action at t ended the ep
    features   float32 (T, N, trunk_dim)    # trunk feature h_t
    ep_starts  bool    (T, N)               # True if this step began a new ep
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Rollout:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    features: torch.Tensor
    ep_starts: torch.Tensor
    bootstrap_value: torch.Tensor
    frame: int = 64
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None


class RolloutBuffer:
    def __init__(self, T: int, N: int, trunk_dim: int, frame: int = 64):
        self.T, self.N, self.F = T, N, frame
        self.obs = np.zeros((T, N, frame, frame), dtype=np.uint8)
        self.next_obs = np.zeros((T, N, frame, frame), dtype=np.uint8)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=bool)
        self.features = np.zeros((T, N, trunk_dim), dtype=np.float32)
        self.ep_starts = np.zeros((T, N), dtype=bool)

    def store(self, t, obs, next_obs, actions, log_probs, values, rewards,
              dones, features, ep_starts):
        self.obs[t] = obs
        self.next_obs[t] = next_obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.features[t] = features
        self.ep_starts[t] = ep_starts

    def finalise(self, bootstrap_value: np.ndarray) -> Rollout:
        return Rollout(
            obs=torch.from_numpy(self.obs),
            next_obs=torch.from_numpy(self.next_obs),
            actions=torch.from_numpy(self.actions),
            log_probs=torch.from_numpy(self.log_probs),
            values=torch.from_numpy(self.values),
            rewards=torch.from_numpy(self.rewards),
            dones=torch.from_numpy(self.dones),
            features=torch.from_numpy(self.features),
            ep_starts=torch.from_numpy(self.ep_starts),
            bootstrap_value=torch.from_numpy(bootstrap_value),
            frame=self.F,
        )


def compute_gae(rollout: Rollout, gamma: float, lam: float) -> Rollout:
    """Fill advantages and returns. `dones[t]` True iff the action at step t
    terminated the episode, so V(s_{t+1}) is masked by (1 - dones[t]). This is
    the GAE-off-by-one-fixed convention from exp_007/shared/rollout.py."""
    T, N = rollout.rewards.shape
    advantages = torch.zeros(T, N, dtype=torch.float32)
    last_gae = torch.zeros(N, dtype=torch.float32)
    next_value = rollout.bootstrap_value.float()
    for t in reversed(range(T)):
        nonterminal = (~rollout.dones[t]).float()
        delta = rollout.rewards[t] + gamma * next_value * nonterminal - rollout.values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = rollout.values[t]
    rollout.advantages = advantages
    rollout.returns = advantages + rollout.values
    return rollout


def collect_rollout(envs, model, device, T: int) -> Rollout:
    """Collect T steps from the vec env using model.act. Rewards are terminal-
    only (computed inside the vec env), so no shaping hook is needed."""
    N = envs.n_envs
    buf = RolloutBuffer(T=T, N=N, trunk_dim=model.encoder.trunk_dim, frame=envs.FRAME
                        if hasattr(envs, "FRAME") else 64)
    obs_np = envs.current_obs()
    prev_done = np.zeros(N, dtype=bool)

    for t in range(T):
        obs_t = torch.from_numpy(obs_np).to(device)
        action_t, logp_t, value_t, feat_t = model.act(obs_t)
        action_np = action_t.cpu().numpy().astype(np.int64)

        next_obs_np, raw_r, dones, infos = envs.step(action_np)

        buf.store(
            t,
            obs=obs_np,
            next_obs=next_obs_np,
            actions=action_np,
            log_probs=logp_t.cpu().numpy(),
            values=value_t.cpu().numpy(),
            rewards=raw_r,
            dones=dones,
            features=feat_t.cpu().numpy(),
            ep_starts=prev_done,
        )
        prev_done = dones
        obs_np = next_obs_np

    with torch.no_grad():
        last_obs = torch.from_numpy(obs_np).to(device)
        _, last_value, _ = model.forward(last_obs)
    bootstrap = last_value.cpu().numpy().astype(np.float32)
    return buf.finalise(bootstrap)
