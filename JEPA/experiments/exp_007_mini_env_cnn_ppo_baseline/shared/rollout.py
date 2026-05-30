"""Rollout buffer + GAE.

Stores T steps from N parallel envs. All tensors are on CPU during
collection; the trainer moves minibatches to device.

Layout (T, N, ...):
    obs           uint8   (T, N, 32, 32)
    actions       int64   (T, N)
    log_probs     float32 (T, N)
    values        float32 (T, N)
    rewards       float32 (T, N)            # shaped rewards
    dones         bool    (T, N)
    features      float32 (T, N, trunk_dim) # trunk feature h_t
    ep_starts     bool    (T, N)            # True if this step BEGAN a new ep
                                            # (i.e., previous step was done)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Rollout:
    obs: torch.Tensor          # (T, N, 32, 32) uint8
    actions: torch.Tensor      # (T, N) int64
    log_probs: torch.Tensor    # (T, N) float32
    values: torch.Tensor       # (T, N) float32
    rewards: torch.Tensor      # (T, N) float32
    dones: torch.Tensor        # (T, N) bool
    features: torch.Tensor     # (T, N, D) float32
    ep_starts: torch.Tensor    # (T, N) bool
    bootstrap_value: torch.Tensor  # (N,) float32 — V(s_T) for GAE
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None


class RolloutBuffer:
    def __init__(self, T: int, N: int, trunk_dim: int):
        self.T, self.N = T, N
        self.obs = np.zeros((T, N, 32, 32), dtype=np.uint8)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=bool)
        self.features = np.zeros((T, N, trunk_dim), dtype=np.float32)
        self.ep_starts = np.zeros((T, N), dtype=bool)

    def store(self, t: int, obs, actions, log_probs, values, rewards, dones, features, ep_starts):
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.features[t] = features
        self.ep_starts[t] = ep_starts

    def finalise(self, bootstrap_value: np.ndarray, device: torch.device) -> Rollout:
        return Rollout(
            obs=torch.from_numpy(self.obs),
            actions=torch.from_numpy(self.actions),
            log_probs=torch.from_numpy(self.log_probs),
            values=torch.from_numpy(self.values),
            rewards=torch.from_numpy(self.rewards),
            dones=torch.from_numpy(self.dones),
            features=torch.from_numpy(self.features),
            ep_starts=torch.from_numpy(self.ep_starts),
            bootstrap_value=torch.from_numpy(bootstrap_value),
        )


def compute_gae(rollout: Rollout, gamma: float, lam: float) -> Rollout:
    """In-place fill advantages and returns on the rollout.

    In our convention, ``rollout.dones[t]`` is True iff the action at step t
    terminated the episode. So at step t, V(s_{t+1}) is zero exactly when
    ``dones[t]`` is True, and the mask for delta_t and the GAE propagation
    is ``(1 - dones[t])``. (An earlier version of this loop took the mask
    from the *previous* iteration via a ``next_nonterminal`` carryover,
    which is correct only if dones[t] semantically means "is the next-step
    starting a new episode" — that is NOT our convention here. The off-by-
    one had the effect of zeroing the bootstrap at the step BEFORE a
    terminal, severing GAE propagation of terminal reward backward through
    the trajectory.)
    """
    T, N = rollout.rewards.shape
    advantages = torch.zeros(T, N, dtype=torch.float32)
    last_gae = torch.zeros(N, dtype=torch.float32)
    next_value = rollout.bootstrap_value.float()
    # Walk backward through time.
    for t in reversed(range(T)):
        nonterminal = (~rollout.dones[t]).float()
        delta = rollout.rewards[t] + gamma * next_value * nonterminal - rollout.values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = rollout.values[t]
    rollout.advantages = advantages
    rollout.returns = advantages + rollout.values
    return rollout


def collect_rollout(envs, model, device, T: int, shape_fn=None) -> Rollout:
    """Collect T steps from the vec env using model.act.

    `shape_fn(raw_rewards, infos) -> shaped_rewards` is applied inline so
    that the stored rollout rewards are the actual shaped signal seen by
    PPO. If None, rewards are stored unshaped (terminal-only).
    """
    N = envs.n_envs
    buf = RolloutBuffer(T=T, N=N, trunk_dim=model.encoder.trunk_dim)

    obs_np = envs.current_obs()  # (N, 32, 32) uint8
    prev_done = np.zeros(N, dtype=bool)  # initially no env just finished

    for t in range(T):
        obs_t = torch.from_numpy(obs_np).to(device)
        action_t, logp_t, value_t, feat_t = model.act(obs_t)
        action_np = action_t.cpu().numpy().astype(np.int64)

        next_obs_np, raw_r, dones, infos = envs.step(action_np)
        shaped_r = shape_fn(raw_r, infos) if shape_fn is not None else raw_r

        buf.store(
            t,
            obs=obs_np,
            actions=action_np,
            log_probs=logp_t.cpu().numpy(),
            values=value_t.cpu().numpy(),
            rewards=shaped_r,
            dones=dones,
            features=feat_t.cpu().numpy(),
            ep_starts=prev_done,
        )

        # The just-finished episode is the boundary for the NEXT step.
        prev_done = dones
        obs_np = next_obs_np

    # Bootstrap value for GAE at the end of the rollout.
    with torch.no_grad():
        last_obs = torch.from_numpy(obs_np).to(device)
        _, last_value, _ = model.forward(last_obs)
    bootstrap = last_value.cpu().numpy().astype(np.float32)

    return buf.finalise(bootstrap, device=device)
