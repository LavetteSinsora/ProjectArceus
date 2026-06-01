"""Dual-stream rollout buffer + per-stream GAE for RND PPO.

Generalises exp_010's rollout to two reward streams:
    * extrinsic — terminal-only {0,1}, EPISODIC value head V_E, gamma_ext
    * intrinsic — RND prediction error (normalised), NON-EPISODIC head V_I, gamma_int

`next_obs` is stored so the trainer can compute the intrinsic reward and the
predictor distillation loss from the same buffer.

Layout (T, N, ...):
    obs / next_obs   uint8   (T, N, F, F)
    actions          int64   (T, N)
    log_probs        float32 (T, N)
    values_ext/int   float32 (T, N)
    rewards_ext      float32 (T, N)            # terminal-only extrinsic reward
    dones            bool    (T, N)            # True iff action at t ended the ep
    features         float32 (T, N, trunk_dim)
    ep_starts        bool    (T, N)            # True if this step began a new ep
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
    values_ext: torch.Tensor
    values_int: torch.Tensor
    rewards_ext: torch.Tensor
    dones: torch.Tensor
    features: torch.Tensor
    ep_starts: torch.Tensor
    bootstrap_value_ext: torch.Tensor
    bootstrap_value_int: torch.Tensor
    frame: int = 64
    # filled in by the trainer / compute_gae
    target_feats: torch.Tensor | None = None   # cached frozen-target embedding of next_obs (T, N, D)
    rewards_int: torch.Tensor | None = None
    adv_ext: torch.Tensor | None = None
    ret_ext: torch.Tensor | None = None
    adv_int: torch.Tensor | None = None
    ret_int: torch.Tensor | None = None


class RolloutBuffer:
    def __init__(self, T: int, N: int, trunk_dim: int, frame: int = 64):
        self.T, self.N, self.F = T, N, frame
        self.obs = np.zeros((T, N, frame, frame), dtype=np.uint8)
        self.next_obs = np.zeros((T, N, frame, frame), dtype=np.uint8)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.values_ext = np.zeros((T, N), dtype=np.float32)
        self.values_int = np.zeros((T, N), dtype=np.float32)
        self.rewards_ext = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=bool)
        self.features = np.zeros((T, N, trunk_dim), dtype=np.float32)
        self.ep_starts = np.zeros((T, N), dtype=bool)

    def store(self, t, obs, next_obs, actions, log_probs, values_ext, values_int,
              rewards_ext, dones, features, ep_starts):
        self.obs[t] = obs
        self.next_obs[t] = next_obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values_ext[t] = values_ext
        self.values_int[t] = values_int
        self.rewards_ext[t] = rewards_ext
        self.dones[t] = dones
        self.features[t] = features
        self.ep_starts[t] = ep_starts

    def finalise(self, bootstrap_ext: np.ndarray, bootstrap_int: np.ndarray) -> Rollout:
        return Rollout(
            obs=torch.from_numpy(self.obs),
            next_obs=torch.from_numpy(self.next_obs),
            actions=torch.from_numpy(self.actions),
            log_probs=torch.from_numpy(self.log_probs),
            values_ext=torch.from_numpy(self.values_ext),
            values_int=torch.from_numpy(self.values_int),
            rewards_ext=torch.from_numpy(self.rewards_ext),
            dones=torch.from_numpy(self.dones),
            features=torch.from_numpy(self.features),
            ep_starts=torch.from_numpy(self.ep_starts),
            bootstrap_value_ext=torch.from_numpy(bootstrap_ext),
            bootstrap_value_int=torch.from_numpy(bootstrap_int),
            frame=self.F,
        )


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, bootstrap: torch.Tensor,
                dones: torch.Tensor, gamma: float, lam: float, episodic: bool):
    """GAE for one reward stream.

    episodic=True  (extrinsic): V(s_{t+1}) and the accumulator are masked by
                   (1 - done_t) -> returns reset at episode boundaries.
    episodic=False (intrinsic): NO done-mask -> the intrinsic stream flows
                   across episode boundaries (RND's non-episodic intrinsic).
    """
    T, N = rewards.shape
    adv = torch.zeros(T, N, dtype=torch.float32)
    last_gae = torch.zeros(N, dtype=torch.float32)
    next_value = bootstrap.float()
    ones = torch.ones(N, dtype=torch.float32)
    for t in reversed(range(T)):
        nonterminal = (~dones[t]).float() if episodic else ones
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
        next_value = values[t]
    return adv, adv + values


def collect_rollout(envs, model, device, T: int) -> Rollout:
    """Collect T steps from the vec env. Extrinsic reward is terminal-only
    (from the env); intrinsic reward is added later by the trainer from RND."""
    N = envs.n_envs
    frame = envs.FRAME if hasattr(envs, "FRAME") else 64
    buf = RolloutBuffer(T=T, N=N, trunk_dim=model.encoder.trunk_dim, frame=frame)
    obs_np = envs.current_obs()
    prev_done = np.zeros(N, dtype=bool)

    for t in range(T):
        obs_t = torch.from_numpy(obs_np).to(device)
        action_t, logp_t, vext_t, vint_t, feat_t = model.act(obs_t)
        action_np = action_t.cpu().numpy().astype(np.int64)

        next_obs_np, raw_r, dones, infos = envs.step(action_np)

        buf.store(
            t,
            obs=obs_np,
            next_obs=next_obs_np,
            actions=action_np,
            log_probs=logp_t.cpu().numpy(),
            values_ext=vext_t.cpu().numpy(),
            values_int=vint_t.cpu().numpy(),
            rewards_ext=raw_r,
            dones=dones,
            features=feat_t.cpu().numpy(),
            ep_starts=prev_done,
        )
        prev_done = dones
        obs_np = next_obs_np

    with torch.no_grad():
        last_obs = torch.from_numpy(obs_np).to(device)
        _, last_vext, last_vint, _ = model.forward(last_obs)
    return buf.finalise(
        last_vext.cpu().numpy().astype(np.float32),
        last_vint.cpu().numpy().astype(np.float32),
    )
