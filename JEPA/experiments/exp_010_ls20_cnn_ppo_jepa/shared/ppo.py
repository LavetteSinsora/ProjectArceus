"""PPO loss + update step for real-LS20 CNN+PPO.

A 64x64-generalised port of exp_007/shared/ppo.py. The optional `extra_params`
argument lets a caller (the JEPA variants) include JEPA predictor / IDM
parameters in the same optimiser without changing the PPO maths — the JEPA
loss itself is added by the trainer, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .model import ActorCritic
from .rollout import Rollout


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    vf_clip_eps: float | None = 0.2
    c_value: float = 0.5
    c_entropy: float = 0.01
    grad_clip: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    norm_advantages: bool = True


@dataclass
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    grad_norm_total: float


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
               rollout: Rollout, cfg: PPOConfig, device: torch.device,
               clip_params=None) -> UpdateStats:
    """Run cfg.epochs epochs of cfg.minibatches updates over the rollout.

    `clip_params` is the parameter set passed to clip_grad_norm_; default is
    model.parameters(). Callers that share the optimiser with JEPA heads pass
    the full union so gradient clipping covers everything.
    """
    T, N = rollout.actions.shape
    F = rollout.frame
    batch_size = T * N
    mb_size = batch_size // cfg.minibatches

    b_obs = rollout.obs.reshape(batch_size, F, F)
    b_actions = rollout.actions.reshape(batch_size)
    b_logp_old = rollout.log_probs.reshape(batch_size)
    b_advantages = rollout.advantages.reshape(batch_size)
    b_returns = rollout.returns.reshape(batch_size)
    b_values_old = rollout.values.reshape(batch_size)

    if clip_params is None:
        clip_params = list(model.parameters())

    pl_sum = vl_sum = ent_sum = kl_sum = clip_sum = gn_sum = 0.0
    n_updates = 0

    idx = np.arange(batch_size)
    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for start in range(0, batch_size, mb_size):
            mb = idx[start:start + mb_size]
            mb_obs = b_obs[mb].to(device)
            mb_actions = b_actions[mb].to(device)
            mb_logp_old = b_logp_old[mb].to(device)
            mb_advantages = b_advantages[mb].to(device)
            mb_returns = b_returns[mb].to(device)
            mb_values_old = b_values_old[mb].to(device)

            if cfg.norm_advantages:
                a = mb_advantages
                mb_advantages = (a - a.mean()) / (a.std() + 1e-8)

            logp_new, entropy, value_new, _ = model.evaluate(mb_obs, mb_actions)
            ratio = (logp_new - mb_logp_old).exp()

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            if cfg.vf_clip_eps is not None:
                v_clipped = mb_values_old + torch.clamp(
                    value_new - mb_values_old, -cfg.vf_clip_eps, cfg.vf_clip_eps
                )
                vl_unclipped = (value_new - mb_returns).pow(2)
                vl_clipped = (v_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
            else:
                value_loss = 0.5 * (value_new - mb_returns).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + cfg.c_value * value_loss + cfg.c_entropy * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_logp_old - logp_new).mean()
                clipfrac = ((ratio - 1).abs() > cfg.clip_eps).float().mean()

            pl_sum += policy_loss.item()
            vl_sum += value_loss.item()
            ent_sum += (-entropy_loss).item()
            kl_sum += approx_kl.item()
            clip_sum += clipfrac.item()
            gn_sum += float(gn)
            n_updates += 1

    return UpdateStats(
        policy_loss=pl_sum / n_updates,
        value_loss=vl_sum / n_updates,
        entropy=ent_sum / n_updates,
        approx_kl=kl_sum / n_updates,
        clipfrac=clip_sum / n_updates,
        grad_norm_total=gn_sum / n_updates,
    )
