"""PPO loss + update step.

Includes a `grad_norm_decomp` helper that performs two extra backward passes
to attribute gradient magnitude on the shared encoder to (a) the actor loss
and (b) the critic loss separately. Run rarely (every `grad_decomp_every`
updates).
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
    vf_clip_eps: float | None = 0.2   # value-function clip range; None disables
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
               rollout: Rollout, cfg: PPOConfig, device: torch.device) -> UpdateStats:
    """Run cfg.epochs epochs of cfg.minibatches updates over the rollout."""
    T, N = rollout.actions.shape
    batch_size = T * N
    mb_size = batch_size // cfg.minibatches

    # Flatten time/env axes.
    b_obs = rollout.obs.reshape(batch_size, 32, 32)
    b_actions = rollout.actions.reshape(batch_size)
    b_logp_old = rollout.log_probs.reshape(batch_size)
    b_advantages = rollout.advantages.reshape(batch_size)
    b_returns = rollout.returns.reshape(batch_size)
    b_values_old = rollout.values.reshape(batch_size)

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
            gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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


def grad_norm_decomp(model: ActorCritic, rollout: Rollout,
                     cfg: PPOConfig, device: torch.device,
                     n_samples: int = 512) -> dict[str, float]:
    """Attribute encoder-parameter gradient norms to actor vs critic.

    Computes ‖∂(L_policy + c_ent·L_ent)/∂θ_enc‖₂ and ‖∂(c_v·L_value)/∂θ_enc‖₂
    over a random subset of `n_samples` transitions from `rollout`. Uses
    two separate backward passes, both with no `optimizer.step()`. Cheap
    enough to call every ~10 updates.
    """
    T, N = rollout.actions.shape
    batch_size = T * N
    n = min(n_samples, batch_size)
    idx = np.random.choice(batch_size, size=n, replace=False)

    obs = rollout.obs.reshape(batch_size, 32, 32)[idx].to(device)
    actions = rollout.actions.reshape(batch_size)[idx].to(device)
    advantages = rollout.advantages.reshape(batch_size)[idx].to(device)
    returns = rollout.returns.reshape(batch_size)[idx].to(device)
    logp_old = rollout.log_probs.reshape(batch_size)[idx].to(device)
    values_old = rollout.values.reshape(batch_size)[idx].to(device)

    if cfg.norm_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    encoder_params = list(model.encoder.parameters())

    # Re-build the graph fresh for each backward so the .grad fields don't
    # get conflated.
    def encoder_grad_norm(loss: torch.Tensor) -> float:
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=False)
        sq = 0.0
        for p in encoder_params:
            if p.grad is not None:
                sq += p.grad.detach().pow(2).sum().item()
        return float(np.sqrt(sq))

    # Actor branch: policy loss + entropy loss.
    logp_new, entropy, value_new, _ = model.evaluate(obs, actions)
    ratio = (logp_new - logp_old).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy_loss = -entropy.mean()
    actor_loss = policy_loss + cfg.c_entropy * entropy_loss
    g_actor = encoder_grad_norm(actor_loss)

    # Critic branch.
    logp_new, entropy, value_new, _ = model.evaluate(obs, actions)
    if cfg.vf_clip_eps is not None:
        v_clipped = values_old + torch.clamp(
            value_new - values_old, -cfg.vf_clip_eps, cfg.vf_clip_eps
        )
        vl_unclipped = (value_new - returns).pow(2)
        vl_clipped = (v_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
    else:
        value_loss = 0.5 * (value_new - returns).pow(2).mean()
    critic_loss = cfg.c_value * value_loss
    g_critic = encoder_grad_norm(critic_loss)

    model.zero_grad(set_to_none=True)

    return {
        "grad_norm_encoder_from_actor": g_actor,
        "grad_norm_encoder_from_critic": g_critic,
        "grad_norm_ratio_critic_over_actor": (g_critic / g_actor) if g_actor > 1e-12 else float("inf"),
    }
