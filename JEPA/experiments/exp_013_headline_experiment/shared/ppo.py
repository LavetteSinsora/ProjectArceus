"""Pure dual-stream PPO update for exp_013.

This is exp_012's RND PPO with the predictor-distillation coupling removed: the
intrinsic module owns its own network + optimiser (see intrinsic.py), so the
policy/value optimiser here trains ONLY the actor-critic (shared encoder + two
value heads + policy head). Everything else — clipped surrogate, combined-and-
normalised advantage, per-head value clip — is the exp_010/exp_012 recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class UpdateStats:
    policy_loss: float
    value_loss_ext: float
    value_loss_int: float
    entropy: float
    approx_kl: float
    clipfrac: float
    grad_norm_total: float


def _value_loss(value_new, value_old, returns, vf_clip):
    if vf_clip is not None:
        v_clipped = value_old + torch.clamp(value_new - value_old, -vf_clip, vf_clip)
        return 0.5 * torch.max((value_new - returns).pow(2),
                               (v_clipped - returns).pow(2)).mean()
    return 0.5 * (value_new - returns).pow(2).mean()


def ppo_update(model, optimizer, rollout, cfg, device) -> UpdateStats:
    T, N = rollout.actions.shape
    Fz = rollout.frame
    B = T * N
    mb_size = max(1, B // cfg.minibatches)

    b_obs = rollout.obs.reshape(B, Fz, Fz)
    b_actions = rollout.actions.reshape(B)
    b_logp_old = rollout.log_probs.reshape(B)
    b_adv_ext = rollout.adv_ext.reshape(B)
    b_adv_int = rollout.adv_int.reshape(B)
    b_ret_ext = rollout.ret_ext.reshape(B)
    b_ret_int = rollout.ret_int.reshape(B)
    b_vext_old = rollout.values_ext.reshape(B)
    b_vint_old = rollout.values_int.reshape(B)

    pl = vle = vli = ent = kl = clip = gn = 0.0
    n_updates = 0

    idx = np.arange(B)
    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for start in range(0, B, mb_size):
            mb = idx[start:start + mb_size]
            mb_obs = b_obs[mb].to(device)
            mb_actions = b_actions[mb].to(device)
            mb_logp_old = b_logp_old[mb].to(device)
            mb_ret_ext = b_ret_ext[mb].to(device)
            mb_ret_int = b_ret_int[mb].to(device)
            mb_vext_old = b_vext_old[mb].to(device)
            mb_vint_old = b_vint_old[mb].to(device)

            adv = cfg.ext_coef * b_adv_ext[mb] + cfg.int_coef * b_adv_int[mb]
            adv = adv.to(device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            logp_new, entropy, v_ext_new, v_int_new, _ = model.evaluate(mb_obs, mb_actions)
            ratio = (logp_new - mb_logp_old).exp()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            vloss_ext = _value_loss(v_ext_new, mb_vext_old, mb_ret_ext, cfg.vf_clip_eps)
            vloss_int = _value_loss(v_int_new, mb_vint_old, mb_ret_int, cfg.vf_clip_eps)
            value_loss = vloss_ext + vloss_int
            entropy_loss = -entropy.mean()

            loss = policy_loss + cfg.c_value * value_loss + cfg.c_entropy * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_logp_old - logp_new).mean()
                clipfrac = ((ratio - 1).abs() > cfg.clip_eps).float().mean()

            pl += policy_loss.item(); vle += vloss_ext.item(); vli += vloss_int.item()
            ent += entropy.mean().item(); kl += approx_kl.item()
            clip += clipfrac.item(); gn += float(grad_norm)
            n_updates += 1

    k = max(1, n_updates)
    return UpdateStats(pl / k, vle / k, vli / k, ent / k, kl / k, clip / k, gn / k)
