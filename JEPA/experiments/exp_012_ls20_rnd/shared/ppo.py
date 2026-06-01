"""Dual-stream PPO update + RND predictor distillation, for exp_012_1.

A port of exp_010's clipped-surrogate PPO with two changes for faithful RND:

  1. Combined advantage  A = ext_coef * A_E + int_coef * A_I  drives the policy
     surrogate; the two value heads are each trained against their own returns
     (each with the exp_010 value-clip).
  2. The RND predictor is trained in the same minibatch loop: a distillation
     loss MSE(f_hat(s'), sg(f(s'))) is added to the total loss, optionally on a
     random `predictor_update_proportion` fraction of the batch.

The clipped-surrogate maths, advantage normalisation (on the *combined*
advantage), value-clip, grad-clip, and epoch/minibatch schedule are otherwise
identical to exp_010.
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
    rnd_loss: float


def _value_loss(value_new, value_old, returns, vf_clip):
    if vf_clip is not None:
        v_clipped = value_old + torch.clamp(value_new - value_old, -vf_clip, vf_clip)
        return 0.5 * torch.max((value_new - returns).pow(2),
                               (v_clipped - returns).pow(2)).mean()
    return 0.5 * (value_new - returns).pow(2).mean()


def ppo_update(model, predictor, optimizer, rollout, cfg, device) -> UpdateStats:
    """cfg is the experiment Config (carries clip/coef/RND fields). The frozen
    RND target is NOT passed: its embedding of next_obs is precomputed once per
    rollout (rollout.target_feats) and indexed per minibatch."""
    T, N = rollout.actions.shape
    Fz = rollout.frame
    B = T * N
    mb_size = B // cfg.minibatches

    b_obs = rollout.obs.reshape(B, Fz, Fz)
    b_next = rollout.next_obs.reshape(B, Fz, Fz)
    b_target_feats = rollout.target_feats.reshape(B, -1)
    b_actions = rollout.actions.reshape(B)
    b_logp_old = rollout.log_probs.reshape(B)
    b_adv_ext = rollout.adv_ext.reshape(B)
    b_adv_int = rollout.adv_int.reshape(B)
    b_ret_ext = rollout.ret_ext.reshape(B)
    b_ret_int = rollout.ret_int.reshape(B)
    b_vext_old = rollout.values_ext.reshape(B)
    b_vint_old = rollout.values_int.reshape(B)

    clip_params = list(model.parameters()) + list(predictor.parameters())

    pl = vle = vli = ent = kl = clip = gn = rl = 0.0
    n_updates = 0

    idx = np.arange(B)
    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for start in range(0, B, mb_size):
            mb = idx[start:start + mb_size]
            mb_obs = b_obs[mb].to(device)
            mb_next = b_next[mb].to(device)
            mb_actions = b_actions[mb].to(device)
            mb_logp_old = b_logp_old[mb].to(device)
            mb_ret_ext = b_ret_ext[mb].to(device)
            mb_ret_int = b_ret_int[mb].to(device)
            mb_vext_old = b_vext_old[mb].to(device)
            mb_vint_old = b_vint_old[mb].to(device)

            # Combined advantage, then normalise (exp_010 convention).
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

            # RND predictor distillation against the cached frozen-target feats.
            mb_tgt = b_target_feats[mb].to(device)
            pred = predictor(mb_next)
            rnd_per = (pred - mb_tgt).pow(2).mean(dim=-1)
            if cfg.predictor_update_proportion < 1.0:
                keep = (torch.rand(rnd_per.shape[0], device=device)
                        < cfg.predictor_update_proportion).float()
                rnd_loss = (rnd_per * keep).sum() / torch.clamp(keep.sum(), min=1.0)
            else:
                rnd_loss = rnd_per.mean()

            loss = (policy_loss + cfg.c_value * value_loss
                    + cfg.c_entropy * entropy_loss + cfg.rnd_loss_coef * rnd_loss)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_logp_old - logp_new).mean()
                clipfrac = ((ratio - 1).abs() > cfg.clip_eps).float().mean()

            pl += policy_loss.item(); vle += vloss_ext.item(); vli += vloss_int.item()
            ent += entropy.mean().item(); kl += approx_kl.item()
            clip += clipfrac.item(); gn += float(grad_norm); rl += rnd_loss.item()
            n_updates += 1

    k = max(1, n_updates)
    return UpdateStats(pl / k, vle / k, vli / k, ent / k, kl / k, clip / k, gn / k, rl / k)
