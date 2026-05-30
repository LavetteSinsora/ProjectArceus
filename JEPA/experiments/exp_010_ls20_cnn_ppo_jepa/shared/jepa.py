"""JEPA loss + update over (s_t, a_t, s_{t+1}) transitions.

Used by:
  * exp_010_1 — online, on the PPO agent's own rollout transitions, jointly
    with PPO (encoder shared & unfrozen).
  * exp_010_2 — offline, on a fixed buffer of random-policy transitions, to
    pretrain the encoder before PPO.

JEPA loss (stop-gradient target, à la exp_007_3 / exp_007_4):
    h_t      = encoder(one_hot(s_t))
    h_next   = encoder(one_hot(s_{t+1}))            # target branch is detached
    h_pred   = predictor(h_t, a_t)
    L_jepa   = MSE(h_pred, sg(h_next))
    L_idm    = CE(idm(h_t, h_next), a_t)            # encoder gets this grad too
    L        = L_jepa + idm_coef * L_idm

Transitions where the step ended an episode are excluded: their `s_{t+1}` is a
reset frame, not a real env transition (same rule as exp_008).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def jepa_losses_on_batch(model, predictor, idm, obs, next_obs, actions,
                         idm_coef: float):
    """Compute (L_jepa, L_idm, idm_acc) on a minibatch of palette-index frames.

    obs / next_obs: (B, H, W) uint8 on device. actions: (B,) int64 on device.
    """
    h_t = model.features(obs)
    h_next = model.features(next_obs)
    h_pred = predictor(h_t, actions)
    l_jepa = F.mse_loss(h_pred, h_next.detach())
    idm_logits = idm(h_t, h_next)
    l_idm = F.cross_entropy(idm_logits, actions)
    with torch.no_grad():
        idm_acc = (idm_logits.argmax(-1) == actions).float().mean().item()
    return l_jepa, l_idm, idm_acc


def jepa_update_from_rollout(model, predictor, idm, optimizer, rollout, cfg,
                             device) -> dict:
    """One pass of JEPA training over a PPO rollout's transitions (online,
    on-policy). Shares `optimizer` with PPO so the encoder is trained by both.
    Returns a dict of mean losses for logging."""
    T, N = rollout.actions.shape
    Fz = rollout.frame
    valid = (~rollout.dones).reshape(-1)               # exclude episode-ending steps
    obs = rollout.obs.reshape(-1, Fz, Fz)[valid]
    next_obs = rollout.next_obs.reshape(-1, Fz, Fz)[valid]
    actions = rollout.actions.reshape(-1)[valid]
    n = obs.shape[0]
    if n == 0:
        return {"jepa_loss": float("nan"), "idm_loss": float("nan"), "idm_acc": float("nan")}

    mb = max(1, n // cfg.minibatches)
    idx = np.arange(n)
    jl = il = ia = 0.0
    steps = 0
    for _ in range(cfg.jepa_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, mb):
            sel = idx[start:start + mb]
            o = obs[sel].to(device)
            no = next_obs[sel].to(device)
            a = actions[sel].to(device)
            l_jepa, l_idm, acc = jepa_losses_on_batch(model, predictor, idm, o, no, a,
                                                      cfg.idm_coef)
            loss = cfg.jepa_coef * l_jepa + cfg.idm_coef * l_idm
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(predictor.parameters())
                + list(idm.parameters()), cfg.grad_clip)
            optimizer.step()
            jl += l_jepa.item(); il += l_idm.item(); ia += acc; steps += 1
    return {"jepa_loss": jl / steps, "idm_loss": il / steps, "idm_acc": ia / steps}
