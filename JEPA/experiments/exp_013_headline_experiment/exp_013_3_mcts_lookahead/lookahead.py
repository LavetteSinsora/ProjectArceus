"""Proposal D — B1 lookahead-softmax controller (MCTS-organized, depth-1).

ACTOR-FREE control (no PPO policy net). At each state, for each action we use the ICM
forward model to predict the next latent, score it by novelty + value, and act by a
softmax over those per-action scores:

    Q(s,a) = novelty(φ̂'_a) + γ · V_int(φ̂'_a),   φ̂'_a = ICM.forward(φ(s), a)   # MODEL for the DECISION
    π(a|s) = softmax( standardize_a(Q) / τ )                                   # Boltzmann, sampled

Key accuracy invariants (SYSTEM_CARD §4.7, §"how V_int is trained"):
  * the MODEL is used ONLY for the action DECISION (ranking predicted next states);
  * V_int is trained by TD/GAE on the REAL novelty reward at the REAL next state — model
    error never enters the value target ("act with the model, learn from reality");
  * there is NO policy gradient, so the value-lag / phantom-advantage entropy collapse
    cannot arise (the policy is recomputed each step from Q, not pushed by an advantage);
  * softmax (not argmax) → stochastic → loop-free; τ is the exploration temperature.
    Q is standardised across the A actions per state so τ is robust to the V_int scale.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import _orth
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import RolloutBuffer


class ValueMLP(nn.Module):
    """Intrinsic value over the latent: V_int(φ) → scalar. Evaluated at BOTH the real
    φ(s) (for GAE / training) and the predicted φ̂'_a (for the lookahead Q)."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _orth(nn.Linear(dim, hidden), 2 ** 0.5), nn.ReLU(inplace=True),
            _orth(nn.Linear(hidden, 1), 1.0),
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi).squeeze(-1)


@torch.no_grad()
def _lookahead_Q(icm, rndphi, value, phi_s, n_actions, gamma):
    """phi_s (N, dim) → Q (N, A): for each action, predict φ̂' and score nov+γV."""
    N = phi_s.shape[0]
    cols = []
    for a in range(n_actions):
        a_vec = torch.full((N,), a, dtype=torch.long, device=phi_s.device)
        phi_hat = icm.predict_next(phi_s, a_vec)          # MODEL 1-step (predicted next latent)
        cols.append(rndphi.novelty(phi_hat) + gamma * value(phi_hat))
    return torch.stack(cols, dim=-1)                       # (N, A)


@torch.no_grad()
def lookahead_act(icm, rndphi, value, phi_s, n_actions, gamma, tau):
    Q = _lookahead_Q(icm, rndphi, value, phi_s, n_actions, gamma)
    Qn = (Q - Q.mean(-1, keepdim=True)) / (Q.std(-1, keepdim=True) + 1e-6)   # per-state standardise
    dist = torch.distributions.Categorical(logits=Qn / tau)
    a = dist.sample()
    return a, dist.log_prob(a), dist.entropy(), Q


def collect_lookahead_rollout(envs, icm, rndphi, value, device, T, n_actions, gamma, tau):
    """Like exp_010.collect_rollout, but actions come from the lookahead-softmax policy
    and `values` stores V_int(φ(s)) for GAE. Records the REAL transitions."""
    N = envs.n_envs
    D = icm.trunk_dim
    Fz = envs.FRAME if hasattr(envs, "FRAME") else 64
    buf = RolloutBuffer(T=T, N=N, trunk_dim=D, frame=Fz)
    obs_np = envs.current_obs()
    prev_done = np.zeros(N, dtype=bool)
    ent_sum = 0.0
    for t in range(T):
        obs_t = torch.from_numpy(obs_np).to(device)
        with torch.no_grad():
            phi_s = icm.encode(obs_t)
            a, logp, ent, _Q = lookahead_act(icm, rndphi, value, phi_s, n_actions, gamma, tau)
            v_s = value(phi_s)                              # value of the CURRENT state (GAE)
        ent_sum += float(ent.mean())
        a_np = a.cpu().numpy().astype(np.int64)
        next_obs_np, raw_r, dones, infos = envs.step(a_np)
        buf.store(t, obs=obs_np, next_obs=next_obs_np, actions=a_np,
                  log_probs=logp.cpu().numpy(), values=v_s.cpu().numpy(),
                  rewards=raw_r, dones=dones, features=phi_s.cpu().numpy(), ep_starts=prev_done)
        prev_done = dones
        obs_np = next_obs_np
    with torch.no_grad():
        v_last = value(icm.encode(torch.from_numpy(obs_np).to(device)))
    rollout = buf.finalise(v_last.cpu().numpy().astype(np.float32))
    rollout.policy_entropy_mean = ent_sum / T               # attach for logging
    return rollout


def value_update(value, opt, rollout, icm, cfg, device):
    """Regress V_int(φ(s)) → GAE returns (the REAL intrinsic returns). φ is detached, so
    the value head trains on top of a fixed φ (φ is trained only by the ICM update)."""
    Fz = rollout.frame
    B = rollout.actions.numel()
    obs = rollout.obs.reshape(B, Fz, Fz)
    returns = rollout.returns.reshape(B)
    old_v = rollout.values.reshape(B)
    mb = max(1, B // cfg.minibatches)
    idx = np.arange(B)
    tot = 0.0
    steps = 0
    for _ in range(cfg.value_epochs):
        np.random.shuffle(idx)
        for s in range(0, B, mb):
            sel = idx[s:s + mb]
            with torch.no_grad():
                phi = icm.encode(obs[sel].to(device))       # φ detached from value training
            v = value(phi)
            tgt = returns[sel].to(device)
            if cfg.vf_clip_eps is not None:
                vo = old_v[sel].to(device)
                vc = vo + torch.clamp(v - vo, -cfg.vf_clip_eps, cfg.vf_clip_eps)
                loss = 0.5 * torch.max((v - tgt).pow(2), (vc - tgt).pow(2)).mean()
            else:
                loss = 0.5 * (v - tgt).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item()
            steps += 1
    return tot / max(1, steps)
