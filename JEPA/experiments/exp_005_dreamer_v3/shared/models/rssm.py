"""Recurrent State-Space Model (RSSM) — the heart of Dreamer V3.

State at time t is the pair (h_t, z_t):
  h_t ∈ R^deter        — deterministic recurrent state (GRU hidden)
  z_t ∈ {0,1}^{G×C}    — stochastic state: G independent categoricals, each C-way,
                          sampled with straight-through one-hot

Modules:
  sequence model      h_t   = f_φ(h_{t-1}, z_{t-1}, a_{t-1})              — GRU
  prior (dynamics)    ẑ_t   ~ p_φ(ẑ_t | h_t)                               — MLP
  posterior (repr)    z_t   ~ q_φ(z_t | h_t, e_φ(x_t))                     — MLP

Public methods:
  obs_step(h_prev, z_prev, a_prev, x_emb) → posterior dict + prior dict
  img_step(h_prev, z_prev, a_prev)        → prior dict (no x_emb)
  observe(seq_x_emb, seq_a, h0, z0)       → roll a full (B, T, ...) batch
  imagine(start_h, start_z, actor, H)     → roll H imagined steps using actor
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .distributions import UnimixCategorical
from .functional import kl_balance_loss


class RSSMState(NamedTuple):
    h: torch.Tensor          # (..., deter)
    z: torch.Tensor          # (..., n_groups, n_classes)
    logits: torch.Tensor     # (..., n_groups, n_classes)


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 1) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


class RSSM(nn.Module):
    """Discrete-latent RSSM (Dreamer V3 §3, Eqs. 1-3)."""

    def __init__(
        self,
        embed_dim: int,
        n_actions: int,
        deter: int = 512,
        n_groups: int = 32,
        n_classes: int = 32,
        hidden: int = 512,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.deter = deter
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.stoch_dim = n_groups * n_classes
        self.n_actions = n_actions
        self.unimix = unimix

        # GRU input: stoch flat + one-hot action
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_dim + n_actions, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(input_size=hidden, hidden_size=deter)

        # Prior p(ẑ | h)
        self.prior_net = _mlp(deter, hidden, n_groups * n_classes, layers=1)
        # Posterior q(z | h, x_emb)
        self.post_net = _mlp(deter + embed_dim, hidden, n_groups * n_classes, layers=1)

    # ── Initial state ─────────────────────────────────────────────────────────

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        h = torch.zeros(batch_size, self.deter, device=device)
        z = torch.zeros(batch_size, self.n_groups, self.n_classes, device=device)
        logits = torch.zeros_like(z)
        return RSSMState(h=h, z=z, logits=logits)

    # ── Single-step helpers ───────────────────────────────────────────────────

    def _gru_step(self, h_prev: torch.Tensor, z_prev: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """Compute h_t from (h_{t-1}, z_{t-1}, a_{t-1})."""
        z_flat = z_prev.reshape(z_prev.shape[0], -1)
        inp = torch.cat([z_flat, a_prev], dim=-1)
        inp = self.pre_gru(inp)
        return self.gru(inp, h_prev)

    def _prior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h_t → (z̃_t, logits_t)."""
        logits = self.prior_net(h).reshape(-1, self.n_groups, self.n_classes)
        dist = UnimixCategorical(logits, mix=self.unimix)
        return dist.sample(), dist.logits

    def _posterior(self, h: torch.Tensor, x_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(h_t, x_emb) → (z_t, logits_t)."""
        logits = self.post_net(torch.cat([h, x_emb], dim=-1)).reshape(-1, self.n_groups, self.n_classes)
        dist = UnimixCategorical(logits, mix=self.unimix)
        return dist.sample(), dist.logits

    def obs_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
        x_emb: torch.Tensor,
    ) -> tuple[RSSMState, RSSMState]:
        """One step with an observation available.

        Args:
            h_prev: (B, deter)
            z_prev: (B, G, C)
            a_prev: (B, n_actions)  one-hot
            x_emb:  (B, embed_dim)
        Returns:
            post (RSSMState), prior (RSSMState) at time t.
        """
        h = self._gru_step(h_prev, z_prev, a_prev)
        z_post, lp_post = self._posterior(h, x_emb)
        _, lp_prior = self._prior(h)              # prior logits at same h
        post = RSSMState(h=h, z=z_post, logits=lp_post)
        prior = RSSMState(h=h, z=z_post, logits=lp_prior)  # h same; z_post stored on prior is unused
        return post, prior

    def img_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> RSSMState:
        """One imagination step (no observation; sample from the prior)."""
        h = self._gru_step(h_prev, z_prev, a_prev)
        z, lp = self._prior(h)
        return RSSMState(h=h, z=z, logits=lp)

    # ── Sequence rollouts ─────────────────────────────────────────────────────

    def observe(
        self,
        x_emb_seq: torch.Tensor,   # (B, T, embed_dim)
        a_seq: torch.Tensor,       # (B, T, n_actions) one-hot
        init: RSSMState | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        """Roll the RSSM through a (B, T) batch with observations.

        Action a_seq[:, t] is the action taken BEFORE seeing obs x_emb_seq[:, t].
        Returns concatenated post and prior states with shape (B, T, ...).
        """
        B, T, _ = x_emb_seq.shape
        if init is None:
            init = self.initial_state(B, x_emb_seq.device)
        h_t, z_t = init.h, init.z
        posts_h, posts_z, posts_lp = [], [], []
        priors_lp = []
        for t in range(T):
            post, prior = self.obs_step(h_t, z_t, a_seq[:, t], x_emb_seq[:, t])
            posts_h.append(post.h); posts_z.append(post.z); posts_lp.append(post.logits)
            priors_lp.append(prior.logits)
            h_t, z_t = post.h, post.z
        post_seq = RSSMState(
            h=torch.stack(posts_h, dim=1),
            z=torch.stack(posts_z, dim=1),
            logits=torch.stack(posts_lp, dim=1),
        )
        prior_seq = RSSMState(
            h=post_seq.h,                              # h is shared
            z=post_seq.z,                              # placeholder (unused)
            logits=torch.stack(priors_lp, dim=1),
        )
        return post_seq, prior_seq

    def imagine(
        self,
        start_h: torch.Tensor,    # (N, deter)
        start_z: torch.Tensor,    # (N, G, C)
        actor,                    # callable: (h, z) → action one-hot, log_prob, entropy
        horizon: int,
    ) -> tuple[RSSMState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll `horizon` steps purely from the prior, sampling actions from `actor`.

        Returns:
            traj    RSSMState with .h, .z, .logits each of shape (H, N, ...)
            actions one-hot (H, N, n_actions)
            log_pi  (H, N)
            entropy (H, N)
        """
        h, z = start_h, start_z
        hs, zs, lps = [], [], []
        acts, log_pis, ents = [], [], []
        for _ in range(horizon):
            a_dist = actor.distribution(h, z)
            a = a_dist.sample()                       # (N, n_actions) one-hot
            lp_a = a_dist.log_prob(a)                 # (N,)
            ent  = a_dist.entropy()                   # (N,)
            new = self.img_step(h, z, a)
            h, z = new.h, new.z
            hs.append(h); zs.append(z); lps.append(new.logits)
            acts.append(a); log_pis.append(lp_a); ents.append(ent)
        traj = RSSMState(
            h=torch.stack(hs, dim=0),
            z=torch.stack(zs, dim=0),
            logits=torch.stack(lps, dim=0),
        )
        return traj, torch.stack(acts, dim=0), torch.stack(log_pis, dim=0), torch.stack(ents, dim=0)

    # ── KL loss ───────────────────────────────────────────────────────────────

    def kl_loss(self, post: RSSMState, prior: RSSMState, free_nats: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (L_dyn, L_rep) with KL balancing + free bits — see functional.kl_balance_loss."""
        return kl_balance_loss(post.logits, prior.logits, free_nats=free_nats)

    # ── Feature ──────────────────────────────────────────────────────────────

    @staticmethod
    def feature(state: RSSMState) -> torch.Tensor:
        """Concatenate (h, z.flatten()) → feature for downstream heads/actor/critic."""
        z = state.z.reshape(*state.z.shape[:-2], -1)
        return torch.cat([state.h, z], dim=-1)
