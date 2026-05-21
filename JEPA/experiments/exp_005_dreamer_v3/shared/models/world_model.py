"""Composed Dreamer V3 world model: encoder + RSSM + decoder + reward + continue + P2E ensemble.

Public methods:
  observe(obs_seq, action_seq, prev_state)
      → dict with post, prior, recon dist, reward dist, continue dist, features
  imagine(start_h, start_z, actor, horizon)
      → traj (RSSMState), actions, log_pi, entropy, reward_dist, continue_dist, value-input features
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .decoder import ConvDecoder
from .encoder import ConvEncoder
from .ensemble import DynamicsEnsemble
from .heads import ContinueHead, RewardHead
from .rssm import RSSM, RSSMState


class WMObserveOutput(NamedTuple):
    post: RSSMState                # (B, T, ...)
    prior: RSSMState
    features: torch.Tensor         # (B, T, feat_dim)
    recon_dist: object             # SymlogMSEDist
    reward_dist: object            # TwohotSymlogDist
    continue_dist: object          # BernoulliDist


class WMImagineOutput(NamedTuple):
    traj: RSSMState                # (H, N, ...)
    features: torch.Tensor         # (H, N, feat_dim)
    actions: torch.Tensor          # (H, N, n_actions) one-hot
    log_pi: torch.Tensor           # (H, N)
    entropy: torch.Tensor          # (H, N)
    reward_dist: object            # TwohotSymlogDist over (H, N, ...)
    continue_dist: object          # BernoulliDist over (H, N, ...)


class WorldModel(nn.Module):
    """Encoder + RSSM + decoder + reward + continue + (P2E ensemble)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = ConvEncoder(
            in_channels=cfg.obs_channels,
            depth=cfg.cnn_depth,
            embed_dim=cfg.embed_dim,
        )
        self.rssm = RSSM(
            embed_dim=cfg.embed_dim,
            n_actions=cfg.n_actions,
            deter=cfg.deter,
            n_groups=cfg.n_groups,
            n_classes=cfg.n_classes,
            hidden=cfg.hidden_units,
            unimix=cfg.unimix,
        )
        feat_dim = cfg.deter + cfg.n_groups * cfg.n_classes
        self.decoder = ConvDecoder(feat_dim=feat_dim, depth=cfg.cnn_depth, out_channels=cfg.obs_channels)
        self.reward_head = RewardHead(feat_dim, hidden=cfg.hidden_units,
                                      K=cfg.twohot_bins, low=cfg.twohot_low, high=cfg.twohot_high)
        self.continue_head = ContinueHead(feat_dim, hidden=cfg.hidden_units)

        # Plan2Explore ensemble (always present; sub-experiment toggles its usage).
        self.ensemble = DynamicsEnsemble(
            deter=cfg.deter,
            n_groups=cfg.n_groups,
            n_classes=cfg.n_classes,
            n_actions=cfg.n_actions,
            hidden=cfg.p2e_hidden_units,
            K=cfg.p2e_n_heads,
        )

    # ── Observation pipeline ─────────────────────────────────────────────────

    def encode_seq(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq: (B, T, C, H, W) → (B, T, embed_dim)."""
        B, T, C, H, W = obs_seq.shape
        emb = self.encoder(obs_seq.reshape(B * T, C, H, W))
        return emb.reshape(B, T, -1)

    def observe(
        self,
        obs_seq: torch.Tensor,        # (B, T, C, H, W)
        action_seq: torch.Tensor,     # (B, T, n_actions) one-hot
        prev_state: RSSMState | None = None,
    ) -> WMObserveOutput:
        x_emb = self.encode_seq(obs_seq)
        post, prior = self.rssm.observe(x_emb, action_seq, init=prev_state)
        feat = RSSM.feature(post)                                # (B, T, feat_dim)

        B, T, _ = feat.shape
        feat_flat = feat.reshape(B * T, -1)
        recon_dist = self._decoder_dist(feat_flat, image_shape=obs_seq.shape[-3:])
        rew_dist = self.reward_head(feat_flat)
        cont_dist = self.continue_head(feat_flat)

        return WMObserveOutput(
            post=post,
            prior=prior,
            features=feat,
            recon_dist=recon_dist,
            reward_dist=rew_dist,
            continue_dist=cont_dist,
        )

    def _decoder_dist(self, feat_flat: torch.Tensor, image_shape: tuple[int, int, int]):
        # Lazy import to avoid cycles.
        from .distributions import SymlogMSEDist
        out = self.decoder(feat_flat)                # (BT, C, H, W) in symlog space
        return SymlogMSEDist(out, event_dims=3)

    # ── Imagination ──────────────────────────────────────────────────────────

    def imagine(self, start_h: torch.Tensor, start_z: torch.Tensor, actor, horizon: int) -> WMImagineOutput:
        """Roll H steps from (start_h, start_z) using `actor`."""
        traj, actions, log_pi, entropy = self.rssm.imagine(start_h, start_z, actor, horizon)

        H, N = traj.h.shape[:2]
        feat = RSSM.feature(traj)                    # (H, N, feat_dim)
        feat_flat = feat.reshape(H * N, -1)
        rew_dist = self.reward_head(feat_flat)
        cont_dist = self.continue_head(feat_flat)
        return WMImagineOutput(
            traj=traj,
            features=feat,
            actions=actions,
            log_pi=log_pi,
            entropy=entropy,
            reward_dist=rew_dist,
            continue_dist=cont_dist,
        )

    # ── P2E disagreement reward over an imagined trajectory ──────────────────

    def p2e_intrinsic_reward(self, traj: RSSMState, actions: torch.Tensor) -> torch.Tensor:
        """Compute ensemble-disagreement intrinsic reward along an imagined trajectory.

        Args:
            traj.h:   (H, N, deter)
            traj.z:   (H, N, G, C)
            actions:  (H, N, n_actions)  one-hot taken at time t (alignment: action[t] led to traj[t+1]; we use actions[t] with state[t] as a sanity-aligned predictor input)
        Returns:
            (H, N) intrinsic reward.
        """
        H, N = traj.h.shape[:2]
        h_flat = traj.h.reshape(H * N, -1)
        z_flat = traj.z.reshape(H * N, *traj.z.shape[2:])
        a_flat = actions.reshape(H * N, -1)
        dis = self.ensemble.disagreement(h_flat, z_flat, a_flat)
        return dis.reshape(H, N)
