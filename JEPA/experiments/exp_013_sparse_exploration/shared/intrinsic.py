"""Pluggable intrinsic-reward modules for exp_013.

Both methods expose the same tiny interface so the dual-stream trainer is
method-agnostic:

    bonus.compute(rollout, device) -> raw_i : np.ndarray (T, N)   # no grad
    bonus.update(rollout, device)  -> dict (scalars to log)        # trains own nets
    bonus.state_dict() / .load_state_dict(sd)

`raw_i` is the UN-normalised per-transition novelty; the trainer divides it by a
running std of the intrinsic *returns* (the RND normaliser) — the SAME for both
methods, so the only thing that differs between the ICM and RND runs is how the
novelty itself is measured.

Done-step transitions (s' is a reset frame) are zeroed in `raw_i` for both
methods, so spurious reset-frame novelty never enters the intrinsic stream.

ICM here is the NORMALIZED variant: no frozen η, no one-time calibration. The
raw forward error goes straight into the shared normaliser (the 2018
large-scale-curiosity fix for the frozen-η collapse seen in exp_011).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# Reuse the audited modules verbatim (single source of truth).
from JEPA.experiments.exp_011_ls20_icm.shared.icm import (
    ICMModule, intrinsic_raw_error, icm_update_from_rollout,
)
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import (
    RNDTarget, RNDPredictor, batched_features, intrinsic_from_features,
)


class IntrinsicBonus:
    """Interface (not abstract — just documents the contract)."""

    def compute(self, rollout, device) -> np.ndarray:        # (T, N) float32
        raise NotImplementedError

    def update(self, rollout, device) -> dict:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, sd: dict) -> None:
        raise NotImplementedError


# ── ICM (normalized) ─────────────────────────────────────────────────────────

class ICMBonus(IntrinsicBonus):
    def __init__(self, cfg, device):
        self.icm = ICMModule(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                             frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim,
                             hidden=cfg.icm_hidden).to(device)
        self.opt = torch.optim.Adam(self.icm.parameters(), lr=cfg.icm_lr)
        # icm_update_from_rollout reads .minibatches/.icm_epochs/.beta/.grad_clip
        self._ucfg = SimpleNamespace(minibatches=cfg.minibatches,
                                     icm_epochs=cfg.icm_epochs,
                                     beta=cfg.icm_beta, grad_clip=cfg.grad_clip)

    def compute(self, rollout, device) -> np.ndarray:
        # intrinsic_raw_error already zeroes done-step transitions.
        raw, _mean = intrinsic_raw_error(self.icm, rollout, device)   # (T, N) torch cpu
        return raw.numpy().astype(np.float32)

    def update(self, rollout, device) -> dict:
        return icm_update_from_rollout(self.icm, self.opt, rollout, self._ucfg, device)

    def state_dict(self):
        return {"icm": self.icm.state_dict(), "opt": self.opt.state_dict()}

    def load_state_dict(self, sd):
        self.icm.load_state_dict(sd["icm"]); self.opt.load_state_dict(sd["opt"])


# ── RND ────────────────────────────────────────────────────────────────────────

class RNDBonus(IntrinsicBonus):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.fdim = cfg.rnd_feature_dim
        self.target = RNDTarget(n_colors=cfg.n_colors, frame_size=cfg.frame_size,
                                feature_dim=self.fdim).to(device)
        self.predictor = RNDPredictor(n_colors=cfg.n_colors, frame_size=cfg.frame_size,
                                      feature_dim=self.fdim,
                                      hidden=cfg.rnd_predictor_hidden).to(device)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=cfg.rnd_lr)

    def compute(self, rollout, device) -> np.ndarray:
        T, N = rollout.actions.shape
        Fz = rollout.frame
        flat_next = rollout.next_obs.reshape(T * N, Fz, Fz)
        tgt = batched_features(self.target, flat_next, device, self.fdim)
        pred = batched_features(self.predictor, flat_next, device, self.fdim)
        raw = intrinsic_from_features(pred, tgt).reshape(T, N).astype(np.float32)
        # Cache the frozen-target embedding so update() needn't recompute it,
        # and zero done-step transitions (reset frames) for parity with ICM.
        rollout.target_feats = tgt.reshape(T, N, self.fdim)
        raw = raw * (~rollout.dones).numpy().astype(np.float32)
        return raw

    def update(self, rollout, device) -> dict:
        cfg = self.cfg
        T, N = rollout.actions.shape
        Fz = rollout.frame
        valid = (~rollout.dones).reshape(-1)
        next_obs = rollout.next_obs.reshape(-1, Fz, Fz)[valid]
        tgt = rollout.target_feats.reshape(-1, self.fdim)[valid]
        n = next_obs.shape[0]
        if n == 0:
            return {"rnd_predictor_loss": float("nan")}
        mb = max(1, n // cfg.minibatches)
        idx = np.arange(n)
        tot = 0.0
        steps = 0
        for start in range(0, n, mb):
            sel = idx[start:start + mb]
            no = next_obs[sel].to(device)
            t = tgt[sel].to(device)
            pred = self.predictor(no)
            per = (pred - t).pow(2).mean(dim=-1)
            if cfg.predictor_update_proportion < 1.0:
                keep = (torch.rand(per.shape[0], device=device)
                        < cfg.predictor_update_proportion).float()
                loss = (per * keep).sum() / torch.clamp(keep.sum(), min=1.0)
            else:
                loss = per.mean()
            self.opt.zero_grad()
            (cfg.rnd_loss_coef * loss).backward()
            nn.utils.clip_grad_norm_(self.predictor.parameters(), cfg.grad_clip)
            self.opt.step()
            tot += loss.item(); steps += 1
        return {"rnd_predictor_loss": tot / max(1, steps)}

    def state_dict(self):
        return {"target": self.target.state_dict(),
                "predictor": self.predictor.state_dict(), "opt": self.opt.state_dict()}

    def load_state_dict(self, sd):
        self.target.load_state_dict(sd["target"])
        self.predictor.load_state_dict(sd["predictor"])
        self.opt.load_state_dict(sd["opt"])


def make_bonus(cfg, device) -> IntrinsicBonus:
    if cfg.method == "icm":
        return ICMBonus(cfg, device)
    if cfg.method == "rnd":
        return RNDBonus(cfg, device)
    raise ValueError(f"unknown method {cfg.method!r} (expected 'icm' or 'rnd')")
