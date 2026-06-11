"""Tracker for exp_016_0 — IDM encoder (inverse-dynamics only, no forward model) +
a cross-update replay buffer, plus the leaky-RND count net (reused from exp_013_1b).
See SYSTEM_CARD §2/§3.

The IDM encoder is trained CONTINUOUSLY (no freeze); we measure the drift.
The replay buffer stores masked boards (already timer-masked) and EXCLUDES no-op
transitions (masked s_t == s_{t+1}) — wall-bumps carry no action signal.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import (
    CNNEncoder, one_hot_frame, _orth,
)

from .actor import mask_frames


class IDMEncoder(nn.Module):
    """Tracker encoder + inverse-dynamics head: predict a_t from (h_t, h_{t+1})."""

    def __init__(self, n_actions: int, n_colors: int = 16, frame_size: int = 64,
                 trunk_dim: int = 256, hidden: int = 256, layernorm: bool = False):
        super().__init__()
        self.encoder = CNNEncoder(n_colors=n_colors, frame_size=frame_size,
                                  trunk_dim=trunk_dim)
        # ABLATION: LayerNorm on h before it feeds the inverse head AND RND. Removes
        # the scale lever the inverse CE exploits (no ‖h‖ inflation → novelty no
        # longer ∝ ‖h‖²). Tests whether feature-norm or coverage drives the collapse.
        self.ln = nn.LayerNorm(trunk_dim) if layernorm else nn.Identity()
        self.inverse = nn.Sequential(
            _orth(nn.Linear(2 * trunk_dim, hidden), 2 ** 0.5), nn.ReLU(inplace=True),
            _orth(nn.Linear(hidden, n_actions), 0.01),
        )
        self.n_colors = n_colors

    def encode_masked(self, masked_uint8: torch.Tensor) -> torch.Tensor:
        """Inputs are ALREADY timer-masked boards (B,H,W) uint8 → (B, trunk_dim)."""
        return self.ln(self.encoder(one_hot_frame(masked_uint8, self.n_colors)))

    def inverse_logits(self, h_t: torch.Tensor, h_tp1: torch.Tensor) -> torch.Tensor:
        return self.inverse(torch.cat([h_t, h_tp1], dim=-1))


class ReplayBuffer:
    """FIFO of timer-masked (s, a, s') transitions for IDM training. No-ops dropped."""

    def __init__(self, capacity: int, frame: int):
        self.cap = capacity
        self.F = frame
        self.s = np.zeros((capacity, frame, frame), dtype=np.uint8)
        self.sp = np.zeros((capacity, frame, frame), dtype=np.uint8)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.size = 0
        self.ptr = 0

    def add_batch(self, s_masked: np.ndarray, a: np.ndarray, sp_masked: np.ndarray,
                  dones: np.ndarray, drop_noops: bool):
        """Each array is flat (M, ...). Skips done-steps (reset frame) and, if
        requested, no-ops (masked s == s')."""
        for i in range(s_masked.shape[0]):
            if dones[i]:
                continue
            if drop_noops and np.array_equal(s_masked[i], sp_masked[i]):
                continue
            j = self.ptr
            self.s[j] = s_masked[i]; self.sp[j] = sp_masked[i]; self.a[j] = a[i]
            self.ptr = (self.ptr + 1) % self.cap
            self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=min(batch, self.size))
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.sp[idx]))


def idm_update(idm: IDMEncoder, opt, buf: ReplayBuffer, cfg, device) -> dict:
    """Train the IDM encoder + inverse head from the replay buffer. Returns
    (inverse_loss, inverse_acc_onpolicy)."""
    if buf.size < cfg.idm_batch:
        return {"idm_inverse_loss": float("nan"), "inverse_acc_onpolicy": float("nan")}
    tot_loss = tot_acc = 0.0
    for _ in range(cfg.idm_grad_steps):
        s, a, sp = buf.sample(cfg.idm_batch)
        s = s.to(device); sp = sp.to(device); a = a.to(device)
        h = idm.encode_masked(s)
        hp = idm.encode_masked(sp)
        logits = idm.inverse_logits(h, hp)
        loss = F.cross_entropy(logits, a)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(idm.parameters(), cfg.grad_clip)
        opt.step()
        with torch.no_grad():
            tot_acc += (logits.argmax(-1) == a).float().mean().item()
        tot_loss += loss.item()
    n = cfg.idm_grad_steps
    return {"idm_inverse_loss": tot_loss / n, "inverse_acc_onpolicy": tot_acc / n}


def rnd_update(rnd, opt, h_masked_visited: torch.Tensor, cfg, device) -> float:
    """Distil predictor→target on this rollout's visited features, then leak ONCE.
    h_masked_visited: (M, trunk_dim) detached features of non-done s'."""
    n = h_masked_visited.shape[0]
    if n == 0:
        rnd.apply_leak()
        return float("nan")
    mb = max(1, n // cfg.rnd_grad_steps)
    idx = np.arange(n)
    tot = 0.0; steps = 0
    for k in range(cfg.rnd_grad_steps):
        sel = idx[k * mb:(k + 1) * mb] if (k + 1) * mb <= n else idx[k * mb:]
        if len(sel) == 0:
            break
        h = h_masked_visited[sel].to(device)
        loss = rnd.distill_loss(h)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(rnd.predictor.parameters(), cfg.grad_clip)
        opt.step()
        tot += loss.item(); steps += 1
    rnd.apply_leak()                       # ONCE per update (decoupled from minibatches)
    return tot / max(1, steps)


@torch.no_grad()
def holdout_inverse_acc(idm: IDMEncoder, holdout, device, chunk: int = 512) -> float:
    """Inverse accuracy on a FIXED uniform-random masked transition set (φ's TRUE
    controllability — on-policy acc inflates as the policy narrows)."""
    s, a, sp = holdout
    correct = 0
    for i in range(0, s.shape[0], chunk):
        h = idm.encode_masked(s[i:i + chunk].to(device))
        hp = idm.encode_masked(sp[i:i + chunk].to(device))
        logits = idm.inverse_logits(h, hp)
        correct += (logits.argmax(-1).cpu() == a[i:i + chunk]).sum().item()
    return correct / max(1, s.shape[0])
