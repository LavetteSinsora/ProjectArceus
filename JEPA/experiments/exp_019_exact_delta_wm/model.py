"""exp_019 — DeltaWorldModel: exact sparse-delta dynamics.

Predicts per cell: P(change) and P(new colour | change). Inference copies the
input frame and overwrites only cells with P(change) > tau. Unchanged cells are
exact by construction — the failure mode of exp_006's full-frame U-Net
(argmax over all 4096 cells never yields a fully-correct frame) is removed at
the parameterization level, not by more capacity.

Fully convolutional dilated stack (no pooling): spatial precision is the whole
game, and ARC dynamics are local (sprite movement, tile effects, timer tick).
~0.2M params — sized for CPU training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_COLORS = 16


class DeltaWorldModel(nn.Module):
    def __init__(self, n_colors: int = N_COLORS, n_actions: int = 4,
                 ch: int = 32, a_embed: int = 8):
        super().__init__()
        self.n_colors = n_colors
        self.action_embed = nn.Embedding(n_actions, a_embed)
        c_in = n_colors + a_embed
        dil = [1, 2, 4, 8, 1]
        layers = []
        prev = c_in
        for d in dil:
            layers += [nn.Conv2d(prev, ch, 3, padding=d, dilation=d),
                       nn.GroupNorm(8, ch), nn.ReLU(inplace=True)]
            prev = ch
        self.body = nn.Sequential(*layers)
        self.head_change = nn.Conv2d(ch, 1, 1)
        self.head_color = nn.Conv2d(ch, n_colors, 1)
        # global heads: terminal, and no-op gate ("does ANYTHING change?").
        # The no-op gate exists because blocked moves are ~half of all
        # transitions and per-cell FPs were ruining them (RESEARCH_LOG
        # 2026-06-10): a balanced global classification is far easier than
        # 4096 near-zero-rate cell decisions.
        self.head_term = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, 32),
            nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.head_noop = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(ch, 32),
            nn.ReLU(inplace=True), nn.Linear(32, 1))

    def forward(self, frame_onehot: torch.Tensor, action: torch.Tensor):
        a = self.action_embed(action)[:, :, None, None]
        a = a.expand(-1, -1, frame_onehot.shape[2], frame_onehot.shape[3])
        h = self.body(torch.cat([frame_onehot, a], dim=1))
        return (self.head_change(h)[:, 0],          # (B,64,64) logit
                self.head_color(h),                  # (B,16,64,64) logits
                self.head_term(h)[:, 0],             # (B,) logit
                self.head_noop(h)[:, 0])             # (B,) logit: P(no change)

    # ---------- inference ----------
    @torch.no_grad()
    def predict_batch(self, frames: np.ndarray, actions: np.ndarray,
                      tau: float = 0.5):  # tau sweep 2026-06-10: 0.5 best
        """frames (B,64,64) uint8, actions (B,) → next frames (B,64,64) uint8,
        terminal probs (B,)."""
        x = torch.from_numpy(frames).long()
        oh = F.one_hot(x, self.n_colors).permute(0, 3, 1, 2).float()
        act = torch.from_numpy(np.asarray(actions)).long()
        chg, col, term, noop = self.forward(oh, act)
        change = torch.sigmoid(chg) > tau
        # no-op gate: if P(no change) > 0.5, copy the frame verbatim
        is_noop = torch.sigmoid(noop) > 0.5
        change[is_noop] = False
        new_col = col.argmax(1)
        nxt = x.clone()
        nxt[change] = new_col[change]
        return nxt.numpy().astype(np.uint8), torch.sigmoid(term).numpy()


class FullFrameBaseline(nn.Module):
    """Ablation: identical backbone, but predicts all 4096 cells directly
    (exp_006-style parameterization)."""

    def __init__(self, n_colors: int = N_COLORS, n_actions: int = 4,
                 ch: int = 32, a_embed: int = 8):
        super().__init__()
        self.n_colors = n_colors
        self.action_embed = nn.Embedding(n_actions, a_embed)
        c_in = n_colors + a_embed
        dil = [1, 2, 4, 8, 1]
        layers = []
        prev = c_in
        for d in dil:
            layers += [nn.Conv2d(prev, ch, 3, padding=d, dilation=d),
                       nn.GroupNorm(8, ch), nn.ReLU(inplace=True)]
            prev = ch
        self.body = nn.Sequential(*layers)
        self.head_frame = nn.Conv2d(ch, n_colors, 1)
        self.head_term = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, 32),
            nn.ReLU(inplace=True), nn.Linear(32, 1))

    def forward(self, frame_onehot, action):
        a = self.action_embed(action)[:, :, None, None]
        a = a.expand(-1, -1, frame_onehot.shape[2], frame_onehot.shape[3])
        h = self.body(torch.cat([frame_onehot, a], dim=1))
        return self.head_frame(h), self.head_term(h)[:, 0]

    @torch.no_grad()
    def predict_batch(self, frames, actions, tau=None):
        x = torch.from_numpy(frames).long()
        oh = F.one_hot(x, self.n_colors).permute(0, 3, 1, 2).float()
        act = torch.from_numpy(np.asarray(actions)).long()
        logits, term = self.forward(oh, act)
        return (logits.argmax(1).numpy().astype(np.uint8),
                torch.sigmoid(term).numpy())
