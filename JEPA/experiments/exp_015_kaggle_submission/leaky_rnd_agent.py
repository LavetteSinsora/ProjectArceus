"""
ARC-AGI-3 Kaggle submission — Leaky RND + Policy Gradient.

Architecture:
  - CNN backbone (Goose-style): 16-ch one-hot → conv4 → feature vector
  - Frozen-random projection of conv4 features → RND φ  (Method A, no ICM)
  - Leaky RND: novelty = ½‖P(φ)−T(φ)‖², predictor leaked toward init each update
  - Policy head: linear(feat → n_actions) with available-action masking
  - Value head: linear(feat → 1)  for baseline subtraction
  - Coordinate head: if ACTION6, sample (x,y) from spatial conv map
  - Update every UPDATE_EVERY steps: 1-epoch policy gradient (REINFORCE + baseline)
    + RND distillation + apply_leak()

Online: yes. No holdout set. No φ-freeze (frozen-random φ never needs freezing).

Place this file as agents/leaky_rnd_agent.py in the ARC-AGI-3-Agents fork.
Register: AVAILABLE_AGENTS = {"leakyrnd": LeakyRNDAgent, ...}
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arcengine import FrameData, GameAction, GameState

try:
    from agents.agent import Agent
except ImportError:
    Agent = object  # type: ignore[assignment,misc]


# ── CNN backbone (shared encoder) ─────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """Goose-style conv backbone → flat feature vector."""
    def __init__(self, in_ch: int = 16, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),           # → (B, 256, 4, 4)
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feat_dim), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)   # (B, feat_dim)


# ── Leaky RND ─────────────────────────────────────────────────────────────────
# Identical logic to exp_013_1b/rnd_phi.py — inlined so submission is self-contained.

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=2):
        super().__init__()
        seq = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.ReLU()]
        seq.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*seq)
    def forward(self, x): return self.net(x)


class LeakyRND(nn.Module):
    """
    Frozen random target T + trainable leaky predictor P.
    novelty(φ) = ½‖P(φ) − T(φ)‖²
    apply_leak(): θ_P ← (1−μ)·θ_P + μ·θ_P^init
    """
    def __init__(self, phi_dim: int = 128, hidden: int = 256,
                 out_dim: int = 256, leak: float = 0.05):
        super().__init__()
        self.target    = _MLP(phi_dim, hidden, out_dim, layers=2)
        self.predictor = _MLP(phi_dim, hidden, out_dim, layers=3)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()
        self.leak = leak
        # Snapshot init weights for the leak
        self._pred_init = [p.detach().clone() for p in self.predictor.parameters()]

    @torch.no_grad()
    def novelty(self, phi: torch.Tensor) -> torch.Tensor:
        """(B, phi_dim) → (B,) raw novelty."""
        return 0.5 * (self.predictor(phi) - self.target(phi)).pow(2).mean(dim=-1)

    def distill_loss(self, phi: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t = self.target(phi)
        return (self.predictor(phi) - t).pow(2).mean()

    @torch.no_grad()
    def apply_leak(self):
        for p, p0 in zip(self.predictor.parameters(), self._pred_init):
            p.mul_(1.0 - self.leak).add_(p0.to(p.device), alpha=self.leak)


# ── Policy + value + coordinate heads ─────────────────────────────────────────

ALL_ACTIONS = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
N_ACTIONS   = len(ALL_ACTIONS)   # 7
GRID        = 64

class PolicyHead(nn.Module):
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.action = nn.Linear(feat_dim, N_ACTIONS)  # 7 logits
        self.value  = nn.Linear(feat_dim, 1)
        # Coordinate head: two-stage (x then y|x)
        self.x_head = nn.Linear(feat_dim, GRID)       # P(x|feat)
        self.x_emb  = nn.Embedding(GRID, 32)
        self.y_head = nn.Linear(feat_dim + 32, GRID)  # P(y|feat, x)

    def forward(self, feat, avail_idx):
        logits = self.action(feat)                     # (B, 7)
        mask = torch.full_like(logits, float("-inf"))
        mask[:, avail_idx] = 0.0
        logits = logits + mask
        return logits, self.value(feat).squeeze(-1)    # (B, 7), (B,)

    def sample(self, feat, avail_idx):
        logits, v = self.forward(feat, avail_idx)
        dist  = torch.distributions.Categorical(logits=logits)
        a     = dist.sample()                          # scalar
        logp  = dist.log_prob(a)
        x = y = None
        if ALL_ACTIONS[int(a)] == "ACTION6":
            xd = torch.distributions.Categorical(logits=self.x_head(feat))
            x  = xd.sample()
            yd = torch.distributions.Categorical(
                    logits=self.y_head(torch.cat([feat, self.x_emb(x)], dim=-1)))
            y  = yd.sample()
            logp = logp + xd.log_prob(x) + yd.log_prob(y)
        return int(a), x, y, logp, v


# ── Frozen random φ projection (φ = frozen linear map from CNN features) ──────

class FrozenProjection(nn.Module):
    """Deterministic, frozen random linear: feat_dim → phi_dim. Seeded."""
    def __init__(self, feat_dim: int = 256, phi_dim: int = 128, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(feat_dim, phi_dim, generator=g) / (feat_dim ** 0.5)
        self.register_buffer("W", W)
    def forward(self, feat):
        return feat @ self.W   # (B, phi_dim)


# ── The Agent ─────────────────────────────────────────────────────────────────

class LeakyRNDAgent(Agent):  # type: ignore[misc]
    """
    Online leaky-RND + policy-gradient agent for ARC-AGI-3.

    Each step:
      1. Encode frame → feat (CNN)
      2. feat → φ (frozen random projection)
      3. novelty = LeakyRND.novelty(φ)   [intrinsic reward]
      4. Sample action from PolicyHead (masked softmax + coordinate heads)
      5. Store transition in buffer
      6. Every UPDATE_EVERY steps: 1-epoch REINFORCE + RND distill + apply_leak

    No ICM, no holdout, no φ-freeze — frozen-random φ never needs freezing.
    """

    # ── config ────────────────────────────────────────────────────────────────
    FEAT_DIM      = 256
    PHI_DIM       = 128
    HIDDEN        = 256
    RND_OUT       = 256
    LEAK          = 0.05
    UPDATE_EVERY  = 32     # steps between policy updates
    GAMMA         = 0.99
    ENT_COEF      = 0.05   # entropy bonus
    RND_COEF      = 1.0    # weight on intrinsic reward
    EXT_COEF      = 0.0    # no extrinsic reward signal in ARC (sparse WIN only)
    LR            = 3e-4
    MAX_MINUTES   = 30.0
    # Novelty normalisation: running EMA of std
    NORM_BETA     = 0.99
    NORM_EPS      = 1e-4

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.game_id: str = (
            getattr(self, "game_id", None)
            or (str(args[0]) if args else "unknown")
        )

        # ── networks ──────────────────────────────────────────────────────────
        self.encoder  = CNNEncoder(in_ch=16, feat_dim=self.FEAT_DIM).to(self.device)
        self.phi_proj = FrozenProjection(self.FEAT_DIM, self.PHI_DIM).to(self.device)
        self.rnd      = LeakyRND(self.PHI_DIM, self.HIDDEN, self.RND_OUT, self.LEAK).to(self.device)
        self.policy   = PolicyHead(self.FEAT_DIM).to(self.device)

        # Freeze projection (it's deterministic but mark explicitly)
        for p in self.phi_proj.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.rnd.predictor.parameters()) +
            list(self.policy.parameters()),
            lr=self.LR,
        )

        # ── running novelty normaliser ─────────────────────────────────────
        self._nov_mean_sq = 1.0   # EMA of novelty²

        # ── transition buffer (cleared each episode / level) ───────────────
        self._buf: list[dict] = []
        self._step   = 0
        self._level  = -1
        self._t0     = time.time()

        print(f"[LeakyRND] game={self.game_id}  device={self.device}  "
              f"leak={self.LEAK}  update_every={self.UPDATE_EVERY}", flush=True)

    # ── required interface ────────────────────────────────────────────────────

    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return (time.time() - self._t0) >= self.MAX_MINUTES * 60.0

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        # ── level reset ───────────────────────────────────────────────────────
        lvl = getattr(latest_frame, "levels_completed",
                      getattr(latest_frame, "score", -1)) or -1
        if lvl != self._level:
            print(f"[LeakyRND] level {self._level}→{lvl} @ step {self._step}", flush=True)
            self._buf.clear()
            self._level = lvl

        # ── non-playing ───────────────────────────────────────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        obs  = self._to_tensor(latest_frame)          # (1, 16, 64, 64)
        avail = self._avail_idx(latest_frame)

        with torch.no_grad():
            feat = self.encoder(obs)                  # (1, FEAT_DIM)
            phi  = self.phi_proj(feat)                # (1, PHI_DIM)
            nov  = self.rnd.novelty(phi).item()       # scalar
            a_idx, x, y, logp, value = self.policy.sample(feat, avail)

        # ── normalise novelty ─────────────────────────────────────────────────
        self._nov_mean_sq = (self.NORM_BETA * self._nov_mean_sq
                             + (1 - self.NORM_BETA) * nov ** 2)
        r_int = nov / (self._nov_mean_sq ** 0.5 + self.NORM_EPS)
        r_int = float(np.clip(r_int, -5.0, 5.0))     # clip ±5 σ

        # ── build action ──────────────────────────────────────────────────────
        name = ALL_ACTIONS[a_idx]
        try:
            action = GameAction.from_name(name)
        except AttributeError:
            action = getattr(GameAction, name)
        if name == "ACTION6" and x is not None:
            action.set_data({"x": int(x), "y": int(y)})

        # ── store transition ──────────────────────────────────────────────────
        self._buf.append({
            "logp":  logp.detach(),
            "value": value.detach(),
            "r_int": r_int,
            "phi":   phi.detach(),
            "feat":  feat.detach(),
        })
        self._step += 1

        # ── periodic update ───────────────────────────────────────────────────
        if self._step % self.UPDATE_EVERY == 0 and len(self._buf) >= 4:
            self._update()

        return action

    # ── update ───────────────────────────────────────────────────────────────

    def _update(self):
        buf = self._buf[-self.UPDATE_EVERY:]   # last window only
        if not buf:
            return

        logps  = torch.stack([t["logp"]  for t in buf])
        values = torch.stack([t["value"] for t in buf])
        phis   = torch.cat([t["phi"] for t in buf], dim=0)

        # Discounted intrinsic returns (no extrinsic — ARC is sparse WIN only)
        rewards = [t["r_int"] for t in buf]
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        advantages = returns - values.detach()

        # Policy gradient loss
        pg_loss  = -(logps * advantages).mean()

        # Value loss
        v_loss   = F.mse_loss(values, returns)

        # Entropy bonus (encourage exploration)
        # Recompute logits for entropy — cheap, single forward
        feats_buf = torch.cat([t["feat"] for t in buf], dim=0)
        avail_all = list(range(N_ACTIONS))   # conservative: all unmasked for entropy
        logits_buf, _ = self.policy(feats_buf, avail_all)
        entropy = torch.distributions.Categorical(logits=logits_buf).entropy().mean()

        # RND distillation
        rnd_loss = self.rnd.distill_loss(phis)

        loss = pg_loss + 0.5 * v_loss - self.ENT_COEF * entropy + rnd_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.policy.parameters()), 0.5)
        self.opt.step()
        self.rnd.apply_leak()

        self._buf.clear()   # clear after update

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_tensor(self, fd: FrameData) -> torch.Tensor:
        frame = np.array(fd.frame, dtype=np.int64)[-1]   # latest layer, (64,64)
        t = torch.zeros(16, GRID, GRID, dtype=torch.float32)
        t.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1.0)
        return t.unsqueeze(0).to(self.device)             # (1, 16, 64, 64)

    @staticmethod
    def _avail_idx(fd: FrameData) -> list[int]:
        avail = getattr(fd, "available_actions", [])
        names = {(a.name if hasattr(a, "name") else str(a)) for a in avail}
        names.add("RESET")
        idx = [i for i, n in enumerate(ALL_ACTIONS) if n in names]
        return idx if idx else [0]


AGENT = LeakyRNDAgent
