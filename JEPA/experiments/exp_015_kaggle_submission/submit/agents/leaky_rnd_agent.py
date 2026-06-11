"""
ARC-AGI-3 — Leaky RND exploration agent.

Online policy-gradient (REINFORCE + baseline) with leaky-RND intrinsic reward.
  θ_P ← (1−μ)·θ_P + μ·θ_P^init   (leak, applied each update)
  r^i  = ½‖P(φ(s)) − T(φ(s))‖²   (novelty; φ = frozen-random CNN projection)

No ICM, no holdout set, no freeze gate — frozen-random φ is stable from step 0.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arcengine import FrameData, GameAction, GameState

# ---------------------------------------------------------------------------
# Action constants  (raw ints in available_actions; map to GameAction enums)
# ---------------------------------------------------------------------------
ACTION_MAP: dict[int, GameAction] = {
    i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)
}
ACTION_MAP[0] = GameAction.RESET
# Reverse: GameAction → int
ACTION_IDX = {v: k for k, v in ACTION_MAP.items()}

ALL_SLOTS   = [0, 1, 2, 3, 4, 5, 6, 7]   # 0=RESET, 1-5=ACTION1-5, 6=ACTION6, 7=ACTION7
N_SLOTS     = len(ALL_SLOTS)               # 8 total slots
GRID        = 64
N_COLORS    = 16


# ---------------------------------------------------------------------------
# 1. CNN encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    def __init__(self, in_ch: int = N_COLORS, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32,  3, padding=1), nn.ReLU(),
            nn.Conv2d(32,   64,  3, padding=1), nn.ReLU(),
            nn.Conv2d(64,   128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128,  256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),            # → (B, 256, 4, 4)
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feat_dim), nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Leaky RND  (inlined from exp_013_1b/rnd_phi.py)
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_d, hid, out_d, n_layers=2):
        super().__init__()
        seq = [nn.Linear(in_d, hid), nn.ReLU()]
        for _ in range(n_layers - 1):
            seq += [nn.Linear(hid, hid), nn.ReLU()]
        seq.append(nn.Linear(hid, out_d))
        self.net = nn.Sequential(*seq)
    def forward(self, x): return self.net(x)


class LeakyRND(nn.Module):
    """
    Frozen random target T + trainable leaky predictor P.
        novelty(φ) = ½ ‖P(φ) − T(φ)‖²
        apply_leak(): θ_P ← (1−μ)·θ_P + μ·θ_P^init
    """
    def __init__(self, phi_dim=128, hidden=256, out_dim=256, leak=0.05):
        super().__init__()
        self.target    = _MLP(phi_dim, hidden, out_dim, n_layers=2)
        self.predictor = _MLP(phi_dim, hidden, out_dim, n_layers=3)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()
        self.leak = leak
        self._pred_init = [p.detach().clone() for p in self.predictor.parameters()]

    @torch.no_grad()
    def novelty(self, phi: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.predictor(phi) - self.target(phi)).pow(2).mean(-1)

    def distill_loss(self, phi: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t = self.target(phi)
        return (self.predictor(phi) - t).pow(2).mean()

    @torch.no_grad()
    def apply_leak(self) -> None:
        for p, p0 in zip(self.predictor.parameters(), self._pred_init):
            p.mul_(1.0 - self.leak).add_(p0.to(p.device), alpha=self.leak)


# ---------------------------------------------------------------------------
# 3. Frozen random φ projection  (stable from step 0, no training needed)
# ---------------------------------------------------------------------------

class FrozenProjection(nn.Module):
    def __init__(self, feat_dim=256, phi_dim=128, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(feat_dim, phi_dim, generator=g) / (feat_dim ** 0.5)
        self.register_buffer("W", W)
    def forward(self, feat): return feat @ self.W


# ---------------------------------------------------------------------------
# 4. Policy + value head
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.action = nn.Linear(feat_dim, N_SLOTS)  # 8 logits (0=RESET..7=ACTION7)
        self.value  = nn.Linear(feat_dim, 1)
        # Coordinate sub-heads for ACTION6 (click games)
        self.x_head = nn.Linear(feat_dim, GRID)
        self.x_emb  = nn.Embedding(GRID, 32)
        self.y_head = nn.Linear(feat_dim + 32, GRID)

    def sample(self, feat: torch.Tensor, avail_slots: list[int]):
        """
        avail_slots: list of slot indices (0=RESET, 1-5=ACTION1-5, 6=ACTION6, 7=ACTION7)
        Returns: (slot_idx, x_or_None, y_or_None, log_prob, value)
        """
        logits = self.action(feat)                     # (1, 8)
        mask   = torch.full_like(logits, float("-inf"))
        mask[:, avail_slots] = 0.0
        logits = logits + mask

        dist = torch.distributions.Categorical(logits=logits)
        slot = dist.sample()
        logp = dist.log_prob(slot)
        v    = self.value(feat).squeeze(-1)

        x = y = None
        if int(slot) == 6:                             # ACTION6 → sample coords
            xd = torch.distributions.Categorical(logits=self.x_head(feat))
            x  = xd.sample()
            yd = torch.distributions.Categorical(
                    logits=self.y_head(torch.cat([feat, self.x_emb(x)], dim=-1)))
            y  = yd.sample()
            logp = logp + xd.log_prob(x) + yd.log_prob(y)

        return int(slot), x, y, logp, v


# ---------------------------------------------------------------------------
# 5. The Agent
# ---------------------------------------------------------------------------

class LeakyRNDAgent:
    """
    Standalone Leaky-RND + policy-gradient agent for ARC-AGI-3.

    The episode loop (main()) is in this class for portability — it works
    identically whether called from main.py or the Kaggle harness.
    """

    # ── hyper-parameters ─────────────────────────────────────────────────────
    FEAT_DIM     = 256
    PHI_DIM      = 128
    HIDDEN       = 256
    RND_OUT      = 256
    LEAK         = 0.05
    LR           = 3e-4
    UPDATE_EVERY = 32      # steps between policy updates
    GAMMA        = 0.99
    ENT_COEF     = 0.05
    MAX_ACTIONS  = 500_000
    MAX_MINUTES  = 60.0
    # Novelty EMA normalisation
    NORM_BETA    = 0.99
    NORM_EPS     = 1e-4
    CLIP_RANGE   = 5.0     # clip normalised novelty to ±CLIP_RANGE σ

    def __init__(self, game_id: str = "unknown", **_: Any) -> None:
        self.game_id = game_id
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder  = CNNEncoder(N_COLORS, self.FEAT_DIM).to(self.device)
        self.phi_proj = FrozenProjection(self.FEAT_DIM, self.PHI_DIM).to(self.device)
        self.rnd      = LeakyRND(self.PHI_DIM, self.HIDDEN, self.RND_OUT, self.LEAK).to(self.device)
        self.policy   = PolicyHead(self.FEAT_DIM).to(self.device)

        for p in self.phi_proj.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.rnd.predictor.parameters()) +
            list(self.policy.parameters()),
            lr=self.LR,
        )

        self._nov_ema_sq = 1.0   # running EMA of novelty²
        self._buf: list[dict]  = []
        self._step  = 0
        self._level = 0
        self._t0    = time.time()

        print(f"[LeakyRND] game={game_id} device={self.device} "
              f"leak={self.LEAK} update_every={self.UPDATE_EVERY}", flush=True)

    # ── required interface (called by episode loop) ───────────────────────────

    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return (time.time() - self._t0) >= self.MAX_MINUTES * 60.0

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        # ── level advance: reset buffer ───────────────────────────────────────
        lvl = int(getattr(latest_frame, "levels_completed", 0) or 0)
        if lvl > self._level:
            print(f"[LeakyRND] level {self._level}→{lvl} @ step {self._step}", flush=True)
            self._buf.clear()
            self._level = lvl

        # ── non-playing states ────────────────────────────────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        # ── available actions → slot indices ──────────────────────────────────
        avail_raw: list[int] = getattr(latest_frame, "available_actions", [1])
        avail_slots = [0] + [int(a) for a in avail_raw]  # 0=RESET always valid
        avail_slots = list(set(s for s in avail_slots if 0 <= s < N_SLOTS))

        # ── forward pass ──────────────────────────────────────────────────────
        obs   = self._to_tensor(latest_frame)            # (1, 16, 64, 64)
        with torch.no_grad():
            feat  = self.encoder(obs)                    # (1, FEAT_DIM)
            phi   = self.phi_proj(feat)                  # (1, PHI_DIM)
            nov   = float(self.rnd.novelty(phi).item())

        # normalise novelty
        self._nov_ema_sq = (self.NORM_BETA * self._nov_ema_sq
                            + (1 - self.NORM_BETA) * nov ** 2)
        r_int = float(np.clip(
            nov / (self._nov_ema_sq ** 0.5 + self.NORM_EPS),
            -self.CLIP_RANGE, self.CLIP_RANGE,
        ))

        # sample action
        slot, x, y, logp, value = self.policy.sample(feat, avail_slots)

        # ── store transition ──────────────────────────────────────────────────
        self._buf.append({"logp": logp.detach(), "value": value.detach(),
                          "r_int": r_int, "phi": phi.detach(),
                          "feat": feat.detach()})
        self._step += 1

        # ── periodic update ───────────────────────────────────────────────────
        if self._step % self.UPDATE_EVERY == 0 and len(self._buf) >= 4:
            self._update()

        # ── build return action ───────────────────────────────────────────────
        if slot == 0:
            return GameAction.RESET
        action = ACTION_MAP.get(slot, GameAction.ACTION1)
        if slot == 6 and x is not None:                 # ACTION6 with coordinates
            action.set_data({"x": int(x), "y": int(y)})
        return action

    # ── update ────────────────────────────────────────────────────────────────

    def _update(self) -> None:
        buf = self._buf[-self.UPDATE_EVERY:]
        if not buf:
            return

        logps  = torch.stack([t["logp"]        for t in buf])
        values = torch.stack([t["value"].squeeze(-1) for t in buf])
        phis   = torch.cat([t["phi"]   for t in buf], dim=0)

        # discounted intrinsic returns
        G, returns = 0.0, []
        for r in reversed([t["r_int"] for t in buf]):
            G = r + self.GAMMA * G
            returns.insert(0, G)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        advantages = ret - values.detach()
        pg_loss    = -(logps * advantages).mean()
        v_loss     = F.mse_loss(values, ret)

        # entropy bonus — reuse buffered features (no re-encode cost)
        feats_buf  = torch.cat([t["feat"] for t in buf], dim=0)
        avail_all  = list(range(N_SLOTS))
        ent_logits = self.policy.action(feats_buf)
        # zero the mask so all slots contribute to entropy
        entropy    = torch.distributions.Categorical(logits=ent_logits).entropy().mean()

        rnd_loss = self.rnd.distill_loss(phis)
        loss = pg_loss + 0.5 * v_loss - self.ENT_COEF * entropy + rnd_loss

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.policy.parameters()), 0.5)
        self.opt.step()
        self.rnd.apply_leak()
        self._buf.clear()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_tensor(self, fd: FrameData) -> torch.Tensor:
        """FrameData.frame → (1, 16, 64, 64) one-hot float tensor."""
        frame = np.array(fd.frame, dtype=np.int64)[-1]   # last layer (64,64)
        t = torch.zeros(N_COLORS, GRID, GRID, dtype=torch.float32)
        t.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1.0)
        return t.unsqueeze(0).to(self.device)

    # ── standalone episode runner (for main.py) ────────────────────────────

    def run_episode(self, env, max_actions: Optional[int] = None) -> dict:
        """
        Drive a full episode. env is an arcengine EnvironmentWrapper.
        Returns summary dict with levels_completed, total_steps.
        """
        max_a = max_actions or self.MAX_ACTIONS
        frame = env.step(GameAction.RESET)
        frames: list = [frame]
        steps = 0
        last_level = 0

        while steps < max_a and not self.is_done(frames, frame):
            if frame is None or frame.state is GameState.WIN:
                break

            action = self.choose_action(frames, frame)

            # Pass coordinates for ACTION6
            if action == GameAction.ACTION6:
                try:
                    ad = action.action_data
                    frame = env.step(action, data={"x": int(ad.x), "y": int(ad.y)})
                except Exception:
                    frame = env.step(action)
            else:
                frame = env.step(action)

            if frame is None:
                continue

            frames = frames[-9:] + [frame]
            steps += 1

            lvl = int(getattr(frame, "levels_completed", 0) or 0)
            if lvl > last_level:
                print(f"[LeakyRND] cleared level {lvl} @ step {steps}", flush=True)
                last_level = lvl

        return {"levels_completed": last_level, "total_steps": steps,
                "wall_sec": time.time() - self._t0}
