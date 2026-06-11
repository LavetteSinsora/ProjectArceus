"""
ARC-AGI-3 Kaggle submission — Stochastic Goose (Dries Smit / Tufa Labs replication).

This is the file to add to the ARC-AGI-3-Agents fork as agents/stochastic_goose_agent.py.
Then register it in main.py:
    from agents.stochastic_goose_agent import StochasticGooseAgent
    AVAILABLE_AGENTS = {"stochasticgoose": StochasticGooseAgent, ...}

Reference performance: 12.58% (Dries Smit, ARC-AGI-3 preview leaderboard).

Architecture:
  - CNN backbone (16-ch one-hot → 32→64→128→256 filters)
  - Binary BCE: did (state, action) cause a frame change?
  - Action type head (ACTION1–5) + coordinate head (ACTION6, 64×64 spatial)
  - Hierarchical sampling: action type first; if ACTION6, sample (x,y) from coord logits
  - Entropy regularisation; experience buffer with MD5-hash deduplication
  - Model + buffer reset on each level advance (fresh start per level)
  - Online learning: train every 5 actions on the accumulated buffer
"""

from __future__ import annotations

import hashlib
import random
import time
from collections import deque
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arcengine import FrameData, GameAction, GameState

try:
    from agents.agent import Agent  # provided by ARC-AGI-3-Agents framework
except ImportError:
    Agent = object  # type: ignore[assignment,misc]  -- for local dev without the framework


# ── CNN model ─────────────────────────────────────────────────────────────────

class ActionModel(nn.Module):
    def __init__(self, input_channels: int = 16, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.relu(self.conv4(x))

        af = self.action_pool(feat).view(feat.size(0), -1)
        af = F.relu(self.action_fc(af))
        af = self.dropout(af)
        action_logits = self.action_head(af)

        cf = F.relu(self.coord_conv1(feat))
        cf = F.relu(self.coord_conv2(cf))
        cf = F.relu(self.coord_conv3(cf))
        coord_logits = self.coord_conv4(cf).view(cf.size(0), -1)   # (B, 4096)

        return torch.cat([action_logits, coord_logits], dim=1)       # (B, 4101)


# ── Agent ─────────────────────────────────────────────────────────────────────

class StochasticGooseAgent(Agent):  # type: ignore[misc]
    """
    Stochastic Goose as an ARC-AGI-3 Agent subclass.

    The Agent base class requires:
        is_done(self, frames, latest_frame) -> bool
        choose_action(self, frames, latest_frame) -> GameAction

    Both are implemented below.  The agent does ONLINE LEARNING (backprop every
    5 actions) — if the competition disallows gradient updates set TRAIN=False.
    """

    TRAIN = True            # set False if the harness bans in-episode gradients
    TRAIN_EVERY = 5         # actions between model updates
    BATCH_SIZE = 64
    BUFFER_CAP = 200_000
    MAX_MINUTES = 30.0      # wall-clock kill-switch (secondary; harness has its own)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_size = 64
        self.num_colours = 16
        self.num_coords = self.grid_size * self.grid_size

        # game_id: Agent base class stores it (check with super().__init__)
        # fall back to inspecting args[0] if the framework passes it positionally
        self.game_id: str = (
            getattr(self, "game_id", None)
            or (str(args[0]) if args else "unknown")
        )

        self.experience_buffer: deque = deque(maxlen=self.BUFFER_CAP)
        self.experience_hashes: set[str] = set()

        self.action_model: Optional[ActionModel] = None
        self.optimizer: Optional[optim.Adam] = None
        self._init_model()

        self.action_list = [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ]

        self.prev_frame: Optional[np.ndarray] = None
        self.prev_action_idx: Optional[int] = None
        self.current_level: int = -1
        self.action_counter: int = 0
        self.start_time: float = time.time()

        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed); np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))

        print(f"[StochasticGoose] game={self.game_id}  device={self.device}  "
              f"train={self.TRAIN}", flush=True)

    # ── interface: required by Agent base class ───────────────────────────────

    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        """Stop on WIN or wall-clock timeout."""
        if latest_frame.state is GameState.WIN:
            return True
        return (time.time() - self.start_time) >= self.MAX_MINUTES * 60.0

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        # ── level-advance: reset model + buffer ──────────────────────────────
        lvl = getattr(latest_frame, "levels_completed",
                      getattr(latest_frame, "score", -1))
        if lvl is None:
            lvl = -1
        if lvl != self.current_level:
            print(f"[StochasticGoose] level {self.current_level} → {lvl} "
                  f"(step {self.action_counter})", flush=True)
            if self.TRAIN:
                self.experience_buffer.clear()
                self.experience_hashes.clear()
                self._init_model()
            self.prev_frame = None
            self.prev_action_idx = None
            self.current_level = lvl

        # ── non-playing states ────────────────────────────────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_frame = None; self.prev_action_idx = None
            return GameAction.RESET

        current_tensor = self._frame_to_tensor(latest_frame)

        # ── store transition from previous step ───────────────────────────────
        if self.TRAIN and self.prev_frame is not None and self.prev_action_idx is not None:
            h = hashlib.md5(self.prev_frame.tobytes()
                            + str(self.prev_action_idx).encode()).hexdigest()
            if h not in self.experience_hashes:
                cur_np = current_tensor.cpu().numpy().astype(bool)
                changed = not np.array_equal(self.prev_frame, cur_np)
                self.experience_buffer.append(
                    {"state": self.prev_frame,
                     "action_idx": self.prev_action_idx,
                     "reward": 1.0 if changed else 0.0})
                self.experience_hashes.add(h)

        # ── inference ────────────────────────────────────────────────────────
        with torch.no_grad():
            logits = self.action_model(current_tensor.unsqueeze(0)).squeeze(0)

        avail = getattr(latest_frame, "available_actions", [])
        action_idx, coords, coord_idx = self._sample(logits, avail)

        if action_idx < 5:
            selected = self.action_list[action_idx]
        else:
            selected = GameAction.ACTION6
            y, x = coords
            selected.set_data({"x": int(x), "y": int(y)})

        # ── save state + periodic train ───────────────────────────────────────
        self.prev_frame = current_tensor.cpu().numpy().astype(bool)
        self.prev_action_idx = action_idx if action_idx < 5 else (5 + coord_idx)
        self.action_counter += 1
        if self.TRAIN and self.action_counter % self.TRAIN_EVERY == 0:
            self._train()

        return selected

    # ── internals ─────────────────────────────────────────────────────────────

    def _init_model(self) -> None:
        self.action_model = ActionModel(self.num_colours, self.grid_size).to(self.device)
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=1e-4)

    def _frame_to_tensor(self, fd: FrameData) -> torch.Tensor:
        frame = np.array(fd.frame, dtype=np.int64)[-1]
        t = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
        t.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1.0)
        return t.to(self.device)

    def _sample(self, logits: torch.Tensor, avail: list) -> tuple:
        a_logits = logits[:5].clone()
        c_logits = logits[5:].clone()

        action6_ok = False
        if avail:
            mask = torch.full_like(a_logits, float("-inf"))
            for a in avail:
                aid = a.value if hasattr(a, "value") else int(a)
                if 1 <= aid <= 5:
                    mask[aid - 1] = 0.0
                elif aid == 6:
                    action6_ok = True
            a_logits = a_logits + mask
        if not action6_ok:
            c_logits = c_logits + torch.full_like(c_logits, float("-inf"))

        a_probs = torch.sigmoid(a_logits)
        c_probs = torch.sigmoid(c_logits) / self.num_coords
        all_p = torch.cat([a_probs, c_probs])
        total = all_p.sum()
        if total <= 0 or torch.isnan(total):
            all_p = torch.ones_like(all_p) / len(all_p)
        else:
            all_p = all_p / total

        sel = int(np.random.choice(len(all_p), p=all_p.cpu().numpy()))
        if sel < 5:
            return sel, None, None
        ci = sel - 5
        y, x = divmod(ci, self.grid_size)
        return 5, (y, x), ci

    def _train(self) -> None:
        if len(self.experience_buffer) < self.BATCH_SIZE:
            return
        idx = np.random.choice(len(self.experience_buffer), self.BATCH_SIZE, replace=False)
        batch = [self.experience_buffer[i] for i in idx]

        states = torch.stack(
            [torch.from_numpy(e["state"]).float().to(self.device) for e in batch])
        actions = torch.tensor(
            [e["action_idx"] for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            [e["reward"] for e in batch], dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        logits = self.action_model(states)
        sel_logits = logits.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(sel_logits, rewards)
        loss = loss - 0.0001 * torch.sigmoid(logits[:, :5]).mean() \
                    - 0.00001 * torch.sigmoid(logits[:, 5:]).mean()
        loss.backward()
        self.optimizer.step()


# Registration name used by ARC-AGI-3-Agents main.py
AGENT = StochasticGooseAgent
