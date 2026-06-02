"""
exp_015 — ARC-AGI-3 (ARC Prize 2026) submission skeleton.

This is a STUB showing the required entry point/interface and where our
leaky-RND + PPO agent and the hierarchical action head plug in.

GROUNDING / ASSUMPTIONS
-----------------------
- The Agent base class, FrameData, GameAction, GameState come from the official
  framework (`arcprize/ARC-AGI-3-Agents`). At submission you subclass `Agent`
  and implement `is_done` + `choose_action`; the harness runs the episode loop
  and the env `step()` for you. (CONFIRMED: agents/agent.py in that repo,
  https://docs.arcprize.org/)
- Imports below mirror the repo:
      from agents.agent import Agent
      from arcengine import FrameData, GameAction, GameState
  (CONFIRMED from agents/templates/random_agent.py).
- `latest_frame.frame` is list[ list[list[int]] ] : a stack of 64x64 grids of
  color ints 0..15. (CONFIRMED: _convert_raw_frame_data in agents/agent.py.)
- `latest_frame.available_actions` is list[GameAction] valid THIS frame.
  (CONFIRMED: agents/agent.py.)
- ACTION6 is the only complex action; set coords via
      action = GameAction.ACTION6; action.set_data({"x": x, "y": y})   # x,y in [0,63]
  (CONFIRMED: agents/templates/random_agent.py, https://docs.arcprize.org/actions.)
- [UNCONFIRMED] Whether in-process gradient updates are allowed at eval and the
  exact Kaggle packaging (Notebook vs repo). See README §5, §7. Code is written
  so the online-update step can be disabled with ONLINE_LEARNING=False to fall
  back to a frozen pre-trained policy if the rules require it.

Everything marked PSEUDOCODE is illustrative; wire it to our real
exp_013 leaky-RND + PPO modules when porting.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# --- Framework imports (provided by the competition harness) ----------------
# In the real submission these resolve inside the ARC-AGI-3-Agents package.
try:
    from agents.agent import Agent  # type: ignore
    from arcengine import FrameData, GameAction, GameState  # type: ignore
except Exception:  # local dev without the framework installed
    Agent = object  # type: ignore
    FrameData = Any  # type: ignore
    GameAction = Any  # type: ignore
    GameState = Any  # type: ignore

# torch is available on Kaggle GPU; guard import for skeleton readability.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore

GRID = 64          # ARC-AGI-3 grid is 64x64 (https://docs.arcprize.org/actions)
N_COLORS = 16      # color ints 0..15
ONLINE_LEARNING = os.getenv("ARC_ONLINE_LEARNING", "True") == "True"
UPDATE_EVERY = 8   # PPO+RND update cadence (actions); small due to RHAE budget


# ---------------------------------------------------------------------------
# 1. Observation encoding: FrameData.frame -> tensor [C, 64, 64]
# ---------------------------------------------------------------------------
def encode_frame(latest_frame: "FrameData", max_layers: int = 4) -> "np.ndarray":
    """Stack of 64x64 int grids -> one-hot color channels per layer.

    Returns array shape [max_layers * N_COLORS, 64, 64]. Pads/truncates the
    number of grid layers so a single CNN handles any game. PSEUDOCODE-ish but
    runnable.
    """
    grids = latest_frame.frame if latest_frame is not None else []
    grids = list(grids)[:max_layers]
    chans = []
    for g in grids:
        arr = np.asarray(g, dtype=np.int64)               # [64,64]
        oh = np.eye(N_COLORS, dtype=np.float32)[arr]       # [64,64,16]
        chans.append(np.transpose(oh, (2, 0, 1)))          # [16,64,64]
    while len(chans) < max_layers:
        chans.append(np.zeros((N_COLORS, GRID, GRID), dtype=np.float32))
    return np.concatenate(chans, axis=0)                   # [max_layers*16,64,64]


# ---------------------------------------------------------------------------
# 2. Hierarchical action head (README §6 — recommended design)
#    top-level: 7 logits {RESET, ACTION1..ACTION6}; masked by available_actions
#    if ACTION6: x ~ P(x|rep); y ~ P(y|rep, x)   (autoregressive)
# ---------------------------------------------------------------------------
ALL_ACTIONS = ["RESET", "ACTION1", "ACTION2", "ACTION3",
               "ACTION4", "ACTION5", "ACTION6"]  # index = top-level logit slot


if torch is not None:

    class Encoder(nn.Module):
        def __init__(self, in_ch: int, feat: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(feat), nn.ReLU(),
            )

        def forward(self, x):  # x: [B, C, 64, 64]
            return self.net(x)

    class HierarchicalActionHead(nn.Module):
        """Sound per README §6: softmax over 6 actions, then autoregressive
        (x then y|x) for ACTION6. Spatial conv-map head is a drop-in alt for the
        (x,y) sub-policy if click-grounding is poor."""

        def __init__(self, feat: int = 256):
            super().__init__()
            self.action_logits = nn.Linear(feat, len(ALL_ACTIONS))  # 7
            self.x_head = nn.Linear(feat, GRID)                      # 64
            self.x_emb = nn.Embedding(GRID, 32)
            self.y_head = nn.Linear(feat + 32, GRID)                 # 64, cond. on x
            self.value = nn.Linear(feat, 1)                          # PPO critic

        @staticmethod
        def _mask(logits: "torch.Tensor", available_idx: list[int]) -> "torch.Tensor":
            """Additive -inf mask: variable n_actions (4/5/6) -> one network."""
            mask = torch.full_like(logits, float("-inf"))
            mask[..., available_idx] = 0.0
            return logits + mask

        def forward(self, rep, available_idx):
            a_logits = self._mask(self.action_logits(rep), available_idx)
            a_dist = torch.distributions.Categorical(logits=a_logits)
            return a_dist, self.value(rep)

        def sample(self, rep, available_idx):
            a_dist, value = self.forward(rep, available_idx)
            a = a_dist.sample()
            logp = a_dist.log_prob(a)
            x = y = None
            if ALL_ACTIONS[int(a)] == "ACTION6":
                x_dist = torch.distributions.Categorical(logits=self.x_head(rep))
                x = x_dist.sample()
                xy_in = torch.cat([rep, self.x_emb(x)], dim=-1)
                y_dist = torch.distributions.Categorical(logits=self.y_head(xy_in))
                y = y_dist.sample()
                logp = logp + x_dist.log_prob(x) + y_dist.log_prob(y)  # joint log-prob
            return int(a), (None if x is None else int(x)), \
                   (None if y is None else int(y)), logp, value


# ---------------------------------------------------------------------------
# 3. Leaky-RND intrinsic reward (PSEUDOCODE — port from exp_013)
#    frozen random target f_target, trainable predictor f_pred;
#    r^i = ||f_pred(s') - f_target(s')||^2, normalized by EMA std (the
#    warm-up + leaky normalizer that cured ICM collapse in exp_013).
# ---------------------------------------------------------------------------
class LeakyRND:  # PSEUDOCODE wrapper around our exp_013 implementation
    def __init__(self, encoder_out: int = 256):
        self.ema_std = 1.0  # leaky EMA normalizer (see finding_phi_drift.md)
        # self.target = frozen random net; self.pred = trainable net  (torch)
        # self.opt = optimizer over self.pred only

    def intrinsic_reward(self, next_rep) -> float:
        raise NotImplementedError("port from exp_013 leaky-RND")

    def update(self, batch) -> None:
        raise NotImplementedError("predictor gradient step + EMA-std update")


# ---------------------------------------------------------------------------
# 4. THE SUBMISSION ENTRY POINT — subclass Agent, implement the 2 methods.
# ---------------------------------------------------------------------------
class LeakyRNDAgent(Agent):  # type: ignore[misc]
    """Selected at eval via: uv run main.py --agent=leakyrndagent --game=<id>
    (CONFIRMED CLI shape: docs.arcprize.org Agents quickstart, main.py)."""

    MAX_ACTIONS = 80  # harness guard; tune. NOT the scoring rule (RHAE counts actions).

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        in_ch = 4 * N_COLORS
        if torch is not None:
            self.encoder = Encoder(in_ch)
            self.head = HierarchicalActionHead()
            # Optional pre-trained init (shipped as a packaged weights file).
            # _load_pretrained(self.encoder, self.head)  # README §5 step 6
        self.rnd = LeakyRND()
        self._buffer: list[dict] = []   # transitions for in-episode PPO+RND
        self._steps = 0

    # --- required: when to stop -------------------------------------------
    def is_done(self, frames, latest_frame) -> bool:
        # Stop on WIN; avoid wasting actions post-solve (hurts RHAE). (README §3)
        return latest_frame.state is GameState.WIN

    # --- required: pick next action ---------------------------------------
    def choose_action(self, frames, latest_frame):
        # Reset at game start / after GAME_OVER (pattern from random_agent.py).
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        available_idx = self._available_indices(latest_frame)

        if torch is not None:
            obs = torch.from_numpy(encode_frame(latest_frame)).unsqueeze(0)
            rep = self.encoder(obs)
            a_idx, x, y, logp, value = self.head.sample(rep, available_idx)
        else:  # skeleton fallback
            a_idx, x, y = available_idx[0], 32, 32

        action = GameAction.from_name(ALL_ACTIONS[a_idx])
        if ALL_ACTIONS[a_idx] == "ACTION6":
            action.set_data({"x": int(x), "y": int(y)})  # x,y in [0,63]

        # ---- online learning (README §5; gate via ONLINE_LEARNING) --------
        if ONLINE_LEARNING:
            # r^i = leaky-RND novelty; r^e = +1 on Δlevels_completed (sparse).
            # self._buffer.append({...}); periodic PPO + RND predictor update:
            self._steps += 1
            if self._steps % UPDATE_EVERY == 0:
                # self.rnd.update(batch); ppo_update(self.encoder, self.head, batch)
                pass

        return action

    # --- helper: map per-frame available_actions -> logit indices ---------
    @staticmethod
    def _available_indices(latest_frame) -> list[int]:
        """Variable n_actions (4/5/6) handled by masking (README §6).
        available_actions is a per-frame list[GameAction]."""
        names = {a.name for a in latest_frame.available_actions}
        names.add("RESET")  # RESET always usable for restart
        idx = [i for i, n in enumerate(ALL_ACTIONS) if n in names]
        return idx or [0]   # never empty


# Some harnesses register agents by class discovery in AVAILABLE_AGENTS; others
# import by name. Keep the class importable at module top level.
AGENT = LeakyRNDAgent
