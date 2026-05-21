"""Shared world model — the transferable artifact.

`FrameWorldModel` predicts `(next_frame, terminal, completed)` from
`(frame, action)`. Trained on transitions pooled from several solved levels,
it learns the *dynamics* of the game (avatar moves one cell, walls block,
modifier tiles transform the avatar, energy drains, the matching goal
completes the level). Because the maze layout is part of the input frame, the
model is layout-agnostic and transfers to unseen levels of the same game.

`ModelEnv` wraps a trained model behind the exact env interface Go-Explore
needs (`reset`, `step`, `n_actions`, `level_completed`), so the search can run
entirely *in imagination* — zero real-environment interaction.

Game-agnostic: nothing here encodes a specific level or goal location.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from claude_automate.framework.env_api import frame_to_tensor


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )


class FrameWorldModel(nn.Module):
    """U-Net dynamics model: (frame one-hot, action) → next-frame logits.

    Skip connections make "copy the unchanged maze pixels" trivial, so the
    model can focus capacity on the small region the action actually changes.
    Auxiliary heads predict episode `terminal` and `level_completed`.
    """

    def __init__(self, n_colors: int = 16, n_actions: int = 5,
                 action_embed: int = 16, base_ch: int = 48):
        super().__init__()
        self.n_colors = n_colors
        self.action_embed = nn.Embedding(max(n_actions, 8), action_embed)

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.inc = _conv_block(n_colors + action_embed, c1)
        self.down1 = _conv_block(c1, c2)
        self.down2 = _conv_block(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.up2 = _conv_block(c3 + c2, c2)
        self.up1 = _conv_block(c2 + c1, c1)
        self.outc = nn.Conv2d(c1, n_colors, 1)
        self.aux = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c3, 64), nn.ReLU(inplace=True), nn.Linear(64, 2),
        )

    def forward(self, frame_onehot: torch.Tensor, action: torch.Tensor):
        """frame_onehot: (B,C,64,64), action: (B,) long.

        Returns (frame_logits (B,C,64,64), terminal_logit (B,), completed (B,))."""
        B = frame_onehot.shape[0]
        a = self.action_embed(action)                       # (B, E)
        a = a[:, :, None, None].expand(-1, -1, *frame_onehot.shape[2:])
        x0 = self.inc(torch.cat([frame_onehot, a], dim=1))   # (B,c1,64,64)
        x1 = self.down1(self.pool(x0))                       # (B,c2,32,32)
        x2 = self.down2(self.pool(x1))                       # (B,c3,16,16)
        u1 = self.up2(torch.cat([
            nn.functional.interpolate(x2, scale_factor=2, mode="nearest"),
            x1], dim=1))                                     # (B,c2,32,32)
        u0 = self.up1(torch.cat([
            nn.functional.interpolate(u1, scale_factor=2, mode="nearest"),
            x0], dim=1))                                     # (B,c1,64,64)
        frame_logits = self.outc(u0)
        aux = self.aux(x2)
        return frame_logits, aux[:, 0], aux[:, 1]

    @torch.no_grad()
    def predict(self, frame: np.ndarray, action: int, device,
                completed_thresh: float = 0.5, terminal_thresh: float = 0.5):
        """Single-step prediction. frame: (64,64) uint8 → (next_frame uint8,
        terminal bool, completed bool)."""
        obs = frame_to_tensor(frame, self.n_colors).unsqueeze(0).to(device)
        act = torch.tensor([action], dtype=torch.long, device=device)
        logits, term, comp = self.forward(obs, act)
        next_frame = logits.argmax(1)[0].to("cpu").numpy().astype(np.uint8)
        return (next_frame,
                bool(torch.sigmoid(term)[0] > terminal_thresh),
                bool(torch.sigmoid(comp)[0] > completed_thresh))


class ModelEnv:
    """Exposes a trained `FrameWorldModel` behind the Go-Explore env interface.

    Deterministic (argmax predictions), so trajectories replay exactly — the
    property Go-Explore relies on. Search through a `ModelEnv` costs zero real
    environment steps.
    """

    def __init__(self, model: FrameWorldModel, initial_frame: np.ndarray,
                 n_actions: int, device, masked_rows=None,
                 max_steps: int = 200):
        self.model = model
        self.initial_frame = np.asarray(initial_frame, dtype=np.uint8)
        self.n_actions = n_actions
        self.device = device
        self._MASKED_ROWS = masked_rows
        self.max_steps = max_steps
        self._frame = None
        self._steps = 0
        self._completed = False

    def reset(self) -> np.ndarray:
        self._frame = self.initial_frame.copy()
        self._steps = 0
        self._completed = False
        return self._frame

    def step(self, action: int):
        next_frame, terminal, completed = self.model.predict(
            self._frame, int(action), self.device)
        self._frame = next_frame
        self._steps += 1
        self._completed = self._completed or completed
        is_terminal = bool(terminal or completed
                           or self._steps >= self.max_steps)
        return next_frame, is_terminal

    @property
    def level_completed(self) -> bool:
        return self._completed
