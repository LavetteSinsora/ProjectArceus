"""
Stochastic Goose replication — CNN-RL action learner.

Original by Dries Smit / Tufa Labs, 1st place ARC-AGI-3 preview competition (12.58%).
Source: https://github.com/DriesSmit/ARC3-solution

Architecture:
  - CNN backbone: 16-ch one-hot input → 32 → 64 → 128 → 256 filters (3×3, padding=1)
  - Action head  (5 logits): MaxPool4 → flatten → FC(512) → Linear(5) for ACTION1–5
  - Coord head (4096 logits): 4 conv layers (spatial) → Linear(1) at each position for ACTION6 clicks
  - Binary BCE classification: did this (state, action) pair cause a frame change?
  - Entropy regularisation: − 0.0001 * action_entropy − 0.00001 * coord_entropy
  - Hierarchical sampling: sample action type first; if ACTION6, sample (x,y) from coord logits
  - Experience buffer with MD5-hash deduplication; cleared on level advance
  - Model + optimizer reset on level advance (fresh start per level)
"""

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


class ActionModel(nn.Module):
    """CNN that predicts which actions will result in new frames (shared conv backbone)."""

    def __init__(self, input_channels: int = 16, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        # Shared backbone
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Action head — pool + FC
        self.action_pool = nn.MaxPool2d(4, 4)
        action_flat = 256 * 16 * 16
        self.action_fc = nn.Linear(action_flat, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        # Coordinate head — fully convolutional (preserves spatial bias)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))

        # Action head
        af = self.action_pool(conv_features)
        af = af.view(af.size(0), -1)
        af = F.relu(self.action_fc(af))
        af = self.dropout(af)
        action_logits = self.action_head(af)

        # Coordinate head
        cf = F.relu(self.coord_conv1(conv_features))
        cf = F.relu(self.coord_conv2(cf))
        cf = F.relu(self.coord_conv3(cf))
        coord_logits = self.coord_conv4(cf).view(cf.size(0), -1)  # (B, 4096)

        return torch.cat([action_logits, coord_logits], dim=1)  # (B, 5 + 4096)


class Action:
    """
    Stochastic Goose agent.  Standalone version — does not depend on the
    ARC-AGI-3-Agents submodule; works directly with arcengine's FrameData /
    GameAction types and any EnvironmentWrapper that exposes `.step()`.
    """

    def __init__(
        self,
        game_id: str,
        max_minutes: float = 30.0,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        inference_only: bool = False,
    ) -> None:
        self.game_id = game_id
        self.start_time = time.time()
        self.max_seconds = max_minutes * 60.0
        self.save_dir = save_dir
        self.inference_only = inference_only

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mode = "inference" if inference_only else "training"
        print(f"[StochasticGoose] device={self.device}  game={game_id}  limit={max_minutes:.0f} min  mode={mode}")

        self.grid_size = 64
        self.num_coords = self.grid_size * self.grid_size
        self.num_colours = 16

        # Experience buffer — cleared each level (unused in inference_only mode)
        self.experience_buffer: deque = deque(maxlen=200_000)
        self.experience_hashes: set = set()
        self.batch_size = 64
        self.train_frequency = 5

        # Model state — reset each level (unless loading a checkpoint)
        self.action_model: Optional[ActionModel] = None
        self.optimizer: Optional[optim.Adam] = None
        self._init_model()

        # Per-step tracking
        self.prev_frame: Optional[np.ndarray] = None  # (16, 64, 64) bool
        self.prev_action_idx: Optional[int] = None
        self.current_score: int = -1
        self.action_counter: int = 0

        self.action_list = [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ]

        # Seed for reproducibility (per-game)
        seed = int(time.time() * 1_000_000) + hash(game_id) % 1_000_000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        self.action_model = ActionModel(
            input_channels=self.num_colours, grid_size=self.grid_size
        ).to(self.device)
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=1e-4)

    def save_model(self, path: str) -> None:
        """Save current model weights to a .pt file."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.action_model.state_dict(),
            "action_counter": self.action_counter,
            "levels_completed": self.current_score,
            "game_id": self.game_id,
        }, path)
        print(f"[StochasticGoose] Model saved → {path}")

    def load_model(self, path: str) -> None:
        """Load model weights from a .pt file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.action_model.load_state_dict(checkpoint["model_state_dict"])
        self.action_model.eval()
        saved_game = checkpoint.get("game_id", "?")
        saved_level = checkpoint.get("levels_completed", "?")
        saved_step = checkpoint.get("action_counter", "?")
        print(f"[StochasticGoose] Model loaded ← {path}")
        print(f"                  (saved from game={saved_game}  level={saved_level}  step={saved_step})")

    def _frame_to_tensor(self, frame_data: FrameData) -> torch.Tensor:
        """Convert FrameData → one-hot float tensor (16, 64, 64) on device."""
        frame = np.array(frame_data.frame, dtype=np.int64)[-1]  # latest layer
        assert frame.shape == (self.grid_size, self.grid_size), frame.shape
        t = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
        t.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1.0)
        return t.to(self.device)

    def _compute_experience_hash(self, frame_np: np.ndarray, action_idx: int) -> str:
        return hashlib.md5(frame_np.tobytes() + str(action_idx).encode()).hexdigest()

    def _sample_from_combined_output(
        self,
        combined_logits: torch.Tensor,
        available_actions: list,
    ) -> tuple[int, Optional[tuple[int, int]], Optional[int], np.ndarray]:
        """
        Returns (action_idx, coords_or_None, coord_flat_idx_or_None, all_probs_viz).
        action_idx < 5 → ACTION1–5; action_idx == 5 → ACTION6 with coords (y, x).
        """
        action_logits = combined_logits[:5].clone()
        coord_logits = combined_logits[5:].clone()

        if available_actions:
            mask = torch.full_like(action_logits, float("-inf"))
            action6_available = False
            for a in available_actions:
                # available_actions can be GameAction enums or raw ints depending on API version
                aid = a.value if hasattr(a, "value") else int(a)
                if 1 <= aid <= 5:
                    mask[aid - 1] = 0.0
                elif aid == 6:
                    action6_available = True
            action_logits = action_logits + mask
            if not action6_available:
                coord_logits = coord_logits + torch.full_like(coord_logits, float("-inf"))

        action_probs = torch.sigmoid(action_logits)
        coord_probs_raw = torch.sigmoid(coord_logits)
        coord_probs_scaled = coord_probs_raw / self.num_coords

        all_probs = torch.cat([action_probs, coord_probs_scaled])
        # Guard against NaN / all-zero
        total = all_probs.sum()
        if total <= 0 or torch.isnan(total):
            all_probs = torch.ones_like(all_probs) / len(all_probs)
        else:
            all_probs = all_probs / total

        probs_np = all_probs.cpu().numpy()
        selected = int(np.random.choice(len(probs_np), p=probs_np))

        viz = torch.cat([action_probs, torch.sigmoid(coord_logits)]).cpu().numpy()

        if selected < 5:
            return selected, None, None, viz
        coord_idx = selected - 5
        y, x = divmod(coord_idx, self.grid_size)
        return 5, (y, x), coord_idx, viz

    def _train_action_model(self) -> None:
        if len(self.experience_buffer) < self.batch_size:
            return
        indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        states = torch.stack(
            [torch.from_numpy(e["state"]).float().to(self.device) for e in batch]
        )
        action_indices = torch.tensor(
            [e["action_idx"] for e in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [e["reward"] for e in batch], dtype=torch.float32, device=self.device
        )

        self.optimizer.zero_grad()
        combined_logits = self.action_model(states)
        selected_logits = combined_logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        main_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)

        all_probs = torch.sigmoid(combined_logits)
        action_entropy = all_probs[:, :5].mean()
        coord_entropy = all_probs[:, 5:].mean()
        total_loss = main_loss - 0.0001 * action_entropy - 0.00001 * coord_entropy

        total_loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_done(self, latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return (time.time() - self.start_time) >= self.max_seconds

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        # --- Level advance: reset model + buffer ---
        current_levels = getattr(latest_frame, "levels_completed", None)
        if current_levels is None:
            current_levels = getattr(latest_frame, "score", -1)

        if current_levels != self.current_score:
            prev = self.current_score
            print(
                f"[StochasticGoose] levels completed: {prev} → {current_levels} "
                f"(step {self.action_counter})"
            )
            # Auto-save model when a level is completed (score went up)
            if self.save_dir is not None and current_levels > max(prev, 0):
                import os
                path = os.path.join(
                    self.save_dir,
                    f"{self.game_id}_level{current_levels}_step{self.action_counter}.pt",
                )
                self.save_model(path)

            if not self.inference_only:
                self.experience_buffer.clear()
                self.experience_hashes.clear()
                self._init_model()
            self.prev_frame = None
            self.prev_action_idx = None
            self.current_score = current_levels

        # --- Handle non-playing states ---
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_frame = None
            self.prev_action_idx = None
            return GameAction.RESET

        current_tensor = self._frame_to_tensor(latest_frame)

        # --- Store experience from previous step (skip in inference mode) ---
        if not self.inference_only and self.prev_frame is not None and self.prev_action_idx is not None:
            exp_hash = self._compute_experience_hash(self.prev_frame, self.prev_action_idx)
            if exp_hash not in self.experience_hashes:
                current_np = current_tensor.cpu().numpy().astype(bool)
                frame_changed = not np.array_equal(self.prev_frame, current_np)
                self.experience_buffer.append(
                    {
                        "state": self.prev_frame,
                        "action_idx": self.prev_action_idx,
                        "reward": 1.0 if frame_changed else 0.0,
                    }
                )
                self.experience_hashes.add(exp_hash)

        # --- Inference ---
        with torch.no_grad():
            logits = self.action_model(current_tensor.unsqueeze(0)).squeeze(0)

        action_idx, coords, coord_idx, viz = self._sample_from_combined_output(
            logits, getattr(latest_frame, "available_actions", [])
        )

        # --- Build GameAction ---
        if action_idx < 5:
            selected = self.action_list[action_idx]
        else:
            selected = GameAction.ACTION6
            y, x = coords
            selected.set_data({"x": x, "y": y})  # stores coords on enum singleton

        # --- Save state for next step ---
        self.prev_frame = current_tensor.cpu().numpy().astype(bool)
        self.prev_action_idx = action_idx if action_idx < 5 else (5 + coord_idx)

        # --- Periodic training (skip in inference mode) ---
        self.action_counter += 1
        if not self.inference_only and self.action_counter % self.train_frequency == 0:
            self._train_action_model()

        return selected
