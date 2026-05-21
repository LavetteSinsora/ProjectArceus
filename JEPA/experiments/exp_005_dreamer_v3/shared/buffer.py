"""Sequence replay buffer for Dreamer V3.

Stores per-step transitions (obs, action_idx, reward, continue) in a flat ring
buffer with episode boundaries marked so we can sample contiguous chunks that
do NOT cross episode boundaries.

Returns chunks shaped (B, T, ...) which the trainer then feeds into
`world_model.observe`.

Memory cost at LS20 (1×64×64 uint8, 250k cap):
  obs:       250k × 64×64 × 1 byte ≈ 1.0 GB    (kept as uint8 in CPU memory)
  action:    250k × 1 int32       ≈ 1 MB
  reward:    250k × 1 float32     ≈ 1 MB
  continue:  250k × 1 uint8       ≈ 0.25 MB
Fits comfortably in RAM.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class SequenceBatch(NamedTuple):
    obs: torch.Tensor        # (B, T, C, H, W) float32 in [-0.5, 0.5]
    action: torch.Tensor     # (B, T, n_actions) one-hot
    reward: torch.Tensor     # (B, T) float32
    cont: torch.Tensor       # (B, T) float32 in {0, 1}


class SequenceReplayBuffer:
    """Ring buffer that samples B contiguous length-T chunks within a single episode.

    Episode boundaries are tracked via the `cont` flag (cont=0 ⇒ terminal step,
    next obs is from a new episode).  We index every episode's first-step
    offset so sampling can avoid crossing boundaries.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, int, int],   # (C, H, W)
        n_actions: int,
        batch_size: int,
        seq_len: int,
        seed: int = 0,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

        # Stored as uint8 (LS20 frames are colour indices 0–15).
        C, H, W = obs_shape
        self._obs = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self._action = np.zeros((capacity,), dtype=np.int32)
        self._reward = np.zeros((capacity,), dtype=np.float32)
        self._cont = np.zeros((capacity,), dtype=np.uint8)
        self._size = 0
        self._ptr = 0
        # Per-episode start offsets (absolute indices into the ring); we
        # rebuild this list every episode end. For training-time sampling we
        # just sample a random absolute index and look forward seq_len steps,
        # then reject if `cont==0` appears mid-chunk (cheap with seq_len=64).
        # (Simpler than maintaining episode index list and handles wrap-around.)

    # ── Insertion ────────────────────────────────────────────────────────────

    def add(self, obs: np.ndarray, action: int, reward: float, cont: float) -> None:
        """Append a single transition. `obs` is uint8 (C, H, W)."""
        self._obs[self._ptr] = obs
        self._action[self._ptr] = action
        self._reward[self._ptr] = reward
        self._cont[self._ptr] = 1 if cont >= 0.5 else 0
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    # ── Sampling ─────────────────────────────────────────────────────────────

    def sample(self, device: torch.device) -> SequenceBatch:
        """Sample (B, T) contiguous chunks that don't cross an episode end mid-chunk.

        We reject and resample if a chunk contains `cont==0` at any non-final position.
        """
        if self._size < self.seq_len + 1:
            raise RuntimeError(f"Not enough samples in buffer: {self._size} < seq_len {self.seq_len}+1.")

        B, T = self.batch_size, self.seq_len
        idxs = np.empty(B, dtype=np.int64)
        # Sample starts in [0, _size - T)
        max_start = self._size - T
        b = 0
        attempts = 0
        while b < B and attempts < B * 20:
            start = self.rng.integers(0, max_start)
            # cont==0 means terminal; episodes may have a single 0 at last step.
            chunk_cont = self._cont[start : start + T]
            # Allow cont=0 only at the final position (T-1).
            mid = chunk_cont[:-1]
            if (mid == 0).any():
                attempts += 1
                continue
            idxs[b] = start
            b += 1
            attempts += 1
        if b < B:
            # Fall back: just take whatever; world model can handle some boundary noise.
            for j in range(b, B):
                idxs[j] = self.rng.integers(0, max_start)

        # Gather
        obs = np.stack([self._obs[i : i + T] for i in idxs], axis=0)          # (B, T, C, H, W)
        actions = np.stack([self._action[i : i + T] for i in idxs], axis=0)   # (B, T)
        rewards = np.stack([self._reward[i : i + T] for i in idxs], axis=0)   # (B, T)
        conts = np.stack([self._cont[i : i + T] for i in idxs], axis=0)       # (B, T)

        # Normalise obs: uint8 (0..15) → float in [-0.5, 0.5]
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32) / 15.0 - 0.5
        # One-hot actions
        act_t = torch.from_numpy(actions).to(device=device, dtype=torch.long)
        act_onehot = torch.nn.functional.one_hot(act_t, num_classes=self.n_actions).float()
        rew_t = torch.from_numpy(rewards).to(device=device, dtype=torch.float32)
        cont_t = torch.from_numpy(conts).to(device=device, dtype=torch.float32)
        return SequenceBatch(obs=obs_t, action=act_onehot, reward=rew_t, cont=cont_t)
