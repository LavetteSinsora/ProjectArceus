from typing import NamedTuple, List
import numpy as np
import torch


# ── LatentBuffer (exp_003+) ───────────────────────────────────────────────────

class LatentBatch(NamedTuple):
    frames:    torch.Tensor   # (B, 64, 64) uint8 — frame_t, for re-encoding h_t
    h_queries: torch.Tensor   # (B, n_latents, d_model) float32 — h_{t-1} from rollout
    actions:   torch.Tensor   # (B,) long
    h_targets: torch.Tensor   # (B, n_latents, d_model) float32 — h_{t+1} from rollout


class LatentBuffer:
    """
    Replay buffer for recurrent JEPA training (exp_003+).

    Stores (frame_t, h_query, action_t, h_target) per transition where:
      frame_t:  raw (64,64) uint8 observation at time t
      h_query:  h_{t-1} from rollout (recurrent encoding) — used as Perceiver query
                to re-encode frame_t during training, matching the rollout path exactly
      action_t: action taken at time t
      h_target: h_{t+1} from rollout — detached target for flow matching loss

    Training usage:
      h_t_fresh = encoder(frame_t_batch, h_queries.detach())  # encoder gets gradient
      loss, _ = predictor.compute_loss(h_t_fresh, h_targets.detach(), a_emb)

    Why store frame_t and h_query (not just h_t):
      Storing h_t directly would freeze the encoder (stored tensor has no grad_fn).
      Re-encoding frame_t with stored h_query gives a fresh h_t that has gradient
      through the encoder parameters while still using the correct recurrent query.

    Memory: 50K × [(64×64×1) + (4×128×4) + 8 + (4×128×4)] bytes ≈ 410 MB
    """

    def __init__(
        self,
        n_latents: int = 4,
        d_model: int = 128,
        capacity: int = 50_000,
        recency_fraction: float = 0.2,
        recent_window: int = 10_000,
    ):
        self.capacity = capacity
        self.recency_fraction = recency_fraction
        self.recent_window = min(recent_window, capacity)

        self._frames    = np.zeros((capacity, 64, 64), dtype=np.uint8)
        self._h_queries = np.zeros((capacity, n_latents, d_model), dtype=np.float32)
        self._actions   = np.zeros(capacity, dtype=np.int64)
        self._h_targets = np.zeros((capacity, n_latents, d_model), dtype=np.float32)

        self._pos  = 0
        self._size = 0

    def add(
        self,
        frame: np.ndarray,
        h_query: np.ndarray,
        action_idx: int,
        h_target: np.ndarray,
    ) -> None:
        """
        frame:    (64, 64) uint8
        h_query:  (n_latents, d_model) float32 — h_{t-1} from rollout
        action_idx: int 0-indexed
        h_target: (n_latents, d_model) float32 — h_{t+1} from rollout
        """
        self._frames[self._pos]    = frame
        self._h_queries[self._pos] = h_query
        self._actions[self._pos]   = action_idx
        self._h_targets[self._pos] = h_target
        self._pos  = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> LatentBatch:
        n_recent  = int(batch_size * self.recency_fraction)
        n_uniform = batch_size - n_recent
        uniform_idx = np.random.randint(0, self._size, size=n_uniform)

        recent_size = min(self._size, self.recent_window)
        start = (self._pos - recent_size) % self.capacity
        if start + recent_size <= self.capacity:
            recent_pool = np.arange(start, start + recent_size)
        else:
            recent_pool = np.concatenate([
                np.arange(start, self.capacity),
                np.arange(0, (start + recent_size) % self.capacity),
            ])
        recent_idx = recent_pool[np.random.randint(0, len(recent_pool), size=n_recent)]
        idx = np.concatenate([uniform_idx, recent_idx])

        return LatentBatch(
            frames=torch.from_numpy(self._frames[idx]).to(device),
            h_queries=torch.from_numpy(self._h_queries[idx]).to(device),
            actions=torch.from_numpy(self._actions[idx]).to(device),
            h_targets=torch.from_numpy(self._h_targets[idx]).to(device),
        )

    def __len__(self) -> int:
        return self._size


# ── NextFrameLatentBuffer (exp_003_2) ────────────────────────────────────────

class NextFrameLatentBatch(NamedTuple):
    frames:      torch.Tensor   # (B, 64, 64) uint8     — frame_t
    h_queries:   torch.Tensor   # (B, n_latents, d_model) float32 — h_{t-1} from rollout
    actions:     torch.Tensor   # (B,) long
    next_frames: torch.Tensor   # (B, 64, 64) uint8     — frame_{t+1}


class NextFrameLatentBuffer:
    """
    Replay buffer for exp_003_2 (action-predictor JEPA, no EMA target).

    Stores (frame_t, h_query=h_{t-1}, action_t, next_frame=frame_{t+1}) per
    transition. The fourth field is the raw next_frame (uint8) — not a
    precomputed h_target — so the JEPA training step can re-encode both h_t
    and h_{t+1} with the *current* encoder weights every step.

    Sampling is purely uniform — there is no recency mix. Re-encoding from the
    live encoder removes the stale-encoder problem that motivated recency
    weighting in exp_003_0.

    Memory: 50K × [(64×64) + (4×128×4) + 8 + (64×64)] bytes ≈ 510 MB.
    """

    def __init__(
        self,
        n_latents: int = 4,
        d_model: int = 128,
        capacity: int = 50_000,
    ):
        self.capacity = capacity

        self._frames      = np.zeros((capacity, 64, 64), dtype=np.uint8)
        self._h_queries   = np.zeros((capacity, n_latents, d_model), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._next_frames = np.zeros((capacity, 64, 64), dtype=np.uint8)

        self._pos  = 0
        self._size = 0

    def add(
        self,
        frame: np.ndarray,
        h_query: np.ndarray,
        action_idx: int,
        next_frame: np.ndarray,
    ) -> None:
        self._frames[self._pos]      = frame
        self._h_queries[self._pos]   = h_query
        self._actions[self._pos]     = action_idx
        self._next_frames[self._pos] = next_frame
        self._pos  = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> NextFrameLatentBatch:
        idx = np.random.randint(0, self._size, size=batch_size)
        return NextFrameLatentBatch(
            frames=torch.from_numpy(self._frames[idx]).to(device),
            h_queries=torch.from_numpy(self._h_queries[idx]).to(device),
            actions=torch.from_numpy(self._actions[idx]).to(device),
            next_frames=torch.from_numpy(self._next_frames[idx]).to(device),
        )

    def __len__(self) -> int:
        return self._size


class Batch(NamedTuple):
    frames: torch.Tensor       # (B, 64, 64) uint8
    actions: torch.Tensor      # (B,) long (0-indexed action indices)
    next_frames: torch.Tensor  # (B, 64, 64) uint8


class ReplayBuffer:
    """
    FIFO ring buffer for off-policy JEPA training.

    Frames are stored as (64, 64) uint8 (color index 0–15) to save memory.
    One-hot/color-embedding conversion happens at sample time inside the encoder.

    Memory: 50K × 2 × (64×64) uint8 ≈ 400 MB  (frames + next_frames)

    Sampling uses recency weighting: a fraction of each batch is drawn from
    the most recent `recent_window` transitions so the encoder stays aligned
    with the current policy's state distribution as training progresses.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        recency_fraction: float = 0.2,
        recent_window: int = 10_000,
    ):
        self.capacity = capacity
        self.recency_fraction = recency_fraction
        self.recent_window = min(recent_window, capacity)

        self._frames = np.zeros((capacity, 64, 64), dtype=np.uint8)
        self._next_frames = np.zeros((capacity, 64, 64), dtype=np.uint8)
        self._actions = np.zeros(capacity, dtype=np.int64)

        self._pos = 0   # next write index
        self._size = 0  # current number of valid entries

    def add(self, frame: np.ndarray, action_idx: int, next_frame: np.ndarray) -> None:
        """frame, next_frame: (64, 64) uint8 arrays; action_idx: int 0–3."""
        self._frames[self._pos] = frame
        self._next_frames[self._pos] = next_frame
        self._actions[self._pos] = action_idx
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        """Sample with recency-weighted mixing."""
        n_recent = int(batch_size * self.recency_fraction)
        n_uniform = batch_size - n_recent

        # Uniform sample from full buffer
        uniform_idx = np.random.randint(0, self._size, size=n_uniform)

        # Recent sample: the most recent `recent_window` entries
        recent_size = min(self._size, self.recent_window)
        start = (self._pos - recent_size) % self.capacity
        if start + recent_size <= self.capacity:
            recent_pool = np.arange(start, start + recent_size)
        else:
            recent_pool = np.concatenate([
                np.arange(start, self.capacity),
                np.arange(0, (start + recent_size) % self.capacity),
            ])
        recent_idx = recent_pool[np.random.randint(0, len(recent_pool), size=n_recent)]

        idx = np.concatenate([uniform_idx, recent_idx])
        return Batch(
            frames=torch.from_numpy(self._frames[idx]).to(device),
            actions=torch.from_numpy(self._actions[idx]).to(device),
            next_frames=torch.from_numpy(self._next_frames[idx]).to(device),
        )

    def __len__(self) -> int:
        return self._size


class PolicyBuffer:
    """
    On-policy trajectory buffer for REINFORCE.

    Stores (log_prob, reward, entropy) triples for the current episode window.
    Entropy is stored so the policy update can apply entropy regularisation
    (encourages exploration, prevents the policy collapsing to a single action).
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []
        self._entropies: List[torch.Tensor] = []

    def add(self, log_prob: torch.Tensor, reward: float,
            entropy: torch.Tensor) -> None:
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._entropies.append(entropy)

    def full(self) -> bool:
        return len(self._log_probs) >= self.capacity

    def get(self, device: torch.device):
        """Returns (log_probs, rewards, entropies) as tensors on device."""
        log_probs = torch.stack(self._log_probs).to(device)
        rewards   = torch.tensor(self._rewards, dtype=torch.float32, device=device)
        entropies = torch.stack(self._entropies).to(device)
        return log_probs, rewards, entropies

    def clear(self) -> None:
        self._log_probs.clear()
        self._rewards.clear()
        self._entropies.clear()

    def __len__(self) -> int:
        return len(self._log_probs)
