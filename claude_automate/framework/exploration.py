"""Count-based exploration — the generalizable exploration driver.

Knows nothing about any specific game. It maps each observation to a discrete
code, keeps a global visit count per code, and returns a novelty bonus that
decays as 1/sqrt(count). New observations ⇒ large bonus ⇒ the policy is pulled
toward unexplored regions of *any* environment.

Two interchangeable counters (both UI-mask the frame first so the
always-changing step counter does not make every frame look novel):

  ExactFrameCounter  — hashes the exact masked frame. For ARC games the
      observation is discrete (64×64, values 0–15) and the environment is
      deterministic, so identical screens recur exactly. This gives one count
      bucket per genuinely distinct screen — the sharpest possible novelty
      signal. This is the default.

  SimHashCounter     — Tang et al. 2017 "#Exploration": a random-projection
      SimHash. Appropriate when observations are continuous / never recur
      exactly. Kept for environments where ExactFrameCounter would never see
      a repeat. NOTE: on raw ARC pixels it is too coarse — a small sprite
      moving barely perturbs the projection — so it is NOT the default.

`make_counter(mode, ...)` picks one. Both share the same interface:
`visit(frame) -> count`, `novelty(count) -> bonus`, `visit_and_score(frame)`,
and the `n_distinct` property.
"""

from __future__ import annotations

import numpy as np


def _mask_frame(frame: np.ndarray, masked_rows: slice | None) -> np.ndarray:
    f = np.asarray(frame)
    if masked_rows is None:
        return f
    f = f.copy()
    f[masked_rows, :] = 0
    return f


class _BaseCounter:
    """Shared visit-count bookkeeping and 1/sqrt(count) novelty."""

    def __init__(self, masked_rows: slice | None = None):
        self.masked_rows = masked_rows
        self.counts: dict[int, int] = {}

    def code(self, frame: np.ndarray) -> int:        # pragma: no cover
        raise NotImplementedError

    def visit(self, frame: np.ndarray) -> int:
        """Register a visit; return the post-increment count for this frame."""
        c = self.code(frame)
        n = self.counts.get(c, 0) + 1
        self.counts[c] = n
        return n

    @staticmethod
    def novelty(count: int) -> float:
        """Count-based bonus 1/sqrt(count). count>=1 ⇒ bonus in (0, 1]."""
        return 1.0 / np.sqrt(max(count, 1))

    def visit_and_score(self, frame: np.ndarray) -> float:
        """Register a visit and return its novelty bonus."""
        return self.novelty(self.visit(frame))

    @property
    def n_distinct(self) -> int:
        return len(self.counts)


class ExactFrameCounter(_BaseCounter):
    """Exact-match counting over the UI-masked frame (default for ARC games)."""

    def code(self, frame: np.ndarray) -> int:
        masked = _mask_frame(frame, self.masked_rows)
        return hash(np.ascontiguousarray(masked, dtype=np.uint8).tobytes())


class SimHashCounter(_BaseCounter):
    """Random-projection SimHash counting (for continuous observations)."""

    def __init__(self, hash_bits: int = 28, frame_size: int = 64,
                 seed: int = 0, masked_rows: slice | None = None):
        super().__init__(masked_rows)
        self.hash_bits = hash_bits
        self.frame_size = frame_size
        rng = np.random.default_rng(seed)
        # Fixed random projection: (hash_bits, frame_size*frame_size).
        self.projection = rng.standard_normal(
            (hash_bits, frame_size * frame_size)
        ).astype(np.float32)

    def code(self, frame: np.ndarray) -> int:
        """Frame → integer SimHash code (sign of a random projection).

        Deterministic across processes: the code is the packed bit pattern
        read directly as a big-endian integer (no Python `hash()`, which is
        salted per process)."""
        flat = _mask_frame(frame, self.masked_rows).astype(np.float32).reshape(-1)
        bits = (self.projection @ flat) > 0.0           # (hash_bits,) bool
        return int.from_bytes(np.packbits(bits).tobytes(), "big")


def make_counter(mode: str = "exact", *, hash_bits: int = 28,
                 frame_size: int = 64, seed: int = 0,
                 masked_rows: slice | None = None) -> _BaseCounter:
    """Factory: 'exact' → ExactFrameCounter, 'simhash' → SimHashCounter."""
    if mode == "exact":
        return ExactFrameCounter(masked_rows=masked_rows)
    if mode == "simhash":
        return SimHashCounter(hash_bits=hash_bits, frame_size=frame_size,
                              seed=seed, masked_rows=masked_rows)
    raise ValueError(f"unknown count mode {mode!r} (expected 'exact'/'simhash')")
