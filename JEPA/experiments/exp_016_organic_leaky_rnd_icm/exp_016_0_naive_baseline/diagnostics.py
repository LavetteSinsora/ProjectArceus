"""Diagnostics for exp_016_0 — the instrumentation that makes the failure modes
*observable*. See SYSTEM_CARD §5.

  * StateRegistry  — global table of distinct masked board states (LS20-L1 has ~110),
                     with cumulative + per-update visit counts. Enables the full-state
                     novelty landscape and the coverage metrics.
  * harvest_states — short random roam to seed the registry and pick the fixed probe set.
  * drift_rel_l2   — relative-L2 drift of an encoder on the probe states (NOT cosine:
                     the ReLU output makes cosine ≈1 for everything — see the probe).
"""
from __future__ import annotations

import numpy as np
import torch

from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel


def mask_board(frame: np.ndarray, rows: tuple) -> np.ndarray:
    m = frame.copy()
    m[..., int(rows[0]):int(rows[1]) + 1, :] = 0
    return m


def state_key(masked: np.ndarray) -> bytes:
    return masked.tobytes()


class StateRegistry:
    """Distinct masked-board states + visit counts (ground-truth oracle counts)."""

    def __init__(self):
        self.ids: dict[bytes, int] = {}
        self.exemplars: list[np.ndarray] = []     # id -> masked board
        self.visits = np.zeros(0, dtype=np.int64)

    def _ensure(self, k: bytes, board: np.ndarray) -> int:
        if k not in self.ids:
            self.ids[k] = len(self.exemplars)
            self.exemplars.append(board.copy())
            self.visits = np.append(self.visits, 0)
        return self.ids[k]

    def observe(self, masked_boards: np.ndarray) -> dict:
        """masked_boards: (M, H, W). Updates counts; returns per-update coverage stats."""
        seen_this, new_this = set(), 0
        for b in masked_boards:
            k = state_key(b)
            was_new = k not in self.ids
            i = self._ensure(k, b)
            self.visits[i] += 1
            if was_new:
                new_this += 1
            seen_this.add(i)
        return {"unique_states_this_update": len(seen_this),
                "new_states_this_update": new_this,
                "cumulative_unique_states": len(self.exemplars)}

    def all_masked(self) -> np.ndarray:
        return np.stack(self.exemplars) if self.exemplars else np.zeros((0, 64, 64), np.uint8)


def harvest_states(game: str, level: int, seed: int, roam_steps: int, n_envs: int,
                   mask_rows: tuple, n_probe: int):
    """Random roam → seed a StateRegistry and pick `n_probe` diverse probe states.
    Diversity = greedy farthest-point in raw masked-pixel Hamming space."""
    env = VecLS20EnvLevel(env_name=game, n_envs=n_envs, max_episode_steps=200,
                          seed=seed + 31, level_index=level)
    rng = np.random.default_rng(seed + 31)
    reg = StateRegistry()
    for _ in range(roam_steps):
        a = rng.integers(0, env.n_actions, size=env.n_envs)
        nobs, _r, dones, _i = env.step(a)
        m = mask_board(nobs, mask_rows)
        reg.observe(m[~dones])
    boards = reg.all_masked()
    if len(boards) == 0:
        raise RuntimeError("harvest found no states")
    # greedy farthest-point selection seeded with the most-visited state
    flat = boards.reshape(len(boards), -1).astype(np.int16)
    chosen = [int(np.argmax(reg.visits))]
    while len(chosen) < min(n_probe, len(boards)):
        d = np.min([np.abs(flat - flat[c]).sum(1) for c in chosen], axis=0)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    probe = boards[chosen]                       # (n_probe, H, W) masked
    return reg, probe


@torch.no_grad()
def encode_all(encode_masked_fn, masked_boards: np.ndarray, device, chunk: int = 512):
    """Encode (S,H,W) masked boards → (S, D) cpu tensor with the given encoder fn."""
    S = masked_boards.shape[0]
    if S == 0:
        return torch.zeros(0)
    x = torch.from_numpy(masked_boards.astype(np.int64))
    out = []
    for i in range(0, S, chunk):
        out.append(encode_masked_fn(x[i:i + chunk].to(device)).cpu())
    return torch.cat(out, 0)


def drift_rel_l2(h_before: torch.Tensor, h_after: torch.Tensor) -> dict:
    """Per-state relative L2 drift ‖after-before‖/‖before‖, plus the pairwise-distance
    context (drift relative to how far apart states are = the meaningful ratio)."""
    if h_before.numel() == 0:
        return {"drift_rel_l2": float("nan"), "mean_pairwise_l2": float("nan"),
                "drift_over_pairdist": float("nan")}
    num = (h_after - h_before).norm(dim=-1)
    den = h_before.norm(dim=-1) + 1e-8
    rel = float((num / den).mean())
    # mean pairwise L2 among the (after) states
    S = h_after.shape[0]
    if S >= 2:
        d = torch.cdist(h_after, h_after)
        pair = float(d[~torch.eye(S, dtype=bool)].mean())
    else:
        pair = float("nan")
    drift_abs = float(num.mean())
    return {"drift_rel_l2": rel, "mean_pairwise_l2": pair,
            "drift_over_pairdist": drift_abs / pair if pair and pair == pair else float("nan")}
