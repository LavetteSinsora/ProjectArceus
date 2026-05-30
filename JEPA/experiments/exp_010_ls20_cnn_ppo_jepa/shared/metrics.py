"""Metric helpers + JSONL writer.

`mean_feature_cosine` is the encoder-health diagnostic carried over from
exp_007: average cosine similarity between consecutive same-episode trunk
features. Near 1.0 => representational rigidity / collapse; very low => noisy.

`MetricsWriter` appends one JSON object per line to runs/<run>/metrics.jsonl,
the format the JEPA dashboard's /api/training/metrics endpoint reads. Each
record carries a `step` (env steps) and `update` index so both the exp_007-
style and exp_001-style dashboards can plot it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


def mean_feature_cosine(features: torch.Tensor, ep_starts: torch.Tensor) -> float:
    """features: (T, N, D), ep_starts: (T, N) bool. Mean cosine(h_t, h_{t+1})
    over consecutive steps that stay within the same episode."""
    T, N, _ = features.shape
    if T < 2:
        return float("nan")
    f = torch.nn.functional.normalize(features, dim=-1)
    cos = (f[:-1] * f[1:]).sum(-1)            # (T-1, N)
    # A pair (t, t+1) is same-episode iff step t+1 did NOT begin a new episode.
    same = ~ep_starts[1:]                     # (T-1, N)
    vals = cos[same]
    if vals.numel() == 0:
        return float("nan")
    return float(vals.mean().item())


def feature_health(features: torch.Tensor) -> dict:
    """Collapse diagnostics on a (M, D) feature matrix: per-dim std mean,
    effective rank, mean pairwise L2. Cheap; computed in numpy on CPU."""
    x = features.reshape(-1, features.shape[-1]).detach().cpu().numpy().astype(np.float64)
    if x.shape[0] < 2:
        return {"feat_std": float("nan"), "feat_effective_rank": float("nan"),
                "feat_pairwise_l2": float("nan")}
    std = float(x.std(axis=0).mean())
    xc = x - x.mean(axis=0)
    # effective rank via singular value entropy
    try:
        s = np.linalg.svd(xc, compute_uv=False)
        s2 = s ** 2
        p = s2 / (s2.sum() + 1e-12)
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    except np.linalg.LinAlgError:
        eff_rank = float("nan")
    # subsample for pairwise L2
    m = min(256, x.shape[0])
    sub = x[np.random.choice(x.shape[0], m, replace=False)]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    pairwise = float(d[np.triu_indices(m, k=1)].mean())
    return {"feat_std": std, "feat_effective_rank": eff_rank, "feat_pairwise_l2": pairwise}


def scrub(obj):
    """Replace NaN/Inf with None so the record is strict-JSON for the dashboard."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


class MetricsWriter:
    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)

    def write(self, record: dict):
        self._f.write(json.dumps(scrub(record)) + "\n")

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
