"""
MetricsWriter — emits one JSON line per LOG_FREQ tick.

Keys are flat but section-prefixed (e.g. `sec1/placeholder_pairwise_cossim`),
so a downstream dashboard can group them by tab without parsing nested
structures.

`write_eval_record(...)` writes an extra one-shot line whose keys live only
in sec1/2/3/4/6 — useful when the eval pass runs on a step that does not
coincide with a normal log tick.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .health import HealthMonitor


def _r(v: float, dp: int = 6):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:    # NaN
        return None
    return round(f, dp)


class MetricsWriter:
    def __init__(self, run_dir: Path):
        self._path = run_dir / "metrics.jsonl"
        self._fh = open(self._path, "w", buffering=1, encoding="utf-8")
        print(f"[exp003_3] Metrics file: {self._path}")

    def _record_from_health(self, health: HealthMonitor) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # sec1 ───────────────────────────────────────────────────────────────
        for k, q in health.sec1.items():
            out[f"sec1/{k}"] = _r(health._mean(q))
        for i, q in enumerate(health.sec1_per_placeholder):
            out[f"sec1/placeholder_drift_from_init_cossim_{i}"] = _r(health._mean(q))
        for i, q in enumerate(health.sec1_latent_norms):
            out[f"sec1/latent_norm_{i}"] = _r(health._mean(q))

        # sec2 ───────────────────────────────────────────────────────────────
        for k, q in health.sec2.items():
            out[f"sec2/{k}"] = _r(health._mean(q))

        # sec3 ───────────────────────────────────────────────────────────────
        for i, q in enumerate(health.sec3_latent_self_attn_row_jsd):
            out[f"sec3/latent_self_attn_row_jsd_r{i}"] = _r(health._mean(q))

        # sec4 ───────────────────────────────────────────────────────────────
        for k, q in health.sec4.items():
            out[f"sec4/{k}"] = _r(health._mean(q))
        out["sec4/action_pred_entropy_max"] = _r(health.action_pred_entropy_max)

        # sec5 ───────────────────────────────────────────────────────────────
        for k, q in health.sec5.items():
            out[f"sec5/{k}"] = _r(health._mean(q))
        out["sec5/policy_entropy_max"] = _r(health.policy_entropy_max)

        # sec6 ───────────────────────────────────────────────────────────────
        for k, q in health.sec6.items():
            out[f"sec6/{k}"] = _r(health._mean(q))
        for i, q in enumerate(health.sec6_per_latent_state_loss):
            out[f"sec6/L_state_per_latent_{i}"] = _r(health._mean(q))
        for a, q in enumerate(health.sec6_action_ce_per_class):
            out[f"sec6/L_action_per_class_{a}"] = _r(health._mean(q))
        for a, q in enumerate(health.sec6_action_count_per_class):
            out[f"sec6/L_action_count_per_class_{a}"] = _r(health._mean(q))
        out["sec6/completion_rate"] = _r(health.completion_rate())

        # sec7 ───────────────────────────────────────────────────────────────
        for k, q in health.sec7.items():
            out[f"sec7/{k}"] = _r(health._mean(q))

        return out

    def write(
        self,
        step: int,
        fps: float,
        buf_size: int,
        ep_count: int,
        health: HealthMonitor,
    ) -> None:
        rec = {
            "step": step,
            "fps": round(fps, 1),
            "buf_size": buf_size,
            "ep_count": ep_count,
        }
        rec.update(self._record_from_health(health))
        self._fh.write(json.dumps(rec) + "\n")

    def write_eval(self, step: int, eval_dict: Dict[str, Any]) -> None:
        """One-shot extra line for an eval pass; keys must be already section-tagged."""
        rec = {"step": step, "_kind": "eval_pass"}
        for k, v in eval_dict.items():
            rec[k] = _r(v) if isinstance(v, (int, float)) else v
        self._fh.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        self._fh.close()
