"""
HealthMonitor — section-tagged streaming deques for exp_003_3.

Each metric in metrics.md lives in one of seven sections:
  sec1 — Representation health (placeholder + latent stats)
  sec2 — Image-patch SA attention
  sec3 — Perceiver self-attention
  sec4 — Predictors (state ODE + action predictor diagnostics)
  sec5 — Policy entropy
  sec6 — Performance + losses + per-class CE + exploration
  sec7 — Gradient norms (per-source × per-sub-block) + UWR + cosines

This monitor only stores deques and provides mean readouts; the actual
metric computations live in the sibling monitors.* modules.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List

import numpy as np


# ── Critical thresholds (lifted from exp_003_0) ──────────────────────────────
LATENT_NORM_CRITICAL = 34.0   # ~3 × sqrt(128); fires only if output_norm fails
LATENT_STD_CRITICAL  = 0.01
LOSS_CV_CRITICAL     = 2.0
TIME_GRAD_WARN       = 1e-4
GRAD_NORM_CRITICAL   = 200.0
ENTROPY_WARN         = 0.30
ODE_COSSIM_WARN      = 0.99


def _make_deque(window: int) -> deque:
    return deque(maxlen=window)


class HealthMonitor:
    def __init__(self, window_fast: int = 200, window_slow: int = 50,
                 window_eval: int = 5, n_latents: int = 4, n_actions: int = 4,
                 n_perceiver_rounds: int = 2):
        self.w_fast = window_fast
        self.w_slow = window_slow
        self.w_eval = window_eval
        self.n_latents = n_latents
        self.n_actions = n_actions
        self.n_perceiver_rounds = n_perceiver_rounds

        # ── sec1: representation ────────────────────────────────────────────
        self.sec1: Dict[str, deque] = {
            "placeholder_pairwise_cossim": _make_deque(window_fast),
            "placeholder_drift_from_init_mean": _make_deque(window_fast),
            "latent_pairwise_cossim_buf": _make_deque(window_slow),
            "latent_pairwise_l2_buf":     _make_deque(window_slow),
            "latent_eff_rank":            _make_deque(window_slow),
            "ht_htp1_cossim_rollout":     _make_deque(window_fast),
            # eval-time series (populated by eval_pass)
            "latent_pairwise_cossim_t1":  _make_deque(window_eval),
            "latent_pairwise_cossim_t10": _make_deque(window_eval),
            "latent_pairwise_cossim_t20": _make_deque(window_eval),
            "round0_postSA_pairwise_cossim_t1": _make_deque(window_eval),
            "ht_htp1_cossim_eval":        _make_deque(window_eval),
            "H1_HT_cossim":               _make_deque(window_eval),
            "T_episode_eval":             _make_deque(window_eval),
        }
        self.sec1_per_placeholder = [
            _make_deque(window_fast) for _ in range(n_latents)
        ]
        self.sec1_latent_norms = [_make_deque(window_slow) for _ in range(n_latents)]

        # ── sec2: image-patch SA ────────────────────────────────────────────
        self.sec2: Dict[str, deque] = {
            "patch_sa_row_jsd":      _make_deque(window_slow),
            "patch_sa_temporal_jsd": _make_deque(window_eval),
        }

        # ── sec3: perceiver self-attn ───────────────────────────────────────
        self.sec3_latent_self_attn_row_jsd = [
            _make_deque(window_eval) for _ in range(n_perceiver_rounds)
        ]

        # ── sec4: predictors ────────────────────────────────────────────────
        self.sec4: Dict[str, deque] = {
            "ode_step_cossim":           _make_deque(window_slow),
            "ode_first_vs_final_cossim": _make_deque(window_slow),
            "predictor_velocity_norm":   _make_deque(window_slow),
            "action_pred_input_cossim":  _make_deque(window_fast),
            "action_pred_entropy":       _make_deque(window_fast),
            "action_pred_entropy_eval":  _make_deque(window_eval),
        }
        self.action_pred_entropy_max = float(math.log(max(n_actions, 1)))

        # ── sec5: policy ────────────────────────────────────────────────────
        self.sec5: Dict[str, deque] = {
            "policy_entropy":            _make_deque(window_fast),
            "policy_entropy_normalized": _make_deque(window_fast),
        }
        self.policy_entropy_max = float(math.log(max(n_actions, 1)))

        # ── sec6: performance ───────────────────────────────────────────────
        self.sec6: Dict[str, deque] = {
            "L_state":                   _make_deque(window_fast),
            "L_action":                  _make_deque(window_fast),
            "L_total":                   _make_deque(window_fast),
            "pol_loss":                  _make_deque(window_fast),
            "reward_total":              _make_deque(window_fast),
            "reward_state_component":    _make_deque(window_fast),
            "reward_action_component":   _make_deque(window_fast),
            "reachable_tile_coverage_pct": _make_deque(window_eval),
            "cross_hits_per_episode":      _make_deque(window_eval),
        }
        self.sec6_per_latent_state_loss = [
            _make_deque(window_fast) for _ in range(n_latents)
        ]
        self.sec6_action_ce_per_class = [
            _make_deque(window_fast) for _ in range(n_actions)
        ]
        self.sec6_action_count_per_class = [
            _make_deque(window_fast) for _ in range(n_actions)
        ]

        # ── sec7: gradients (filled dynamically — see writer) ───────────────
        # gnorm_<sub>_total / _from_state / _from_action_via_Ht / _from_action_via_Htp1
        # gcossim_state_vs_action_<sub>, gcossim_action_Ht_vs_Htp1_<sub>
        # uwr_<sub>
        self.sec7: Dict[str, deque] = {}

        # ── Episode / reward tracking (legacy keys, used by check()) ────────
        self.episodes_done: List[int] = []
        self._low_std_counts = [0] * n_latents
        self._action_ce_max_streak = 0

    # ── Utility ─────────────────────────────────────────────────────────────

    def _mean(self, q: deque) -> float:
        return float(np.mean(q)) if q else float("nan")

    def push_sec7(self, key: str, value: float, window: int = 50) -> None:
        """Lazy push into sec7 deque, creating it on first use."""
        q = self.sec7.get(key)
        if q is None:
            q = _make_deque(window)
            self.sec7[key] = q
        q.append(value)

    # ── Headline aggregates used by print_stats ─────────────────────────────

    def mean_L_total(self) -> float:    return self._mean(self.sec6["L_total"])
    def mean_L_state(self) -> float:    return self._mean(self.sec6["L_state"])
    def mean_L_action(self) -> float:   return self._mean(self.sec6["L_action"])
    def mean_reward(self) -> float:     return self._mean(self.sec6["reward_total"])
    def mean_entropy(self) -> float:    return self._mean(self.sec5["policy_entropy"])

    def completion_rate(self, window: int = 20) -> float:
        recent = self.episodes_done[-window:]
        return float(np.mean(recent)) if recent else 0.0

    # ── Critical / warning checks ───────────────────────────────────────────

    def check(self) -> tuple:
        criticals, warnings = [], []

        # NaN in losses
        for k in ("L_state", "L_action", "L_total"):
            q = self.sec6[k]
            if q and not np.isfinite(q[-1]):
                criticals.append(f"NaN in {k}")

        # Exploding gradients (use the patch_sa total if logged)
        gn_patch = self.sec7.get("gnorm_patch_sa_total")
        if gn_patch and self._mean(gn_patch) > GRAD_NORM_CRITICAL:
            criticals.append(f"Exploding patch_sa grad: {self._mean(gn_patch):.1f}")

        # Latent norm explosion
        for i, q in enumerate(self.sec1_latent_norms):
            if q and self._mean(q) > LATENT_NORM_CRITICAL:
                criticals.append(f"Latent {i} norm explosion: {self._mean(q):.2f}")

        # Action predictor saturated at log(n_actions) for >50 steps
        ent_q = self.sec4["action_pred_entropy"]
        if ent_q and len(ent_q) >= 50:
            recent_ent = list(ent_q)[-50:]
            if np.mean(recent_ent) >= self.action_pred_entropy_max - 0.01:
                self._action_ce_max_streak += 1
                if self._action_ce_max_streak >= 5:
                    warnings.append(
                        f"Action predictor near max entropy "
                        f"({np.mean(recent_ent):.4f} vs Hmax={self.action_pred_entropy_max:.4f}) "
                        "— possible collapse"
                    )
            else:
                self._action_ce_max_streak = 0

        # Standard exp_003_0 warnings
        time_q = self.sec7.get("gnorm_state_pred_time_emb_total")
        if time_q and self._mean(time_q) < TIME_GRAD_WARN:
            warnings.append(f"Time embedding grad={self._mean(time_q):.2e}")

        ode_q = self.sec4["ode_step_cossim"]
        if ode_q and self._mean(ode_q) > ODE_COSSIM_WARN:
            warnings.append(f"ODE step cos-sim={self._mean(ode_q):.4f} > {ODE_COSSIM_WARN}")

        ent_pol = self.sec5["policy_entropy"]
        if ent_pol and self._mean(ent_pol) < ENTROPY_WARN:
            warnings.append(f"Low policy entropy: {self._mean(ent_pol):.3f}")

        return criticals, warnings
