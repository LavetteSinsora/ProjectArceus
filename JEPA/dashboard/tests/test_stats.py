"""
Unit tests for the pure-numpy stat functions in debug_runner.py.
No torch, no JEPA models, no arc environment needed.
"""
import math
import numpy as np
import pytest

# conftest.py adds repo root to sys.path
from JEPA.dashboard.debug_runner import (
    _activation_entropy,
    _fmt_first_last3,
    _r,
    compute_patch_stats,
    compute_embedding_summary,
    compute_reasoning_stats,
)


# ── _activation_entropy ──────────────────────────────────────────────────────

class TestActivationEntropy:
    def test_uniform_vector_entropy_is_one(self):
        """Uniform activation across all dims → entropy = 1.0."""
        v = np.ones(128, dtype=np.float32) / math.sqrt(128)  # unit-norm uniform
        ent = _activation_entropy(v)
        # p = v² / sum(v²) = 1/128 for all → H = log(128)/log(128) = 1.0
        assert abs(ent - 1.0) < 1e-4

    def test_spike_vector_entropy_near_zero(self):
        """All mass on one dimension → entropy ≈ 0."""
        v = np.zeros(128, dtype=np.float32)
        v[0] = 1.0  # entire activation on dim 0
        ent = _activation_entropy(v)
        assert ent < 0.01

    def test_zero_vector_entropy_is_zero(self):
        """All-zero vector → entropy = 0 (safe fallback)."""
        v = np.zeros(128, dtype=np.float32)
        ent = _activation_entropy(v)
        assert ent == 0.0

    def test_entropy_range(self):
        """Entropy is always in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = rng.standard_normal(128).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            ent = _activation_entropy(v)
            assert 0.0 <= ent <= 1.0 + 1e-6


# ── _fmt_first_last3 ─────────────────────────────────────────────────────────

class TestFmtFirstLast3:
    def test_returns_six_values(self):
        v = np.arange(128, dtype=np.float32)
        result = _fmt_first_last3(v)
        assert len(result) == 6

    def test_first_and_last_values_correct(self):
        v = np.zeros(128, dtype=np.float32)
        v[0], v[1], v[2] = 1.0, 2.0, 3.0
        v[125], v[126], v[127] = 10.0, 20.0, 30.0
        result = _fmt_first_last3(v)
        assert result[:3] == [1.0, 2.0, 3.0]
        assert result[3:] == [10.0, 20.0, 30.0]


# ── _r (rounding helper) ──────────────────────────────────────────────────────

class TestRounding:
    def test_rounds_to_4dp(self):
        assert _r(math.pi) == round(math.pi, 4)

    def test_accepts_numpy_scalar(self):
        x = np.float32(1.23456789)
        assert abs(_r(x) - round(float(x), 4)) < 1e-6


# ── compute_patch_stats ───────────────────────────────────────────────────────

def _make_unit(rng, d=128):
    v = rng.standard_normal(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _make_patch():
    rng = np.random.default_rng(0)
    return rng.integers(0, 16, (16, 16), dtype=np.uint8)


class TestComputePatchStats:
    rng = np.random.default_rng(7)

    def test_returns_all_required_keys(self):
        z = _make_unit(self.rng)
        z_pred = _make_unit(self.rng)
        z_next = _make_unit(self.rng)
        patch  = _make_patch()
        stats  = compute_patch_stats(z, None, z_pred, z_next, patch, patch)
        required = {
            'norm','mean','std','min_val','max_val','activation_entropy',
            'cos_sim_prev','l2_dist_prev','mean_abs_diff_prev','max_abs_diff_prev',
            'pred_error','pixel_change_frac',
        }
        assert required.issubset(set(stats.keys()))

    def test_no_prev_gives_null_prev_fields(self):
        z = _make_unit(self.rng)
        z_p, z_n = _make_unit(self.rng), _make_unit(self.rng)
        patch = _make_patch()
        stats = compute_patch_stats(z, None, z_p, z_n, patch, patch)
        assert stats['cos_sim_prev'] is None
        assert stats['l2_dist_prev'] is None
        assert stats['mean_abs_diff_prev'] is None
        assert stats['max_abs_diff_prev'] is None

    def test_identical_prev_gives_cos1_l2_zero(self):
        z = _make_unit(self.rng)
        z_p, z_n = _make_unit(self.rng), _make_unit(self.rng)
        patch = _make_patch()
        stats = compute_patch_stats(z, z.copy(), z_p, z_n, patch, patch)
        assert abs(stats['cos_sim_prev'] - 1.0) < 1e-4
        assert abs(stats['l2_dist_prev'] - 0.0) < 1e-4
        assert abs(stats['mean_abs_diff_prev'] - 0.0) < 1e-4

    def test_unit_norm_vector_has_norm_near_one(self):
        z = _make_unit(self.rng)
        z_p, z_n = _make_unit(self.rng), _make_unit(self.rng)
        patch = _make_patch()
        stats = compute_patch_stats(z, None, z_p, z_n, patch, patch)
        assert abs(stats['norm'] - 1.0) < 1e-3

    def test_identical_patches_pixel_change_frac_zero(self):
        z = _make_unit(self.rng)
        z_p, z_n = _make_unit(self.rng), _make_unit(self.rng)
        patch = _make_patch()
        stats = compute_patch_stats(z, None, z_p, z_n, patch, patch.copy())
        assert stats['pixel_change_frac'] == 0.0

    def test_all_different_pixels_frac_one(self):
        z = _make_unit(self.rng)
        z_p, z_n = _make_unit(self.rng), _make_unit(self.rng)
        p_curr = np.zeros((16,16), dtype=np.uint8)   # all color 0
        p_next = np.ones((16,16), dtype=np.uint8)    # all color 1
        stats = compute_patch_stats(z, None, z_p, z_n, p_curr, p_next)
        assert abs(stats['pixel_change_frac'] - 1.0) < 1e-6

    def test_pred_error_zero_when_pred_equals_next(self):
        z = _make_unit(self.rng)
        z_pred = _make_unit(self.rng)
        patch = _make_patch()
        stats = compute_patch_stats(z, None, z_pred, z_pred.copy(), patch, patch)
        assert abs(stats['pred_error']) < 1e-5


# ── compute_embedding_summary ─────────────────────────────────────────────────

class TestComputeEmbeddingSummary:
    rng = np.random.default_rng(13)

    def test_required_keys_present(self):
        z = np.stack([_make_unit(self.rng) for _ in range(16)])
        s = compute_embedding_summary(z, None)
        for key in ['mean_pairwise_cos_sim','effective_rank','per_dim_std_mean',
                    'per_dim_std_hist_counts','per_dim_std_hist_edges','dead_dim_count',
                    'mean_embedding_drift']:
            assert key in s

    def test_no_prev_gives_null_drift(self):
        z = np.stack([_make_unit(self.rng) for _ in range(16)])
        s = compute_embedding_summary(z, None)
        assert s['mean_embedding_drift'] is None

    def test_collapse_has_low_effective_rank(self):
        """All 16 patches identical → effective rank ≈ 1."""
        base = _make_unit(self.rng)
        z = np.stack([base.copy() for _ in range(16)])
        s = compute_embedding_summary(z, None)
        assert s['effective_rank'] < 2.0

    def test_collapse_has_high_pairwise_cos_sim(self):
        """All 16 patches identical → pairwise cosine ≈ 1."""
        base = _make_unit(self.rng)
        z = np.stack([base.copy() for _ in range(16)])
        s = compute_embedding_summary(z, None)
        assert s['mean_pairwise_cos_sim'] > 0.99

    def test_diverse_has_higher_effective_rank(self):
        """Random unit vectors → effective rank significantly above 1."""
        z = np.stack([_make_unit(self.rng) for _ in range(16)])
        s = compute_embedding_summary(z, None)
        assert s['effective_rank'] > 3.0

    def test_drift_zero_when_prev_is_same(self):
        z = np.stack([_make_unit(self.rng) for _ in range(16)])
        s = compute_embedding_summary(z, z.copy())
        assert abs(s['mean_embedding_drift']) < 1e-5

    def test_hist_has_20_bins(self):
        z = np.stack([_make_unit(self.rng) for _ in range(16)])
        s = compute_embedding_summary(z, None)
        assert len(s['per_dim_std_hist_counts']) == 20
        assert len(s['per_dim_std_hist_edges']) == 21

    def test_dead_dim_count_all_identical(self):
        """If all patches are identical, every dim has std=0 → all 128 dead."""
        base = _make_unit(self.rng)
        z = np.stack([base.copy() for _ in range(16)])
        s = compute_embedding_summary(z, None)
        assert s['dead_dim_count'] == 128

    def test_effective_rank_range(self):
        """Effective rank must be in [1, 16]."""
        z = np.stack([_make_unit(self.rng) for _ in range(16)])
        s = compute_embedding_summary(z, None)
        assert 1.0 <= s['effective_rank'] <= 16.0 + 1e-4


# ── compute_reasoning_stats ───────────────────────────────────────────────────

class TestComputeReasoningStats:
    rng = np.random.default_rng(99)

    def test_keys_present(self):
        h = _make_unit(self.rng)
        s = compute_reasoning_stats(h, None)
        for k in ['norm','mean','std','activation_entropy',
                  'cos_sim_prev','l2_dist_prev','mean_abs_diff_prev','max_abs_diff_prev']:
            assert k in s

    def test_null_prev_fields_at_t0(self):
        h = _make_unit(self.rng)
        s = compute_reasoning_stats(h, None)
        assert s['cos_sim_prev'] is None
        assert s['l2_dist_prev'] is None

    def test_identical_prev_cos1_l2_zero(self):
        h = _make_unit(self.rng)
        s = compute_reasoning_stats(h, h.copy())
        assert abs(s['cos_sim_prev'] - 1.0) < 1e-4
        assert abs(s['l2_dist_prev']) < 1e-4

    def test_norm_correct(self):
        h = _make_unit(self.rng)
        s = compute_reasoning_stats(h, None)
        assert abs(s['norm'] - 1.0) < 1e-3
