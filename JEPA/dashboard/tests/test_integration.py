"""
Integration tests: run a real debug episode with a real checkpoint.
These tests require the arc environment and at least one checkpoint.
They are skipped automatically if no checkpoint is found.
"""
import math
from pathlib import Path

import pytest

# conftest.py adds repo root to sys.path
EXP_NAME = "exp_001_vit_jepa_baseline"
CKPT_DIR = Path(__file__).parent.parent.parent / "experiments" / EXP_NAME / "checkpoints"

def latest_checkpoint():
    ckpts = sorted(CKPT_DIR.glob("step_*.pt"), reverse=True)
    return str(ckpts[0]) if ckpts else None


@pytest.fixture(scope="module")
def episode():
    """Run a 3-step debug episode once for the whole test module."""
    ckpt = latest_checkpoint()
    if ckpt is None:
        pytest.skip(f"No checkpoint found in {CKPT_DIR}")
    from JEPA.dashboard.debug_runner import run_debug_episode
    return run_debug_episode(ckpt, experiment=EXP_NAME, max_steps=3)


# ── Top-level schema ─────────────────────────────────────────────────────────

class TestEpisodeSchema:
    def test_top_level_keys(self, episode):
        for key in ['checkpoint','checkpoint_step','episode_steps',
                    'level_completed','truncated','arc_colors','timesteps']:
            assert key in episode

    def test_arc_colors_has_16_entries(self, episode):
        assert len(episode['arc_colors']) == 16

    def test_each_arc_color_is_rgb_triple(self, episode):
        for c in episode['arc_colors']:
            assert len(c) == 3
            for v in c:
                assert 0 <= v <= 255

    def test_episode_steps_positive(self, episode):
        assert episode['episode_steps'] >= 1

    def test_timesteps_length_matches_episode_steps(self, episode):
        assert len(episode['timesteps']) == episode['episode_steps']

    def test_checkpoint_step_is_int(self, episode):
        assert isinstance(episode['checkpoint_step'], int)


# ── Per-timestep schema ───────────────────────────────────────────────────────

class TestTimestepSchema:
    def test_step_zero_required_keys(self, episode):
        ts = episode['timesteps'][0]
        for key in [
            't','frame','action_taken','is_terminal','available_actions',
            'reward','jepa_loss','patch_embeddings','next_patch_embeddings',
            'predicted_next_embeddings','reasoning_token',
            'reasoning_token_first_last3','action_probs','action_entropy',
            'patch_weights','per_patch_stats','per_patch_pred_error',
            'embedding_summary','reasoning_stats',
            'encoder_attn_block1','encoder_attn_block2','policy_attn_weights',
        ]:
            assert key in ts, f"Missing key in timestep: {key}"

    def test_frame_shape(self, episode):
        frame = episode['timesteps'][0]['frame']
        assert len(frame) == 64
        assert all(len(row) == 64 for row in frame)

    def test_frame_values_in_range(self, episode):
        frame = episode['timesteps'][0]['frame']
        for row in frame:
            for v in row:
                assert 0 <= v <= 15, f"Color index out of range: {v}"

    def test_patch_embeddings_shape(self, episode):
        emb = episode['timesteps'][0]['patch_embeddings']
        assert len(emb) == 16
        assert all(len(row) == 128 for row in emb)

    def test_patch_embeddings_are_unit_norm(self, episode):
        """Encoder applies F.normalize so all embeddings should have norm ≈ 1."""
        import math
        emb = episode['timesteps'][0]['patch_embeddings']
        for i, row in enumerate(emb):
            norm = math.sqrt(sum(v**2 for v in row))
            assert abs(norm - 1.0) < 0.01, f"Patch {i} norm = {norm:.4f} (expected ≈1)"

    def test_action_probs_sum_to_one(self, episode):
        probs = episode['timesteps'][0]['action_probs']
        assert len(probs) == 4
        assert abs(sum(probs) - 1.0) < 1e-4

    def test_action_taken_in_range(self, episode):
        for ts in episode['timesteps']:
            assert 0 <= ts['action_taken'] <= 3

    def test_reasoning_token_length(self, episode):
        h = episode['timesteps'][0]['reasoning_token']
        assert len(h) == 128

    def test_reasoning_token_first_last3_length(self, episode):
        fl = episode['timesteps'][0]['reasoning_token_first_last3']
        assert len(fl) == 6

    def test_patch_weights_length(self, episode):
        assert len(episode['timesteps'][0]['patch_weights']) == 16

    def test_per_patch_stats_count(self, episode):
        assert len(episode['timesteps'][0]['per_patch_stats']) == 16

    def test_per_patch_pred_error_count(self, episode):
        assert len(episode['timesteps'][0]['per_patch_pred_error']) == 16

    def test_encoder_attn_shape(self, episode):
        ts = episode['timesteps'][0]
        for key in ['encoder_attn_block1','encoder_attn_block2']:
            mat = ts[key]
            assert len(mat) == 16
            assert all(len(row) == 16 for row in mat)

    def test_policy_attn_weights_length(self, episode):
        assert len(episode['timesteps'][0]['policy_attn_weights']) == 16

    def test_policy_attn_weights_sum_to_one(self, episode):
        """Softmax output should sum to ≈ 1."""
        weights = episode['timesteps'][0]['policy_attn_weights']
        # May not sum exactly to 1 due to rounding to 4dp
        assert abs(sum(weights) - 1.0) < 0.05


# ── Step-to-step consistency ──────────────────────────────────────────────────

class TestStepConsistency:
    def test_t_index_increments(self, episode):
        for i, ts in enumerate(episode['timesteps']):
            assert ts['t'] == i

    def test_prev_fields_null_at_step_zero(self, episode):
        ps = episode['timesteps'][0]['per_patch_stats']
        for i, p in enumerate(ps):
            assert p['cos_sim_prev'] is None, f"Patch {i} should have null cos_sim_prev at t=0"
            assert p['l2_dist_prev'] is None

    def test_prev_fields_not_null_after_step_zero(self, episode):
        if episode['episode_steps'] < 2:
            pytest.skip("Episode too short to check step >0")
        ps = episode['timesteps'][1]['per_patch_stats']
        # At least some patches should have non-null prev fields (even if pixel_change_frac=0)
        assert any(p['cos_sim_prev'] is not None for p in ps)

    def test_reasoning_stats_null_prev_at_t0(self, episode):
        rs = episode['timesteps'][0]['reasoning_stats']
        assert rs['cos_sim_prev'] is None
        assert rs['l2_dist_prev'] is None


# ── Value sanity checks ───────────────────────────────────────────────────────

class TestValueSanity:
    def test_jepa_loss_positive(self, episode):
        for ts in episode['timesteps']:
            assert ts['jepa_loss'] >= 0, "JEPA loss should be non-negative"

    def test_reward_non_negative(self, episode):
        """Intrinsic reward is jepa_loss ≥ 0; completion bonus is 50 ≥ 0."""
        for ts in episode['timesteps']:
            assert ts['reward'] >= 0

    def test_action_entropy_non_negative(self, episode):
        for ts in episode['timesteps']:
            assert ts['action_entropy'] >= 0

    def test_pixel_change_frac_in_range(self, episode):
        for ts in episode['timesteps']:
            for ps in ts['per_patch_stats']:
                frac = ps['pixel_change_frac']
                assert 0.0 <= frac <= 1.0, f"pixel_change_frac out of [0,1]: {frac}"

    def test_effective_rank_in_range(self, episode):
        for ts in episode['timesteps']:
            rank = ts['embedding_summary']['effective_rank']
            assert 1.0 - 0.01 <= rank <= 16.0 + 0.01, f"effective_rank out of [1,16]: {rank}"

    def test_dead_dim_count_in_range(self, episode):
        for ts in episode['timesteps']:
            dead = ts['embedding_summary']['dead_dim_count']
            assert 0 <= dead <= 128

    def test_encoder_attn_rows_sum_near_one(self, episode):
        """Each row of the attention matrix should sum to ≈ 1 (softmax output)."""
        ts = episode['timesteps'][0]
        for key in ['encoder_attn_block1','encoder_attn_block2']:
            mat = ts[key]
            for i, row in enumerate(mat):
                row_sum = sum(row)
                # Rounding to 4dp means some tolerance needed
                assert abs(row_sum - 1.0) < 0.1, \
                    f"{key} row {i} sums to {row_sum:.4f} (expected ≈1)"
