"""
Server endpoint tests using FastAPI's TestClient.
Mocks run_debug_episode so no GPU / arc environment is needed.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

# conftest.py adds JEPA/ to sys.path
from fastapi.testclient import TestClient
from dashboard.server import app

client = TestClient(app, raise_server_exceptions=True)


# ── /api/checkpoints ─────────────────────────────────────────────────────────

class TestListCheckpoints:
    def test_returns_200(self):
        resp = client.get('/api/checkpoints')
        assert resp.status_code == 200

    def test_response_has_checkpoints_key(self):
        resp = client.get('/api/checkpoints')
        data = resp.json()
        assert 'checkpoints' in data

    def test_checkpoints_is_a_list(self):
        resp = client.get('/api/checkpoints')
        assert isinstance(resp.json()['checkpoints'], list)

    def test_checkpoints_are_pt_filenames(self):
        resp = client.get('/api/checkpoints')
        for name in resp.json()['checkpoints']:
            assert name.endswith('.pt'), f"Expected .pt file, got {name}"

    def test_checkpoints_sorted_newest_first(self):
        resp = client.get('/api/checkpoints')
        names = resp.json()['checkpoints']
        if len(names) >= 2:
            # step_235000.pt > step_230000.pt when sorted descending
            assert names[0] >= names[1], "Checkpoints should be newest first"


# ── /api/run_episode ─────────────────────────────────────────────────────────

def _fake_episode():
    """Minimal valid episode dict matching the schema debug_runner returns."""
    ts = {
        "t": 0, "frame": [[0]*64 for _ in range(64)],
        "action_taken": 0, "is_terminal": False,
        "available_actions": [1,2,3,4], "reward": 0.05, "jepa_loss": 0.05,
        "patch_embeddings": [[0.0]*128 for _ in range(16)],
        "next_patch_embeddings": [[0.0]*128 for _ in range(16)],
        "predicted_next_embeddings": [[0.0]*128 for _ in range(16)],
        "reasoning_token": [0.0]*128,
        "reasoning_token_first_last3": [0.0]*6,
        "action_probs": [0.25,0.25,0.25,0.25],
        "action_entropy": 1.386, "patch_weights": [0.1]*16,
        "per_patch_stats": [{
            "norm":1.0,"mean":0.0,"std":0.08,"min_val":-0.3,"max_val":0.3,
            "activation_entropy":0.8,"cos_sim_prev":None,"l2_dist_prev":None,
            "mean_abs_diff_prev":None,"max_abs_diff_prev":None,
            "pred_error":0.1,"pixel_change_frac":0.0,
        }]*16,
        "per_patch_pred_error": [0.1]*16,
        "embedding_summary": {
            "mean_pairwise_cos_sim":0.1,"effective_rank":12.0,
            "per_dim_std_mean":0.09,"per_dim_std_hist_counts":[0.0]*20,
            "per_dim_std_hist_edges":[0.0]*21,"dead_dim_count":3,
            "mean_embedding_drift":None,
        },
        "reasoning_stats": {
            "norm":1.0,"mean":0.0,"std":0.08,"activation_entropy":0.7,
            "cos_sim_prev":None,"l2_dist_prev":None,
            "mean_abs_diff_prev":None,"max_abs_diff_prev":None,
        },
        "encoder_attn_block1": [[1/16]*16 for _ in range(16)],
        "encoder_attn_block2": [[1/16]*16 for _ in range(16)],
        "policy_attn_weights": [1/16]*16,
    }
    return {
        "checkpoint": "step_235000.pt",
        "checkpoint_step": 235000,
        "episode_steps": 1,
        "level_completed": False,
        "truncated": False,
        "arc_colors": [[0,0,0]]*16,
        "timesteps": [ts],
    }


class TestRunEpisode:
    def test_missing_checkpoint_returns_404(self):
        resp = client.post('/api/run_episode', json={
            "checkpoint": "nonexistent_step_999.pt",
            "max_steps": 5,
        })
        assert resp.status_code == 404

    def test_invalid_payload_returns_422(self):
        # Missing required 'checkpoint' field
        resp = client.post('/api/run_episode', json={"max_steps": 5})
        assert resp.status_code == 422

    def test_mocked_episode_returns_200(self):
        """With mocked runner, server should return 200 and valid JSON."""
        with patch('dashboard.server.run_debug_episode', return_value=_fake_episode()):
            # We still need the checkpoint file to 'exist' for the path check
            from dashboard.server import CKPT_DIR
            fake_path = CKPT_DIR / 'step_235000.pt'
            if not fake_path.exists():
                pytest.skip("Real checkpoint not present; skipping mock test that needs path check.")
            resp = client.post('/api/run_episode', json={
                "checkpoint": "step_235000.pt",
                "max_steps": 2,
            })
        assert resp.status_code == 200

    def test_mocked_episode_schema(self):
        """Verify top-level keys in the response."""
        with patch('dashboard.server.run_debug_episode', return_value=_fake_episode()):
            from dashboard.server import CKPT_DIR
            fake_path = CKPT_DIR / 'step_235000.pt'
            if not fake_path.exists():
                pytest.skip("Real checkpoint not present.")
            resp = client.post('/api/run_episode', json={
                "checkpoint": "step_235000.pt",
                "max_steps": 2,
            })
        data = resp.json()
        for key in ['checkpoint','checkpoint_step','episode_steps','level_completed',
                    'truncated','arc_colors','timesteps']:
            assert key in data, f"Missing key: {key}"

    def test_serve_ui_returns_html(self):
        resp = client.get('/')
        assert resp.status_code == 200
        assert 'text/html' in resp.headers.get('content-type','')
        assert 'JEPA Debug Dashboard' in resp.text
