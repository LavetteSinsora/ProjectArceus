"""Dashboard episode runner for exp_016_0 (naive leaky-RND on IDM features).

The shared JEPA dashboard (JEPA/dashboard/server.py) auto-discovers this module and
calls `run_debug_episode(checkpoint_path, env_name=..., max_steps=...)` to replay the
agent's behavior at a selected checkpoint. We return the same per-timestep schema the
exp_010/011/012 runners use (frame, action_taken, action_probs, entropy), so the
generic frame-scrubbing viewer works with NO panel.js — plus two exp_016 extras per
step: `novelty` (the leaky-RND bonus the agent sees) and the masked board it actually
encodes. The full metrics page is served separately from this run's metrics.jsonl.

Only the actor is needed to ACT; the tracker (IDM encoder + leaky RND) is loaded so we
can show the intrinsic reward signal alongside the behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from JEPA.shared.env_wrapper import (  # noqa: E402
    make_env, resolve_dashboard_env, short_env_name, full_game_id,
)
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device  # noqa: E402
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import _LevelStartWrapper  # noqa: E402
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi  # noqa: E402

from .config import Config
from .actor import Actor, mask_frames
from .tracker import IDMEncoder

ARC_COLORS_RGB = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (61, 153, 112), (255, 255, 255),
    (0, 31, 63), (1, 255, 112), (133, 20, 75), (1, 75, 101),
]

CAPABILITIES = {
    "has_encoder_attention": False, "has_policy_attention": False,
    "has_patch_embeddings": False, "has_latent_vectors": False,
    "n_patches": 0, "extra": {"intrinsic": "leaky_rnd_on_idm"},
}


def _load(checkpoint_path: str, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw = ck.get("config", {})
    cfg = Config(**raw) if isinstance(raw, dict) else raw
    actor = Actor(cfg.n_actions, cfg.n_colors, cfg.frame_size, cfg.trunk_dim,
                  tuple(cfg.timer_mask_rows),
                  value_head=getattr(cfg, "use_value_head", False)).to(device)
    actor.load_state_dict(ck["actor"]); actor.eval()
    idm = IDMEncoder(cfg.n_actions, cfg.n_colors, cfg.frame_size, cfg.trunk_dim,
                     cfg.idm_hidden, layernorm=getattr(cfg, "idm_layernorm", False)).to(device)
    idm.load_state_dict(ck["idm"]); idm.eval()
    rnd = RNDPhi(dim=cfg.trunk_dim, hidden=cfg.rnd_hidden, out=cfg.rnd_out,
                 leak=cfg.leak).to(device)
    rnd.target.load_state_dict(ck["rnd_target"])
    rnd.predictor.load_state_dict(ck["rnd_predictor"])
    rnd.eval()
    return cfg, actor, idm, rnd, int(ck.get("step", 0))


def run_debug_episode(checkpoint_path: str, env_name: str | None = None,
                      max_steps: int = 200) -> dict:
    device = get_device()
    cfg, actor, idm, rnd, ckpt_step = _load(checkpoint_path, device)

    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(_REPO_ROOT / "environment_files"))
    full_gid, warning = resolve_dashboard_env(env_name, full_game_id(cfg.game))
    env = make_env(arc.make(full_gid), full_gid)
    level_index = int(getattr(cfg, "level_index", 0) or 0)
    if level_index > 0:
        env = _LevelStartWrapper(env, level_index)
    rows = tuple(cfg.timer_mask_rows)

    frame_np = env.reset()
    timesteps = []
    for t in range(max_steps):
        obs_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = actor.logits(obs_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = int(dist.sample().item())
            entropy = float(dist.entropy().item())
            # intrinsic reward the agent sees on THIS state (leaky-RND on IDM-φ)
            h = idm.encode_masked(mask_frames(obs_t, rows))
            novelty = float(rnd.novelty(h).item())

        next_np, is_terminal = env.step(action_idx)
        success = bool(env.level_completed)
        reward = 1.0 if (is_terminal and success) else 0.0

        timesteps.append({
            "t": t,
            "frame": frame_np.tolist(),
            "action_taken": action_idx,
            "is_terminal": bool(is_terminal),
            "available_actions": list(env.available_actions),
            "reward": round(reward, 4),
            "value": 0.0,                          # no critic (pure REINFORCE)
            "novelty": round(novelty, 6),          # exp_016 extra: leaky-RND bonus
            "action_probs": [round(float(p), 4) for p in probs.tolist()],
            "action_entropy": round(entropy, 4),
        })
        frame_np = next_np
        if is_terminal:
            break

    out = {
        "checkpoint": Path(checkpoint_path).name,
        "checkpoint_step": ckpt_step,
        "experiment": cfg.exp_name,
        "env_name": short_env_name(full_gid),
        "level_index": level_index,
        "capabilities": CAPABILITIES,
        "episode_steps": len(timesteps),
        "level_completed": bool(env.level_completed),
        "truncated": len(timesteps) >= max_steps and not timesteps[-1]["is_terminal"],
        "arc_colors": ARC_COLORS_RGB,
        "timesteps": timesteps,
    }
    if warning:
        out["warning"] = warning
    return out
