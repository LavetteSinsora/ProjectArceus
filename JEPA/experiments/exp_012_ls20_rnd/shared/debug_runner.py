"""Shared dashboard episode runner for the exp_012 family.

Mirrors exp_010's runner but loads the dual-head ActorCriticRND and surfaces
both critic values (extrinsic + intrinsic) per step. The JEPA/ViT capability
flags are False so the ViT-only dashboard cards hide cleanly.
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
from .model import ActorCriticRND  # noqa: E402
from .config_base import Config  # noqa: E402
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device  # noqa: E402

ARC_COLORS_RGB = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (61, 153, 112), (255, 255, 255),
    (0, 31, 63), (1, 255, 112), (133, 20, 75), (1, 75, 101),
]

CAPABILITIES = {
    "has_encoder_attention": False,
    "has_policy_attention": False,
    "has_patch_embeddings": False,
    "has_latent_vectors": False,
    "n_patches": 0,
    "extra": {},
}


def _load_model(checkpoint_path: str, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_raw = ck.get("config", {})
    cfg = Config(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw
    model = ActorCriticRND(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                           frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return cfg, model, int(ck.get("step", 0))


def run_debug_episode(checkpoint_path: str, env_name: str | None = None,
                      max_steps: int = 200) -> dict:
    device = get_device()
    cfg, model, ckpt_step = _load_model(checkpoint_path, device)

    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(_REPO_ROOT / "environment_files"))
    cfg_full = full_game_id(cfg.env_name)
    full_gid, warning = resolve_dashboard_env(env_name, cfg_full)
    env = make_env(arc.make(full_gid), full_gid)

    frame_np = env.reset()
    timesteps = []
    for t in range(max_steps):
        obs_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, v_ext, v_int, _ = model.forward(obs_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = int(dist.sample().item())
            entropy = float(dist.entropy().item())

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
            "value": round(float(v_ext.item()), 4),
            "value_intrinsic": round(float(v_int.item()), 4),
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
