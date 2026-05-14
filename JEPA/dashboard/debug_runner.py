"""
Debug episode runner for the JEPA dashboard.

Loads a checkpoint, runs one episode, and returns a fully serialized dict
containing per-timestep data for all dashboard visualizations.

Usage (standalone test):
    cd "Code Repo"
    uv run python JEPA/dashboard/debug_runner.py \\
        JEPA/experiments/exp_001_vit_jepa_baseline/checkpoints/step_235000.pt \\
        exp_001_vit_jepa_baseline
"""

import copy
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root (Code Repo/) is on sys.path so JEPA package is importable
_REPO_ROOT = Path(__file__).parent.parent.parent  # dashboard → JEPA → Code Repo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from JEPA.shared.env_wrapper import LS20Env  # noqa: E402


def _get_experiment_models(experiment: str):
    """Dynamically import the experiment's models package and return (module, compute_patch_weights)."""
    models_mod = importlib.import_module(f"JEPA.experiments.{experiment}.models")
    train_mod  = importlib.import_module(f"JEPA.experiments.{experiment}.train")
    return models_mod, train_mod.compute_patch_weights


# ── ARC-AGI 16-color palette (indices 0–15) ────────────────────────────────
# Colors 0–9: official ARC-AGI palette; 10–15: extended for LS20.
ARC_COLORS_RGB = [
    (0,   0,   0),    # 0  black
    (0,  116, 217),   # 1  blue
    (255,  65,  54),  # 2  red
    (46,  204,  64),  # 3  green
    (255, 220,   0),  # 4  yellow
    (170, 170, 170),  # 5  gray
    (240,  18, 190),  # 6  magenta
    (255, 133,  27),  # 7  orange
    (127, 219, 255),  # 8  azure
    (135,  12,  37),  # 9  maroon
    (61,  153, 112),  # 10 teal-green
    (255, 255, 255),  # 11 white
    (0,   31,  63),   # 12 navy
    (1,  255, 112),   # 13 lime
    (133,  20,  75),  # 14 burgundy
    (1,   75, 101),   # 15 dark-teal
]


# ── Model loading ───────────────────────────────────────────────────────────

def load_checkpoint(path: str, experiment: str, device: torch.device):
    """Load checkpoint and instantiate all models via the experiment's load_models factory."""
    # Old checkpoints pickle Config as 'config.Config' (from sys.path including JEPA/).
    # Temporarily add JEPA/ so unpickling finds the shim at JEPA/config.py.
    _jepa_dir = str(_REPO_ROOT / "JEPA")
    _inserted = _jepa_dir not in sys.path
    if _inserted:
        sys.path.insert(0, _jepa_dir)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    finally:
        if _inserted:
            sys.path.remove(_jepa_dir)

    # Config: new checkpoints store a plain dict; old ones store a pickled dataclass object
    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, dict):
        config_mod = importlib.import_module(f"JEPA.experiments.{experiment}.config")
        cfg = config_mod.Config(**cfg_raw)
    else:
        cfg = cfg_raw

    models_mod, _ = _get_experiment_models(experiment)
    encoder, target_encoder, predictor, action_embed, policy = models_mod.load_models(cfg, device)

    encoder.load_state_dict(ckpt["encoder"])
    target_encoder.load_state_dict(ckpt["target_encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    action_embed.load_state_dict(ckpt["action_embed"])
    policy.load_state_dict(ckpt["policy"])

    encoder.eval()
    target_encoder.eval()
    predictor.eval()
    action_embed.eval()
    policy.eval()

    return cfg, encoder, target_encoder, predictor, action_embed, policy


# ── Statistics helpers (all numpy, avoids MPS SVD issues) ──────────────────

def _r(x: float, dp: int = 4) -> float:
    return round(float(x), dp)


def _activation_entropy(v: np.ndarray) -> float:
    """Normalized entropy of v²: -sum(p*log(p)) / log(D), where p = v²/sum(v²)."""
    v2 = v.astype(np.float64) ** 2
    s = v2.sum()
    if s < 1e-12:
        return 0.0
    p = v2 / s
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / math.log(len(v)))


def compute_patch_stats(
    z: np.ndarray,           # (128,) current embedding
    z_prev: np.ndarray | None,  # (128,) or None
    z_pred: np.ndarray,      # (128,) predictor output
    z_next: np.ndarray,      # (128,) actual next embedding
    patch_curr: np.ndarray,  # (16,16) uint8
    patch_next: np.ndarray,  # (16,16) uint8
) -> dict:
    stats = {
        "norm": _r(float(np.linalg.norm(z))),
        "mean": _r(float(z.mean())),
        "std": _r(float(z.std())),
        "min_val": _r(float(z.min())),
        "max_val": _r(float(z.max())),
        "activation_entropy": _r(_activation_entropy(z)),
        "pred_error": _r(float(np.linalg.norm(z_pred - z_next))),
        "pixel_change_frac": _r(float((patch_curr != patch_next).mean())),
        "cos_sim_prev": None,
        "l2_dist_prev": None,
        "mean_abs_diff_prev": None,
        "max_abs_diff_prev": None,
    }
    if z_prev is not None:
        diff = z - z_prev
        dot = float(np.dot(z, z_prev))
        norms = float(np.linalg.norm(z)) * float(np.linalg.norm(z_prev))
        stats["cos_sim_prev"] = _r(dot / (norms + 1e-12))
        stats["l2_dist_prev"] = _r(float(np.linalg.norm(diff)))
        stats["mean_abs_diff_prev"] = _r(float(np.abs(diff).mean()))
        stats["max_abs_diff_prev"] = _r(float(np.abs(diff).max()))
    return stats


def compute_embedding_summary(
    z: np.ndarray,           # (16,128) current step
    z_prev: np.ndarray | None,  # (16,128) or None
) -> dict:
    # Gram matrix — z is already unit-norm, so G[i,j] = cosine_sim(i,j)
    G = z @ z.T  # (16,16)
    idx = np.triu_indices(16, k=1)
    mean_pw_cos = float(G[idx].mean())

    # Effective rank via SVD of centered embeddings
    z_c = z - z.mean(axis=0)
    S = np.linalg.svd(z_c, compute_uv=False)  # (16,)
    S2 = S ** 2
    p = S2 / (S2.sum() + 1e-12)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    # Per-dim std across the 16 patches
    per_dim_std = z.std(axis=0)  # (128,)
    counts, edges = np.histogram(per_dim_std, bins=20)

    dead = int((per_dim_std < 0.01).sum())

    summary = {
        "mean_pairwise_cos_sim": _r(mean_pw_cos),
        "effective_rank": _r(eff_rank),
        "per_dim_std_mean": _r(float(per_dim_std.mean())),
        "per_dim_std_hist_counts": [_r(c) for c in counts.tolist()],
        "per_dim_std_hist_edges": [_r(e) for e in edges.tolist()],
        "dead_dim_count": dead,
        "mean_embedding_drift": None,
    }
    if z_prev is not None:
        drifts = np.linalg.norm(z - z_prev, axis=-1)  # (16,)
        summary["mean_embedding_drift"] = _r(float(drifts.mean()))
    return summary


def compute_reasoning_stats(h: np.ndarray, h_prev: np.ndarray | None) -> dict:
    stats = {
        "norm": _r(float(np.linalg.norm(h))),
        "mean": _r(float(h.mean())),
        "std": _r(float(h.std())),
        "activation_entropy": _r(_activation_entropy(h)),
        "cos_sim_prev": None,
        "l2_dist_prev": None,
        "mean_abs_diff_prev": None,
        "max_abs_diff_prev": None,
    }
    if h_prev is not None:
        diff = h - h_prev
        dot = float(np.dot(h, h_prev))
        norms = float(np.linalg.norm(h)) * float(np.linalg.norm(h_prev))
        stats["cos_sim_prev"] = _r(dot / (norms + 1e-12))
        stats["l2_dist_prev"] = _r(float(np.linalg.norm(diff)))
        stats["mean_abs_diff_prev"] = _r(float(np.abs(diff).mean()))
        stats["max_abs_diff_prev"] = _r(float(np.abs(diff).max()))
    return stats


def _fmt_first_last3(v: np.ndarray) -> list[float]:
    """First 3 and last 3 entries of a vector, rounded to 4 dp."""
    return [_r(float(x)) for x in list(v[:3]) + list(v[-3:])]


# ── Encoder attention weight extraction ────────────────────────────────────

def _encode_with_attn(encoder, frame_t: torch.Tensor):
    """Run encoder forward, capture per-block attention weights via hooks."""
    attn_capture: list[np.ndarray | None] = [None, None]
    hooks = []

    for i, block in enumerate(encoder.blocks):
        def make_hook(idx: int):
            def hook(module, inp, output):
                # nn.MultiheadAttention returns (attn_out, attn_weights)
                # attn_weights: (B, 16, 16) averaged over heads (default need_weights=True)
                if isinstance(output, tuple) and output[1] is not None:
                    attn_capture[idx] = output[1].squeeze(0).detach().cpu().numpy()
            return hook
        hooks.append(block.attn.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        z_t = encoder(frame_t).squeeze(0)  # (16,128)

    for h in hooks:
        h.remove()

    # Round and convert; fall back to uniform if hook didn't fire
    def _safe(w):
        if w is None:
            return [[_r(1/16)] * 16 for _ in range(16)]
        return [[_r(float(w[r, c])) for c in range(16)] for r in range(16)]

    return z_t, _safe(attn_capture[0]), _safe(attn_capture[1])


# ── Main episode runner ─────────────────────────────────────────────────────

def run_debug_episode(checkpoint_path: str, experiment: str,
                      env_name: str | None = None,
                      env: str | None = None,
                      max_steps: int = 200) -> dict:
    # Accept either `env` or `env_name` (server.py passes `env`).
    if env_name is None:
        env_name = env
    # Try experiment-specific runner first; force-reload to pick up any code changes
    # without requiring a server restart.
    mod_name = f"JEPA.experiments.{experiment}.debug_runner"
    try:
        import inspect as _inspect
        import sys as _sys
        if mod_name in _sys.modules:
            exp_mod = importlib.reload(_sys.modules[mod_name])
        else:
            exp_mod = importlib.import_module(mod_name)
        if hasattr(exp_mod, "run_debug_episode"):
            fn = exp_mod.run_debug_episode
            params = _inspect.signature(fn).parameters
            # Forward env_name only if the per-experiment runner accepts it.
            # Older runners (exp_003_X before this revision) keep their original
            # signature `(checkpoint_path, max_steps)` and silently use their
            # config's env.
            if "env_name" in params:
                return fn(checkpoint_path, env_name=env_name, max_steps=max_steps)
            return fn(checkpoint_path, max_steps)
    except ModuleNotFoundError:
        pass
    # Fall through to existing exp_001-compatible runner below

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cfg, encoder, target_encoder, predictor, action_embed, policy = \
        load_checkpoint(checkpoint_path, experiment, device)

    # Capability flags — what this experiment's architecture can visualize
    models_mod, compute_patch_weights = _get_experiment_models(experiment)
    capabilities = getattr(models_mod, "CAPABILITIES", {
        "has_encoder_attention": False,
        "has_policy_attention": False,
        "has_patch_embeddings": True,
        "n_patches": 16,
        "extra": {},
    })

    # Set up environment
    from arc_agi import Arcade, OperationMode
    from JEPA.shared.env_wrapper import make_env, resolve_dashboard_env
    repo_root = _REPO_ROOT
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root / "environment_files"),
    )
    full_gid, env_warning = resolve_dashboard_env(env_name, cfg.game_id)
    raw_env = arc.make(full_gid)
    env = make_env(raw_env, full_gid)

    frame_np = env.reset()
    h = policy.initial_state().to(device)
    h_prev_np: np.ndarray | None = None
    z_prev_np: np.ndarray | None = None
    prev_frame_np: np.ndarray | None = None

    timesteps = []

    for t in range(max_steps):
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

        # Encode with attention weight capture
        z_t, attn_b1, attn_b2 = _encode_with_attn(encoder, frame_t)
        z_t_np = z_t.detach().cpu().numpy()

        # Policy cross-attention weights (recomputed from projections)
        h_before = h.detach().clone()
        h_before_np = h_before.cpu().numpy()
        with torch.no_grad():
            Q = policy.q_proj(h_before)         # (128,)
            K = policy.k_proj(z_t)              # (16,128)
            pol_attn = F.softmax(
                Q.unsqueeze(0) @ K.T * policy.scale, dim=-1
            ).squeeze(0).cpu().numpy()          # (16,)

        # Action selection
        action_idx, log_prob, h_new, entropy = policy.act(
            h_before, z_t, env.available_actions
        )

        # Clean action probs (run forward once more with masking but no sampling)
        with torch.no_grad():
            h_upd = policy._cross_attn_update(h_before, z_t)
            logits = policy.action_head(h_upd)
            avail = env.available_actions
            if avail:
                mask = torch.full_like(logits, float("-inf"))
                for a in avail:
                    idx = int(a) - 1
                    if 0 <= idx < cfg.n_actions:
                        mask[idx] = 0.0
                logits = logits + mask
            probs_np = F.softmax(logits, dim=-1).cpu().numpy()

        # Environment step
        next_np, is_terminal = env.step(action_idx)
        next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)

        with torch.no_grad():
            z_next = target_encoder(next_t).squeeze(0)
            a_emb = action_embed(torch.tensor([action_idx], device=device))
            z_pred = predictor(z_t.unsqueeze(0), a_emb).squeeze(0)

            # Patch weights + JEPA loss — floor at 0.1 matches training loop
            pixel_w = compute_patch_weights(frame_t, next_t)   # (1,16) in [0,1]
            w = 0.1 + 0.9 * pixel_w                            # (1,16) in [0.1,1.0]
            w_np = w.squeeze(0).cpu().numpy()
            err_sq = (z_pred - z_next).pow(2).sum(dim=-1)   # (16,)
            jepa_loss = float((w.squeeze(0) * err_sq).mean().item())
            reward = jepa_loss + (50.0 if (is_terminal and env.level_completed) else 0.0)

        z_next_np = z_next.detach().cpu().numpy()
        z_pred_np = z_pred.detach().cpu().numpy()
        per_patch_pred_error = [
            _r(float(np.linalg.norm(z_pred_np[i] - z_next_np[i])))
            for i in range(16)
        ]

        # Per-patch stats: compare S_{t-1} → S_t so pixel_change_frac is backward-looking,
        # matching the t-1/t thumbnails and embedding drift stats in the dashboard.
        p = cfg.patch_size
        _ref = prev_frame_np if prev_frame_np is not None else frame_np
        per_patch_stats = []
        for i in range(16):
            r_i, c_i = divmod(i, 4)
            patch_curr = _ref[r_i*p:(r_i+1)*p, c_i*p:(c_i+1)*p]     # S_{t-1} (or S_0 at t=0)
            patch_next = frame_np[r_i*p:(r_i+1)*p, c_i*p:(c_i+1)*p]  # S_t
            z_prev_i = z_prev_np[i] if z_prev_np is not None else None
            per_patch_stats.append(compute_patch_stats(
                z_t_np[i], z_prev_i, z_pred_np[i], z_next_np[i],
                patch_curr, patch_next,
            ))

        # Embedding summary
        emb_summary = compute_embedding_summary(z_t_np, z_prev_np)

        # Reasoning token stats
        reasoning_stats = compute_reasoning_stats(h_before_np, h_prev_np)

        # h first3/last3 and reasoning token first/last 3
        h_first_last3 = _fmt_first_last3(h_before_np)

        timesteps.append({
            "t": t,
            "frame": frame_np.tolist(),
            "action_taken": action_idx,
            "is_terminal": bool(is_terminal),
            "available_actions": list(env.available_actions),
            "reward": _r(reward),
            "jepa_loss": _r(jepa_loss),
            "patch_embeddings": [[_r(v) for v in row] for row in z_t_np.tolist()],
            "next_patch_embeddings": [[_r(v) for v in row] for row in z_next_np.tolist()],
            "predicted_next_embeddings": [[_r(v) for v in row] for row in z_pred_np.tolist()],
            "reasoning_token": [_r(v) for v in h_before_np.tolist()],
            "reasoning_token_first_last3": h_first_last3,
            "action_probs": [_r(float(p)) for p in probs_np.tolist()],
            "action_entropy": _r(float(entropy.item())),
            "patch_weights": [_r(float(w)) for w in w_np.tolist()],
            "per_patch_stats": per_patch_stats,
            "per_patch_pred_error": per_patch_pred_error,
            "embedding_summary": emb_summary,
            "reasoning_stats": reasoning_stats,
            "encoder_attn_block1": attn_b1,
            "encoder_attn_block2": attn_b2,
            "policy_attn_weights": [_r(float(x)) for x in pol_attn.tolist()],
        })

        # Advance state
        h_prev_np = h_before_np.copy()
        z_prev_np = z_t_np.copy()
        h = h_new.detach()
        prev_frame_np = frame_np
        frame_np = next_np

        if is_terminal:
            break

    ckpt_name = Path(checkpoint_path).name
    ckpt_step = int(ckpt_name.replace("step_", "").replace(".pt", "").split("_")[0]) \
        if ckpt_name.startswith("step_") else 0

    from JEPA.shared.env_wrapper import short_env_name
    out = {
        "checkpoint": ckpt_name,
        "checkpoint_step": ckpt_step,
        "experiment": experiment,
        "env_name": short_env_name(full_gid),
        "capabilities": capabilities,
        "episode_steps": len(timesteps),
        "level_completed": bool(env.level_completed),
        "truncated": len(timesteps) >= max_steps and not timesteps[-1]["is_terminal"],
        "arc_colors": ARC_COLORS_RGB,
        "timesteps": timesteps,
    }
    if env_warning:
        out["warning"] = env_warning
    return out


if __name__ == "__main__":
    _ckpt = (sys.argv[1] if len(sys.argv) > 1
             else "JEPA/experiments/exp_001_vit_jepa_baseline/checkpoints/step_235000.pt")
    _exp  = sys.argv[2] if len(sys.argv) > 2 else "exp_001_vit_jepa_baseline"
    data = run_debug_episode(_ckpt, experiment=_exp, max_steps=10)
    print(f"Episode: {data['episode_steps']} steps, "
          f"completed={data['level_completed']}, truncated={data['truncated']}")
    t0 = data["timesteps"][0]
    print(f"Step 0: action={t0['action_taken']}, reward={t0['reward']}, "
          f"jepa_loss={t0['jepa_loss']}")
    print(f"Embedding summary: {t0['embedding_summary']}")
    print(f"Policy attn (first 4): {t0['policy_attn_weights'][:4]}")
    print("JSON size estimate:", len(json.dumps(data)) // 1024, "KB")
