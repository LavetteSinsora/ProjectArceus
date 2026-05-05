"""
Debug episode runner for the JEPA dashboard.

Loads a checkpoint, runs one episode, and returns a fully serialized dict
containing per-timestep data for all dashboard visualizations.

Usage (standalone test):
    cd "Code Repo"
    uv run python JEPA/dashboard/debug_runner.py JEPA/checkpoints/step_235000.pt
"""

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Put JEPA/ on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from encoder import Encoder
from predictor import Predictor
from action_embed import ActionEmbedding
from policy import PolicyNetwork
from env_wrapper import LS20Env
from train import compute_patch_weights


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

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg: Config = ckpt["config"]

    encoder = Encoder(
        cfg.d_model, cfg.d_color, cfg.n_heads, cfg.n_blocks, cfg.ffn_dim, cfg.patch_size
    ).to(device)
    target_encoder = copy.deepcopy(encoder)
    predictor = Predictor(cfg.d_model, cfg.d_action).to(device)
    action_embed = ActionEmbedding(cfg.n_actions, cfg.d_action).to(device)
    policy = PolicyNetwork(cfg.d_model, cfg.n_actions).to(device)

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

def run_debug_episode(checkpoint_path: str, max_steps: int = 200) -> dict:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cfg, encoder, target_encoder, predictor, action_embed, policy = \
        load_checkpoint(checkpoint_path, device)

    # Set up environment
    from arc_agi import Arcade, OperationMode
    repo_root = Path(__file__).parent.parent.parent
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)

    frame_np = env.reset()
    h = policy.initial_state().to(device)
    h_prev_np: np.ndarray | None = None
    z_prev_np: np.ndarray | None = None

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

            # Patch weights + JEPA loss
            w = compute_patch_weights(frame_t, next_t)  # (1,16)
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

        # Per-patch stats
        p = cfg.patch_size
        per_patch_stats = []
        for i in range(16):
            r_i, c_i = divmod(i, 4)
            patch_curr = frame_np[r_i*p:(r_i+1)*p, c_i*p:(c_i+1)*p]
            patch_next = next_np[r_i*p:(r_i+1)*p, c_i*p:(c_i+1)*p]
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
        frame_np = next_np

        if is_terminal:
            break

    ckpt_name = Path(checkpoint_path).name
    ckpt_step = int(ckpt_name.replace("step_", "").replace(".pt", "").split("_")[0]) \
        if ckpt_name.startswith("step_") else 0

    return {
        "checkpoint": ckpt_name,
        "checkpoint_step": ckpt_step,
        "episode_steps": len(timesteps),
        "level_completed": bool(env.level_completed),
        "truncated": len(timesteps) >= max_steps and not timesteps[-1]["is_terminal"],
        "arc_colors": ARC_COLORS_RGB,
        "timesteps": timesteps,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "JEPA/checkpoints/step_235000.pt"
    data = run_debug_episode(path, max_steps=10)
    print(f"Episode: {data['episode_steps']} steps, "
          f"completed={data['level_completed']}, truncated={data['truncated']}")
    t0 = data["timesteps"][0]
    print(f"Step 0: action={t0['action_taken']}, reward={t0['reward']}, "
          f"jepa_loss={t0['jepa_loss']}")
    print(f"Embedding summary: {t0['embedding_summary']}")
    print(f"Policy attn (first 4): {t0['policy_attn_weights'][:4]}")
    print("JSON size estimate:", len(json.dumps(data)) // 1024, "KB")
