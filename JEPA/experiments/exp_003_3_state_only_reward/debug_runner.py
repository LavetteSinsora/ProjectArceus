"""
Debug episode runner for exp_003_3_state_only_reward.

Per-timestep data collected (same as exp_003_0 plus action predictor + dual-path reward):
  patch_embeddings, per_patch_stats, embedding_summary
  encoder_sa_block1/2, latent_vectors, latent_norms
  perceiver_cross_attn_r0/r1, perceiver_self_attn_r0/r1
  ode_trajectory, h_target_latents (= h_{t+1} from online encoder)
  per_latent_pred_error, flow_loss
  action_probs (policy), action_entropy
  NEW:
    action_predictor_probs     — softmax(action_predictor(h_t, h_{t+1}))
    action_predictor_entropy
    action_predictor_predicted — argmax
    action_predictor_ce        — CE vs ground-truth action
    reward_state_component, reward_action_component, reward_total

Usage:
    cd "Code Repo"
    uv run python JEPA/experiments/exp_003_3_state_only_reward/debug_runner.py \\
        JEPA/experiments/exp_003_3_state_only_reward/checkpoints/step_005000.pt
"""

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_3_state_only_reward.config import Config
from JEPA.experiments.exp_003_3_state_only_reward.models import load_models
from JEPA.experiments.exp_003_3_state_only_reward.reward_shaping import is_end_of_life
from JEPA.shared.env_wrapper import (
    LS20Env, make_env, resolve_dashboard_env, short_env_name,
)

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


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _r(x: float, dp: int = 4) -> float:
    return round(float(x), dp)


def _activation_entropy(v: np.ndarray) -> float:
    v2 = v.astype(np.float64) ** 2
    s = v2.sum()
    if s < 1e-12:
        return 0.0
    p = v2 / s
    return float(-np.sum(p * np.log(p + 1e-12)) / math.log(len(v)))


def _patch_stats(z: np.ndarray, z_prev: np.ndarray | None,
                 patch_curr: np.ndarray, patch_prev: np.ndarray | None) -> dict:
    stats = {
        "norm": _r(float(np.linalg.norm(z))),
        "mean": _r(float(z.mean())),
        "std":  _r(float(z.std())),
        "min_val": _r(float(z.min())),
        "max_val": _r(float(z.max())),
        "activation_entropy": _r(_activation_entropy(z)),
        "pixel_change_frac": None,
        "cos_sim_prev": None,
        "l2_dist_prev": None,
        "mean_abs_diff_prev": None,
        "max_abs_diff_prev": None,
    }
    if patch_prev is not None:
        stats["pixel_change_frac"] = _r(float((patch_curr != patch_prev).mean()))
    if z_prev is not None:
        diff = z - z_prev
        n1, n2 = float(np.linalg.norm(z)), float(np.linalg.norm(z_prev))
        stats["cos_sim_prev"]       = _r(float(np.dot(z, z_prev)) / (n1 * n2 + 1e-12))
        stats["l2_dist_prev"]       = _r(float(np.linalg.norm(diff)))
        stats["mean_abs_diff_prev"] = _r(float(np.abs(diff).mean()))
        stats["max_abs_diff_prev"]  = _r(float(np.abs(diff).max()))
    return stats


def _embedding_summary(z: np.ndarray, z_prev: np.ndarray | None) -> dict:
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    G = z_norm @ z_norm.T  # cosine similarity matrix (not dot products)
    idx = np.triu_indices(len(z), k=1)
    mean_pw = float(G[idx].mean())
    z_c = z - z.mean(axis=0)
    sv = np.linalg.svd(z_c, compute_uv=False)
    sv2 = sv ** 2
    p = sv2 / (sv2.sum() + 1e-12)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    per_dim_std = z.std(axis=0)
    counts, edges = np.histogram(per_dim_std, bins=20)
    dead = int((per_dim_std < 0.01).sum())
    summary = {
        "mean_pairwise_cos_sim":     _r(mean_pw),
        "effective_rank":            _r(eff_rank),
        "per_dim_std_mean":          _r(float(per_dim_std.mean())),
        "per_dim_std_hist_counts":   [int(c) for c in counts.tolist()],
        "per_dim_std_hist_edges":    [_r(e) for e in edges.tolist()],
        "dead_dim_count":            dead,
        "mean_embedding_drift":      None,
    }
    if z_prev is not None:
        summary["mean_embedding_drift"] = _r(float(np.linalg.norm(z - z_prev, axis=-1).mean()))
    return summary


# ── Attention extraction helpers ──────────────────────────────────────────────

def _safe_sa_attn(block, n_tokens: int = 16) -> list:
    """Read cached _debug_attn from a SelfAttentionBlock → (n_tokens, n_tokens) list."""
    if not hasattr(block, "_debug_attn") or block._debug_attn is None:
        u = 1.0 / n_tokens
        return [[_r(u)] * n_tokens for _ in range(n_tokens)]
    w = block._debug_attn.mean(dim=1).squeeze(0)  # avg over heads → (n_tokens, n_tokens)
    return [[_r(float(w[i, j])) for j in range(n_tokens)] for i in range(n_tokens)]


def _safe_self_attn(module, n: int = 4) -> list:
    """Read cached _debug_attn from _SelfAttentionAmongLatents → (n, n) list."""
    if not hasattr(module, "_debug_attn") or module._debug_attn is None:
        u = 1.0 / n
        return [[_r(u)] * n for _ in range(n)]
    w = module._debug_attn.mean(dim=1).squeeze(0)  # avg heads → (n, n)
    return [[_r(float(w[i, j])) for j in range(n)] for i in range(n)]


def _safe_cross_attn(captured, n_latents: int = 4, n_patches: int = 16) -> list:
    """Convert hook-captured (1, n_heads, 4, 16) → [[float]*16]*4."""
    if captured[0] is None:
        u = 1.0 / n_patches
        return [[_r(u)] * n_patches for _ in range(n_latents)]
    w = captured[0].mean(dim=1).squeeze(0)  # (4, 16)
    return [[_r(float(w[i, j])) for j in range(n_patches)] for i in range(n_latents)]


# ── Main runner ───────────────────────────────────────────────────────────────

def run_debug_episode(checkpoint_path: str,
                      env_name: str | None = None,
                      max_steps: int = 200) -> dict:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, dict):
        # Drop fields unknown to this Config (e.g. exp_003_1's ema_decay_*)
        valid = {f.name for f in dataclasses.fields(Config)}
        cfg = Config(**{k: v for k, v in cfg_raw.items() if k in valid})
    else:
        cfg = cfg_raw

    encoder, state_predictor, action_predictor, action_embed, policy, _baseline = \
        load_models(cfg, device)
    encoder.load_state_dict(ckpt["encoder"])
    state_predictor.load_state_dict(ckpt["state_predictor"])
    action_predictor.load_state_dict(ckpt["action_predictor"])
    action_embed.load_state_dict(ckpt["action_embed"])
    policy.load_state_dict(ckpt["policy"])
    encoder.eval(); state_predictor.eval(); action_predictor.eval()
    action_embed.eval(); policy.eval()
    # Alias for the lines below that refer to a generic "predictor"
    predictor = state_predictor

    ckpt_name = Path(checkpoint_path).name
    ckpt_step = ckpt.get("step", 0)

    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    full_gid, env_warning = resolve_dashboard_env(env_name, cfg.game_id)
    raw_env = arc.make(full_gid)
    env = make_env(raw_env, full_gid)
    frame_np = env.reset()

    # Perceiver cross-attention hooks on rounds[0/1]
    attn_r0_cap = [None]
    attn_r1_cap = [None]
    def _make_ca_hook(store):
        def _hook(_mod, _inp, out):
            if isinstance(out, tuple) and len(out) == 2:
                store[0] = out[1].detach().cpu()  # (1, n_heads, 4, 16)
        return _hook

    with torch.no_grad():
        h_t = encoder.perceiver.get_initial_queries(1, device)  # (1, 4, 128)

    z_prev_np: np.ndarray | None = None  # (16, 128)
    prev_frame_np: np.ndarray | None = None
    timesteps = []
    patch_size = cfg.patch_size  # 16

    for t in range(max_steps):
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

        # Capture h_t input (h_{t-1} or placeholder at t=0)
        h_t_input_np = h_t.squeeze(0).detach().cpu().numpy()  # (4, 128)

        # Hooks to capture intermediate states inside the perceiver
        inter_r0_cap     = [None]  # output of Round 0 (queries after cross+self attn)
        ca_out_r0_cap    = [None]  # queries after Round 0 cross-attn (before self-attn)
        ca_out_r1_cap    = [None]  # queries after Round 1 cross-attn (before self-attn)

        def _make_round_hook(store):
            def _hook(_mod, _inp, out):
                store[0] = out[0].detach().cpu()  # (1, 4, 128) — queries output
            return _hook

        def _make_ca_out_hook(store):
            def _hook(_mod, _inp, out):
                store[0] = out[0].detach().cpu()  # (1, 4, 128) — updated queries from cross-attn
            return _hook

        hook0        = encoder.perceiver.rounds[0].cross_attn.register_forward_hook(_make_ca_hook(attn_r0_cap))
        hook1        = encoder.perceiver.rounds[1].cross_attn.register_forward_hook(_make_ca_hook(attn_r1_cap))
        hook_r0_out  = encoder.perceiver.rounds[0].register_forward_hook(_make_round_hook(inter_r0_cap))
        hook_ca_out0 = encoder.perceiver.rounds[0].cross_attn.register_forward_hook(_make_ca_out_hook(ca_out_r0_cap))
        hook_ca_out1 = encoder.perceiver.rounds[1].cross_attn.register_forward_hook(_make_ca_out_hook(ca_out_r1_cap))

        with torch.no_grad():
            latents, sa_out, _ = encoder(frame_t, h_t)

        hook0.remove(); hook1.remove(); hook_r0_out.remove()
        hook_ca_out0.remove(); hook_ca_out1.remove()

        inter_r0_np     = inter_r0_cap[0].squeeze(0).numpy()  if inter_r0_cap[0]  is not None else h_t_input_np
        after_cross_r0  = ca_out_r0_cap[0].squeeze(0).numpy() if ca_out_r0_cap[0] is not None else inter_r0_np
        after_cross_r1  = ca_out_r1_cap[0].squeeze(0).numpy() if ca_out_r1_cap[0] is not None else inter_r0_np

        # ── Patch embeddings (from SA output) ───────────────────────────────
        z_t_np = sa_out.squeeze(0).detach().cpu().numpy()  # (16, 128)

        # ── SA block attention weights ───────────────────────────────────────
        sa_attn_b1 = _safe_sa_attn(encoder.sa_blocks[0])   # (16, 16) list
        sa_attn_b2 = _safe_sa_attn(encoder.sa_blocks[1])   # (16, 16) list

        # ── Perceiver cross-attention ────────────────────────────────────────
        ca_r0 = _safe_cross_attn(attn_r0_cap)  # (4, 16)
        ca_r1 = _safe_cross_attn(attn_r1_cap)  # (4, 16)

        # ── Perceiver self-attention ─────────────────────────────────────────
        sa_perc_r0 = _safe_self_attn(encoder.perceiver.rounds[0].self_attn)  # (4, 4)
        sa_perc_r1 = _safe_self_attn(encoder.perceiver.rounds[1].self_attn)  # (4, 4)

        # ── Latent vectors and norms ─────────────────────────────────────────
        latents_np = latents.squeeze(0).detach().cpu().numpy()  # (4, 128)
        latent_vectors = [[_r(float(v)) for v in latents_np[i]] for i in range(4)]
        latent_norms   = [_r(float(np.linalg.norm(latents_np[i]))) for i in range(4)]

        # ── Policy ───────────────────────────────────────────────────────────
        with torch.no_grad():
            logits = policy(latents).squeeze(0)
            avail = env.available_actions
            if avail:
                mask = torch.full_like(logits, float("-inf"))
                for a in avail:
                    idx2 = int(a) - 1
                    if 0 <= idx2 < cfg.n_actions:
                        mask[idx2] = 0.0
                masked_logits = logits + mask
            else:
                masked_logits = logits
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            action_idx = action_tensor.item()
            entropy = dist.entropy()

        action_probs   = [_r(float(p)) for p in probs.cpu().tolist()]
        action_entropy = _r(float(entropy.item()))

        # ── Environment step ─────────────────────────────────────────────────
        next_np, is_terminal = env.step(action_idx)

        # ── Predictor: ODE trajectory + target ───────────────────────────────
        with torch.no_grad():
            a_emb = action_embed(torch.tensor([action_idx], device=device))
            next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)
            h_next, _, _ = encoder(next_t, latents)
            h_target_np = h_next.squeeze(0).detach().cpu().numpy()  # (4, 128)

            # ODE trajectory: x_0 → x_{1/N} → ... → x_1 (N+1 tensors)
            h_pred, traj = predictor.predict_with_trajectory(latents, a_emb)
            # per-latent MSE
            per_latent_mse = (h_next - h_pred).pow(2).mean(dim=-1).squeeze(0)
            per_latent_pred_error = [_r(float(per_latent_mse[i].item())) for i in range(4)]
            flow_loss = _r(float(per_latent_mse.mean().item()))

            # ── Action predictor: distribution over a_t given (h_t, h_{t+1}) ──
            ap_logits = action_predictor(latents, h_next).squeeze(0)
            ap_probs = F.softmax(ap_logits, dim=-1)
            ap_entropy = -(ap_probs * (ap_probs + 1e-12).log()).sum().item()
            ap_predicted = int(ap_probs.argmax().item())
            ap_ce = F.cross_entropy(
                ap_logits.unsqueeze(0),
                torch.tensor([action_idx], device=device),
            ).item()

            # ── Dual-path curiosity reward components ──────────────────────────
            state_err = float(per_latent_mse.mean().item())
            action_err = float(ap_ce)
            reward_total = min(
                cfg.reward_w_state * state_err
                + cfg.reward_w_action * action_err,
                cfg.reward_clamp,
            )

        action_predictor_probs = [_r(float(p)) for p in ap_probs.cpu().tolist()]
        action_predictor_entropy = _r(float(ap_entropy))
        action_predictor_ce = _r(float(ap_ce))
        reward_state_component = _r(state_err)
        reward_action_component = _r(action_err)
        reward_total_r = _r(float(reward_total))

        # traj: list of (1, 4, 128) cpu tensors → list of (4, 128) lists
        ode_trajectory = [
            [[_r(float(v)) for v in step[0][i].tolist()] for i in range(4)]
            for step in traj
        ]
        h_target_list = [[_r(float(v)) for v in h_target_np[i]] for i in range(4)]

        # ── Per-patch stats ──────────────────────────────────────────────────
        p = patch_size
        per_patch_stats = []
        for i in range(16):
            ri, ci = divmod(i, 4)
            patch_curr = frame_np[ri*p:(ri+1)*p, ci*p:(ci+1)*p]
            patch_prev = prev_frame_np[ri*p:(ri+1)*p, ci*p:(ci+1)*p] if prev_frame_np is not None else None
            per_patch_stats.append(_patch_stats(
                z_t_np[i],
                z_prev_np[i] if z_prev_np is not None else None,
                patch_curr,
                patch_prev,
            ))

        emb_summary = _embedding_summary(z_t_np, z_prev_np)

        timesteps.append({
            "t": t,
            "frame": frame_np.tolist(),
            "action_taken": action_idx,
            "is_terminal": bool(is_terminal),
            "available_actions": list(env.available_actions),
            "reward": flow_loss,
            # ── Encoder SA ──────────────────────────────────────────────────
            "patch_embeddings":  [[_r(float(v)) for v in row] for row in z_t_np.tolist()],
            "per_patch_stats":   per_patch_stats,
            "embedding_summary": emb_summary,
            "encoder_sa_block1": sa_attn_b1,
            "encoder_sa_block2": sa_attn_b2,
            # ── Perceiver ───────────────────────────────────────────────────
            "latent_vectors":            latent_vectors,
            "latent_norms":              latent_norms,
            "perceiver_input_queries":   [[_r(float(v)) for v in h_t_input_np[i]]  for i in range(4)],
            "perceiver_after_cross_r0":  [[_r(float(v)) for v in after_cross_r0[i]] for i in range(4)],
            "perceiver_inter_r0":        [[_r(float(v)) for v in inter_r0_np[i]]   for i in range(4)],
            "perceiver_after_cross_r1":  [[_r(float(v)) for v in after_cross_r1[i]] for i in range(4)],
            "perceiver_cross_attn_r0":   ca_r0,
            "perceiver_cross_attn_r1":   ca_r1,
            "perceiver_self_attn_r0":    sa_perc_r0,
            "perceiver_self_attn_r1":    sa_perc_r1,
            # ── State predictor ─────────────────────────────────────────────
            "ode_trajectory":         ode_trajectory,
            "h_target_latents":       h_target_list,
            "per_latent_pred_error":  per_latent_pred_error,
            "flow_loss":              flow_loss,
            # ── Action predictor (NEW for exp_003_3) ────────────────────────
            "action_predictor_probs":     action_predictor_probs,
            "action_predictor_entropy":   action_predictor_entropy,
            "action_predictor_predicted": ap_predicted,
            "action_predictor_ce":        action_predictor_ce,
            # ── Dual-path curiosity reward components ───────────────────────
            "reward_state_component":  reward_state_component,
            "reward_action_component": reward_action_component,
            "reward_total":            reward_total_r,
            # ── Policy ──────────────────────────────────────────────────────
            "action_probs":    action_probs,
            "action_entropy":  action_entropy,
        })

        # Cross-env play: LS20 life-end heuristic does not apply on other games.
        if env_warning is not None:
            eol = bool(is_terminal)
        else:
            eol = is_end_of_life(frame_np, next_np, is_terminal)
        h_t = latents.detach()
        z_prev_np = z_t_np.copy()
        prev_frame_np = frame_np
        frame_np = next_np

        if eol:
            break

    from JEPA.experiments.exp_003_3_state_only_reward.models import CAPABILITIES
    out = {
        "checkpoint":       ckpt_name,
        "checkpoint_step":  ckpt_step,
        "experiment":       "exp_003_3_state_only_reward",
        "env_name":         short_env_name(full_gid),
        "capabilities":     CAPABILITIES,
        "episode_steps":    len(timesteps),
        "level_completed":  bool(env.level_completed),
        "truncated":        len(timesteps) >= max_steps and not timesteps[-1]["is_terminal"],
        "arc_colors":       ARC_COLORS_RGB,
        "timesteps":        timesteps,
    }
    if env_warning:
        out["warning"] = env_warning
    return out


if __name__ == "__main__":
    _ckpt = (sys.argv[1] if len(sys.argv) > 1
             else "JEPA/experiments/exp_003_3_state_only_reward/checkpoints/step_005000.pt")
    data = run_debug_episode(_ckpt, max_steps=10)
    print(f"Episode: {data['episode_steps']} steps, "
          f"completed={data['level_completed']}, truncated={data['truncated']}")
    t0 = data["timesteps"][0]
    print(f"Step 0: action={t0['action_taken']}, flow_loss={t0['flow_loss']}")
    print(f"Latent norms: {t0['latent_norms']}")
    print(f"SA block 1 attn shape: {len(t0['encoder_sa_block1'])}×{len(t0['encoder_sa_block1'][0])}")
    print(f"ODE trajectory steps: {len(t0['ode_trajectory'])}")
    print(f"Perceiver SA r0 shape: {len(t0['perceiver_self_attn_r0'])}×{len(t0['perceiver_self_attn_r0'][0])}")
    print("JSON size estimate:", len(json.dumps(data)) // 1024, "KB")
