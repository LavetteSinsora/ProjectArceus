"""
exp_004_0 training loop — LS20 + TU93 shared-encoder JEPA.

Architecture: encoder + Perceiver Resampler (cross-attn only) + state predictor +
action predictor — all SHARED between the two envs. Action embeddings and policies
are PER-ENV. JEPA loss is exp_003_2's `0.5·L_state + 0.5·L_action`; reward is
exp_003_3's state-only curiosity (action term retained in JEPA loss only).

Data collection: round-robin episodes (one full episode per env per cycle).
JEPA updates: balanced batches with `batch_size//2` transitions from each per-env buffer.
Policy updates: per-env REINFORCE with EMA baseline.

Bug fix (system_card.md §3.3): when the rollout's last step triggered life_end,
the dying step is excluded from replay buffer (existing `[:-1]` slice), AND
from the policy buffer + rollout health metrics (new in this experiment).

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.train
    uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.train --resume <ckpt>
    uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.train --max-steps 5000
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import io
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_004_0_ls20_tu93.config import Config
from JEPA.experiments.exp_004_0_ls20_tu93.models import load_models
from JEPA.experiments.exp_004_0_ls20_tu93.reward_shaping import is_end_of_life

# Reuse exp_003_4's monitor infrastructure verbatim — the encoder/predictor architecture
# is identical, so all sec1-sec7 metrics about the shared modules transfer directly.
from JEPA.experiments.exp_003_4_no_resampler_self_attn.monitors.health import HealthMonitor
from JEPA.experiments.exp_003_4_no_resampler_self_attn.monitors.writer import MetricsWriter
from JEPA.experiments.exp_003_4_no_resampler_self_attn.monitors import (
    gradients as grad_mod,
    predictors as pred_mod,
    representation as repr_mod,
)

from JEPA.shared.buffer import NextFrameLatentBuffer, PolicyBuffer
from JEPA.shared.env_wrapper import make_env


# ── Constants ────────────────────────────────────────────────────────────────

CHECKPOINT_FREQ = 5_000
LOG_FREQ        = 200


# ── Tee logger ───────────────────────────────────────────────────────────────

class _Tee(io.TextIOBase):
    def __init__(self, original: io.TextIOBase, log_path: Path):
        self._orig = original
        self._file = open(log_path, "w", buffering=1, encoding="utf-8")

    def write(self, data: str) -> int:
        self._orig.write(data); self._orig.flush()
        self._file.write(data)
        return len(data)

    def flush(self):
        self._orig.flush(); self._file.flush()

    def close(self):
        super().close()
        self._file.close()


def setup_run_logger(resume_path: str | None) -> tuple[Path, _Tee]:
    runs_root = Path(__file__).parent / "runs"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "resume" if resume_path else "fresh"
    run_dir = runs_root / f"run_{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    tee = _Tee(sys.stdout, log_path)
    sys.stdout = tee
    print(f"[exp004_0] Run dir:  {run_dir}")
    print(f"[exp004_0] Log file: {log_path}")
    return run_dir, tee


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_checkpoint(step, encoder, state_predictor, action_predictor,
                    action_embeds, policies, baselines, cfg,
                    label="", ckpt_dir: Path | None = None):
    if ckpt_dir is None:
        ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"step_{step:06d}" + (f"_{label}" if label else "")
    path = ckpt_dir / f"{tag}.pt"
    tmp = path.with_suffix(".tmp")
    payload = {
        "encoder":          encoder.state_dict(),
        "state_predictor":  state_predictor.state_dict(),
        "action_predictor": action_predictor.state_dict(),
        "action_embeds":    {k: v.state_dict() for k, v in action_embeds.items()},
        "policies":         {k: v.state_dict() for k, v in policies.items()},
        "baselines":        {k: v.value for k, v in baselines.items()},
        "step":             step,
        "config":           dataclasses.asdict(cfg),
    }
    torch.save(payload, tmp)
    tmp.rename(path)
    return path


def load_checkpoint(path, encoder, state_predictor, action_predictor,
                    action_embeds, policies, baselines, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    state_predictor.load_state_dict(ckpt["state_predictor"])
    action_predictor.load_state_dict(ckpt["action_predictor"])
    for k, sd in ckpt["action_embeds"].items():
        if k in action_embeds:
            action_embeds[k].load_state_dict(sd)
    for k, sd in ckpt["policies"].items():
        if k in policies:
            policies[k].load_state_dict(sd)
    for k, v in ckpt.get("baselines", {}).items():
        if k in baselines:
            baselines[k].value = float(v)
    return ckpt.get("step", 0)


# ── Print stats ──────────────────────────────────────────────────────────────

def print_stats(step, health: HealthMonitor, env_names,
                buf_sizes: dict, ep_counts: dict, fps: float,
                lambda_state: float, lambda_action: float):
    Ls = health.mean_L_state(); La = health.mean_L_action(); Lt = health.mean_L_total()
    ent_g = health.mean_entropy()
    print(
        f"\n{'─'*78}\n"
        f"  Step {step:7d}  fps={fps:.0f}\n"
        f"  Buffers: " + "  ".join(f"{e}={buf_sizes[e]}" for e in env_names) + "\n"
        f"  Episodes: " + "  ".join(f"{e}={ep_counts[e]}" for e in env_names) + "\n"
        f"  Loss: L_total={Lt:.5f}  = {lambda_state}·L_state({Ls:.5f}) + "
        f"{lambda_action}·L_action({La:.5f})\n"
        f"  Per-env L_state:  " + "  ".join(
            f"{e}={health._mean(health.sec6.get(f'L_state_{e}', deque())):.5f}"
            for e in env_names) + "\n"
        f"  Per-env L_action: " + "  ".join(
            f"{e}={health._mean(health.sec6.get(f'L_action_{e}', deque())):.5f}"
            for e in env_names) + "\n"
        f"  Per-env reward:   " + "  ".join(
            f"{e}={health._mean(health.sec6.get(f'reward_total_{e}', deque())):.4f}"
            for e in env_names) + "\n"
        f"  Per-env policy_H: " + "  ".join(
            f"{e}={health._mean(health.sec5.get(f'policy_entropy_{e}', deque())):.3f}"
            for e in env_names) + "\n"
        f"  ht_htp1_cos={health._mean(health.sec1['ht_htp1_cossim_rollout']):.4f}  "
        f"placeholder_pcos={health._mean(health.sec1['placeholder_pairwise_cossim']):.4f}\n"
        f"{'─'*78}"
    )


# ── Episode rollout in a single env ──────────────────────────────────────────

def run_one_episode(
    *,
    env_name: str,
    env,
    encoder, state_predictor, action_predictor,
    action_embed, policy,
    latent_buf, policy_buf,
    cfg: Config,
    device,
    health: HealthMonitor,
    step_start: int,
    global_step_box,           # 1-element list mutated to track global step
) -> tuple[int, int]:
    """
    Roll one episode in `env`. Returns (steps_in_episode, completed_int).

    Mutates: latent_buf, policy_buf, health, global_step_box[0].
    """
    frame_np = env.reset()
    h_t: torch.Tensor | None = None
    h_query_np: np.ndarray | None = None
    ep_transitions: list = []
    in_episode = True
    n_steps = 0
    completed = 0

    while in_episode and global_step_box[0] < cfg.max_steps:
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

        # ── Encode current frame (rollout, no grad) ─────────────────────────
        with torch.no_grad():
            if h_t is None:
                queries = encoder.perceiver.get_initial_queries(1, device)
                h_query_np = queries.squeeze(0).cpu().numpy()
            else:
                queries = h_t.detach()
            h_current, _, _ = encoder(frame_t, queries)

        # ── Select action ───────────────────────────────────────────────────
        avail = env.available_actions
        if global_step_box[0] < cfg.warmup_steps:
            action_idx = int(np.random.randint(0, cfg.n_actions))
            log_prob = entropy = None
            entropy_val = None
        else:
            action_idx, log_prob, entropy = policy.act(h_current.squeeze(0), avail)
            entropy_val = entropy.item()

        # ── Step env ────────────────────────────────────────────────────────
        next_np, is_terminal = env.step(action_idx)
        life_end = is_end_of_life(env_name, frame_np, next_np, is_terminal)

        # ── Curiosity reward + health metrics (only if NOT a dying step) ────
        # See system_card.md §3.3.
        if not life_end:
            with torch.no_grad():
                next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)
                h_next, _, _ = encoder(next_t, h_current.detach())
                a_emb_r = action_embed(torch.tensor([action_idx], device=device))
                h_pred = state_predictor.predict(h_current, a_emb_r)
                state_err = (h_next - h_pred).pow(2).mean().item()
                action_logits_r = action_predictor(h_current, h_next)
                action_err = F.cross_entropy(
                    action_logits_r,
                    torch.tensor([action_idx], device=device),
                ).item()
                raw_reward = (cfg.reward_w_state * state_err
                              + cfg.reward_w_action * action_err)
                curiosity_reward = (min(raw_reward, cfg.reward_clamp)
                                    if np.isfinite(raw_reward) else 0.0)
                ht_htp1_cs = repr_mod.ht_htp1_cossim(
                    h_current.squeeze(0), h_next.squeeze(0)
                )

            health.sec1["ht_htp1_cossim_rollout"].append(ht_htp1_cs)
            health.sec6["reward_state_component"].append(state_err)
            health.sec6["reward_action_component"].append(action_err)
            # Per-env breakdowns
            health.sec6.setdefault(f"reward_state_component_{env_name}",
                                   deque(maxlen=200)).append(state_err)
            health.sec6.setdefault(f"reward_action_component_{env_name}",
                                   deque(maxlen=200)).append(action_err)
            health.sec6.setdefault(f"reward_total_{env_name}",
                                   deque(maxlen=200)).append(curiosity_reward)

            # Policy buffer
            if global_step_box[0] >= cfg.warmup_steps and log_prob is not None:
                policy_buf.add(log_prob, curiosity_reward, entropy)
                health.sec6["reward_total"].append(curiosity_reward)
                if entropy_val is not None:
                    health.sec5["policy_entropy"].append(entropy_val)
                    hmax = float(np.log(max(len(avail), 1)))
                    if hmax > 0:
                        health.sec5["policy_entropy_normalized"].append(
                            entropy_val / hmax
                        )
                    health.sec5.setdefault(f"policy_entropy_{env_name}",
                                           deque(maxlen=200)).append(entropy_val)

        # ── Append to episode buffer (unconditional; [:-1] excludes dying step) ─
        ep_transitions.append((
            frame_np.copy(),
            h_query_np.copy() if h_query_np is not None else None,
            action_idx,
            next_np.copy(),
        ))

        # ── Advance recurrent state ─────────────────────────────────────────
        h_t = h_current
        h_query_np = h_current.squeeze(0).cpu().numpy()

        n_steps += 1
        global_step_box[0] += 1

        # ── End-of-life flush ───────────────────────────────────────────────
        if life_end:
            for frame_i, hq_i, action_i, next_i in ep_transitions[:-1]:
                if hq_i is None:
                    # Shouldn't happen — h_query_np is set on first encode.
                    continue
                latent_buf.add(frame_i, hq_i, action_i, next_i)
            completed = int(getattr(env, "level_completed", False))
            in_episode = False
        else:
            frame_np = next_np

    return n_steps, completed


# ── JEPA update on a balanced batch ──────────────────────────────────────────

def jepa_update(
    *,
    cfg: Config,
    latent_bufs,            # dict env -> NextFrameLatentBuffer
    action_embeds,          # dict env -> ActionEmbedding
    encoder, state_predictor, action_predictor,
    enc_opt, state_pred_opt, action_pred_opt,
    sub_blocks, health: HealthMonitor,
    device,
    env_names,
) -> dict:
    """One balanced JEPA gradient step. Returns a small diagnostic dict."""
    half = cfg.batch_size // 2
    batches = {e: latent_bufs[e].sample(half, device) for e in env_names}

    enc_opt.zero_grad(); state_pred_opt.zero_grad(); action_pred_opt.zero_grad()

    L_state_per_env = {}
    L_action_per_env = {}
    per_lat_state_combined = None
    h_t_combined = []
    h_tp1_combined = []
    a_t_combined = []

    for e in env_names:
        b = batches[e]
        h_t_fresh, _, _ = encoder(b.frames, b.h_queries.detach())
        h_tp1_fresh, _, _ = encoder(b.next_frames, h_t_fresh.detach())
        a_emb = action_embeds[e](b.actions)

        L_state_e, per_lat_state_e = state_predictor.compute_loss(
            h_t_fresh, h_tp1_fresh.detach(), a_emb
        )
        action_logits_e = action_predictor(h_t_fresh, h_tp1_fresh)
        L_action_e = F.cross_entropy(action_logits_e, b.actions)

        L_state_per_env[e] = L_state_e
        L_action_per_env[e] = L_action_e
        if per_lat_state_combined is None:
            per_lat_state_combined = per_lat_state_e.detach().clone()
        else:
            per_lat_state_combined = per_lat_state_combined + per_lat_state_e.detach()

        h_t_combined.append(h_t_fresh.detach())
        h_tp1_combined.append(h_tp1_fresh.detach())
        a_t_combined.append(b.actions)

    L_state  = 0.5 * (L_state_per_env[env_names[0]] + L_state_per_env[env_names[1]])
    L_action = 0.5 * (L_action_per_env[env_names[0]] + L_action_per_env[env_names[1]])
    L = cfg.lambda_state * L_state + cfg.lambda_action * L_action

    if not torch.isfinite(L):
        return {"L_state": L_state.item(), "L_action": L_action.item(),
                "L_total": L.item(), "nonfinite": True}

    L.backward()

    # Pre-clip per-sub-block totals
    for sub_name, params in sub_blocks.items():
        health.push_sec7(f"gnorm_{sub_name}_total", grad_mod.grad_norm(params))

    nn.utils.clip_grad_norm_(
        list(encoder.parameters())
        + list(state_predictor.parameters())
        + list(action_predictor.parameters())
        + list(action_embeds[env_names[0]].parameters())
        + list(action_embeds[env_names[1]].parameters()),
        cfg.grad_clip_model,
    )
    enc_opt.step(); state_pred_opt.step(); action_pred_opt.step()

    # Streaming losses (aggregate + per-env)
    health.sec6["L_state"].append(L_state.item())
    health.sec6["L_action"].append(L_action.item())
    health.sec6["L_total"].append(L.item())
    for e in env_names:
        health.sec6.setdefault(f"L_state_{e}", deque(maxlen=200)).append(
            L_state_per_env[e].item()
        )
        health.sec6.setdefault(f"L_action_{e}", deque(maxlen=200)).append(
            L_action_per_env[e].item()
        )
    per_lat_state_avg = per_lat_state_combined / len(env_names)
    for i, v in enumerate(per_lat_state_avg.tolist()):
        health.sec6_per_latent_state_loss[i].append(v)

    return {"L_state": L_state.item(), "L_action": L_action.item(),
            "L_total": L.item(), "nonfinite": False}


# ── Main training loop ───────────────────────────────────────────────────────

def train(cfg: Config, resume_path: str = None, run_dir: Path = None,
          checkpoint_dir: Path | None = None) -> None:
    set_seeds(cfg.seed)
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    env_names = list(cfg.env_names)
    print(f"[exp004_0] device={device}  max_steps={cfg.max_steps}")
    print(f"[exp004_0] envs={env_names}  game_ids={list(cfg.game_ids)}")
    print(f"[exp004_0] lambdas: state={cfg.lambda_state}  action={cfg.lambda_action}")
    print(f"[exp004_0] reward weights: state={cfg.reward_w_state}  action={cfg.reward_w_action}  "
          f"clamp={cfg.reward_clamp}")
    print(f"[exp004_0] buffer_size_per_env={cfg.buffer_size_per_env}")
    print(f"[exp004_0] batch_size={cfg.batch_size}  (per-env half = {cfg.batch_size // 2})")
    assert cfg.batch_size % 2 == 0, "batch_size must be even for balanced sampling"

    encoder, state_predictor, action_predictor, action_embeds, policies, baselines = \
        load_models(cfg, device)

    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(
            resume_path, encoder, state_predictor, action_predictor,
            action_embeds, policies, baselines, device,
        )
        print(f"[exp004_0] Resumed from step {start_step}")

    # Placeholder init snapshot (encoder placeholders are shared across envs)
    if run_dir is not None:
        ph_init_path = run_dir / "placeholder_init.pt"
        if not ph_init_path.exists():
            torch.save(encoder.perceiver.placeholders.detach().cpu().clone(), ph_init_path)
            print(f"[exp004_0] Cached placeholder init → {ph_init_path}")
        placeholder_init = torch.load(ph_init_path, map_location=device, weights_only=True)
    else:
        placeholder_init = encoder.perceiver.placeholders.detach().clone()

    # ── Per-component optimizers ────────────────────────────────────────────
    enc_s1_params = (
        list(encoder.color_embed.parameters())
        + list(encoder.patch_proj.parameters())
        + list(encoder.sa_blocks.parameters())
        + list(encoder.sa_norm.parameters())
    )
    enc_s2_params = list(encoder.perceiver.parameters())

    enc_opt = torch.optim.AdamW([
        {"params": enc_s1_params, "lr": cfg.sa_lr,        "weight_decay": cfg.encoder_wd},
        {"params": enc_s2_params, "lr": cfg.perceiver_lr, "weight_decay": cfg.encoder_wd},
    ])
    # Both action-embedding tables go into the state-predictor optimizer (their
    # gradients flow through L_state only — the action predictor doesn't read them).
    sp_params = list(state_predictor.parameters())
    for e in env_names:
        sp_params += list(action_embeds[e].parameters())
    state_pred_opt = torch.optim.AdamW(
        sp_params,
        lr=cfg.state_predictor_lr, weight_decay=cfg.state_predictor_wd,
    )
    action_pred_opt = torch.optim.AdamW(
        action_predictor.parameters(),
        lr=cfg.action_predictor_lr, weight_decay=cfg.action_predictor_wd,
    )
    pol_opts = {e: torch.optim.Adam(policies[e].parameters(), lr=cfg.policy_lr)
                for e in env_names}

    # Sub-block parameter partition (used by gnorm totals). Use exp_003_4's helper
    # with the first env's action embedding for compatibility; cross-env gradient
    # decomposition is left for a follow-up. The shared modules' gnorms are still
    # tracked under the standard sub_blocks names.
    sub_blocks = grad_mod.build_sub_block_params(
        encoder, state_predictor, action_predictor,
        action_embeds[env_names[0]], policies[env_names[0]],
    )

    # ── Buffers (per env) ────────────────────────────────────────────────────
    latent_bufs = {
        e: NextFrameLatentBuffer(
            n_latents=cfg.n_latents, d_model=cfg.d_model,
            capacity=cfg.buffer_size_per_env,
        )
        for e in env_names
    }
    policy_bufs = {e: PolicyBuffer(cfg.policy_update_freq) for e in env_names}

    # ── Environments ─────────────────────────────────────────────────────────
    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    envs = {}
    for name, gid in zip(env_names, cfg.game_ids):
        raw = arc.make(gid)
        envs[name] = make_env(raw, gid)
        print(f"[exp004_0] Loaded env '{name}' ({gid}) — n_actions={envs[name].n_actions}")

    metrics_writer = MetricsWriter(run_dir) if run_dir else None
    health = HealthMonitor(
        window_fast=LOG_FREQ, window_slow=50, window_eval=5,
        n_latents=cfg.n_latents, n_actions=cfg.n_actions,
        n_perceiver_rounds=cfg.n_perceiver_rounds,
    )

    ep_counts = {e: 0 for e in env_names}
    completion_counts = {e: deque(maxlen=20) for e in env_names}
    buf_min_satisfied = False
    jepa_updates = 0
    next_probe_at = cfg.capture_attn_freq    # fire encoder probe at this JEPA-update count
    global_step_box = [start_step]
    t0 = time.time()

    print(f"\n[exp004_0] Training started  warmup={cfg.warmup_steps}  seed={cfg.seed}\n")

    env_cycle_idx = 0

    while global_step_box[0] < cfg.max_steps:
        # ── Pick next env (round-robin per episode) ─────────────────────────
        env_name = env_names[env_cycle_idx % len(env_names)]
        env_cycle_idx += 1

        # ── Roll one full episode in the chosen env ────────────────────────
        steps_done, completed = run_one_episode(
            env_name=env_name,
            env=envs[env_name],
            encoder=encoder,
            state_predictor=state_predictor,
            action_predictor=action_predictor,
            action_embed=action_embeds[env_name],
            policy=policies[env_name],
            latent_buf=latent_bufs[env_name],
            policy_buf=policy_bufs[env_name],
            cfg=cfg,
            device=device,
            health=health,
            step_start=global_step_box[0],
            global_step_box=global_step_box,
        )
        ep_counts[env_name] += 1
        completion_counts[env_name].append(completed)
        health.episodes_done.append(completed)
        health.sec6.setdefault(f"episode_length_{env_name}",
                               deque(maxlen=50)).append(float(steps_done))
        health.sec6.setdefault(f"completion_rate_{env_name}",
                               deque(maxlen=50)).append(float(completed))

        step = global_step_box[0]

        # ── JEPA updates: schedule by global env-step count ─────────────────
        # We do one JEPA update for every `update_freq` env steps that elapsed
        # during this episode (approximate but consistent).
        if not buf_min_satisfied:
            buf_min_satisfied = all(len(latent_bufs[e]) >= cfg.min_buffer_size
                                    for e in env_names)
        if buf_min_satisfied:
            n_updates_to_do = max(steps_done // cfg.update_freq, 1)
            for _ in range(n_updates_to_do):
                info = jepa_update(
                    cfg=cfg, latent_bufs=latent_bufs, action_embeds=action_embeds,
                    encoder=encoder, state_predictor=state_predictor,
                    action_predictor=action_predictor,
                    enc_opt=enc_opt, state_pred_opt=state_pred_opt,
                    action_pred_opt=action_pred_opt,
                    sub_blocks=sub_blocks, health=health,
                    device=device, env_names=env_names,
                )
                jepa_updates += 1
                if info["nonfinite"]:
                    print(f"[WARNING step {step}] Non-finite JEPA loss "
                          f"L_total={info['L_total']:.4f}")
                    break

            # Cheap encoder probe — recycle exp_003_4's representation metrics
            # on a small balanced peek. Use a threshold counter rather than `%` so
            # we don't skip past multiples when many updates run per episode.
            if jepa_updates >= next_probe_at:
                next_probe_at = jepa_updates + cfg.capture_attn_freq
                with torch.no_grad():
                    half = max(1, min(8, len(latent_bufs[env_names[0]]) // 2,
                                      len(latent_bufs[env_names[1]]) // 2))
                    mb_ls20 = latent_bufs[env_names[0]].sample(half, device)
                    mb_tu93 = latent_bufs[env_names[1]].sample(half, device)
                    mb_frames = torch.cat([mb_ls20.frames, mb_tu93.frames], dim=0)
                    mb_actions = torch.cat([mb_ls20.actions, mb_tu93.actions], dim=0)
                    B_m = mb_frames.shape[0]
                    q_m = encoder.perceiver.get_initial_queries(B_m, device)
                    h_m, _, _ = encoder(mb_frames, q_m)

                    norms = repr_mod.latent_norms(h_m)
                    for i, n_ in enumerate(norms):
                        health.sec1_latent_norms[i].append(n_)
                    health.sec1["latent_pairwise_cossim_buf"].append(
                        repr_mod.latent_pairwise_cossim(h_m)
                    )
                    health.sec1["latent_pairwise_l2_buf"].append(
                        repr_mod.latent_pairwise_l2(h_m)
                    )
                    health.sec1["latent_eff_rank"].append(
                        repr_mod.effective_rank(h_m[0])
                    )
                    health.sec1["placeholder_pairwise_cossim"].append(
                        repr_mod.placeholder_pairwise_cossim(
                            encoder.perceiver.placeholders
                        )
                    )
                    drift = repr_mod.placeholder_drift_from_init(
                        encoder.perceiver.placeholders, placeholder_init
                    )
                    for i, d in enumerate(drift):
                        if i < len(health.sec1_per_placeholder):
                            health.sec1_per_placeholder[i].append(d)
                    if drift:
                        health.sec1["placeholder_drift_from_init_mean"].append(
                            float(np.mean(drift))
                        )

        # ── Per-env policy updates ───────────────────────────────────────────
        for e in env_names:
            if global_step_box[0] >= cfg.warmup_steps and policy_bufs[e].full():
                log_probs, rewards, entropies = policy_bufs[e].get(device)
                adv = rewards - baselines[e].value
                adv = adv / (adv.std().clamp(min=1e-8) + 1e-8)
                baselines[e].update(rewards.mean().item())
                pol_loss = (-(adv * log_probs).mean()
                            - cfg.policy_entropy_lambda * entropies.mean())
                pol_opts[e].zero_grad()
                pol_loss.backward()
                nn.utils.clip_grad_norm_(policies[e].parameters(), cfg.grad_clip_policy)
                pol_opts[e].step()
                policy_bufs[e].clear()
                health.sec6.setdefault(f"pol_loss_{e}", deque(maxlen=200)).append(
                    pol_loss.item()
                )

        # ── Logging ──────────────────────────────────────────────────────────
        # We log after each episode (LS20 ~130 steps, TU93 ~50 steps).
        # That's coarser than every-LOG_FREQ but reduces dashboard chatter.
        if step // LOG_FREQ != (step - steps_done) // LOG_FREQ:
            fps = step / (time.time() - t0 + 1e-6)
            buf_sizes = {e: len(latent_bufs[e]) for e in env_names}
            print_stats(step, health, env_names, buf_sizes, ep_counts, fps,
                        cfg.lambda_state, cfg.lambda_action)
            if metrics_writer is not None:
                metrics_writer.write(step, fps, sum(buf_sizes.values()),
                                     sum(ep_counts.values()), health)
            criticals, warnings = health.check()
            for w in warnings:
                print(f"  [WARN] {w}")
            for c in criticals:
                print(f"  [CRITICAL] {c}")
            if criticals:
                print("[exp004_0] Critical issue — saving checkpoint and stopping")
                save_checkpoint(step, encoder, state_predictor, action_predictor,
                                action_embeds, policies, baselines, cfg, "critical",
                                ckpt_dir=checkpoint_dir)
                break

        # ── Checkpointing ────────────────────────────────────────────────────
        if step > start_step and step // CHECKPOINT_FREQ != (step - steps_done) // CHECKPOINT_FREQ:
            path = save_checkpoint(step, encoder, state_predictor, action_predictor,
                                   action_embeds, policies, baselines, cfg,
                                   ckpt_dir=checkpoint_dir)
            print(f"  [ckpt] Saved {path.name}")

    path = save_checkpoint(global_step_box[0], encoder, state_predictor,
                           action_predictor, action_embeds, policies, baselines,
                           cfg, "final", ckpt_dir=checkpoint_dir)
    print(f"\n[exp004_0] Training complete at step {global_step_box[0]}. "
          f"Final checkpoint: {path.name}")
    if metrics_writer is not None:
        metrics_writer.close()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--max-steps",  type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to write checkpoints into. Defaults to "
             "exp_004_0_ls20_tu93/checkpoints/.",
    )
    args = parser.parse_args()

    cfg = Config()
    overrides = {}
    if args.max_steps  is not None: overrides["max_steps"]  = args.max_steps
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    ckpt_dir_path = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None

    _run_dir, _tee = setup_run_logger(args.resume)
    try:
        train(cfg, resume_path=args.resume, run_dir=_run_dir,
              checkpoint_dir=ckpt_dir_path)
    finally:
        sys.stdout = _tee._orig
        _tee.close()


if __name__ == "__main__":
    main()
