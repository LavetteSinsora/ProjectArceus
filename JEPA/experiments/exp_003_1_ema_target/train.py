"""
Exp-003-1 Training Script: EMA Target Encoder + Placeholder Gradient Fix.

Three changes from exp_003:

  [CHANGE 1] EMA target encoder
    H_{T+1} targets are computed by target_encoder (EMA copy of online encoder).
    EMA momentum follows a cosine schedule 0.996 → 0.9999.
    target_encoder.parameters() never receive gradients.

  [CHANGE 2] Placeholder gradient fix
    The LatentBuffer now stores an is_initial flag per transition.
    During the training update, initial-step transitions (those that used
    placeholder queries during rollout) are forward-passed through the online
    encoder using the live encoder.perceiver.placeholders parameter, so
    gradient flows to the placeholder nn.Parameter.
    Recurrent transitions (h_{t-1} queries) are still passed as detached tensors.

  [CHANGE 3] Stop-gradient verification
    H_{T+1} comes from target_encoder (no gradient path, separate network).
    H_T comes from online encoder (gradient flows).
    The .detach() on h_targets is now meaningful: it ensures that even if
    target_encoder somehow appeared in a grad graph it would be stopped.

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_1_ema_target.train
    uv run python -m JEPA.experiments.exp_003_1_ema_target.train --resume checkpoints/step_050000.pt
"""

import argparse
import dataclasses
import datetime
import io
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_1_ema_target.config import Config
from JEPA.experiments.exp_003_1_ema_target.models import load_models_with_target
from JEPA.experiments.exp_003_1_ema_target.reward_shaping import is_end_of_life
from JEPA.shared.buffer import PolicyBuffer
from JEPA.shared.ema import update_ema, ema_momentum
from JEPA.shared.env_wrapper import LS20Env


# ── Buffer with is_initial flag ───────────────────────────────────────────────

class LatentBatch(NamedTuple):
    frames:     torch.Tensor   # (B, 64, 64) uint8
    h_queries:  torch.Tensor   # (B, n_latents, d_model)  — h_{t-1} from rollout
    actions:    torch.Tensor   # (B,) long
    h_targets:  torch.Tensor   # (B, n_latents, d_model)  — h_{t+1} from target encoder
    is_initial: torch.Tensor   # (B,) bool — True if h_query was a placeholder


class LatentBuffer:
    """
    Replay buffer for exp_003_1. Identical to shared LatentBuffer plus
    an `is_initial` boolean field marking episode-start transitions.
    """

    def __init__(self, n_latents, d_model, capacity, recency_fraction, recent_window):
        self.capacity = capacity
        self.recency_fraction = recency_fraction
        self.recent_window = min(recent_window, capacity)

        self._frames     = np.zeros((capacity, 64, 64), dtype=np.uint8)
        self._h_queries  = np.zeros((capacity, n_latents, d_model), dtype=np.float32)
        self._actions    = np.zeros(capacity, dtype=np.int64)
        self._h_targets  = np.zeros((capacity, n_latents, d_model), dtype=np.float32)
        self._is_initial = np.zeros(capacity, dtype=np.bool_)

        self._pos  = 0
        self._size = 0

    def add(self, frame, h_query, action_idx, h_target, is_initial: bool) -> None:
        self._frames[self._pos]     = frame
        self._h_queries[self._pos]  = h_query
        self._actions[self._pos]    = action_idx
        self._h_targets[self._pos]  = h_target
        self._is_initial[self._pos] = is_initial
        self._pos  = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> LatentBatch:
        n_recent  = int(batch_size * self.recency_fraction)
        n_uniform = batch_size - n_recent
        uniform_idx = np.random.randint(0, self._size, size=n_uniform)

        recent_size = min(self._size, self.recent_window)
        start = (self._pos - recent_size) % self.capacity
        if start + recent_size <= self.capacity:
            recent_pool = np.arange(start, start + recent_size)
        else:
            recent_pool = np.concatenate([
                np.arange(start, self.capacity),
                np.arange(0, (start + recent_size) % self.capacity),
            ])
        recent_idx = recent_pool[np.random.randint(0, len(recent_pool), n_recent)]
        idx = np.concatenate([uniform_idx, recent_idx])

        return LatentBatch(
            frames=torch.from_numpy(self._frames[idx]).to(device),
            h_queries=torch.from_numpy(self._h_queries[idx]).to(device),
            actions=torch.from_numpy(self._actions[idx]).to(device),
            h_targets=torch.from_numpy(self._h_targets[idx]).to(device),
            is_initial=torch.from_numpy(self._is_initial[idx]).to(device),
        )

    def __len__(self) -> int:
        return self._size


# ── Re-use monitoring / utility code from exp_003 ─────────────────────────────
# Rather than copy these verbatim we import the classes directly.
from JEPA.experiments.exp_003_0_normalized_latent_jepa.train import (
    _Tee,
    MetricsWriter,
    ActivationMonitor,
    HealthMonitor,
    compute_ode_step_cossim,
    print_stats,
    effective_rank,
    set_seeds,
    grad_norm,
    CHECKPOINT_FREQ,
    LOG_FREQ,
    EMBED_METRIC_FREQ,
    MAX_EP_STEPS,
    REWARD_CAP,
    LATENT_NORM_CRITICAL,
    LATENT_STD_CRITICAL,
    LOSS_CV_CRITICAL,
    TIME_GRAD_WARN,
    GRAD_NORM_CRITICAL,
    ENTROPY_WARN,
    ODE_COSSIM_WARN,
)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(step, encoder, target_encoder, predictor, action_embed, policy, cfg, label=""):
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    tag = f"step_{step:06d}" + (f"_{label}" if label else "")
    path = ckpt_dir / f"{tag}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({
        "encoder":        encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor":      predictor.state_dict(),
        "action_embed":   action_embed.state_dict(),
        "policy":         policy.state_dict(),
        "step":           step,
        "config":         dataclasses.asdict(cfg),
    }, tmp)
    tmp.rename(path)
    return path


def load_checkpoint(path, encoder, target_encoder, predictor, action_embed, policy, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    if "target_encoder" in ckpt:
        target_encoder.load_state_dict(ckpt["target_encoder"])
    else:
        # Resuming from an exp_003 checkpoint: initialise target from online encoder
        import copy
        target_encoder.load_state_dict(copy.deepcopy(ckpt["encoder"]))
        print("[exp003-1] No target_encoder in checkpoint — initialised from online encoder")
    predictor.load_state_dict(ckpt["predictor"])
    action_embed.load_state_dict(ckpt["action_embed"])
    policy.load_state_dict(ckpt["policy"])
    return ckpt.get("step", 0)


def setup_run_logger(resume_path):
    runs_root = Path(__file__).parent / "runs"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "resume" if resume_path else "fresh"
    run_dir = runs_root / f"run_{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    tee = _Tee(sys.stdout, log_path)
    sys.stdout = tee
    print(f"[exp003-1] Run dir:  {run_dir}")
    print(f"[exp003-1] Log file: {log_path}")
    return run_dir, tee


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: Config, resume_path: str = None, run_dir: Path = None) -> None:
    set_seeds(cfg.seed)
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[exp003-1] device={device}  max_steps={cfg.max_steps}")

    # [CHANGE 1] Load online + target encoder
    encoder, target_encoder, predictor, action_embed, policy, baseline = \
        load_models_with_target(cfg, device)

    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(
            resume_path, encoder, target_encoder,
            predictor, action_embed, policy, device,
        )
        print(f"[exp003-1] Resumed from step {start_step}")

    # ── Activation monitors ───────────────────────────────────────────────────
    act_mon = ActivationMonitor()
    for i, block in enumerate(encoder.sa_blocks):
        act_mon.register(block.ffn.net[1], f"enc_sa{i}_ffn")
    act_mon.register(encoder.perceiver.rounds[0].cross_attn.ffn.net[1], "perc_r0_cross_ffn")
    act_mon.register(encoder.perceiver.rounds[0].self_attn.ffn.net[1],  "perc_r0_self_ffn")
    if cfg.n_perceiver_rounds > 1:
        act_mon.register(encoder.perceiver.rounds[1].cross_attn.ffn.net[1], "perc_r1_cross_ffn")
        act_mon.register(encoder.perceiver.rounds[1].self_attn.ffn.net[1],  "perc_r1_self_ffn")
    for i, mlp in enumerate(predictor.mlps):
        act_mon.register(mlp.net[1], f"pred_mlp{i}")
    act_mon.register(policy.net[1], "policy_ffn")

    # ── Optimisers ────────────────────────────────────────────────────────────
    enc_s1_params = (
        list(encoder.color_embed.parameters()) +
        list(encoder.patch_proj.parameters()) +
        list(encoder.sa_blocks.parameters()) +
        list(encoder.sa_norm.parameters())
    )
    # [CHANGE 2] placeholders are now part of enc_s2_params and WILL receive
    # gradients through the initial-transition forward pass.
    enc_s2_params = list(encoder.perceiver.parameters())

    enc_opt = torch.optim.AdamW([
        {"params": enc_s1_params, "lr": cfg.sa_lr,        "weight_decay": cfg.encoder_wd},
        {"params": enc_s2_params, "lr": cfg.perceiver_lr, "weight_decay": cfg.encoder_wd},
    ])
    pred_opt = torch.optim.AdamW(
        list(predictor.parameters()) + list(action_embed.parameters()),
        lr=cfg.predictor_lr, weight_decay=cfg.predictor_wd,
    )
    pol_opt = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)

    # ── Buffers ───────────────────────────────────────────────────────────────
    # [CHANGE 2] Use LatentBuffer with is_initial flag
    latent_buf = LatentBuffer(
        n_latents=cfg.n_latents,
        d_model=cfg.d_model,
        capacity=cfg.buffer_size,
        recency_fraction=cfg.recency_fraction,
        recent_window=cfg.recent_buffer_size,
    )
    policy_buf = PolicyBuffer(cfg.policy_update_freq)

    # ── Environment ───────────────────────────────────────────────────────────
    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)

    metrics_writer = MetricsWriter(run_dir) if run_dir else None

    health   = HealthMonitor(window=LOG_FREQ)
    frame_np = env.reset()
    ep_count = 0
    step     = start_step
    t0       = time.time()

    h_t: torch.Tensor | None = None
    is_episode_start = True        # tracks whether next encode uses placeholder
    ep_transitions: list = []

    print(f"\n[exp003-1] Training started  warmup={cfg.warmup_steps}  seed={cfg.seed}")
    print(f"[exp003-1] EMA decay: {cfg.ema_decay_start} → {cfg.ema_decay_end}")
    print(f"[exp003-1] Placeholder gradient: ENABLED via is_initial flag\n")

    while step < cfg.max_steps:
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

        # ── Encode current frame (no grad — rollout only) ─────────────────────
        with torch.no_grad():
            if h_t is None:
                # [CHANGE 2] placeholder path — is_initial=True marks this transition
                queries = encoder.perceiver.get_initial_queries(1, device)
                current_is_initial = True
            else:
                queries = h_t.detach()
                current_is_initial = False

            h_current, _, _ = encoder(frame_t, queries)

        # ── Select action ─────────────────────────────────────────────────────
        avail = env.available_actions
        if step < cfg.warmup_steps:
            action_idx = int(np.random.randint(0, cfg.n_actions))
            log_prob = entropy_val = None
        else:
            action_idx, log_prob, entropy = policy.act(h_current.squeeze(0), avail)
            entropy_val = entropy.item()

        # ── Step environment ──────────────────────────────────────────────────
        next_np, is_terminal = env.step(action_idx)
        life_end = is_end_of_life(frame_np, next_np, is_terminal)

        # ── Compute h_target and curiosity reward ─────────────────────────────
        with torch.no_grad():
            next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)

            # [CHANGE 1] H_{T+1} from TARGET encoder — breaks the collapse cycle.
            # [CHANGE 3] This is the correct stop-gradient: target_encoder has no
            #            grad, and we additionally .detach() h_next when storing.
            h_next, _, _ = target_encoder(next_t, h_current.detach())

            a_emb_r = action_embed(torch.tensor([action_idx], device=device))
            _, per_lat_r = predictor.predict_with_loss(h_current, h_next, a_emb_r)
            raw_reward = per_lat_r.mean().item()
            curiosity_reward = min(raw_reward, REWARD_CAP) if np.isfinite(raw_reward) else 0.0
            h_target_np = h_next.squeeze(0).cpu().numpy()
            ht_ht1_cs = F.cosine_similarity(
                h_current.squeeze(0), h_next.squeeze(0), dim=-1
            ).mean().item()
        health.ht_ht1_cossim.append(ht_ht1_cs)

        # ── Store transition ──────────────────────────────────────────────────
        # h_query stored as the query used to produce h_current.
        # For initial: this is the placeholder values (needed for non-initial recurrent path).
        # The is_initial flag tells training which path to use.
        h_query_np = queries.squeeze(0).detach().cpu().numpy()
        ep_transitions.append((
            frame_np.copy(),
            h_query_np.copy(),
            action_idx,
            h_target_np.copy(),
            current_is_initial,
        ))

        if step >= cfg.warmup_steps and log_prob is not None:
            policy_buf.add(log_prob, curiosity_reward, entropy)
            health.rewards.append(curiosity_reward)
            if entropy_val is not None:
                health.entropy.append(entropy_val)

        h_t = h_current
        ep_len = len(ep_transitions)
        force_flush = (ep_len >= MAX_EP_STEPS) and not life_end

        if life_end or force_flush:
            for frame_i, hq_i, action_i, ht_i, init_i in ep_transitions[:-1]:
                latent_buf.add(frame_i, hq_i, action_i, ht_i, init_i)

            if force_flush:
                ep_transitions = [ep_transitions[-1]]
            else:
                ep_count += 1
                health.episodes_done.append(int(env.level_completed))
                ep_transitions = []
                h_t = None

            frame_np = env.reset() if (life_end and is_terminal) else next_np
        else:
            frame_np = next_np

        step += 1

        # ── JEPA / flow-matching training ─────────────────────────────────────
        if step % cfg.update_freq == 0 and len(latent_buf) >= cfg.min_buffer_size:
            batch = latent_buf.sample(cfg.batch_size, device)

            enc_opt.zero_grad()
            pred_opt.zero_grad()

            # [CHANGE 2] Placeholder gradient fix.
            # Split batch into initial-step and recurrent-step transitions.
            # Initial: use live encoder.perceiver.placeholders (gradient flows to it).
            # Recurrent: use stored h_queries (detached — h_{t-1} needs no gradient).
            init_mask = batch.is_initial          # (B,) bool
            rec_mask  = ~init_mask
            B = batch.frames.shape[0]

            h_t_fresh = torch.empty(
                B, cfg.n_latents, cfg.d_model, device=device
            )

            if init_mask.any():
                n_init = int(init_mask.sum().item())
                # Live placeholders: gradient WILL flow back through these
                q_init = encoder.perceiver.get_initial_queries(n_init, device)
                h_init, _, _ = encoder(batch.frames[init_mask], q_init)
                h_t_fresh[init_mask] = h_init

            if rec_mask.any():
                # Stored h_{t-1}: detached, no gradient intended
                h_rec, _, _ = encoder(
                    batch.frames[rec_mask],
                    batch.h_queries[rec_mask].detach(),
                )
                h_t_fresh[rec_mask] = h_rec

            a_emb = action_embed(batch.actions)

            # [CHANGE 3] h_targets come from target_encoder (stored numpy → tensor,
            # already no gradient). .detach() is a belt-and-suspenders guard.
            flow_loss, per_lat = predictor.compute_loss(
                h_t_fresh, batch.h_targets.detach(), a_emb
            )

            if not torch.isfinite(flow_loss):
                print(f"[WARNING step {step}] Non-finite flow loss: {flow_loss.item():.4f}")
                enc_opt.zero_grad()
                pred_opt.zero_grad()
            else:
                flow_loss.backward()

                gn_s1   = grad_norm(enc_s1_params)
                gn_s2   = grad_norm(enc_s2_params)
                gn_mlps = [grad_norm(list(mlp.parameters())) for mlp in predictor.mlps]
                gn_time = grad_norm(list(predictor.time_embed.parameters()))

                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters())
                    + list(action_embed.parameters()),
                    cfg.grad_clip_model,
                )
                enc_opt.step()
                pred_opt.step()

                # [CHANGE 1] EMA update: θ_target ← m·θ_target + (1−m)·θ_online
                m = ema_momentum(step, cfg.max_steps, cfg.ema_decay_start, cfg.ema_decay_end)
                update_ema(encoder, target_encoder, m)

                health.flow_loss_total.append(flow_loss.item())
                for i, v in enumerate(per_lat.tolist()):
                    health.per_latent_loss[i].append(v)
                health.grad_enc_s1.append(gn_s1)
                health.grad_enc_s2.append(gn_s2)
                for i, gn in enumerate(gn_mlps):
                    health.grad_pred_mlps[i].append(gn)
                health.time_emb_grad.append(gn_time)

            # Embedding diversity metrics (use placeholder path for a consistent baseline)
            if (step // cfg.update_freq) % EMBED_METRIC_FREQ == 0:
                with torch.no_grad():
                    mb = latent_buf.sample(min(16, len(latent_buf)), device)
                    B_m = mb.frames.shape[0]
                    q_m = encoder.perceiver.get_initial_queries(B_m, device)
                    h_m, _, _ = encoder(mb.frames, q_m)

                    for i in range(cfg.n_latents):
                        health.latent_norms[i].append(h_m[:, i, :].norm(dim=-1).mean().item())
                        health.per_latent_std[i].append(h_m[:, i, :].std().item())

                    pairs = [
                        (h_m[:, i, :] - h_m[:, j, :]).norm(dim=-1).mean().item()
                        for i in range(cfg.n_latents) for j in range(i + 1, cfg.n_latents)
                    ]
                    health.latent_pairwise_l2.append(float(np.mean(pairs)))
                    health.latent_eff_rank.append(effective_rank(h_m[0]))
                    health.across_state_std.append(h_m.std(dim=0).mean().item())

                    a_m = action_embed(mb.actions[:1])
                    health.ode_step_cossim.append(
                        compute_ode_step_cossim(predictor, h_m[:1], a_m)
                    )

        # ── Policy training ───────────────────────────────────────────────────
        if step >= cfg.warmup_steps and policy_buf.full():
            log_probs, rewards, entropies = policy_buf.get(device)
            adv = rewards - baseline.value
            adv = adv / (adv.std().clamp(min=1e-8) + 1e-8)
            baseline.update(rewards.mean().item())

            pol_loss = (-(adv * log_probs).mean()
                        - cfg.policy_entropy_lambda * entropies.mean())
            pol_opt.zero_grad()
            pol_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_policy)
            pol_opt.step()
            policy_buf.clear()

            health.pol_loss.append(pol_loss.item())
            health.grad_policy.append(grad_norm(policy.parameters()))

        # ── Logging ───────────────────────────────────────────────────────────
        if step % LOG_FREQ == 0 and step > 0:
            fps = step / (time.time() - t0 + 1e-6)
            print_stats(step, health, act_mon, len(latent_buf), ep_count, fps)
            if metrics_writer:
                metrics_writer.write(step, fps, len(latent_buf), ep_count, health)
            criticals, warnings = health.check()
            for w in warnings:
                print(f"  [WARN] {w}")
            for c in criticals:
                print(f"  [CRITICAL] {c}")
            if criticals:
                print("[exp003-1] Critical issue — saving checkpoint and stopping")
                save_checkpoint(step, encoder, target_encoder,
                                predictor, action_embed, policy, cfg, "critical")
                break

        # ── Checkpointing ─────────────────────────────────────────────────────
        if step % CHECKPOINT_FREQ == 0 and step > start_step:
            path = save_checkpoint(step, encoder, target_encoder,
                                   predictor, action_embed, policy, cfg)
            print(f"  [ckpt] Saved {path.name}")

    path = save_checkpoint(step, encoder, target_encoder,
                           predictor, action_embed, policy, cfg, "final")
    print(f"\n[exp003-1] Training complete at step {step}. Final checkpoint: {path.name}")
    act_mon.remove()
    if metrics_writer:
        metrics_writer.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--max-steps",  type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    overrides = {}
    if args.max_steps  is not None: overrides["max_steps"]  = args.max_steps
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    _run_dir, _tee = setup_run_logger(args.resume)
    try:
        train(cfg, resume_path=args.resume, run_dir=_run_dir)
    finally:
        sys.stdout = _tee._orig
        _tee.close()


if __name__ == "__main__":
    main()
