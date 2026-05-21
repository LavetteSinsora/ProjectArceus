"""Dreamer V3 training loop.

Generic over reward source: callers pass `reward_fn(prev_frame, frame, env) → (r, info)`.
Sub-experiments only swap that function + their config; the trainer here is identical.

High-level structure (see exp_005_dreamer_v3/system_card.md for the rationale):

    prefill replay with random policy
    while step < max_env_steps:
        env interaction (carry recurrent state across steps; reset on done)
        every `env_steps_per_update` env steps:
            world-model update on a (B, T) sequence batch
            actor + critic update on imagined trajectories (H steps)
            P2E ensemble update; if enabled, exploration actor + critic update
            critic EMA update
        log + checkpoint + eval
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from JEPA.shared.env_wrapper import make_env as _make_arc_env

from .buffer import SequenceReplayBuffer
from .device import resolve_device
from .models import load_models
from .models.functional import (
    PercentileReturnScale,
    lambda_returns,
    symlog,
)
from .models.rssm import RSSMState


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    while p.name != "Code Repo" and p.parent != p:
        p = p.parent
    return p


def _build_env(cfg):
    """Build a fresh LS20 env using the shared wrapper."""
    from arc_agi import Arcade, OperationMode

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root() / "environment_files"),
    )
    raw = arc.make(cfg.game_id)
    return _make_arc_env(raw, cfg.game_id)


def _obs_to_uint8(frame: np.ndarray) -> np.ndarray:
    """(H, W) uint8 frame → (C=1, H, W) uint8."""
    return frame[None, :, :]


def _obs_to_float_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W) uint8 → (1, 1, H, W) float in [-0.5, 0.5]."""
    return torch.from_numpy(frame).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 15.0 - 0.5


def _twohot_ce(value_dist, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of soft twohot(symlog(target)) against value_dist logits."""
    from .models.functional import twohot_encode
    soft = twohot_encode(symlog(target), value_dist.bins)
    return -(soft * value_dist.log_probs).sum(dim=-1).mean()


def train(cfg, reward_fn: Callable, run_dir: Path | str | None = None) -> dict:
    """Main training entrypoint.

    Args:
        cfg: a ConfigBase subclass instance.
        reward_fn: callable (prev_frame, frame, env, info) → (reward: float, done_override: bool|None, info_out: dict)
            done_override lets reward_fn end an episode early; pass None to use env's `is_terminal`.
        run_dir:  where to put training.log, metrics.jsonl, checkpoints/.  Auto-created if None.

    Returns:
        Final summary dict.
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if run_dir is None:
        run_dir = Path(__file__).resolve().parent.parent / "runs" / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    log_path = run_dir / "training.log"
    metrics_path = run_dir / "metrics.jsonl"
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2, default=str))

    log = lambda msg: (print(msg), open(log_path, "a").write(msg + "\n"))
    log(f"[setup] run_dir={run_dir} device={device}")

    env = _build_env(cfg)
    buffer = SequenceReplayBuffer(
        capacity=cfg.replay_capacity,
        obs_shape=(cfg.obs_channels, cfg.obs_size, cfg.obs_size),
        n_actions=cfg.n_actions,
        batch_size=cfg.batch_size,
        seq_len=cfg.batch_length,
        seed=cfg.seed,
    )

    wm, actor, critic, critic_ema, actor_p2e, critic_p2e, critic_p2e_ema = load_models(cfg, device)
    critic_ema.to(device)
    if critic_p2e_ema is not None:
        critic_p2e_ema.to(device)

    # Optimizers
    wm_no_ensemble = [p for n, p in wm.named_parameters() if not n.startswith("ensemble")]
    opt_wm = torch.optim.Adam(wm_no_ensemble, lr=cfg.wm_lr, eps=cfg.wm_adam_eps)
    opt_ensemble = torch.optim.Adam(wm.ensemble.parameters(), lr=cfg.p2e_ensemble_lr, eps=cfg.wm_adam_eps)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.actor_adam_eps)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, eps=cfg.critic_adam_eps)
    if actor_p2e is not None:
        opt_actor_p2e = torch.optim.Adam(actor_p2e.parameters(), lr=cfg.actor_lr, eps=cfg.actor_adam_eps)
        opt_critic_p2e = torch.optim.Adam(critic_p2e.parameters(), lr=cfg.critic_lr, eps=cfg.critic_adam_eps)
    else:
        opt_actor_p2e = opt_critic_p2e = None

    ret_scale_task = PercentileReturnScale(decay=cfg.return_scale_decay)
    ret_scale_p2e = PercentileReturnScale(decay=cfg.return_scale_decay)

    # ── Env-interaction state ───────────────────────────────────────────────
    frame = env.reset()
    state = wm.rssm.initial_state(1, device)
    last_action = torch.zeros(1, cfg.n_actions, device=device)
    prev_frame = frame.copy()
    ep_steps = 0
    ep_count = 0
    ep_completed = 0
    ep_total_reward = 0.0
    recent_eps: list[tuple[int, float, bool]] = []   # (len, ret, completed)

    # ── Prefill with random policy ───────────────────────────────────────────
    log(f"[prefill] {cfg.prefill_steps} random steps")
    for _ in range(cfg.prefill_steps):
        a_idx = int(np.random.randint(cfg.n_actions))
        next_frame, done = env.step(a_idx)
        r, done_override, _info = reward_fn(prev_frame=frame, frame=next_frame, env=env, info={})
        if done_override is not None:
            done = done or done_override
        buffer.add(_obs_to_uint8(frame), a_idx, r, 0.0 if done else 1.0)
        if done:
            frame = env.reset()
            state = wm.rssm.initial_state(1, device)
            last_action = torch.zeros(1, cfg.n_actions, device=device)
        else:
            frame = next_frame

    log(f"[prefill] done, buffer size={buffer.size}")

    # ── Main loop ────────────────────────────────────────────────────────────
    # Gradient-step cadence: train_ratio is (transitions trained / env steps).
    # With batch_size*batch_length transitions per gradient step, this means
    # env_steps_per_update = batch_size*batch_length / train_ratio.
    env_steps_per_update = max(1, (cfg.batch_size * cfg.batch_length) // cfg.train_ratio)
    log(f"[setup] env_steps_per_update={env_steps_per_update} (train_ratio={cfg.train_ratio})")

    env_step = 0
    grad_step = 0
    t_start = time.time()

    while env_step < cfg.max_env_steps:
        # ── Env-interaction window ──────────────────────────────────────────
        for _ in range(env_steps_per_update):
            # Encode + observe step to get a fresh posterior z_t.
            x_emb = wm.encoder(_obs_to_float_tensor(frame, device))
            with torch.no_grad():
                post, _ = wm.rssm.obs_step(state.h, state.z, last_action, x_emb)
                state = post
                # Choose acting actor: P2E warmup, then task actor.
                if actor_p2e is not None and env_step < cfg.p2e_acting_steps:
                    a_onehot = actor_p2e.act(state.h, state.z, deterministic=False)
                else:
                    a_onehot = actor.act(state.h, state.z, deterministic=False)
            a_idx = int(a_onehot.argmax(dim=-1).item())
            last_action = a_onehot

            next_frame, done = env.step(a_idx)
            r, done_override, _info = reward_fn(prev_frame=frame, frame=next_frame, env=env, info={})
            if done_override is not None:
                done = done or done_override
            buffer.add(_obs_to_uint8(frame), a_idx, r, 0.0 if done else 1.0)
            env_step += 1
            ep_steps += 1
            ep_total_reward += r

            if done:
                completed = bool(env.level_completed)
                recent_eps.append((ep_steps, ep_total_reward, completed))
                if completed:
                    ep_completed += 1
                ep_count += 1
                frame = env.reset()
                state = wm.rssm.initial_state(1, device)
                last_action = torch.zeros(1, cfg.n_actions, device=device)
                ep_steps = 0
                ep_total_reward = 0.0
            else:
                frame = next_frame
            prev_frame = frame

            if env_step >= cfg.max_env_steps:
                break

        # ── Single gradient update ──────────────────────────────────────────
        if buffer.size < cfg.batch_length + 1:
            continue
        metrics = _train_step(
            cfg=cfg,
            wm=wm, actor=actor, critic=critic, critic_ema=critic_ema,
            actor_p2e=actor_p2e, critic_p2e=critic_p2e, critic_p2e_ema=critic_p2e_ema,
            opt_wm=opt_wm, opt_ensemble=opt_ensemble,
            opt_actor=opt_actor, opt_critic=opt_critic,
            opt_actor_p2e=opt_actor_p2e, opt_critic_p2e=opt_critic_p2e,
            buffer=buffer,
            ret_scale_task=ret_scale_task, ret_scale_p2e=ret_scale_p2e,
            device=device,
        )
        grad_step += 1

        # Logging
        if grad_step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sps = env_step / max(elapsed, 1e-6)
            recent = recent_eps[-20:] if recent_eps else []
            recent_compl = sum(1 for _, _, c in recent if c) / max(len(recent), 1)
            entry = {
                "env_step": env_step, "grad_step": grad_step, "elapsed_s": elapsed, "env_sps": sps,
                "buffer_size": buffer.size,
                "recent20_compl": recent_compl,
                "n_episodes": ep_count, "total_completed": ep_completed,
                **metrics,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            log(f"[step {env_step:>7}] grad {grad_step:>6} | "
                f"L_wm {metrics['L_wm']:.3f} (pred {metrics['L_pred']:.3f}, dyn {metrics['L_dyn']:.3f}, rep {metrics['L_rep']:.3f}) "
                f"| L_actor {metrics['L_actor']:.4f} L_critic {metrics['L_critic']:.3f} "
                f"| H[π] {metrics['policy_entropy']:.3f} "
                f"| recent20 compl {recent_compl:.0%} | sps {sps:.1f}")

        # Checkpoint
        if grad_step % cfg.ckpt_every == 0:
            _save_checkpoint(ckpt_dir, env_step, grad_step, wm, actor, critic, critic_ema, actor_p2e, critic_p2e)

    # Final checkpoint
    _save_checkpoint(ckpt_dir, env_step, grad_step, wm, actor, critic, critic_ema, actor_p2e, critic_p2e, tag="final")
    summary = {
        "env_step": env_step,
        "grad_step": grad_step,
        "n_episodes": ep_count,
        "total_completed": ep_completed,
        "elapsed_s": time.time() - t_start,
    }
    log(f"[done] {summary}")
    return summary


# ── Single gradient update ───────────────────────────────────────────────────

def _train_step(
    cfg,
    wm, actor, critic, critic_ema,
    actor_p2e, critic_p2e, critic_p2e_ema,
    opt_wm, opt_ensemble, opt_actor, opt_critic, opt_actor_p2e, opt_critic_p2e,
    buffer, ret_scale_task, ret_scale_p2e, device,
) -> dict:
    batch = buffer.sample(device)
    B, T = cfg.batch_size, cfg.batch_length

    # ── World model forward ─────────────────────────────────────────────────
    out = wm.observe(batch.obs, batch.action, prev_state=None)

    # L_pred — reconstruction NLL + reward NLL + continue NLL
    obs_flat = batch.obs.reshape(B * T, *batch.obs.shape[2:])
    log_p_x = out.recon_dist.log_prob(obs_flat)              # (BT,)
    log_p_r = out.reward_dist.log_prob(batch.reward.reshape(-1))  # (BT,)
    log_p_c = out.continue_dist.log_prob(batch.cont.reshape(-1))  # (BT,)
    L_pred = -(log_p_x.mean() + log_p_r.mean() + log_p_c.mean())

    # L_dyn / L_rep
    L_dyn, L_rep = wm.rssm.kl_loss(out.post, out.prior, free_nats=cfg.free_nats)

    L_wm = cfg.beta_pred * L_pred + cfg.beta_dyn * L_dyn + cfg.beta_rep * L_rep

    opt_wm.zero_grad(set_to_none=True)
    L_wm.backward()
    nn.utils.clip_grad_norm_([p for g in opt_wm.param_groups for p in g["params"]], cfg.wm_grad_clip)
    opt_wm.step()

    # ── P2E ensemble update (predict next posterior z) ──────────────────────
    L_ensemble = torch.tensor(0.0, device=device)
    if cfg.use_p2e_ensemble:
        with torch.no_grad():
            # Inputs: (h_t, z_t, a_t) at times 0..T-2 → target z_{t+1}
            h_in = out.post.h[:, :-1].detach()
            z_in = out.post.z[:, :-1].detach()
            a_in = batch.action[:, :-1]
            z_target = out.post.z[:, 1:].detach()
        h_flat = h_in.reshape(-1, h_in.shape[-1])
        z_flat = z_in.reshape(-1, *z_in.shape[2:])
        a_flat = a_in.reshape(-1, a_in.shape[-1])
        z_tgt_flat = z_target.reshape(-1, *z_target.shape[2:])
        L_ensemble = wm.ensemble.train_loss(h_flat, z_flat, a_flat, z_tgt_flat)
        opt_ensemble.zero_grad(set_to_none=True)
        L_ensemble.backward()
        nn.utils.clip_grad_norm_(wm.ensemble.parameters(), cfg.wm_grad_clip)
        opt_ensemble.step()

    # ── Imagination (detach posteriors as start state) ──────────────────────
    start_h = out.post.h.reshape(B * T, -1).detach()
    start_z = out.post.z.reshape(B * T, cfg.n_groups, cfg.n_classes).detach()
    H = cfg.imag_horizon

    # Task imagination uses task actor
    imag = wm.imagine(start_h, start_z, actor, horizon=H)
    feat_seq = imag.features                     # (H, N, feat_dim)
    rew_mean = imag.reward_dist.mean().reshape(H, -1)  # (H, N)
    cont_mean = imag.continue_dist.mean().reshape(H, -1)   # (H, N)

    # Critic values at every state in the imagined traj — need (H+1, N).
    # We use the value at start state (post) plus values along imag.
    with torch.no_grad():
        start_feat = torch.cat([start_h, start_z.reshape(start_z.shape[0], -1)], dim=-1)  # (N, feat_dim)
        v_start = critic(start_feat).mean()                                                # (N,)
    v_imag_dist_for_grad = critic(feat_seq.reshape(H * imag.features.shape[1], -1))         # gradient flows for critic loss
    v_imag = v_imag_dist_for_grad.mean().reshape(H, -1)                                     # (H, N)
    # For the actor we need a stop-gradient version.
    v_for_actor = v_imag.detach()

    # Bootstrap: last value
    v_extended = torch.cat([v_for_actor, v_for_actor[-1:].detach()], dim=0)                  # (H+1, N)
    R_lambda = lambda_returns(rew_mean.detach(), v_extended.detach(), cont_mean.detach(),
                              gamma=cfg.gamma, lam=cfg.lam)                                  # (H, N)

    # Percentile-scale and compute task actor loss
    S_task = ret_scale_task.update(R_lambda)
    advantage = (R_lambda - v_for_actor) / max(1.0, S_task)
    L_actor = -(advantage.detach() * imag.log_pi + cfg.entropy_eta * imag.entropy).mean()

    # Critic loss — twohot CE against R_lambda + EMA target regularisation
    L_critic_main = _twohot_ce(v_imag_dist_for_grad, R_lambda.detach().reshape(-1))
    with torch.no_grad():
        v_target = critic_ema(feat_seq.reshape(H * imag.features.shape[1], -1)).mean().detach()
    L_critic_reg = _twohot_ce(v_imag_dist_for_grad, v_target)
    L_critic = L_critic_main + L_critic_reg

    # Single backward — actor and critic share the imagination graph (log_pi
    # and features both depend on the recurrent rollout through the GRU), so
    # we cannot backward through them twice. Separate optimizers still update
    # only their own params.
    opt_actor.zero_grad(set_to_none=True)
    opt_critic.zero_grad(set_to_none=True)
    (L_actor + L_critic).backward()
    nn.utils.clip_grad_norm_(actor.parameters(), cfg.actor_grad_clip)
    nn.utils.clip_grad_norm_(critic.parameters(), cfg.critic_grad_clip)
    opt_actor.step()
    opt_critic.step()
    critic_ema.update(critic)

    # ── Exploration actor + critic (P2E) ────────────────────────────────────
    L_actor_p2e = torch.tensor(0.0, device=device)
    L_critic_p2e = torch.tensor(0.0, device=device)
    if actor_p2e is not None:
        imag_e = wm.imagine(start_h, start_z, actor_p2e, horizon=H)
        feat_e = imag_e.features
        with torch.no_grad():
            r_int = wm.p2e_intrinsic_reward(imag_e.traj, imag_e.actions)                    # (H, N)
            cont_e = imag_e.continue_dist.mean().reshape(H, -1)
        v_e_dist = critic_p2e(feat_e.reshape(H * feat_e.shape[1], -1))
        v_e = v_e_dist.mean().reshape(H, -1)
        v_e_ext = torch.cat([v_e.detach(), v_e.detach()[-1:]], dim=0)
        R_lambda_e = lambda_returns(r_int.detach(), v_e_ext.detach(), cont_e.detach(),
                                    gamma=cfg.gamma, lam=cfg.lam)
        S_e = ret_scale_p2e.update(R_lambda_e)
        adv_e = (R_lambda_e - v_e.detach()) / max(1.0, S_e)
        L_actor_p2e = -(adv_e.detach() * imag_e.log_pi + cfg.entropy_eta * imag_e.entropy).mean()

        L_critic_p2e_main = _twohot_ce(v_e_dist, R_lambda_e.detach().reshape(-1))
        with torch.no_grad():
            v_e_target = critic_p2e_ema(feat_e.reshape(H * feat_e.shape[1], -1)).mean().detach()
        L_critic_p2e_reg = _twohot_ce(v_e_dist, v_e_target)
        L_critic_p2e = L_critic_p2e_main + L_critic_p2e_reg

        # Combined backward — same reasoning as task actor/critic above.
        opt_actor_p2e.zero_grad(set_to_none=True)
        opt_critic_p2e.zero_grad(set_to_none=True)
        (L_actor_p2e + L_critic_p2e).backward()
        nn.utils.clip_grad_norm_(actor_p2e.parameters(), cfg.actor_grad_clip)
        nn.utils.clip_grad_norm_(critic_p2e.parameters(), cfg.critic_grad_clip)
        opt_actor_p2e.step()
        opt_critic_p2e.step()
        critic_p2e_ema.update(critic_p2e)

    return {
        "L_wm": L_wm.item(),
        "L_pred": L_pred.item(),
        "L_dyn": L_dyn.item(),
        "L_rep": L_rep.item(),
        "L_ensemble": L_ensemble.item(),
        "L_actor": L_actor.item(),
        "L_critic": L_critic.item(),
        "L_actor_p2e": L_actor_p2e.item(),
        "L_critic_p2e": L_critic_p2e.item(),
        "policy_entropy": imag.entropy.mean().item(),
        "ret_scale_task": ret_scale_task.scale,
    }


def _save_checkpoint(ckpt_dir, env_step, grad_step, wm, actor, critic, critic_ema, actor_p2e, critic_p2e, tag: str | None = None):
    name = f"step_{env_step:07d}.pt" if tag is None else f"{tag}.pt"
    blob = {
        "env_step": env_step,
        "grad_step": grad_step,
        "wm": wm.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "critic_ema": critic_ema.state_dict(),
    }
    if actor_p2e is not None:
        blob["actor_p2e"] = actor_p2e.state_dict()
        blob["critic_p2e"] = critic_p2e.state_dict()
    torch.save(blob, ckpt_dir / name)
