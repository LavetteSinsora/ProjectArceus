"""Entry point for exp_007_3_jepa_sg.

CNN encoder is decoupled from PPO:
  - Policy + value heads see model.encoder(obs).detach() (or equivalently a
    no_grad encoder forward) — they never push gradients into the encoder.
  - Encoder + predictor are trained jointly by an action-conditioned
    self-prediction loss: MSE(predictor(h_t, a_t), sg(h_{t+1})).

Two optimizers, separated by parameter group:
  opt_pp  — policy_head + value_head
  opt_enc — encoder + predictor

Usage:
    uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.train
    uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.train --smoke
    uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.train --short
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.config_base import Config as _Base
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.device import pick_device
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.model import ActorCritic, one_hot_frame
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.vec_env import VecMiniEnv
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.rewards import make_shaping_fn
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.rollout import (
    Rollout, collect_rollout, compute_gae,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.metrics import (
    mean_feature_cosine, run_eval_episodes, summarise_completed,
)

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.config import Config
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.models import (
    ActionConditionedPredictor,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.diagnostics import (
    all_diagnostics,
)


# ───────────────────────────────────────────────────────────────────────────
# PPO update with stop-gradient encoder
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    grad_norm_pp: float


def _ppo_update_sg(model: ActorCritic, opt_pp: torch.optim.Optimizer,
                   rollout: Rollout, cfg: Config, device: torch.device,
                   assert_no_encoder_grad: bool = False) -> PPOStats:
    """PPO update over policy + value heads only; encoder is no_grad."""
    T, N = rollout.actions.shape
    batch_size = T * N
    mb_size = batch_size // cfg.minibatches

    b_obs = rollout.obs.reshape(batch_size, 32, 32)
    b_actions = rollout.actions.reshape(batch_size)
    b_logp_old = rollout.log_probs.reshape(batch_size)
    b_advantages = rollout.advantages.reshape(batch_size)
    b_returns = rollout.returns.reshape(batch_size)
    b_values_old = rollout.values.reshape(batch_size)

    pp_params = list(model.policy_head.parameters()) + list(model.value_head.parameters())

    pl_sum = vl_sum = ent_sum = kl_sum = clip_sum = gn_sum = 0.0
    n_updates = 0

    idx = np.arange(batch_size)
    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for start in range(0, batch_size, mb_size):
            mb = idx[start:start + mb_size]
            mb_obs = b_obs[mb].to(device)
            mb_actions = b_actions[mb].to(device)
            mb_logp_old = b_logp_old[mb].to(device)
            mb_advantages = b_advantages[mb].to(device)
            mb_returns = b_returns[mb].to(device)
            mb_values_old = b_values_old[mb].to(device)

            a = mb_advantages
            mb_advantages = (a - a.mean()) / (a.std() + 1e-8)

            # ── encoder forward with NO autograd graph ──
            with torch.no_grad():
                feat_sg = model.encoder(one_hot_frame(mb_obs))

            logits = model.policy_head(feat_sg)
            value_new = model.value_head(feat_sg).squeeze(-1)
            dist = torch.distributions.Categorical(logits=logits)
            logp_new = dist.log_prob(mb_actions)
            entropy = dist.entropy()

            ratio = (logp_new - mb_logp_old).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            if cfg.vf_clip_eps is not None:
                v_clipped = mb_values_old + torch.clamp(
                    value_new - mb_values_old, -cfg.vf_clip_eps, cfg.vf_clip_eps
                )
                vl_unclipped = (value_new - mb_returns).pow(2)
                vl_clipped = (v_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
            else:
                value_loss = 0.5 * (value_new - mb_returns).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + cfg.c_value * value_loss + cfg.c_entropy * entropy_loss

            opt_pp.zero_grad(set_to_none=True)
            # Also zero encoder grads to make the stop-gradient assertion
            # checkable (so we know any non-zero is from THIS backward).
            model.encoder.zero_grad(set_to_none=True)
            loss.backward()

            if assert_no_encoder_grad:
                for p in model.encoder.parameters():
                    assert p.grad is None or float(p.grad.abs().sum()) == 0.0, (
                        "PPO step leaked gradient into encoder")

            gn = nn.utils.clip_grad_norm_(pp_params, cfg.grad_clip)
            opt_pp.step()

            with torch.no_grad():
                approx_kl = (mb_logp_old - logp_new).mean()
                clipfrac = ((ratio - 1).abs() > cfg.clip_eps).float().mean()

            pl_sum += policy_loss.item()
            vl_sum += value_loss.item()
            ent_sum += (-entropy_loss).item()
            kl_sum += approx_kl.item()
            clip_sum += clipfrac.item()
            gn_sum += float(gn)
            n_updates += 1

    return PPOStats(
        policy_loss=pl_sum / n_updates,
        value_loss=vl_sum / n_updates,
        entropy=ent_sum / n_updates,
        approx_kl=kl_sum / n_updates,
        clipfrac=clip_sum / n_updates,
        grad_norm_pp=gn_sum / n_updates,
    )


# ───────────────────────────────────────────────────────────────────────────
# JEPA update (encoder + predictor)
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class JEPAStats:
    jepa_loss: float
    grad_norm_encoder: float
    valid_fraction: float


def _jepa_update(model: ActorCritic, predictor: ActionConditionedPredictor,
                 opt_enc: torch.optim.Optimizer, rollout: Rollout, cfg: Config,
                 device: torch.device,
                 assert_no_pv_grad: bool = False) -> JEPAStats:
    """Train encoder + predictor on MSE(predictor(h_t, a_t), sg(h_{t+1}))."""
    T, N = rollout.actions.shape
    if T < 2:
        return JEPAStats(jepa_loss=float("nan"), grad_norm_encoder=0.0,
                         valid_fraction=0.0)

    # Pairs over t in [0, T-2]. Drop the last rollout step (no obs_{t+1}
    # available within the buffer); ≈ 1/T of transitions lost.
    obs_t = rollout.obs[:-1].reshape(-1, 32, 32)          # ((T-1)·N, 32, 32)
    obs_tp1 = rollout.obs[1:].reshape(-1, 32, 32)
    actions_t = rollout.actions[:-1].reshape(-1)
    # If dones[t] is True the vec env auto-reset means obs[t+1] is from a new
    # episode — that transition is invalid for predictor training.
    valid = (~rollout.dones[:-1]).reshape(-1)

    n = obs_t.shape[0]
    n_valid = int(valid.sum().item())
    valid_fraction = n_valid / max(1, n)

    if n_valid == 0:
        return JEPAStats(jepa_loss=float("nan"), grad_norm_encoder=0.0,
                         valid_fraction=0.0)

    enc_params = list(model.encoder.parameters()) + list(predictor.parameters())
    pv_params = list(model.policy_head.parameters()) + list(model.value_head.parameters())

    mb_size = max(1, n // cfg.minibatches)

    loss_sum = gn_sum = 0.0
    n_updates = 0

    idx = np.arange(n)
    for _ in range(cfg.jepa_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, mb_size):
            mb = idx[start:start + mb_size]
            v = valid[mb]
            if not bool(v.any()):
                continue

            mb_obs_t = obs_t[mb].to(device)
            mb_obs_tp1 = obs_tp1[mb].to(device)
            mb_actions = actions_t[mb].to(device)
            mb_valid = v.to(device).float()

            h_t = model.encoder(one_hot_frame(mb_obs_t))                    # grads ON
            with torch.no_grad():
                h_tp1 = model.encoder(one_hot_frame(mb_obs_tp1))            # target: sg

            pred = predictor(h_t, mb_actions)
            sq = (pred - h_tp1).pow(2).sum(dim=-1)                          # per-sample
            denom = mb_valid.sum().clamp(min=1.0)
            jepa_loss = (sq * mb_valid).sum() / denom

            opt_enc.zero_grad(set_to_none=True)
            # Zero pv grads to make the stop-gradient assertion checkable.
            for p in pv_params:
                p.grad = None
            jepa_loss.backward()

            if assert_no_pv_grad:
                for p in pv_params:
                    assert p.grad is None or float(p.grad.abs().sum()) == 0.0, (
                        "JEPA step leaked gradient into policy/value heads")

            gn = nn.utils.clip_grad_norm_(enc_params, cfg.jepa_grad_clip)
            opt_enc.step()

            loss_sum += float(jepa_loss.item())
            gn_sum += float(gn)
            n_updates += 1

    if n_updates == 0:
        return JEPAStats(jepa_loss=float("nan"), grad_norm_encoder=0.0,
                         valid_fraction=valid_fraction)
    return JEPAStats(
        jepa_loss=loss_sum / n_updates,
        grad_norm_encoder=gn_sum / n_updates,
        valid_fraction=valid_fraction,
    )


# ───────────────────────────────────────────────────────────────────────────
# Outer loop
# ───────────────────────────────────────────────────────────────────────────


def _make_run_dir(cfg: Config, extra_tag: str | None = None) -> Path:
    base = Path(cfg.runs_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    vf_tag = (f"vfclip{cfg.vf_clip_eps:g}".replace(".", "p")
              if cfg.vf_clip_eps is not None else "novfclip")
    parts = [cfg.exp_name, vf_tag]
    if extra_tag:
        parts.append(extra_tag)
    parts.append(ts)
    run = base / "_".join(parts)
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    return run


def _log_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def train(cfg: Config, max_updates: int | None = None,
          assert_grads: bool = False, tag: str | None = None) -> Path:
    """Full training loop. Returns the run directory."""
    device = pick_device()
    print(f"[exp_007_3] device={device}  exp={cfg.exp_name}  mode={cfg.reward_mode}  tag={tag}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = _make_run_dir(cfg, extra_tag=tag)
    print(f"[exp_007_3] run_dir={run_dir}")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    envs = VecMiniEnv(cfg.level_path, n_envs=cfg.n_envs, seed=cfg.seed)
    envs.reset_all()

    model = ActorCritic().to(device)
    predictor = ActionConditionedPredictor(
        d_feat=model.encoder.trunk_dim,
        n_actions=model.n_actions,
        d_action=cfg.d_action,
        hidden=cfg.predictor_hidden,
    ).to(device)

    opt_pp = torch.optim.Adam(
        list(model.policy_head.parameters()) + list(model.value_head.parameters()),
        lr=cfg.policy_value_lr,
    )
    opt_enc = torch.optim.Adam(
        list(model.encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.encoder_lr,
    )

    shape_fn = make_shaping_fn(
        cfg.reward_mode,
        wall_penalty=cfg.wall_penalty,
        match_bonus=cfg.match_bonus,
        unmatch_penalty=cfg.unmatch_penalty,
    )

    log_path = run_dir / "metrics.jsonl"
    total_updates = max_updates if max_updates is not None else cfg.total_updates
    print(f"[exp_007_3] total_updates={total_updates}  rollout_steps={cfg.rollout_steps}  n_envs={cfg.n_envs}")

    env_step = 0
    t0 = time.time()

    for update in range(1, total_updates + 1):
        rollout = collect_rollout(envs, model, device=device,
                                   T=cfg.rollout_steps, shape_fn=shape_fn)
        env_step += cfg.rollout_steps * cfg.n_envs

        completed = envs.drain_completed_episodes()
        train_summary = summarise_completed(completed)

        rollout = compute_gae(rollout, gamma=cfg.gamma, lam=cfg.gae_lambda)

        feat_cos = mean_feature_cosine(rollout)

        # PPO update — encoder is no_grad'd, only policy+value heads learn.
        ppo_stats = _ppo_update_sg(model, opt_pp, rollout, cfg, device=device,
                                    assert_no_encoder_grad=assert_grads)

        # JEPA update — encoder + predictor learn from MSE(predictor(h_t, a_t), sg(h_{t+1})).
        jepa_stats = _jepa_update(model, predictor, opt_enc, rollout, cfg,
                                   device=device,
                                   assert_no_pv_grad=assert_grads)

        # Collapse diagnostics on rollout features (already detached, on CPU).
        feat_flat = rollout.features.reshape(-1, rollout.features.shape[-1])
        diag = all_diagnostics(feat_flat)

        rec = {
            "update": update,
            "env_step": env_step,
            "policy_loss": ppo_stats.policy_loss,
            "value_loss": ppo_stats.value_loss,
            "entropy": ppo_stats.entropy,
            "approx_kl": ppo_stats.approx_kl,
            "clipfrac": ppo_stats.clipfrac,
            "grad_norm_total": ppo_stats.grad_norm_pp,   # alias for dashboard
            "grad_norm_pp": ppo_stats.grad_norm_pp,
            "grad_norm_encoder": jepa_stats.grad_norm_encoder,
            "jepa_loss": jepa_stats.jepa_loss,
            "jepa_valid_fraction": jepa_stats.valid_fraction,
            "mean_feature_cosine": feat_cos,
            "feat_std": diag["feat_std"],
            "feat_pairwise_l2": diag["feat_pairwise_l2"],
            "feat_effective_rank": diag["feat_effective_rank"],
            "wall_clock_s": time.time() - t0,
        }
        rec.update(train_summary)

        if update % cfg.eval_every == 0 or update == 1:
            eval_metrics = run_eval_episodes(cfg.level_path, model, device=device,
                                              n_episodes=cfg.eval_episodes)
            rec.update(eval_metrics)
            print(f"[exp_007_3] update={update:5d} env_step={env_step:9d} "
                  f"eval_success={eval_metrics['eval_success_rate']:.2f} "
                  f"entropy={ppo_stats.entropy:.3f} feat_cos={feat_cos:.3f} "
                  f"feat_std={diag['feat_std']:.3f} eff_rank={diag['feat_effective_rank']:.1f} "
                  f"jepa_loss={jepa_stats.jepa_loss:.4f}")

        _log_jsonl(log_path, rec)

        if update % cfg.save_every == 0 or update == total_updates:
            ckpt_path = run_dir / "checkpoints" / f"update_{update:06d}.pt"
            torch.save({
                "update": update,
                "env_step": env_step,
                "model_state_dict": model.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "opt_pp_state_dict": opt_pp.state_dict(),
                "opt_enc_state_dict": opt_enc.state_dict(),
                "config": asdict(cfg),
            }, ckpt_path)

    final_path = run_dir / "checkpoints" / "final.pt"
    torch.save({
        "update": total_updates,
        "env_step": env_step,
        "model_state_dict": model.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "opt_pp_state_dict": opt_pp.state_dict(),
        "opt_enc_state_dict": opt_enc.state_dict(),
        "config": asdict(cfg),
    }, final_path)

    print(f"[exp_007_3] done. run_dir={run_dir}")
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="run 5 updates only (with grad-flow asserts)")
    p.add_argument("--short", action="store_true", help="run 50 updates only")
    p.add_argument("--updates", type=int, default=None, help="override total updates")
    p.add_argument("--tag", type=str, default=None, help="extra suffix on run dir name")
    args = p.parse_args()

    cfg = Config()
    if args.smoke:
        max_updates = 5
        assert_grads = True
    elif args.short:
        max_updates = 50
        assert_grads = False
    else:
        max_updates = args.updates
        assert_grads = False
    train(cfg, max_updates=max_updates, assert_grads=assert_grads, tag=args.tag)


if __name__ == "__main__":
    main()
