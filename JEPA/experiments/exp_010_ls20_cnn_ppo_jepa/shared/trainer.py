"""Central PPO(+optional online JEPA) training loop for exp_010.

One entry point, `train(cfg)`, drives all three sub-experiments:

  * exp_010_0  — cfg.jepa_mode="none"                         (plain CNN+PPO)
  * exp_010_1  — cfg.jepa_mode="online"                       (joint PPO + JEPA
                 on the agent's own rollout transitions)
  * exp_010_2  — cfg.jepa_mode="none" + cfg.init_encoder_ckpt (PPO from a
                 random-data-pretrained encoder, unfrozen by default)

Artifacts (dashboard-compatible layout):
    <exp_dir>/checkpoints/step_<env_step>.pt    — torch checkpoint (flat)
    <exp_dir>/runs/<run_name>/metrics.jsonl     — one JSON record per line
    <exp_dir>/runs/<run_name>/config.json       — config snapshot
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .device import get_device
from .model import ActorCritic, ActionConditionedPredictor, InverseDynamicsModel
from .ls20_vec_env import VecLS20Env
from .rollout import collect_rollout, compute_gae
from .ppo import PPOConfig, ppo_update
from .jepa import jepa_update_from_rollout
from .evaluator import evaluate
from .metrics import MetricsWriter, mean_feature_cosine, feature_health


def _repo_root() -> Path:
    # shared -> exp_010_.. -> experiments -> JEPA -> Code Repo
    return Path(__file__).resolve().parents[4]


def build_model(cfg, device):
    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    if cfg.init_encoder_ckpt:
        ck = torch.load(cfg.init_encoder_ckpt, map_location=device, weights_only=False)
        enc_sd = ck["encoder"] if isinstance(ck, dict) and "encoder" in ck else ck
        model.encoder.load_state_dict(enc_sd)
        print(f"[exp010] loaded encoder init from {cfg.init_encoder_ckpt}")
    if cfg.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.encoder.eval()
        print("[exp010] encoder FROZEN")
    return model


def save_checkpoint(cfg, model, predictor, idm, global_step, update, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{global_step:08d}.pt"
    torch.save({
        "step": int(global_step),
        "update": int(update),
        "config": dataclasses.asdict(cfg),
        "model": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "predictor": predictor.state_dict() if predictor is not None else None,
        "idm": idm.state_dict() if idm is not None else None,
    }, path)
    return path


def train(cfg, smoke: bool = False):
    if smoke:
        cfg = cfg.smoke()
    device = get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    exp_dir = _repo_root() / cfg.exp_dir
    run_name = f"{cfg.exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = exp_dir / "runs" / run_name
    ckpt_dir = exp_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    writer = MetricsWriter(run_dir)

    print(f"[exp010] device={device}  run={run_name}")
    print(f"[exp010] env={cfg.env_name}  jepa_mode={cfg.jepa_mode}  "
          f"updates={cfg.total_updates}  steps/update={cfg.rollout_steps*cfg.n_envs}")

    # Envs (separate train / eval so eval never perturbs training episode state).
    envs = VecLS20Env(cfg.env_name, n_envs=cfg.n_envs,
                      max_episode_steps=cfg.max_episode_steps, seed=cfg.seed)
    eval_envs = VecLS20Env(cfg.env_name, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=cfg.seed + 777)

    model = build_model(cfg, device)

    predictor = idm = None
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.jepa_mode == "online":
        predictor = ActionConditionedPredictor(cfg.trunk_dim, cfg.n_actions,
                                               cfg.action_emb_dim).to(device)
        idm = InverseDynamicsModel(cfg.trunk_dim, cfg.n_actions).to(device)
        params = params + list(predictor.parameters()) + list(idm.parameters())

    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps,
                        c_value=cfg.c_value, c_entropy=cfg.c_entropy,
                        grad_clip=cfg.grad_clip, epochs=cfg.epochs,
                        minibatches=cfg.minibatches)
    # JEPA update needs jepa_epochs/coefs/minibatches/grad_clip — pass cfg through.
    clip_params = [p for p in params]

    global_step = 0
    t_start = time.time()
    consec_success = 0          # consecutive evals at/above the early-stop threshold
    for update in range(1, cfg.total_updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)
        compute_gae(rollout, cfg.gamma, cfg.gae_lambda)
        ustats = ppo_update(model, optimizer, rollout, ppo_cfg, device,
                            clip_params=clip_params)
        global_step += cfg.rollout_steps * cfg.n_envs

        jstats = {}
        if cfg.jepa_mode == "online":
            jstats = jepa_update_from_rollout(model, predictor, idm, optimizer,
                                              rollout, cfg, device)

        # Drain episodes finished during this rollout for train-side stats.
        done_eps = envs.drain_completed_episodes()
        train_succ = (float(np.mean([e.success for e in done_eps]))
                      if done_eps else float("nan"))

        record = {
            "step": global_step,
            "update": update,
            "policy_loss": ustats.policy_loss,
            "value_loss": ustats.value_loss,
            "policy_entropy": ustats.entropy,
            "approx_kl": ustats.approx_kl,
            "clipfrac": ustats.clipfrac,
            "grad_norm_total": ustats.grad_norm_total,
            "mean_feature_cosine": mean_feature_cosine(rollout.features, rollout.ep_starts),
            "train_success_rate": train_succ,
            "train_episodes": len(done_eps),
            "sps": global_step / max(1e-6, time.time() - t_start),
            **jstats,
        }

        is_eval = (update % cfg.eval_every == 0 or update == cfg.total_updates)
        if is_eval:
            record.update(evaluate(model, eval_envs, device, cfg.eval_episodes))
            record.update(feature_health(rollout.features))

        if update % cfg.log_every == 0:
            writer.write(record)

        # Decide early stop on sustained eval success.
        stop_now = False
        if is_eval:
            esr = record.get("success_rate", float("nan"))
            print(f"[exp010] update {update}/{cfg.total_updates} step={global_step} "
                  f"eval_success_rate={esr if esr == esr else 'nan'} "
                  f"avg_steps_to_solve={record.get('avg_steps_to_solve')}")
            if cfg.early_stop_enabled:
                # esr == esr filters out NaN (no successful episodes).
                consec_success = (consec_success + 1
                                  if (esr == esr and esr >= cfg.early_stop_success_rate)
                                  else 0)
                stop_now = consec_success >= cfg.early_stop_patience

        if update % cfg.save_every == 0 or update == cfg.total_updates or stop_now:
            p = save_checkpoint(cfg, model, predictor, idm, global_step, update, ckpt_dir)
            print(f"[exp010]   saved {p.name}")

        if stop_now:
            print(f"[exp010] early stop: eval success_rate >= {cfg.early_stop_success_rate} "
                  f"for {cfg.early_stop_patience} consecutive evals (step={global_step})")
            break

    writer.close()
    print(f"[exp010] done. checkpoints in {ckpt_dir}")
    return ckpt_dir
