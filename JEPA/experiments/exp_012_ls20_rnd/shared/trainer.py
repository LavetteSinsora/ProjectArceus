"""RND dual-stream PPO training loop for exp_012_1.

One entry point, `train(cfg)`:
    collect rollout -> compute intrinsic reward (RND) + normalise
    -> per-stream GAE (extrinsic episodic, intrinsic non-episodic)
    -> dual-stream PPO update (+ predictor distillation)
    -> eval -> log -> checkpoint.

Compute: tuned for Apple-silicon MPS (and CUDA on Colab). We (a) pick the best
device, (b) let Torch use every CPU core for the env loop / CPU-side ops, and
(c) compute the intrinsic reward in one batched GPU pass per rollout. Env
stepping (~1.6k steps/s/env) is not the bottleneck per exp_010; the gradient
step dominates, so we keep collection serial and feed the GPU a large batch
(n_envs=16 -> 2048 transitions/update).

Artifacts (dashboard-compatible layout):
    <exp_dir>/checkpoints/step_<env_step>.pt
    <exp_dir>/runs/<run_name>/metrics.jsonl
    <exp_dir>/runs/<run_name>/config.json
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Unchanged infrastructure reused from exp_010 (single source of truth).
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.metrics import (
    MetricsWriter, mean_feature_cosine, feature_health,
)

from .ls20_vec_env import MultiLevelVecLS20Env
from .model import ActorCriticRND
from .rnd import (RNDTarget, RNDPredictor, batched_features,
                  intrinsic_from_features, RunningMeanStd, RewardForwardFilter)
from .rollout import collect_rollout, compute_gae
from .ppo import ppo_update
from .evaluator import evaluate


def _repo_root() -> Path:
    # shared -> exp_012_.. -> experiments -> JEPA -> Code Repo
    return Path(__file__).resolve().parents[4]


def _setup_compute() -> torch.device:
    """Use every CPU core for Torch CPU ops / the env loop, and pick the device."""
    n_cpu = os.cpu_count() or 1
    torch.set_num_threads(n_cpu)
    try:
        torch.set_num_interop_threads(max(1, n_cpu // 2))
    except RuntimeError:
        pass  # interop thread count can only be set once per process
    return get_device()


def save_checkpoint(cfg, model, target, predictor, global_step, update, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{global_step:08d}.pt"
    torch.save({
        "step": int(global_step),
        "update": int(update),
        "config": dataclasses.asdict(cfg),
        "model": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "rnd_target": target.state_dict(),
        "rnd_predictor": predictor.state_dict(),
    }, path)
    return path


def train(cfg, smoke: bool = False):
    if smoke:
        cfg = cfg.smoke()
    device = _setup_compute()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    exp_dir = _repo_root() / cfg.exp_dir
    run_name = f"{cfg.exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = exp_dir / "runs" / run_name
    ckpt_dir = exp_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    writer = MetricsWriter(run_dir)

    print(f"[exp012_1] device={device}  threads={torch.get_num_threads()}  run={run_name}")
    print(f"[exp012_1] env={cfg.env_name}  n_envs={cfg.n_envs}  "
          f"updates={cfg.total_updates}  steps/update={cfg.rollout_steps*cfg.n_envs}")

    envs = MultiLevelVecLS20Env(cfg.env_name, n_envs=cfg.n_envs,
                                max_episode_steps=cfg.max_episode_steps,
                                stop_levels=cfg.stop_levels, seed=cfg.seed)
    eval_envs = MultiLevelVecLS20Env(cfg.env_name, n_envs=cfg.n_envs,
                                     max_episode_steps=cfg.max_episode_steps,
                                     stop_levels=cfg.stop_levels, seed=cfg.seed + 777)

    model = ActorCriticRND(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                           frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    target = RNDTarget(n_colors=cfg.n_colors, frame_size=cfg.frame_size,
                       feature_dim=cfg.rnd_feature_dim).to(device)
    predictor = RNDPredictor(n_colors=cfg.n_colors, frame_size=cfg.frame_size,
                             feature_dim=cfg.rnd_feature_dim,
                             hidden=cfg.rnd_predictor_hidden).to(device)

    # Warm-start (e.g. continue an L1 agent into L2): load policy/value, the
    # frozen RND target, and the (L1-distilled) predictor so L1 states read as
    # familiar and the bonus drives exploration toward the novel L2 states.
    if cfg.init_ckpt:
        ckpt_path = cfg.init_ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = str(_repo_root() / ckpt_path)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        target.load_state_dict(ck["rnd_target"])
        predictor.load_state_dict(ck["rnd_predictor"])
        print(f"[exp012_1] warm-started model+RND from {ckpt_path} "
              f"(step {ck.get('step')})")

    # Optimiser trains policy/value heads + encoder + predictor (target is frozen).
    params = list(model.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    # RND intrinsic-reward normalisers.
    rff = RewardForwardFilter(cfg.gamma_int)
    int_ret_rms = RunningMeanStd()

    global_step = 0
    first_reward_step: int | None = None      # first +1 (any level cleared)
    first_success_step: int | None = None     # first episode reaching stop_levels
    t_start = time.time()
    consec_success = 0
    for update in range(1, cfg.total_updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)

        # ── Intrinsic reward: one batched GPU pass over next_obs, then normalise.
        # The frozen-target embedding is cached on the rollout and reused by
        # every PPO minibatch (no per-minibatch target recompute).
        T, N = rollout.actions.shape
        Fz = rollout.frame
        flat_next = rollout.next_obs.reshape(T * N, Fz, Fz)
        target_feats = batched_features(target, flat_next, device, cfg.rnd_feature_dim)
        pred_feats = batched_features(predictor, flat_next, device, cfg.rnd_feature_dim)
        rollout.target_feats = target_feats.reshape(T, N, cfg.rnd_feature_dim)
        raw_i = intrinsic_from_features(pred_feats, target_feats).reshape(T, N)
        # Update the running std of the intrinsic *returns* (RND normaliser).
        rems = np.stack([rff.update(raw_i[t]) for t in range(T)])  # (T, N)
        int_ret_rms.update(rems)
        norm_i = raw_i / (np.sqrt(int_ret_rms.var) + cfg.int_norm_eps)
        rollout.rewards_int = torch.from_numpy(norm_i.astype(np.float32))

        # ── Per-stream GAE: extrinsic episodic, intrinsic non-episodic.
        rollout.adv_ext, rollout.ret_ext = compute_gae(
            rollout.rewards_ext, rollout.values_ext, rollout.bootstrap_value_ext,
            rollout.dones, cfg.gamma_ext, cfg.gae_lambda, episodic=True)
        rollout.adv_int, rollout.ret_int = compute_gae(
            rollout.rewards_int, rollout.values_int, rollout.bootstrap_value_int,
            rollout.dones, cfg.gamma_int, cfg.gae_lambda, episodic=False)

        ustats = ppo_update(model, predictor, optimizer, rollout, cfg, device)
        global_step += cfg.rollout_steps * cfg.n_envs

        # First extrinsic reward = the headline exploration metric.
        if first_reward_step is None and float(rollout.rewards_ext.sum()) > 0.0:
            first_reward_step = global_step
            print(f"[exp012_1] *** first extrinsic reward at env_step={global_step} "
                  f"(random baseline ~50,000) ***")

        done_eps = envs.drain_completed_episodes()
        train_succ = (float(np.mean([e.success for e in done_eps]))
                      if done_eps else float("nan"))
        if first_success_step is None and any(e.success for e in done_eps):
            first_success_step = global_step
            print(f"[exp012_1] *** first success (cleared {cfg.stop_levels} level(s)) "
                  f"at env_step={global_step} ***")

        record = {
            "step": global_step,
            "update": update,
            "policy_loss": ustats.policy_loss,
            "value_loss": ustats.value_loss_ext + ustats.value_loss_int,
            "value_loss_ext": ustats.value_loss_ext,
            "value_loss_int": ustats.value_loss_int,
            "policy_entropy": ustats.entropy,
            "approx_kl": ustats.approx_kl,
            "clipfrac": ustats.clipfrac,
            "grad_norm_total": ustats.grad_norm_total,
            "rnd_predictor_loss": ustats.rnd_loss,
            "intrinsic_reward_mean": float(raw_i.mean()),
            "intrinsic_reward_std": float(raw_i.std()),
            "intrinsic_return_std": float(np.sqrt(int_ret_rms.var)),
            "v_ext_mean": float(rollout.values_ext.mean()),
            "v_int_mean": float(rollout.values_int.mean()),
            "adv_ext_mean": float(rollout.adv_ext.mean()),
            "adv_int_mean": float(rollout.adv_int.mean()),
            "mean_feature_cosine": mean_feature_cosine(rollout.features, rollout.ep_starts),
            "train_success_rate": train_succ,
            "train_episodes": len(done_eps),
            "env_steps_to_first_extrinsic_reward": first_reward_step,
            "env_steps_to_first_success": first_success_step,
            "sps": global_step / max(1e-6, time.time() - t_start),
        }

        is_eval = (update % cfg.eval_every == 0 or update == cfg.total_updates)
        if is_eval:
            record.update(evaluate(model, eval_envs, device, cfg.eval_episodes))
            record.update(feature_health(rollout.features))

        if update % cfg.log_every == 0:
            writer.write(record)

        stop_now = False
        if is_eval:
            esr = record.get("success_rate", float("nan"))
            print(f"[exp012_1] update {update}/{cfg.total_updates} step={global_step} "
                  f"eval_success_rate={esr if esr == esr else 'nan'} "
                  f"int_rew={record['intrinsic_reward_mean']:.4f} "
                  f"first_reward_step={first_reward_step}")
            if cfg.early_stop_enabled:
                consec_success = (consec_success + 1
                                  if (esr == esr and esr >= cfg.early_stop_success_rate)
                                  else 0)
                stop_now = consec_success >= cfg.early_stop_patience

        if update % cfg.save_every == 0 or update == cfg.total_updates or stop_now:
            p = save_checkpoint(cfg, model, target, predictor, global_step, update, ckpt_dir)
            print(f"[exp012_1]   saved {p.name}")

        if stop_now:
            print(f"[exp012_1] early stop: eval success_rate >= {cfg.early_stop_success_rate} "
                  f"for {cfg.early_stop_patience} consecutive evals (step={global_step})")
            break

    writer.close()
    print(f"[exp012_1] done. first extrinsic reward at env_step={first_reward_step}. "
          f"checkpoints in {ckpt_dir}")
    return ckpt_dir
