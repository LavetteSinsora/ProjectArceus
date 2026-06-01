"""exp_013 unified training loop: dual-stream PPO + pluggable intrinsic bonus.

One `train(cfg)` runs a single (method × game × level × seed):

    collect rollout -> bonus.compute (raw novelty) -> normalise by running std of
    intrinsic RETURNS (shared RND normaliser) -> per-stream GAE (ext episodic,
    int non-episodic) -> dual-stream PPO on the actor-critic -> bonus.update
    (the intrinsic net trains with its OWN optimiser) -> log.

Stop rule (exp_013): stop the instant the FIRST extrinsic reward appears
(`stop_on_first_reward`), else run to the `max_env_steps` cap and report the run
as right-censored. The headline number `env_steps_to_first_reward` is written to
`<run_dir>/result.json`.
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.metrics import MetricsWriter
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.model import ActorCriticRND
from JEPA.experiments.exp_012_ls20_rnd.shared.rollout import collect_rollout, compute_gae
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter
from JEPA.experiments.exp_012_ls20_rnd.shared.evaluator import evaluate

from .intrinsic import make_bonus
from .ppo import ppo_update


def _repo_root() -> Path:
    # shared -> exp_013_.. -> experiments -> JEPA -> Code Repo
    return Path(__file__).resolve().parents[4]


class _EMAStd:
    """EMA of the variance of a scalar stream → a std that tracks the CURRENT
    scale. Unlike a cumulative RMS (exp_012's RunningMeanStd, whose count never
    decays), a single startup transient does not pin it forever — the fix for
    the ICM normaliser collapse documented in SMOKE_TEST_FINDINGS.md."""

    def __init__(self, decay: float):
        self.decay = decay
        self.var: float | None = None

    def update(self, x) -> None:
        v = float(np.asarray(x, dtype=np.float64).var())
        self.var = v if self.var is None else self.decay * self.var + (1.0 - self.decay) * v

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var)) if self.var is not None else 1.0


def train(cfg, smoke: bool = False) -> dict:
    if smoke:
        cfg = cfg.smoke()
    device = get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    exp_dir = _repo_root() / cfg.exp_dir
    run_name = f"{cfg.exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = exp_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = MetricsWriter(run_dir)

    envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=cfg.seed,
                           level_index=cfg.level_index)
    if cfg.n_actions is None:
        cfg.n_actions = envs.n_actions
    assert cfg.n_actions == envs.n_actions, (
        f"cfg.n_actions={cfg.n_actions} but env reports {envs.n_actions} for {cfg.game}")
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    eval_envs = None
    if cfg.eval_every > 0:
        eval_envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=cfg.n_envs,
                                    max_episode_steps=cfg.max_episode_steps,
                                    seed=cfg.seed + 777, level_index=cfg.level_index)

    print(f"[exp013] {cfg.exp_name}  device={device}  n_actions={cfg.n_actions}  "
          f"cap={cfg.max_env_steps}  stop_on_first_reward={cfg.stop_on_first_reward}")

    model = ActorCriticRND(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                           frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    bonus = make_bonus(cfg, device)

    rff = RewardForwardFilter(cfg.gamma_int)
    int_ret_std = _EMAStd(cfg.int_norm_decay)

    global_step = 0
    first_reward_step: int | None = None
    t_start = time.time()
    stop_now = False

    for update in range(1, cfg.total_updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)

        # ── intrinsic reward: raw novelty -> warm-up + EMA-std normalisation ──
        # The bonus net is ALWAYS trained (below), but during warm-up the policy
        # gets ZERO intrinsic reward (an untrained predictor's error is not
        # "surprise") and the normaliser is not polluted by the startup transient.
        raw_i = bonus.compute(rollout, device)                 # (T, N) np float32
        T, N = raw_i.shape
        warming = update <= cfg.norm_warmup_updates
        if warming:
            norm_i = np.zeros_like(raw_i)
        else:
            rems = np.stack([rff.update(raw_i[t]) for t in range(T)])
            int_ret_std.update(rems)                           # EMA std of returns
            norm_i = raw_i / (int_ret_std.std + cfg.int_norm_eps)  # scale, no centering
            if cfg.int_reward_clip is not None:
                norm_i = np.clip(norm_i, 0.0, cfg.int_reward_clip)
        rollout.rewards_int = torch.from_numpy(norm_i.astype(np.float32))

        rollout.adv_ext, rollout.ret_ext = compute_gae(
            rollout.rewards_ext, rollout.values_ext, rollout.bootstrap_value_ext,
            rollout.dones, cfg.gamma_ext, cfg.gae_lambda, episodic=True)
        rollout.adv_int, rollout.ret_int = compute_gae(
            rollout.rewards_int, rollout.values_int, rollout.bootstrap_value_int,
            rollout.dones, cfg.gamma_int, cfg.gae_lambda, episodic=False)

        ustats = ppo_update(model, optimizer, rollout, cfg, device)
        bstats = bonus.update(rollout, device)
        global_step += cfg.rollout_steps * cfg.n_envs

        # ── headline: precise env-step of the FIRST extrinsic reward ─────────
        extrinsic = rollout.rewards_ext
        if first_reward_step is None and bool((extrinsic > 0).any().item()):
            t_idx = int((extrinsic > 0).float().sum(dim=1).nonzero()[0].item())
            base = global_step - cfg.rollout_steps * cfg.n_envs
            first_reward_step = base + (t_idx + 1) * cfg.n_envs
            print(f"[exp013] *** FIRST REWARD at ~{first_reward_step} env steps "
                  f"(update {update}, random baseline e.g. LS20-L1 ~50,000) ***")
            if cfg.stop_on_first_reward:
                stop_now = True

        done_eps = envs.drain_completed_episodes()
        train_succ = (float(np.mean([e.success for e in done_eps]))
                      if done_eps else float("nan"))

        record = {
            "step": global_step, "update": update,
            "policy_loss": ustats.policy_loss,
            "value_loss_ext": ustats.value_loss_ext,
            "value_loss_int": ustats.value_loss_int,
            "policy_entropy": ustats.entropy,
            "approx_kl": ustats.approx_kl,
            "grad_norm_total": ustats.grad_norm_total,
            "intrinsic_reward_raw_mean": float(raw_i.mean()),
            "intrinsic_reward_raw_std": float(raw_i.std()),
            "intrinsic_reward_norm_mean": float(norm_i.mean()),
            "intrinsic_return_std": int_ret_std.std,
            "norm_warming": bool(warming),
            "v_ext_mean": float(rollout.values_ext.mean()),
            "v_int_mean": float(rollout.values_int.mean()),
            "train_success_rate": train_succ,
            "train_episodes": len(done_eps),
            "env_steps_to_first_reward": first_reward_step,
            "sps": global_step / max(1e-6, time.time() - t_start),
            **bstats,
        }
        if cfg.eval_every > 0 and (update % cfg.eval_every == 0 or stop_now):
            record.update(evaluate(model, eval_envs, device, cfg.eval_episodes))
        if cfg.log_every > 0 and (update % cfg.log_every == 0 or stop_now):
            writer.write(record)

        if update % 25 == 0 or stop_now:
            print(f"[exp013] {cfg.exp_name} update {update}/{cfg.total_updates} "
                  f"step={global_step} first_reward_step={first_reward_step} "
                  f"r^i_raw={record['intrinsic_reward_raw_mean']:.4g} "
                  f"entropy={ustats.entropy:.3f}")

        if stop_now:
            break

    writer.close()
    solved = first_reward_step is not None
    result = {
        "exp_name": cfg.exp_name, "method": cfg.method, "game": cfg.game,
        "level_index": cfg.level_index, "seed": cfg.seed,
        "env_steps_to_first_reward": first_reward_step,
        "solved": solved,
        "censored": not solved,                       # never solved within the cap
        "total_env_steps": global_step,
        "max_env_steps": cfg.max_env_steps,
        "wall_seconds": time.time() - t_start,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[exp013] DONE {cfg.exp_name}: "
          f"{'first reward @ ' + str(first_reward_step) if solved else 'CENSORED (no reward)'} "
          f"env steps; total={global_step}")
    return result
