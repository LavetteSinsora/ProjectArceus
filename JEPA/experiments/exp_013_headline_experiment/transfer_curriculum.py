"""ACTUAL transfer — curriculum continuation L1→L2→L3 as ONE continued agent.

Instead of independent per-level runs, we build the agent ONCE (policy + ICM φ/inverse/
forward + RND predictor + value + optimisers) and CONTINUE it across levels: train on L1
until the first reward, then swap the env to L2 *keeping every network and the frozen φ*,
continue, then L3. This transfers BEHAVIOR (policy), the level-invariant game PHYSICS (ICM
dynamics), and the visitation memory (RND predictor) — not just the φ encoder (which the
encoder-only `--init-phi-ckpt` showed does nothing; see probes/transfer_analysis.md).

Metric: cumulative env-steps to the first reward at EACH level (the staircase, literally).
Compare to from-scratch L2/L3 (the frontier sweep: all censored) — a solve here is the
transfer win. φ is frozen once (during L1) and kept (the mechanics are shared across levels).
Timer-mask + all exp_013_1 fixes apply.

    uv run python -m JEPA.experiments.exp_013_headline_experiment.transfer_curriculum \
        --game ls20 --seed 0 --levels 0 1 2 --caps 200000 300000 300000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.metrics import MetricsWriter
from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule, icm_update_from_rollout
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter

from .exp_013_1b_leaky_rnd_on_icm_phi.config import Config, GAME_N_ACTIONS
from .exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi
from .exp_013_1b_leaky_rnd_on_icm_phi.trainer import (
    _EMAStd, _phi_and_novelty, _rnd_update, _gae_nonepisodic, _gae_episodic,
    _collect_holdout, _eval_holdout_inv_acc, _apply_timer_mask,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_curriculum(game: str, seed: int, levels: list[int], caps: list[int]) -> dict:
    cfg = Config(game=game, seed=seed, level_index=levels[0])
    device = get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cfg.n_actions is None:
        cfg.n_actions = GAME_N_ACTIONS[game]

    exp_dir = _repo_root() / "JEPA/experiments/exp_013_headline_experiment/transfer_runs"
    run_name = f"curriculum_{game}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = exp_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = MetricsWriter(run_dir)

    # ── build the agent ONCE; it persists across levels ──────────────────────
    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    icm = ICMModule(n_actions=cfg.n_actions, n_colors=cfg.n_colors, frame_size=cfg.frame_size,
                    trunk_dim=cfg.trunk_dim, hidden=cfg.icm_hidden).to(device)
    rndphi = RNDPhi(dim=cfg.trunk_dim, hidden=cfg.rnd_hidden, out=cfg.rnd_feature_dim,
                    leak=cfg.leak).to(device)
    if cfg.mask_timer:
        _apply_timer_mask(icm, cfg.timer_mask_rows)

    ppo_opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    icm_opt = torch.optim.Adam(icm.parameters(), lr=cfg.icm_lr)
    rnd_opt = torch.optim.Adam(rndphi.predictor.parameters(), lr=cfg.rnd_lr)
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps, c_value=cfg.c_value,
                        c_entropy=cfg.c_entropy, grad_clip=cfg.grad_clip, epochs=cfg.epochs,
                        minibatches=cfg.minibatches)
    rff = RewardForwardFilter(cfg.gamma)
    int_ret_std = _EMAStd(cfg.int_norm_decay)

    print(f"[curriculum] {run_name}  device={device}  mask_timer={cfg.mask_timer}  "
          f"levels={levels} caps={caps}", flush=True)

    phi_frozen = False
    inv_streak = 0
    raw_mean_ema: float | None = None
    global_step = 0
    update = 0
    t_start = time.time()
    stage_results = []

    for stage, (li, cap) in enumerate(zip(levels, caps)):
        envs = VecLS20EnvLevel(env_name=game, n_envs=cfg.n_envs, max_episode_steps=cfg.max_episode_steps,
                               seed=seed + stage, level_index=li)
        assert cfg.n_actions == envs.n_actions
        holdout = _collect_holdout(game, li, seed, cfg.holdout_size, device)
        stage_updates = cap // (cfg.rollout_steps * cfg.n_envs)
        stage_start_step = global_step
        first_reward_step = None
        last_holdout_inv = float("nan")
        print(f"[curriculum] === stage {stage}: L{li+1} (cap {cap}, {stage_updates} updates) "
              f"starting at cum_step={global_step} | φ_frozen={phi_frozen} ===", flush=True)

        for _ in range(stage_updates):
            update += 1
            rollout = collect_rollout(envs, model, device, cfg.rollout_steps)
            phi_cached, nov = _phi_and_novelty(icm, rndphi, rollout, device)
            raw_i = nov.numpy()
            T, N = raw_i.shape
            raw_mean_pre = float(raw_i.mean())

            warming = update <= cfg.norm_warmup_updates
            if warming:
                norm_i = np.zeros_like(raw_i)
            elif raw_mean_pre < getattr(cfg, "novelty_dead_eps", 0.0):
                norm_i = raw_i.astype(np.float32)        # dead field → no std-amplification
            else:
                if cfg.reward_clip_k is not None:
                    raw_mean_ema = (raw_mean_pre if raw_mean_ema is None
                                    else 0.99 * raw_mean_ema + 0.01 * raw_mean_pre)
                    if raw_mean_ema > 0:
                        raw_i = np.minimum(raw_i, cfg.reward_clip_k * raw_mean_ema)
                rems = np.stack([rff.update(raw_i[t]) for t in range(T)])
                int_ret_std.update(rems)
                norm_i = raw_i / (int_ret_std.std + cfg.int_norm_eps)

            extrinsic = rollout.rewards.clone()
            rollout.rewards = torch.from_numpy(norm_i.astype(np.float32))
            (_gae_episodic if cfg.intrinsic_episodic else _gae_nonepisodic)(rollout, cfg.gamma, cfg.gae_lambda)
            ustats = ppo_update(model, ppo_opt, rollout, ppo_cfg, device)

            if not phi_frozen:
                icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)
                last_holdout_inv = _eval_holdout_inv_acc(icm, holdout, device)
                trig = last_holdout_inv if cfg.freeze_metric == "holdout" else last_holdout_inv
                inv_streak = inv_streak + 1 if trig >= cfg.phi_freeze_inverse_acc else 0
                controllable = last_holdout_inv >= cfg.phi_uncontrollable_factor / cfg.n_actions
                if (inv_streak >= cfg.phi_freeze_patience or update >= cfg.phi_freeze_max_updates) and controllable:
                    for p in icm.phi.parameters():
                        p.requires_grad_(False)
                    icm.phi.eval(); phi_frozen = True
                    print(f"[curriculum] φ FROZEN @u{update} (L{li+1}) holdout_inv={last_holdout_inv:.3f}", flush=True)
            elif update % 10 == 0:
                last_holdout_inv = _eval_holdout_inv_acc(icm, holdout, device)   # monitor frozen-φ quality on new level

            _rnd_update(rndphi, rnd_opt, phi_cached, rollout.dones, cfg, device)
            global_step += cfg.rollout_steps * cfg.n_envs

            if first_reward_step is None and bool((extrinsic > 0).any().item()):
                t_idx = int((extrinsic > 0).float().sum(dim=1).nonzero()[0].item())
                first_reward_step = (global_step - cfg.rollout_steps * cfg.n_envs) + (t_idx + 1) * cfg.n_envs
                print(f"[curriculum] *** L{li+1} FIRST REWARD at cum_step ~{first_reward_step} "
                      f"(stage steps {first_reward_step - stage_start_step}) ***", flush=True)

            writer.write({"step": global_step, "update": update, "level": li, "stage": stage,
                          "policy_entropy": ustats.entropy, "novelty_raw_mean": raw_mean_pre,
                          "intrinsic_reward_norm_mean": float(norm_i.mean()),
                          "holdout_inv_acc": last_holdout_inv, "phi_frozen": bool(phi_frozen),
                          "stage_first_reward_step": first_reward_step,
                          "sps": global_step / max(1e-6, time.time() - t_start)})
            if update % 25 == 0:
                print(f"[curriculum] L{li+1} u{update} cum={global_step} frr={first_reward_step} "
                      f"nov={raw_mean_pre:.4g} ent={ustats.entropy:.3f} hold={last_holdout_inv:.2f} "
                      f"frozen={phi_frozen}", flush=True)
            if first_reward_step is not None:
                break   # cleared this level → CONTINUE the warm agent to the next

        solved = first_reward_step is not None
        stage_results.append({"level": li, "solved": solved,
                              "cum_step_to_reward": first_reward_step,
                              "stage_steps": (first_reward_step - stage_start_step) if solved else None,
                              "cap": cap, "holdout_inv_acc": last_holdout_inv})
        print(f"[curriculum] stage {stage} L{li+1} {'SOLVED @cum ' + str(first_reward_step) if solved else 'CENSORED'}", flush=True)
        if not solved:
            print(f"[curriculum] did not clear L{li+1} within cap → stopping curriculum.", flush=True)
            break

    writer.close()
    result = {"run": run_name, "game": game, "seed": seed, "levels": levels, "caps": caps,
              "mask_timer": cfg.mask_timer, "stages": stage_results,
              "total_env_steps": global_step, "wall_seconds": time.time() - t_start}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[curriculum] DONE {run_name}: " +
          " ".join(f"L{s['level']+1}={'✓'+str(s['cum_step_to_reward']) if s['solved'] else '✗'}" for s in stage_results), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description="L1→L2→L3 curriculum-continuation transfer")
    p.add_argument("--game", choices=list(GAME_N_ACTIONS), default="ls20")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--caps", type=int, nargs="+", default=[200_000, 300_000, 300_000])
    args = p.parse_args()
    assert len(args.levels) == len(args.caps), "one cap per level"
    print(run_curriculum(args.game, args.seed, args.levels, args.caps))


if __name__ == "__main__":
    main()
