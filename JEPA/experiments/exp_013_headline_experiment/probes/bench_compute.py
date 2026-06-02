"""Compute-efficiency probe for the exp_013 method sweep.

NOT a method-code change — it imports the real trainers' building blocks
(env, models, collect_rollout, update fns) and TIMES them, splitting each
update into:
    t_collect  — the synchronous N-env Python step loop + per-step GPU forward
    t_update   — the batched GPU gradient step(s) (PPO/value + ICM + RND)

It sweeps n_envs ∈ {8,16,32,64} for two representative methods:
    exp_013_1b_leaky_rnd_on_icm_phi  — the shared PPO+ICM+RND shape (methods 1/2/4 are ~this)
    exp_013_3_mcts_lookahead — the heaviest: A forward-model evals PER env step

Usage:
    uv run python -m JEPA.experiments.exp_013_headline_experiment.probes.bench_compute \
        --updates 6 --warmup 2 --n-envs 8,16,32,64 --method both

Keep it cheap: a handful of updates, no env-step cap reached, no files written
beyond stdout (which the caller captures).
"""

from __future__ import annotations

import argparse
import time
import json

import numpy as np
import torch


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_rnd_icm(n_envs, updates, warmup, rollout_steps, device):
    """Time exp_013_1 RND+ICM updates (PPO + ICM + RND-on-phi)."""
    from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.config import Config
    from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.trainer import (
        _EMAStd, _phi_and_novelty, _rnd_update, _gae_nonepisodic,
    )
    from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
    from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule, icm_update_from_rollout
    from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
    from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter

    cfg = Config(game="ls20", level_index=0, seed=0)
    cfg.n_envs = n_envs
    cfg.rollout_steps = rollout_steps

    envs = VecLS20EnvLevel(env_name="ls20", n_envs=n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=0, level_index=0)
    cfg.n_actions = envs.n_actions
    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    icm = ICMModule(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                    frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim,
                    hidden=cfg.icm_hidden).to(device)
    rndphi = RNDPhi(dim=cfg.trunk_dim, hidden=cfg.rnd_hidden, out=cfg.rnd_feature_dim,
                    leak=cfg.leak).to(device)
    ppo_opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    icm_opt = torch.optim.Adam(icm.parameters(), lr=cfg.icm_lr)
    rnd_opt = torch.optim.Adam(rndphi.predictor.parameters(), lr=cfg.rnd_lr)
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps,
                        c_value=cfg.c_value, c_entropy=cfg.c_entropy,
                        grad_clip=cfg.grad_clip, epochs=cfg.epochs, minibatches=cfg.minibatches)
    rff = RewardForwardFilter(cfg.gamma)
    int_ret_std = _EMAStd(cfg.int_norm_decay)

    return _run_loop(
        device, n_envs, rollout_steps, updates, warmup,
        collect_fn=lambda: collect_rollout(envs, model, device, rollout_steps),
        update_fn=lambda rollout: _rnd_icm_update(
            rollout, model, icm, rndphi, ppo_opt, icm_opt, rnd_opt, ppo_cfg, cfg,
            rff, int_ret_std, _phi_and_novelty, _rnd_update, _gae_nonepisodic,
            icm_update_from_rollout, ppo_update, device),
    )


def _rnd_icm_update(rollout, model, icm, rndphi, ppo_opt, icm_opt, rnd_opt, ppo_cfg, cfg,
                    rff, int_ret_std, _phi_and_novelty, _rnd_update, _gae_nonepisodic,
                    icm_update_from_rollout, ppo_update, device):
    phi_cached, nov = _phi_and_novelty(icm, rndphi, rollout, device)
    raw_i = nov.numpy()
    T, N = raw_i.shape
    rems = np.stack([rff.update(raw_i[t]) for t in range(T)])
    int_ret_std.update(rems)
    norm_i = raw_i / (int_ret_std.std + cfg.int_norm_eps)
    rollout.rewards = torch.from_numpy(norm_i.astype(np.float32))
    _gae_nonepisodic(rollout, cfg.gamma, cfg.gae_lambda)
    ppo_update(model, ppo_opt, rollout, ppo_cfg, device)
    icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)
    _rnd_update(rndphi, rnd_opt, phi_cached, rollout.dones, cfg, device)


def bench_lookahead(n_envs, updates, warmup, rollout_steps, device):
    """Time exp_013_5 lookahead: A forward-model evals per step + value/ICM/RND update."""
    from JEPA.experiments.exp_013_headline_experiment.exp_013_3_mcts_lookahead.config import Config
    from JEPA.experiments.exp_013_headline_experiment.exp_013_3_mcts_lookahead.lookahead import (
        ValueMLP, collect_lookahead_rollout, value_update,
    )
    from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.trainer import (
        _EMAStd, _phi_and_novelty, _rnd_update, _gae_nonepisodic,
    )
    from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi
    from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule, icm_update_from_rollout
    from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
    from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter

    cfg = Config(game="ls20", level_index=0, seed=0)
    cfg.n_envs = n_envs
    cfg.rollout_steps = rollout_steps

    envs = VecLS20EnvLevel(env_name="ls20", n_envs=n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=0, level_index=0)
    cfg.n_actions = envs.n_actions
    icm = ICMModule(n_actions=cfg.n_actions, n_colors=cfg.n_colors, frame_size=cfg.frame_size,
                    trunk_dim=cfg.trunk_dim, hidden=cfg.icm_hidden).to(device)
    rndphi = RNDPhi(dim=cfg.trunk_dim, hidden=cfg.rnd_hidden, out=cfg.rnd_feature_dim,
                    leak=cfg.leak).to(device)
    value = ValueMLP(dim=cfg.trunk_dim, hidden=cfg.value_hidden).to(device)
    icm_opt = torch.optim.Adam(icm.parameters(), lr=cfg.icm_lr)
    rnd_opt = torch.optim.Adam(rndphi.predictor.parameters(), lr=cfg.rnd_lr)
    value_opt = torch.optim.Adam(value.parameters(), lr=cfg.value_lr)
    rff = RewardForwardFilter(cfg.gamma)
    int_ret_std = _EMAStd(cfg.int_norm_decay)

    def collect():
        return collect_lookahead_rollout(envs, icm, rndphi, value, device,
                                         rollout_steps, cfg.n_actions, cfg.gamma, cfg.tau)

    def update(rollout):
        phi_cached, nov = _phi_and_novelty(icm, rndphi, rollout, device)
        raw_i = nov.numpy()
        T, N = raw_i.shape
        rems = np.stack([rff.update(raw_i[t]) for t in range(T)])
        int_ret_std.update(rems)
        norm_i = raw_i / (int_ret_std.std + cfg.int_norm_eps)
        rollout.rewards = torch.from_numpy(norm_i.astype(np.float32))
        _gae_nonepisodic(rollout, cfg.gamma, cfg.gae_lambda)
        value_update(value, value_opt, rollout, icm, cfg, device)
        icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)
        _rnd_update(rndphi, rnd_opt, phi_cached, rollout.dones, cfg, device)

    return _run_loop(device, n_envs, rollout_steps, updates, warmup, collect, update)


def _run_loop(device, n_envs, rollout_steps, updates, warmup, collect_fn, update_fn):
    collect_times, update_times = [], []
    for u in range(updates + warmup):
        _sync(device); t0 = time.time()
        rollout = collect_fn()
        _sync(device); t1 = time.time()
        update_fn(rollout)
        _sync(device); t2 = time.time()
        if u >= warmup:
            collect_times.append(t1 - t0)
            update_times.append(t2 - t1)
    tc = float(np.mean(collect_times)); tu = float(np.mean(update_times))
    steps_per_update = rollout_steps * n_envs
    sps = steps_per_update / (tc + tu)
    return {
        "n_envs": n_envs, "rollout_steps": rollout_steps,
        "steps_per_update": steps_per_update,
        "t_collect_s": round(tc, 4), "t_update_s": round(tu, 4),
        "collect_frac": round(tc / (tc + tu), 3),
        "sps": round(sps, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--n-envs", type=str, default="8,16,32,64")
    p.add_argument("--method", choices=["rnd_icm", "lookahead", "both"], default="both")
    args = p.parse_args()

    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
    device = get_device()
    n_envs_list = [int(x) for x in args.n_envs.split(",")]
    methods = ["rnd_icm", "lookahead"] if args.method == "both" else [args.method]

    results = {}
    for m in methods:
        results[m] = []
        for ne in n_envs_list:
            fn = bench_rnd_icm if m == "rnd_icm" else bench_lookahead
            r = fn(ne, args.updates, args.warmup, args.rollout_steps, device)
            print(f"[{m}] n_envs={ne:3d}  collect={r['t_collect_s']:.3f}s  "
                  f"update={r['t_update_s']:.3f}s  collect_frac={r['collect_frac']:.0%}  "
                  f"sps={r['sps']:.0f}", flush=True)
            results[m].append(r)
    print("RESULTS_JSON=" + json.dumps({"device": device.type, "results": results}))


if __name__ == "__main__":
    main()
