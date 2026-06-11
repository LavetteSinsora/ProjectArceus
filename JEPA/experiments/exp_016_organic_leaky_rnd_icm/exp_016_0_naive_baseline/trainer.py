"""exp_016_0 naive training loop. See SYSTEM_CARD.md §3/§5.

One update (= rollout_steps × n_envs env steps):
  collect rollout → novelty(s') via tracker enc (snapshot) → z-score reward
  → reward-to-go returns (γ, episodic) ÷ batch-std → REINFORCE (no baseline)
  → IDM update from replay (continuous, no freeze) → RND distill + leak
  → full diagnostics (drift, coverage, full-state novelty, normalization-ablation stats).
"""
from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.metrics import MetricsWriter
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import one_hot_frame
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RunningMeanStd
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi

from .actor import Actor
from .tracker import (IDMEncoder, ReplayBuffer, idm_update, rnd_update,
                      holdout_inverse_acc)
from .diagnostics import (StateRegistry, harvest_states, encode_all, drift_rel_l2,
                          mask_board, state_key)


def _repo_root() -> Path:
    # exp_016_0_naive_baseline -> exp_016_organic.. -> experiments -> JEPA -> Code Repo
    return Path(__file__).resolve().parents[4]


def _save_ckpt(ckpt_dir: Path, actor, idm, rnd, cfg, step: int) -> Path:
    """Save actor + tracker (idm + rnd) so the dashboard debug_runner can replay
    the agent's behavior at this checkpoint. See debug_runner.py."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"
    torch.save({"step": int(step), "config": dataclasses.asdict(cfg),
                "actor": actor.state_dict(), "idm": idm.state_dict(),
                "rnd_target": rnd.target.state_dict(),
                "rnd_predictor": rnd.predictor.state_dict()}, path)
    return path


def compute_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Reward-to-go, episodic (reset at done), zero bootstrap at rollout end. (T,N)."""
    T, N = rewards.shape
    G = torch.zeros(T, N)
    run = torch.zeros(N)
    for t in reversed(range(T)):
        run = rewards[t] + gamma * run * (~dones[t]).float()
        G[t] = run
    return G


@torch.no_grad()
def collect_holdout(game, level, seed, n_target, mask_rows):
    """Fixed uniform-random masked (s,a,s') set for held-out inverse accuracy."""
    env = VecLS20EnvLevel(env_name=game, n_envs=16, max_episode_steps=200,
                          seed=seed + 4242, level_index=level)
    obs = env.current_obs()
    S, A, SP = [], [], []
    while len(A) < n_target:
        a = np.random.randint(0, env.n_actions, size=env.n_envs).astype(np.int64)
        nobs, _r, dones, _i = env.step(a)
        ms, msp = mask_board(obs, mask_rows), mask_board(nobs, mask_rows)
        for i in range(env.n_envs):
            if not dones[i] and not np.array_equal(ms[i], msp[i]):
                S.append(ms[i].copy()); A.append(int(a[i])); SP.append(msp[i].copy())
        obs = nobs
    return (torch.from_numpy(np.stack(S[:n_target])),
            torch.from_numpy(np.array(A[:n_target], dtype=np.int64)),
            torch.from_numpy(np.stack(SP[:n_target])))


def _unique_per_episode(masked_TN: np.ndarray, dones: np.ndarray) -> float:
    """Mean distinct masked states per (completed) episode within the rollout."""
    T, N = dones.shape
    lens = []
    for n in range(N):
        cur = set()
        for t in range(T):
            cur.add(state_key(masked_TN[t, n]))
            if dones[t, n]:
                lens.append(len(cur)); cur = set()
    return float(np.mean(lens)) if lens else float("nan")


def train(cfg, smoke: bool = False) -> dict:
    if smoke:
        cfg = cfg.smoke()
    device = get_device()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    exp_dir = _repo_root() / cfg.exp_dir
    run_name = f"{cfg.exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = exp_dir / "runs" / run_name
    ckpt_dir = exp_dir / "checkpoints" / run_name        # dashboard reads experiment/checkpoints/**
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = MetricsWriter(run_dir)
    state_nov_path = run_dir / "state_novelty.jsonl"

    envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=cfg.seed,
                           level_index=cfg.level_index)
    if cfg.n_actions is None:
        cfg.n_actions = envs.n_actions
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    print(f"[exp016_0] {cfg.exp_name} device={device} n_actions={cfg.n_actions} "
          f"cap={cfg.max_env_steps} leak={cfg.leak}")

    actor = Actor(cfg.n_actions, cfg.n_colors, cfg.frame_size, cfg.trunk_dim,
                  cfg.timer_mask_rows, value_head=cfg.use_value_head).to(device)
    idm = IDMEncoder(cfg.n_actions, cfg.n_colors, cfg.frame_size, cfg.trunk_dim,
                     cfg.idm_hidden, layernorm=cfg.idm_layernorm).to(device)
    rnd = RNDPhi(dim=cfg.trunk_dim, hidden=cfg.rnd_hidden, out=cfg.rnd_out,
                 leak=cfg.leak).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    idm_opt = torch.optim.Adam(idm.parameters(), lr=cfg.idm_lr)
    rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr=cfg.rnd_lr)

    buf = ReplayBuffer(cfg.replay_capacity, cfg.frame_size)
    nov_rms = RunningMeanStd()                       # running stats for reward z-score
    holdout = collect_holdout(cfg.game, cfg.level_index, cfg.seed, cfg.holdout_size,
                              cfg.timer_mask_rows)
    registry, probe = harvest_states(cfg.game, cfg.level_index, cfg.seed,
                                     cfg.probe_roam_steps, cfg.n_envs,
                                     cfg.timer_mask_rows, cfg.n_probe_states)
    print(f"[exp016_0] harvested {len(registry.exemplars)} states, {len(probe)} probes")

    def actor_enc(m):  # encode masked boards with the ACTOR encoder
        return actor.encoder(one_hot_frame(m, cfg.n_colors))

    global_step = 0
    first_reward_step = None
    t_start = time.time()
    F = cfg.frame_size

    for update in range(1, cfg.total_updates + 1):
        # ── 1. collect rollout (on-policy) ──────────────────────────────────
        obs = np.zeros((cfg.rollout_steps, cfg.n_envs, F, F), np.uint8)
        nxt = np.zeros_like(obs)
        acts = np.zeros((cfg.rollout_steps, cfg.n_envs), np.int64)
        dones = np.zeros((cfg.rollout_steps, cfg.n_envs), bool)
        ext = np.zeros((cfg.rollout_steps, cfg.n_envs), np.float32)
        o = envs.current_obs()
        for t in range(cfg.rollout_steps):
            a, _lp, _ent = actor.act(torch.from_numpy(o).to(device))
            a = a.cpu().numpy().astype(np.int64)
            no, r, d, _i = envs.step(a)
            obs[t] = o; nxt[t] = no; acts[t] = a; dones[t] = d; ext[t] = r
            o = no

        masked_nxt = mask_board(nxt, cfg.timer_mask_rows)        # (T,N,H,W)
        masked_obs = mask_board(obs, cfg.timer_mask_rows)

        # ── 2. reward: novelty(s') with the CURRENT tracker encoder (snapshot) ─
        with torch.no_grad():
            h_next = encode_all(idm.encode_masked, masked_nxt.reshape(-1, F, F), device)  # (M,D) cpu
            raw = rnd.novelty(h_next.to(device)).cpu().numpy().reshape(cfg.rollout_steps, cfg.n_envs)
        raw = raw * (~dones).astype(np.float32)                  # zero done-steps
        nd = ~dones
        nov_rms.update(raw[nd])                                  # running stats over non-done
        rmean, rstd = nov_rms.mean, float(np.sqrt(nov_rms.var))
        # REWARD: scale by running std for cross-time stability. Centering (subtract
        # the running mean) is OPTIONAL and OFF by default — a lagging cumulative mean
        # was the source of the spurious-negative reward; the batch baseline below
        # re-centers correctly regardless, so reward-level centering is redundant.
        if cfg.reward_zscore:
            center = rmean if cfg.reward_center else 0.0
            reward = (raw - center) / (rstd + cfg.norm_eps)
        else:
            reward = raw.copy()
        reward = reward * (~dones).astype(np.float32)

        # ── 3. returns → BASELINE → advantage ────────────────────────────────
        # G = reward-to-go. Advantage A = G − baseline, baseline = the batch-mean
        # return (the variance-reducing baseline). Subtracting the batch mean removes
        # the constant offset that drives the volatility-drag entropy collapse: when
        # the reward is uninformative (all states ~equally novel) A→0 → no update →
        # the policy HOLDS its entropy instead of diffusing to a corner.
        # NOTE: do NOT then divide by the batch std (return_scale_by_std) — that
        # re-amplifies the residual to unit scale and re-injects the drag.
        G = compute_returns(torch.from_numpy(reward), torch.from_numpy(dones), cfg.gamma)
        G_raw_mean, G_raw_std = float(G.mean()), float(G.std())
        A = G - G.mean() if cfg.use_baseline else G
        if cfg.return_scale_by_std:
            A = A / (A.std() + cfg.norm_eps)
        G_norm = A

        # ── drift: encode probe states BEFORE any update (both encoders) ─────
        with torch.no_grad():
            a_before = encode_all(actor_enc, probe, device)
            i_before = encode_all(idm.encode_masked, probe, device)

        # ── 4. ACTOR update (REINFORCE, minibatched) ────────────────────────
        B = cfg.rollout_steps * cfg.n_envs
        b_obs = torch.from_numpy(obs.reshape(B, F, F))
        b_act = torch.from_numpy(acts.reshape(B))
        b_adv = G_norm.reshape(B)          # constant-baseline advantage (no-value path)
        b_ret = G.reshape(B)               # raw return (value target + V-baseline path)
        idx = np.arange(B); np.random.shuffle(idx)
        mb = max(1, B // 4)
        ent_acc = gn_acc = ploss_acc = vloss_acc = 0.0; nstep = 0
        probs_acc = np.zeros(cfg.n_actions, np.float64)
        for s in range(0, B, mb):
            sel = idx[s:s + mb]
            mo = b_obs[sel].to(device); ma = b_act[sel].to(device)
            logp, ent, probs, v = actor.evaluate(mo, ma)
            if cfg.use_value_head:                          # STATE-DEPENDENT baseline V(s)
                ret_mb = b_ret[sel].to(device)
                adv = ret_mb - v.detach()
                v_loss = 0.5 * (v - ret_mb).pow(2).mean()
            else:                                           # constant / no baseline
                adv = b_adv[sel].to(device)
                v_loss = torch.zeros((), device=device)
            loss = -(logp * adv).mean() - cfg.ent_coef * ent.mean() + cfg.c_value * v_loss
            actor_opt.zero_grad(); loss.backward()
            gn = nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
            actor_opt.step()
            ent_acc += ent.mean().item(); gn_acc += float(gn); ploss_acc += loss.item()
            vloss_acc += v_loss.item()
            probs_acc += probs.mean(0).detach().cpu().numpy().astype(np.float64); nstep += 1
        entropy = ent_acc / nstep; grad_norm = gn_acc / nstep
        per_action = (probs_acc / nstep).tolist(); value_loss = vloss_acc / nstep

        # ── 5. IDM update from replay (continuous) ──────────────────────────
        buf.add_batch(masked_obs.reshape(-1, F, F), acts.reshape(-1),
                      masked_nxt.reshape(-1, F, F), dones.reshape(-1), cfg.drop_noops)
        idm_stats = idm_update(idm, idm_opt, buf, cfg, device)
        hold_acc = holdout_inverse_acc(idm, holdout, device)

        # ── 6. RND distill (on the snapshot h_next, non-done) + leak ────────
        valid = nd.reshape(-1)
        rnd_loss = rnd_update(rnd, rnd_opt, h_next[valid], cfg, device)

        # ── drift AFTER updates ─────────────────────────────────────────────
        with torch.no_grad():
            a_after = encode_all(actor_enc, probe, device)
            i_after = encode_all(idm.encode_masked, probe, device)
        drift_actor = drift_rel_l2(a_before, a_after)
        drift_idm = drift_rel_l2(i_before, i_after)

        # ── coverage + registry visit counts ────────────────────────────────
        cov = registry.observe(masked_nxt[nd])
        uniq_ep = _unique_per_episode(masked_nxt, dones)

        # ── full-state novelty landscape (current tracker encoder) ──────────
        if cfg.log_state_novelty:
            all_m = registry.all_masked()
            with torch.no_grad():
                h_all = encode_all(idm.encode_masked, all_m, device)
                nov_all = rnd.novelty(h_all.to(device)).cpu().numpy()
            # per-state visits THIS update
            vtu = np.zeros(len(all_m), np.int64)
            for b in masked_nxt[nd]:
                vtu[registry.ids[state_key(b)]] += 1
            with open(state_nov_path, "a") as f:
                f.write(json.dumps({"update": update, "step": global_step,
                                    "novelty": [round(float(x), 6) for x in nov_all],
                                    "visits": registry.visits.tolist(),
                                    "visits_this_update": vtu.tolist()}) + "\n")

        global_step += cfg.rollout_steps * cfg.n_envs

        # ── headline: first extrinsic reward ────────────────────────────────
        if first_reward_step is None and (ext > 0).any():
            t_idx = int((ext > 0).sum(1).nonzero()[0][0])
            first_reward_step = (global_step - cfg.rollout_steps * cfg.n_envs) + (t_idx + 1) * cfg.n_envs
            print(f"[exp016_0] *** FIRST REWARD ~{first_reward_step} env steps (u{update}) ***")
        done_eps = envs.drain_completed_episodes()
        succ = float(np.mean([e.success for e in done_eps])) if done_eps else float("nan")

        record = {
            "step": global_step, "update": update,
            # normalization-ablation stats
            "novelty_raw_mean": float(raw[nd].mean()) if nd.any() else 0.0,
            "novelty_raw_std": float(raw[nd].std()) if nd.any() else 0.0,
            "run_mean": rmean, "run_std": rstd,
            "reward_norm_mean": float(reward[nd].mean()) if nd.any() else 0.0,
            "reward_norm_std": float(reward[nd].std()) if nd.any() else 0.0,
            "return_raw_mean": G_raw_mean, "return_raw_std": G_raw_std,
            "return_norm_mean": float(G_norm.mean()), "return_norm_std": float(G_norm.std()),
            "return_variance": float(G.var()),
            # policy
            "entropy": entropy, "grad_norm": grad_norm, "policy_loss": ploss_acc / nstep,
            "value_loss": value_loss, "per_action_prob": per_action,
            # rnd / count
            "rnd_distill_loss": rnd_loss, "novelty_floor": float(raw[nd].min()) if nd.any() else 0.0,
            # idm / encoder
            **idm_stats, "inverse_acc_holdout": hold_acc,
            "noop_fraction": float((masked_obs == masked_nxt).all(axis=(2, 3))[nd].mean()) if nd.any() else 0.0,
            # coverage
            **cov, "unique_per_episode": uniq_ep,
            # drift
            "drift_actor_rel_l2": drift_actor["drift_rel_l2"],
            "drift_idm_rel_l2": drift_idm["drift_rel_l2"],
            "drift_idm_over_pairdist": drift_idm["drift_over_pairdist"],
            "idm_mean_pairwise_l2": drift_idm["mean_pairwise_l2"],
            # headline
            "env_steps_to_first_reward": first_reward_step,
            "train_success_rate": succ, "train_episodes": len(done_eps),
            "sps": global_step / max(1e-6, time.time() - t_start),
        }
        if cfg.log_every > 0 and update % cfg.log_every == 0:
            writer.write(record)
        if update % 10 == 0 or update == 1:
            print(f"[exp016_0] u{update}/{cfg.total_updates} step={global_step} "
                  f"H={entropy:.3f} nov_raw={record['novelty_raw_mean']:.3g} "
                  f"inv(on/hold)={idm_stats['inverse_acc_onpolicy']:.2f}/{hold_acc:.2f} "
                  f"cov={cov['cumulative_unique_states']} drift_idm={drift_idm['drift_rel_l2']:.3g} "
                  f"frr={first_reward_step}")

        if cfg.save_every and update % cfg.save_every == 0:
            _save_ckpt(ckpt_dir, actor, idm, rnd, cfg, global_step)

    _save_ckpt(ckpt_dir, actor, idm, rnd, cfg, global_step)   # always save final
    writer.close()
    result = {
        "exp_name": cfg.exp_name, "game": cfg.game, "level_index": cfg.level_index,
        "seed": cfg.seed, "env_steps_to_first_reward": first_reward_step,
        "solved": first_reward_step is not None, "total_env_steps": global_step,
        "cumulative_unique_states": len(registry.exemplars),
        "wall_seconds": time.time() - t_start,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[exp016_0] DONE {cfg.exp_name}: frr={first_reward_step} "
          f"states={len(registry.exemplars)} total={global_step}")
    return result
