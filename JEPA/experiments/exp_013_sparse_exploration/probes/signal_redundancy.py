"""CHEAP probe: are the exp_013_2 ICM + RND intrinsic signals REDUNDANT and/or
UNINFORMATIVE? Kill-or-confirm BEFORE committing to a 500k w-sweep.

Replicates the exp_013_2-style additive loop (reuses the audited exp_013_1/2
helpers; method/harness code UNMODIFIED) for ~25-30 updates on ls20 L2
(level_index=1). It does NOT train toward any reward goal — it only MEASURES the
two normalised intrinsic signals against a ground-truth visitation oracle.

LS20 is deterministic, so hashing each 64x64 next-frame is an exact global
visitation count -- BUT the bottom rows 61-62 carry a step-timer that marches 1
cell/step regardless of the agent's action (verified: rows 61-62 change on 60/60
fixed-action steps). Raw-byte hashing would therefore treat every in-episode
frame as unique (timer phase), polluting the oracle and hiding wall-bump no-ops.
So we MASK the bottom rows (60-63) before hashing / before the no-op test, giving
a clean agent-state visitation oracle. Per update, over NON-done transitions:
  * n_icm = _SignalNorm(intrinsic_raw_error)              (exp_011 icm.py + exp_013_2 norm)
  * n_rnd = _SignalNorm(_phi_and_novelty RND-on-φ)        (exp_013_1 + exp_013_2 norm)
  * count = visit count of that transition's next-state BEFORE incrementing
  * novelty_target = -log(count)                          (true novelty ↓ with visits)

Logged per update + aggregated:
  1. Redundancy:      corr(n_icm, n_rnd)                  ~1.0 ⇒ additive adds nothing
  2. Informativeness: corr(n_icm, novelty_target),
                      corr(n_rnd, novelty_target)         ≈0 ⇒ signal is noise (φ≈chance)
  3. Context:         no-op fraction (next-frame == frame, wall-bumps)

Run (ONE process; MPS): uv run python -m \
  JEPA.experiments.exp_013_sparse_exploration.probes.signal_redundancy
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
from JEPA.experiments.exp_011_ls20_icm.shared.icm import (
    ICMModule, icm_update_from_rollout, intrinsic_raw_error,
)
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter  # noqa: F401 (used via _SignalNorm)

from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.config import Config
from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.rnd_phi import RNDPhi
from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.trainer import (
    _EMAStd, _phi_and_novelty, _rnd_update, _gae_nonepisodic,
    _collect_holdout, _eval_holdout_inv_acc,
)
# Reuse exp_013_2's EXACT per-signal normaliser (the one the additive loop uses).
from JEPA.experiments.exp_013_sparse_exploration.exp_013_2_rnd_icm_additive.trainer import _SignalNorm


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson corr over 1-D arrays; nan if either is (near-)constant."""
    if a.size < 2:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(updates: int = 28, level_index: int = 1, seed: int = 0):
    cfg = Config(game="ls20", level_index=level_index, seed=seed)
    cfg.c_entropy = 0.05                       # per task: doesn't matter for signal measurement
    device = get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=cfg.seed,
                           level_index=cfg.level_index)
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
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps, c_value=cfg.c_value,
                        c_entropy=cfg.c_entropy, grad_clip=cfg.grad_clip, epochs=cfg.epochs,
                        minibatches=cfg.minibatches)

    # TWO independent normalisers — EXACTLY as exp_013_2 (common ~unit scale).
    norm_icm = _SignalNorm(cfg.gamma, cfg.int_norm_decay, cfg.reward_clip_k, cfg.int_norm_eps)
    norm_rnd = _SignalNorm(cfg.gamma, cfg.int_norm_decay, cfg.reward_clip_k, cfg.int_norm_eps)

    holdout = _collect_holdout(cfg.game, cfg.level_index, cfg.seed, cfg.holdout_size, device)

    # Global exact-frame visitation oracle (next-frame bytes -> count). Deterministic env.
    visit = defaultdict(int)

    phi_frozen = False
    inv_streak = 0
    rows = []
    # Run accumulators (concatenate per-update post-warmup vectors for run-level corr).
    all_nicm, all_nrnd, all_tgt, all_noop = [], [], [], []

    print(f"[probe] signal_redundancy  device={device}  L{cfg.level_index + 1}  "
          f"updates={updates}  ~{updates * cfg.rollout_steps * cfg.n_envs} env steps")
    print(f"[probe] holdout={holdout[0].shape[0]}  reward_clip_k={cfg.reward_clip_k}")
    hdr = (f"\n{'upd':>4}{'corr_icm_rnd':>13}{'corr_icm_tgt':>13}{'corr_rnd_tgt':>13}"
           f"{'noop_frac':>11}{'mean_cnt':>9}{'inv':>7}{'frozen':>7}")
    print(hdr)

    for update in range(1, updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)

        # Both raw signals on the CURRENT models, BEFORE the ICM/RND updates (exp_013_2 order).
        phi_cached, rnd_nov = _phi_and_novelty(icm, rndphi, rollout, device)   # RND-on-φ (T,N)
        rnd_raw = rnd_nov.numpy()
        icm_raw_t, _m = intrinsic_raw_error(icm, rollout, device)              # ICM fwd err (T,N) done-zeroed
        icm_raw = icm_raw_t.numpy()
        T, N = rnd_raw.shape

        warming = update <= cfg.norm_warmup_updates
        if warming:
            n_icm = np.zeros_like(icm_raw)
            n_rnd = np.zeros_like(rnd_raw)
        else:
            n_icm = norm_icm(icm_raw)
            n_rnd = norm_rnd(rnd_raw)

        # ---- visitation oracle over this rollout (BEFORE incrementing per transition) ----
        dones_np = rollout.dones.numpy()                    # (T,N) bool
        obs_np = rollout.obs.numpy()                        # (T,N,F,F) uint8
        nobs_np = rollout.next_obs.numpy()                  # (T,N,F,F) uint8
        # Mask the bottom rows (60-63): they carry a step-timer that marches every
        # step independent of the agent, which would make every frame look unique.
        TIMER_ROW0 = 60
        obs_m = obs_np.copy(); obs_m[:, :, TIMER_ROW0:, :] = 0
        nobs_m = nobs_np.copy(); nobs_m[:, :, TIMER_ROW0:, :] = 0
        counts = np.zeros((T, N), dtype=np.float64)
        noop = np.zeros((T, N), dtype=bool)
        for t in range(T):
            for i in range(N):
                if dones_np[t, i]:
                    continue
                nf = nobs_m[t, i]
                key = nf.tobytes()
                counts[t, i] = visit[key] + 1               # count BEFORE incrementing (>=1)
                visit[key] += 1
                noop[t, i] = bool(np.array_equal(nf, obs_m[t, i]))  # masked: true wall-bump

        valid = ~dones_np
        # novelty_target = -log(count): true novelty decreases with visits.
        novelty_target = -np.log(np.maximum(counts, 1.0))

        # ---- per-update correlations over NON-done transitions (post-warmup only) ----
        if not warming:
            vidx = valid.reshape(-1)
            ni = n_icm.reshape(-1)[vidx]
            nr = n_rnd.reshape(-1)[vidx]
            tg = novelty_target.reshape(-1)[vidx]
            c_ir = _pearson(ni, nr)
            c_it = _pearson(ni, tg)
            c_rt = _pearson(nr, tg)
            all_nicm.append(ni); all_nrnd.append(nr); all_tgt.append(tg)
            all_noop.append(noop.reshape(-1)[vidx])
        else:
            c_ir = c_it = c_rt = float("nan")

        n_valid = int(valid.sum())
        noop_frac = float(noop[valid].mean()) if n_valid else float("nan")
        mean_cnt = float(counts[valid].mean()) if n_valid else float("nan")

        # ---- run the standard updates (keeps the loop faithful to exp_013_2 dynamics) ----
        # Reward = additive mix (w=0.5) so the policy/φ evolve as in the real loop.
        w = 0.5
        r = w * n_icm + (1.0 - w) * n_rnd
        rollout.rewards = torch.from_numpy(r.astype(np.float32))
        _gae_nonepisodic(rollout, cfg.gamma, cfg.gae_lambda)
        ustats = ppo_update(model, ppo_opt, rollout, ppo_cfg, device)

        icm_stats = icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)
        last_inv = icm_stats["inverse_acc"]
        if not phi_frozen:
            hold_inv = _eval_holdout_inv_acc(icm, holdout, device)
            trig = hold_inv if cfg.freeze_metric == "holdout" else last_inv
            inv_streak = inv_streak + 1 if trig >= cfg.phi_freeze_inverse_acc else 0
            if inv_streak >= cfg.phi_freeze_patience or update >= cfg.phi_freeze_max_updates:
                for p in icm.phi.parameters():
                    p.requires_grad_(False)
                icm.phi.eval()
                phi_frozen = True
        _rnd_update(rndphi, rnd_opt, phi_cached, rollout.dones, cfg, device)

        rows.append({
            "update": update, "n_valid": n_valid,
            "corr_icm_rnd": c_ir, "corr_icm_tgt": c_it, "corr_rnd_tgt": c_rt,
            "noop_frac": noop_frac, "mean_count": mean_cnt,
            "unique_states": len(visit), "inverse_acc": last_inv,
            "phi_frozen": bool(phi_frozen), "warming": bool(warming),
            "entropy": ustats.entropy,
        })
        print(f"{update:>4}{c_ir:>13.4f}{c_it:>13.4f}{c_rt:>13.4f}"
              f"{noop_frac:>11.3f}{mean_cnt:>9.2f}{last_inv:>7.2f}{str(phi_frozen):>7}")

    # ---- run-level aggregates (concatenate all post-warmup non-done transitions) ----
    cat_nicm = np.concatenate(all_nicm) if all_nicm else np.array([])
    cat_nrnd = np.concatenate(all_nrnd) if all_nrnd else np.array([])
    cat_tgt = np.concatenate(all_tgt) if all_tgt else np.array([])
    cat_noop = np.concatenate(all_noop) if all_noop else np.array([])
    n_total = int(cat_nicm.size)

    run_corr_icm_rnd = _pearson(cat_nicm, cat_nrnd)
    run_corr_icm_tgt = _pearson(cat_nicm, cat_tgt)
    run_corr_rnd_tgt = _pearson(cat_nrnd, cat_tgt)

    per_upd = [r for r in rows if not r["warming"]]
    mean_corr_icm_rnd = float(np.nanmean([r["corr_icm_rnd"] for r in per_upd]))
    mean_corr_icm_tgt = float(np.nanmean([r["corr_icm_tgt"] for r in per_upd]))
    mean_corr_rnd_tgt = float(np.nanmean([r["corr_rnd_tgt"] for r in per_upd]))
    overall_noop = float(np.mean([r["noop_frac"] for r in rows if not np.isnan(r["noop_frac"])]))
    pooled_noop = float(cat_noop.mean()) if cat_noop.size else float("nan")

    summary = {
        "level_index": cfg.level_index, "updates": updates, "seed": seed,
        "env_steps": updates * cfg.rollout_steps * cfg.n_envs,
        "n_transitions_pooled": n_total,
        "unique_next_states": len(visit),
        "run_corr_icm_rnd": run_corr_icm_rnd,
        "run_corr_icm_tgt": run_corr_icm_tgt,
        "run_corr_rnd_tgt": run_corr_rnd_tgt,
        "mean_perupd_corr_icm_rnd": mean_corr_icm_rnd,
        "mean_perupd_corr_icm_tgt": mean_corr_icm_tgt,
        "mean_perupd_corr_rnd_tgt": mean_corr_rnd_tgt,
        "noop_frac_mean_perupd": overall_noop,
        "noop_frac_pooled": pooled_noop,
        "rows": rows,
    }

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / f"signal_redundancy_L{cfg.level_index + 1}.json"
    res_path.write_text(json.dumps(summary, indent=2))

    print("\n========== RUN SUMMARY ==========")
    print(f"  n (pooled non-done transitions) = {n_total}   unique next-states = {len(visit)}")
    print(f"  REDUNDANCY   corr(n_icm,n_rnd):  pooled={run_corr_icm_rnd:+.4f}  "
          f"mean_per_upd={mean_corr_icm_rnd:+.4f}")
    print(f"  INFORM icm   corr(n_icm,-log cnt): pooled={run_corr_icm_tgt:+.4f}  "
          f"mean_per_upd={mean_corr_icm_tgt:+.4f}")
    print(f"  INFORM rnd   corr(n_rnd,-log cnt): pooled={run_corr_rnd_tgt:+.4f}  "
          f"mean_per_upd={mean_corr_rnd_tgt:+.4f}")
    print(f"  NO-OP frac:  pooled={pooled_noop:.3f}  mean_per_upd={overall_noop:.3f}")
    print(f"  wrote {res_path}")
    return summary


if __name__ == "__main__":
    import sys
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(updates=28, level_index=lvl)
