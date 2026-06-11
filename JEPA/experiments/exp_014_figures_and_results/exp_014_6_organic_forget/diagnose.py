"""exp_014_6 — ORGANIC RND forgetting on a monitored set of near-reset states.

This is the *organic* sibling of exp_014_5. Instead of a hand-controlled per-state
distillation schedule, we run a REAL rollout loop on LS20-L2 with a random policy and
distil exactly like the production trainer (exp_013_1b `_rnd_update`): one distillation
pass per "PPO update", `rnd_epochs` epochs split into `minibatches` minibatches, with the
LEAK applied AFTER EVERY MINIBATCH STEP (this is the *true* cadence in the real loop —
~`minibatches` leaks per update, NOT one per update).

PROTOCOL
--------
  0. HARVEST: short uniform-random roam → pick N_MONITOR (=5) distinct states that sit
     NEAR THE INITIAL RESET (small first-seen episode step), are visited often, and have
     observed incoming transitions (so they can be blocked later). We also record, for
     every observed transition (s, a) → s', whether s' is one of the monitored states;
     that set of "(s,a) leads to a monitored state" pairs is what the mask uses.

  1. PHASE 1 — FREE  (UPDATES_FREE updates):
        random policy. Every update: collect a 2048-transition rollout, count how often
        each monitored state was entered, then distil BOTH predictors on the rollout's
        masked next-states (identical data + minibatch order; the only difference is the
        leak). The monitored states ARE visited, so they get driven down.

  2. PHASE 2 — MASKED  (UPDATES_MASKED updates):
        same loop, but the policy is ACTION-MASKED: before each env step, any action whose
        observed transition (s, a) lands on a monitored state is forbidden; the agent
        picks uniformly among the remaining actions. The monitored states are therefore
        (almost) never visited → never distilled. We count "mask failures" (entered a
        monitored state anyway via a not-yet-observed transition; that (s,a) is then added
        to the block set online).

We run TWO RND predictors over the SAME rollouts / SAME minibatch order / IDENTICAL init,
differing ONLY in the leak:
    • standard RND  (μ = 0)    — never forgets an abandoned state.
    • leaky RND     (μ > 0)    — regenerates novelty on the masked (no-longer-visited) states.

We record the distillation error (½‖P(φ)−T(φ)‖²) of all N_MONITOR states BEFORE the update
and AFTER every minibatch gradient step, for both predictors, plus per-update visit counts.

φ here is a frozen random projection of the masked board (same ruler as exp_014_5/exp_014_1),
NOT a learned ICM φ — so the forgetting we measure is purely the leak, not encoder drift.

CPU-only. Figures via plot.py.

Run:
    uv run python -m \
      JEPA.experiments.exp_014_figures_and_results.exp_014_6_organic_forget.diagnose
    uv run ... --mu 0.05 --updates-free 5 --updates-masked 5 --n-monitor 5
    uv run ... --suffix _seed1 --seed 1            # avoid overwriting another run
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi
from JEPA.experiments.exp_014_figures_and_results.exp_014_1_rnd_saturation.diagnose import (
    mask_board, state_key, project_states, FRAME, N_COLORS,
)

DEVICE = torch.device("cpu")


def build_projection(seed: int, proj_dim: int) -> torch.Tensor:
    """Frozen random linear map one-hot board (64*64*16) → proj_dim (see exp_014_5)."""
    g = torch.Generator().manual_seed(seed)
    in_dim = FRAME * FRAME * N_COLORS
    return torch.randn(in_dim, proj_dim, generator=g) / (in_dim ** 0.5)


def project_chunked(masked: np.ndarray, W: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """project_states builds a dense (B, 4096, 16) one-hot; for the ~2048 frames in a
    rollout that is ~0.5 GB in one go. Project in chunks and concatenate to stay small."""
    outs = []
    for s in range(0, len(masked), chunk):
        outs.append(project_states(masked[s:s + chunk], W))
    return torch.cat(outs, dim=0)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20")
    p.add_argument("--level", type=int, default=1, help="0-indexed (1 = LS20 L2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-monitor", type=int, default=5, help="# near-reset states to track")
    p.add_argument("--monitor-max-dist", type=int, default=6,
                   help="only monitor states first seen within this many steps of reset")
    p.add_argument("--updates-free", type=int, default=8, help="PPO-style updates, free policy")
    p.add_argument("--updates-masked", type=int, default=8, help="updates with the monitored "
                   "states action-masked (never entered → never distilled)")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=128, help="env steps/env per update "
                   "(rollout_steps*n_envs = transitions per update, default 128*16=2048)")
    p.add_argument("--minibatches", type=int, default=4, help="distill minibatches per epoch "
                   "(real loop = 4); leak fires after EACH minibatch step")
    p.add_argument("--rnd-epochs", type=int, default=4, help="distill epochs per update. The "
                   "real loop uses 1, but converging a state to its floor then takes thousands "
                   "of updates; we use 4 (with rnd-lr 1e-3) so the free phase drives the "
                   "monitored states down within ~8 updates. The leak-per-minibatch-step "
                   "cadence (the faithful part) is unchanged. Use --rnd-epochs 1 --rnd-lr 1e-4 "
                   "for the exact production distillation strength (needs many more updates).")
    p.add_argument("--mu", type=float, default=0.05, help="leak for the leaky predictor")
    p.add_argument("--mu2", type=float, default=0.1, help="second leak value, plotted as one "
                   "extra masked line (set <=0 to disable)")
    p.add_argument("--rnd-lr", type=float, default=1e-3, help="Adam lr (accelerated; real=1e-4)")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--proj-dim", type=int, default=1024, help="frozen-projection / RND feature dim")
    p.add_argument("--pre-roam", type=int, default=800, help="per-env steps to harvest states")
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--suffix", default="", help="appended to the level tag so results/figures "
                   "do not overwrite an existing run")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def harvest_monitor_states(envs, rng, n_monitor, max_dist, pre_roam):
    """Short uniform-random roam → N_MONITOR distinct near-reset states + the transition
    info needed to mask them.

    Returns
      monitored_masked : list[(64,64) uint8]   the chosen states' masked boards
      monitored_keys   : list[bytes]
      blocked_sa       : set[(state_key, action)]  observed (s,a) that land on a monitor
      stats            : dict for the summary
    """
    N = envs.n_envs
    visit: dict[bytes, int] = defaultdict(int)
    first_step: dict[bytes, int] = {}                 # min episode-step a state was seen at
    exemplar: dict[bytes, np.ndarray] = {}
    # all observed transitions: (prev_key, action) -> Counter of next_key
    trans: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))

    obs = envs.current_obs()
    obs_mask = mask_board(obs)
    prev_keys = [state_key(obs_mask[i]) for i in range(N)]
    ep_step = np.zeros(N, dtype=np.int64)
    for i in range(N):                                # reset state(s): distance 0
        first_step.setdefault(prev_keys[i], 0)
        exemplar.setdefault(prev_keys[i], obs_mask[i])

    for _ in range(pre_roam):
        a = rng.integers(0, envs.n_actions, size=N)
        nobs, _r, dones, _i = envs.step(a)
        nmask = mask_board(nobs)
        for i in range(N):
            if dones[i]:
                ep_step[i] = 0
                prev_keys[i] = state_key(nmask[i])    # post-reset state
                first_step.setdefault(prev_keys[i], 0)
                exemplar.setdefault(prev_keys[i], nmask[i])
                continue
            ep_step[i] += 1
            nk = state_key(nmask[i])
            visit[nk] += 1
            exemplar.setdefault(nk, nmask[i])
            fs = first_step.get(nk)
            if fs is None or ep_step[i] < fs:
                first_step[nk] = int(ep_step[i])
            trans[(prev_keys[i], int(a[i]))][nk] += 1
            prev_keys[i] = nk

    # candidates: seen NEAR reset (1 <= first_step <= max_dist), excluding the reset state,
    # ranked by visit count (most reliably revisited → cleanest signal).
    cand = [(k, visit[k]) for k, fs in first_step.items()
            if 1 <= fs <= max_dist and visit[k] > 0]
    cand.sort(key=lambda kv: kv[1], reverse=True)
    chosen = [k for k, _ in cand[:n_monitor]]
    if len(chosen) < n_monitor:
        # relax the distance cap if too few near-reset states were found
        extra = [k for k, _ in sorted(visit.items(), key=lambda kv: kv[1], reverse=True)
                 if k not in chosen]
        chosen += extra[: n_monitor - len(chosen)]
    chosen = chosen[:n_monitor]

    monitored_set = set(chosen)
    blocked_sa = {sa for sa, nexts in trans.items()
                  if any(nk in monitored_set for nk in nexts)}

    monitored_masked = [exemplar[k] for k in chosen]
    stats = {
        "first_step": [int(first_step.get(k, -1)) for k in chosen],
        "harvest_visits": [int(visit[k]) for k in chosen],
        "n_blocked_sa": len(blocked_sa),
        "n_states_seen": len(visit),
    }
    return monitored_masked, chosen, blocked_sa, stats


def choose_masked_actions(cur_keys, blocked_sa, n_actions, rng):
    """For each env, forbid any action whose observed transition lands on a monitored
    state; pick uniformly among the rest (fall back to all if every action is blocked)."""
    N = len(cur_keys)
    acts = np.empty(N, dtype=np.int64)
    for i in range(N):
        allowed = [a for a in range(n_actions) if (cur_keys[i], a) not in blocked_sa]
        acts[i] = rng.choice(allowed) if allowed else rng.integers(0, n_actions)
    return acts


def run_update(envs, rng, W, monitored_keys, monitored_set, blocked_sa, cfg, masking):
    """Collect one rollout (cfg.rollout_steps * n_envs transitions) under the given policy.

    Returns
      feats   : (M, dim) projected masked next-states for the VALID (non-done) transitions
      idx_shuffle : a single shuffled index over M (shared by both predictors)
      visits  : (n_monitor,) times each monitored state was entered this update
      mask_fail : int  (entered a monitored state despite masking → (s,a) added to block set)
    """
    N = envs.n_envs
    n_monitor = len(monitored_keys)
    mon_index = {k: j for j, k in enumerate(monitored_keys)}
    visits = np.zeros(n_monitor, dtype=np.int64)
    mask_fail = 0
    next_masked: list[np.ndarray] = []

    obs = envs.current_obs()
    cur_keys = [state_key(mask_board(obs[i])) for i in range(N)]
    for _t in range(cfg.rollout_steps):
        if masking:
            a = choose_masked_actions(cur_keys, blocked_sa, envs.n_actions, rng)
        else:
            a = rng.integers(0, envs.n_actions, size=N)
        nobs, _r, dones, _i = envs.step(a)
        nmask = mask_board(nobs)
        new_keys = cur_keys[:]
        for i in range(N):
            if dones[i]:
                new_keys[i] = state_key(nmask[i])     # post-reset; transition is invalid
                continue
            nk = state_key(nmask[i])
            if nk in monitored_set:
                visits[mon_index[nk]] += 1
                blocked_sa.add((cur_keys[i], int(a[i])))   # learn the block online
                if masking:
                    mask_fail += 1
            next_masked.append(nmask[i])
            new_keys[i] = nk
        cur_keys = new_keys

    if next_masked:
        feats = project_chunked(np.stack(next_masked), W).to(DEVICE)
    else:
        feats = torch.zeros((0, W.shape[1]), device=DEVICE)
    idx_shuffle = np.arange(feats.shape[0])
    rng.shuffle(idx_shuffle)
    return feats, idx_shuffle, visits, mask_fail


def distill_record(rnds, opts, feats, idx_shuffle, mon_feats, cfg, nov_log):
    """One distillation pass (cfg.rnd_epochs epochs × cfg.minibatches minibatches) for
    BOTH predictors on identical data/order, leak AFTER each minibatch step. After every
    minibatch step, append the monitored-state novelty of both predictors to nov_log.

    Emits EXACTLY cfg.rnd_epochs * cfg.minibatches records per call (via np.array_split),
    regardless of how many valid transitions the rollout produced — so the masked and
    unmasked conditions stay aligned on a shared per-record x-axis (empty minibatches just
    re-record the unchanged novelty)."""
    M = feats.shape[0]
    for _ep in range(cfg.rnd_epochs):
        chunks = (np.array_split(idx_shuffle, cfg.minibatches) if M > 0
                  else [np.array([], dtype=int)] * cfg.minibatches)
        for batch_idx in chunks:
            if len(batch_idx) > 0:
                batch = feats[batch_idx]
                for mu, r in rnds.items():
                    opt = opts[mu]
                    opt.zero_grad()
                    r.distill_loss(batch).backward()
                    torch.nn.utils.clip_grad_norm_(r.predictor.parameters(), cfg.grad_clip)
                    opt.step()
                    r.apply_leak()
            for mu, r in rnds.items():
                with torch.no_grad():
                    nov_log[mu].append(r.novelty(mon_feats).cpu().numpy())


def run_condition(envs, W, monitored_keys, monitored_set, blocked_sa_init, mon_feats,
                  cfg, mus, do_mask, label):
    """Run the full free→(masked|free) update loop for ONE condition on a freshly reset env
    with a freshly seeded RNG and freshly initialised predictors, so the two conditions share
    an identical phase 1 and diverge only in phase 2 (masked vs. still-free).

    `mus` is the list of leak values; one predictor per μ, ALL sharing identical init (only
    the leak differs). Returns nov (n_mus, n_monitor, n_records), update_end_step, visits,
    mask_fail."""
    envs.reset_all()
    rng = np.random.default_rng(cfg.seed + 1)           # identical phase-1 RNG for both conds
    blocked_sa = set(blocked_sa_init)
    rnds, opts = {}, {}
    for mu in mus:
        torch.manual_seed(cfg.seed + 100)               # identical init across μ + conditions
        r = RNDPhi(dim=cfg.proj_dim, hidden=256, out=256, leak=mu).to(DEVICE)
        rnds[mu] = r
        opts[mu] = torch.optim.Adam(r.predictor.parameters(), lr=cfg.rnd_lr)

    total_updates = cfg.updates_free + cfg.updates_masked
    n_monitor = len(monitored_keys)
    nov_log = {mu: [] for mu in mus}
    update_end_step = []
    visits_per_update = np.zeros((total_updates, n_monitor), dtype=np.int64)
    mask_fail_per_update = np.zeros(total_updates, dtype=np.int64)

    for mu, r in rnds.items():                           # initial point (before any training)
        with torch.no_grad():
            nov_log[mu].append(r.novelty(mon_feats).cpu().numpy())

    for u in range(total_updates):
        masking = do_mask and (u >= cfg.updates_free)
        feats, idx_shuffle, visits, mask_fail = run_update(
            envs, rng, W, monitored_keys, monitored_set, blocked_sa, cfg, masking)
        visits_per_update[u] = visits
        mask_fail_per_update[u] = mask_fail
        distill_record(rnds, opts, feats, idx_shuffle, mon_feats, cfg, nov_log)
        update_end_step.append(len(nov_log[mus[0]]) - 1)
        phase = "MASK" if masking else "free"
        novs = "  ".join(f"μ{mu}={nov_log[mu][-1].mean():.2e}" for mu in mus)
        print(f"  [{label:8s}] update {u + 1:>2}/{total_updates} [{phase}]  "
              f"visits={int(visits.sum())}  mask_fail={mask_fail}  {novs}", flush=True)

    nov = np.stack([np.stack(nov_log[mu], axis=1) for mu in mus], axis=0)  # (n_mus, n_mon, R)
    return nov, update_end_step, visits_per_update, mask_fail_per_update


def main():
    cfg = _parse()
    HERE = Path(__file__).resolve().parent
    RES_DIR = HERE / "results"
    RES_DIR.mkdir(parents=True, exist_ok=True)
    LEVEL_TAG = f"{cfg.game}_L{cfg.level + 1}{cfg.suffix}"

    rng = np.random.default_rng(cfg.seed)
    envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps,
                           seed=cfg.seed, level_index=cfg.level)
    W = build_projection(cfg.seed, cfg.proj_dim)

    # ── 1. harvest the monitored near-reset states + initial block set ──────────
    monitored_masked, monitored_keys, blocked_sa, hstats = harvest_monitor_states(
        envs, rng, cfg.n_monitor, cfg.monitor_max_dist, cfg.pre_roam)
    n_monitor = len(monitored_keys)
    monitored_set = set(monitored_keys)
    mon_feats = project_chunked(np.stack(monitored_masked), W).to(DEVICE)   # (n_monitor, dim)
    print(f"[exp_014_6/{LEVEL_TAG}] monitoring {n_monitor} states  "
          f"first_step={hstats['first_step']}  harvest_visits={hstats['harvest_visits']}  "
          f"blocked(s,a)={hstats['n_blocked_sa']}  μ={cfg.mu}")

    # leak values: 0 = standard RND, then the leaky values (cfg.mu, optional cfg.mu2)
    mus = [0.0, cfg.mu] + ([cfg.mu2] if cfg.mu2 and cfg.mu2 > 0 else [])

    # ── 2. run BOTH conditions on the same monitored states / identical phase 1 ──
    #   • MASKED   : phase 2 blocks the monitored states (they are abandoned).
    #   • UNMASKED : phase 2 stays free (the monitored states keep being visited) — the
    #                control overlay showing what "still visited" looks like.
    nov_m, ues, vis_m, mf_m = run_condition(
        envs, W, monitored_keys, monitored_set, blocked_sa, mon_feats, cfg, mus,
        do_mask=True, label="masked")
    nov_u, _ues_u, vis_u, mf_u = run_condition(
        envs, W, monitored_keys, monitored_set, blocked_sa, mon_feats, cfg, mus,
        do_mask=False, label="unmasked")

    # ── 3. save raw series (.npz) + summary (.json) ────────────────────────────
    npz_path = RES_DIR / f"organic_forget_series_{LEVEL_TAG}.npz"
    np.savez_compressed(
        npz_path,
        nov_masked=nov_m, nov_unmasked=nov_u,        # (n_mus, n_monitor, n_records)
        mus=np.array(mus, dtype=np.float32),
        update_end_step=np.array(ues, dtype=np.int64),
        visits_per_update_masked=vis_m, visits_per_update_unmasked=vis_u,
        mask_fail_per_update=mf_m,
        updates_free=cfg.updates_free, updates_masked=cfg.updates_masked,
        minibatches=cfg.minibatches, rnd_epochs=cfg.rnd_epochs,
        rollout_env_steps=cfg.rollout_steps * cfg.n_envs,
        mu=cfg.mu, first_step=np.array(hstats["first_step"], dtype=np.int64),
    )
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "game": cfg.game, "level_index": cfg.level, "level_tag": LEVEL_TAG,
        "n_monitor": n_monitor, "monitor_first_step": hstats["first_step"],
        "harvest_visits": hstats["harvest_visits"], "mus": mus,
        "updates_free": cfg.updates_free, "updates_masked": cfg.updates_masked,
        "rollout_env_steps_per_update": cfg.rollout_steps * cfg.n_envs,
        "minibatches": cfg.minibatches, "rnd_epochs": cfg.rnd_epochs,
        "leaks_per_update": cfg.minibatches * cfg.rnd_epochs,
        "mu": cfg.mu, "mu2": cfg.mu2, "rnd_lr": cfg.rnd_lr,
        "proj_dim": cfg.proj_dim, "seed": cfg.seed,
        "visits_per_update_masked": vis_m.tolist(),
        "visits_per_update_unmasked": vis_u.tolist(),
        "mask_fail_per_update": mf_m.tolist(),
        "npz": str(npz_path),
        "end_novelty": {
            f"mu_{mu}": {"masked": float(nov_m[i, :, -1].mean()),
                         "unmasked": float(nov_u[i, :, -1].mean())}
            for i, mu in enumerate(mus)
        },
    }
    json_path = RES_DIR / f"organic_forget_summary_{LEVEL_TAG}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[exp_014_6] wrote series  -> {npz_path}")
    print(f"[exp_014_6] wrote summary -> {json_path}")
    print(f"[exp_014_6] masked visits/update:   {vis_m.sum(axis=1).tolist()}")
    print(f"[exp_014_6] unmasked visits/update: {vis_u.sum(axis=1).tolist()}")
    for i, mu in enumerate(mus):
        print(f"[exp_014_6] μ={mu}: end masked={nov_m[i, :, -1].mean():.3e}  "
              f"unmasked={nov_u[i, :, -1].mean():.3e}")

    # ── 5. plot ────────────────────────────────────────────────────────────────
    if not cfg.no_plot:
        from JEPA.experiments.exp_014_figures_and_results.exp_014_6_organic_forget.plot import (
            make_figures,
        )
        make_figures(npz_path, HERE / "figures", LEVEL_TAG)


if __name__ == "__main__":
    main()
