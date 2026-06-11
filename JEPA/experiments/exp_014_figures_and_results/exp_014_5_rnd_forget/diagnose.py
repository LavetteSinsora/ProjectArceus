"""exp_014_5 — RND FORGETTING under a controlled distillation schedule.

This is a clean, fully-controlled distillation experiment (NOT env roaming). We fix a
small set of real LS20-L2 masked-board states and drive the predictor by hand, one
state per env step, so the visitation schedule is exactly what we specify.

PROTOCOL
--------
  • Pick N_STATES (=20) real, mutually-distinct masked-board states from LS20 Level 2.
  • Phase 1  (env steps 1 .. PHASE1):
        every env step, pick ONE of the 20 states uniformly at random and run ONE
        distillation step on it. (The leaky instance also leaks every step.)
        → all 20 states get driven down.
  • Phase 2  (env steps PHASE1+1 .. PHASE1+PHASE2):
        select N_KEPT (=15) of the 20 states. Every env step, pick ONE of those 15
        uniformly at random and distill it (leak fires on the leak-cadence boundary).
        The other 5 states are ABANDONED — never distilled again.

We run TWO RND predictors over the SAME state set, the SAME schedule and IDENTICAL
init — the ONLY difference is the leak:
    • standard RND   (μ = 0)   — one-way error ratchet, never forgets.
    • leaky RND      (μ > 0)   — shrinks toward init → forgets the abandoned states.

LEAK CADENCE (important): the real exp_013 loop applies the leak ONCE PER PPO UPDATE
(~2048 env steps of distillation), NOT once per env step. So a per-env-step leak with
the headline μ=0.05 is ~2048× too aggressive and pegs every state to the random-init
novelty (no learning is ever retained → kept ≈ abandoned, no signal). We therefore leak
once every `--leak-every` steps. Default μ=0.05 / leak-every=100 reproduces the headline
μ you actually trained with and gives the most prominent effect: between leaks the
predictor is re-distilled ~100 times, so kept states recover their floor while abandoned
states accumulate forgetting per leak-cycle. (Use --leak-every 1 --mu 0.005 for the
gentle per-step regime.)

At EVERY env step we record the novelty ½·‖P(φ)−T(φ)‖² of ALL N_STATES for BOTH
predictors. The raw (N_STATES, total_steps) series for each predictor is saved to an
.npz so any pair of states can be re-plotted instantly (see plot.py).

EXPECTED PATTERN (the thing we are proving)
-------------------------------------------
Standard RND (μ=0):
    • a KEPT state (one of the 5)  → keeps being distilled → novelty goes EVEN LOWER.
    • an ABANDONED state           → predictor never forgets → novelty stays flat-LOW.
Leaky RND (μ>0):
    • a KEPT state (one of the 5)  → re-learned every few steps → novelty stays FLAT.
    • an ABANDONED state           → leak pulls predictor to init → novelty RISES.

So leaky RND is a *recency* signal: it re-opens the abandoned states for exploration,
while standard RND leaves them permanently "known".

CPU-only (other agents train on MPS). matplotlib Agg via plot.py.

Run:
    uv run python -m \
      JEPA.experiments.exp_014_figures_and_results.exp_014_5_rnd_forget.diagnose
    uv run ... --mu 0.05 --leak-every 100          # default: headline μ, prominent
    uv run ... --mu 0.005 --leak-every 1           # gentle per-step regime
    uv run ... --phase1 10000 --phase2 10000 --n-states 20 --n-kept 5
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
    """Frozen random linear map one-hot board (64*64*16) → proj_dim.

    Local, dim-parameterised copy of exp_014_1.build_projection (whose dim is fixed
    at 256). A larger proj_dim yields more nearly-orthogonal per-state features, which
    reduces how much the shared predictor's fit of the kept states generalises to (and
    artificially suppresses) the abandoned states — letting the leak regenerate them."""
    g = torch.Generator().manual_seed(seed)
    in_dim = FRAME * FRAME * N_COLORS
    return torch.randn(in_dim, proj_dim, generator=g) / (in_dim ** 0.5)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20")
    p.add_argument("--level", type=int, default=1, help="0-indexed (1 = LS20 L2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-states", type=int, default=20, help="distinct states in the pool")
    p.add_argument("--n-kept", type=int, default=15, help="states still distilled in phase 2 "
                   "(the rest are abandoned; default 15 kept / 5 abandoned)")
    p.add_argument("--phase1", type=int, default=10000, help="env steps, all states distilled")
    p.add_argument("--phase2", type=int, default=10000, help="env steps, only kept states distilled")
    p.add_argument("--mu", type=float, default=0.05,
                   help="leak for the leaky predictor (the headline value used in exp_013; "
                        "needs --leak-every>1, else a per-step 0.05 pegs everything to init)")
    p.add_argument("--leak-every", type=int, default=100,
                   help="apply the leak once every N env steps instead of every step. "
                        "The real exp_013 loop leaks ONCE PER PPO UPDATE (~2048 env steps "
                        "of distillation), so a faithful per-step rate is ~μ/2048. Set "
                        "--leak-every ~100-200 to use the headline μ=0.05/0.1 you actually "
                        "trained with: the predictor is re-distilled N times between leaks, "
                        "so kept states recover while abandoned ones forget per update-cycle.")
    p.add_argument("--rnd-lr", type=float, default=1e-4,
                   help="Adam lr for the predictor. Kept gentle: a single big Adam step per "
                        "env step (single-sample distill) otherwise causes huge per-step "
                        "novelty swings AND catastrophic interference that lifts the abandoned "
                        "states even at μ=0 — masking the leak as the cause of forgetting.")
    p.add_argument("--proj-dim", type=int, default=1024,
                   help="Frozen-projection / RND feature dim. Larger ⇒ more orthogonal "
                        "per-state features ⇒ less cross-state generalization, so fitting the "
                        "5 kept states does not artificially suppress the 15 abandoned ones.")
    p.add_argument("--pre-roam", type=int, default=600, help="per-env steps to harvest states")
    p.add_argument("--suffix", default="", help="appended to the level tag so results/figures "
                   "do not overwrite an existing run (e.g. --suffix _ppo_update)")
    p.add_argument("--no-plot", action="store_true", help="skip figure generation")
    return p.parse_args()


def harvest_states(envs, rng, W, n_states, pre_roam):
    """Short uniform-random roam → n_states real, mutually-distinct masked states.

    Same idea as exp_014_1: greedily pick the frequently-visited states whose frozen
    projections are farthest apart, so each is individually memorisable and the
    predictor can fit / forget them independently (no mutual-interference floor that
    would hide the leak)."""
    visit: dict[bytes, int] = defaultdict(int)
    exemplar: dict[bytes, np.ndarray] = {}
    for _ in range(pre_roam):
        a = rng.integers(0, envs.n_actions, size=envs.n_envs)
        nobs, _r, dones, _i = envs.step(a)
        m = mask_board(nobs)
        for i in range(envs.n_envs):
            if dones[i]:
                continue
            k = state_key(m[i])
            visit[k] += 1
            exemplar.setdefault(k, m[i])

    ranked = sorted(visit.items(), key=lambda kv: kv[1], reverse=True)
    pool = [k for k, c in ranked if c >= 20]
    if len(pool) < n_states:
        pool = [k for k, _ in ranked]
    pool = pool[: max(n_states * 4, n_states)]          # cap candidate set
    pool_masked = np.stack([exemplar[k] for k in pool])
    feats = project_states(pool_masked, W).numpy()

    chosen_i = [0]                                       # seed with most-visited
    while len(chosen_i) < min(n_states, len(pool)):
        d = np.min([np.linalg.norm(feats - feats[c], axis=1) for c in chosen_i], axis=0)
        d[chosen_i] = -1.0
        chosen_i.append(int(np.argmax(d)))
    chosen = [pool[i] for i in chosen_i]
    return [exemplar[k] for k in chosen], [visit[k] for k in chosen]


def main():
    cfg = _parse()
    HERE = Path(__file__).resolve().parent
    RES_DIR = HERE / "results"
    RES_DIR.mkdir(parents=True, exist_ok=True)
    LEVEL_TAG = f"{cfg.game}_L{cfg.level + 1}{cfg.suffix}"

    rng = np.random.default_rng(cfg.seed)
    envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=16, max_episode_steps=200,
                           seed=cfg.seed, level_index=cfg.level)
    W = build_projection(cfg.seed, cfg.proj_dim)

    # ── 1. harvest the fixed pool of real states ───────────────────────────────
    masked_states, pre_counts = harvest_states(envs, rng, W, cfg.n_states, cfg.pre_roam)
    n_states = len(masked_states)
    if n_states < cfg.n_states:
        print(f"[exp_014_5] WARNING: only {n_states} distinct states available "
              f"(asked {cfg.n_states})")
    feats = project_states(np.stack(masked_states), W).to(DEVICE)     # (n_states, dim)

    # kept set = first n_kept indices (state ordering is arbitrary / by visit rank)
    n_kept = min(cfg.n_kept, n_states)
    kept_idx = np.arange(n_kept)
    abandoned_idx = np.arange(n_kept, n_states)
    print(f"[exp_014_5/{LEVEL_TAG}] {n_states} states  kept={list(kept_idx)}  "
          f"abandoned={list(abandoned_idx)}  μ={cfg.mu}  lr={cfg.rnd_lr}")

    # ── 2. two RND predictors with IDENTICAL init (only the leak differs) ───────
    rnds, opts = {}, {}
    for mu in (0.0, cfg.mu):
        torch.manual_seed(cfg.seed + 100)            # identical target+predictor init
        r = RNDPhi(dim=cfg.proj_dim, hidden=256, out=256, leak=mu).to(DEVICE)
        rnds[mu] = r
        opts[mu] = torch.optim.Adam(r.predictor.parameters(), lr=cfg.rnd_lr)

    total = cfg.phase1 + cfg.phase2
    # nov[mu] : (n_states, total) novelty of every state at every env step (pre-distill)
    nov = {mu: np.zeros((n_states, total), dtype=np.float32) for mu in rnds}
    # distill_count[idx] across each phase (for the JSON summary / sanity)
    distill_p1 = np.zeros(n_states, dtype=np.int64)
    distill_p2 = np.zeros(n_states, dtype=np.int64)

    # ── 3. the controlled per-env-step distillation schedule ───────────────────
    for t in range(total):
        phase2 = t >= cfg.phase1

        # (a) record novelty of ALL states for BOTH predictors, BEFORE this step's distill
        for mu, r in rnds.items():
            with torch.no_grad():
                nov[mu][:, t] = r.novelty(feats).cpu().numpy()

        # (b) choose ONE state to distill this step (shared across both predictors)
        if phase2:
            j = int(rng.choice(kept_idx))
            distill_p2[j] += 1
        else:
            j = int(rng.integers(0, n_states))
            distill_p1[j] += 1
        target_feat = feats[j: j + 1]                # (1, dim)

        # (c) one distill step on the chosen state for both predictors; leak only on
        #     the leak-cadence boundary (every cfg.leak_every steps). With leak_every=1
        #     the leak fires every step; larger values emulate the real per-update leak.
        do_leak = ((t + 1) % cfg.leak_every) == 0
        for mu, r in rnds.items():
            opt = opts[mu]
            opt.zero_grad()
            r.distill_loss(target_feat).backward()
            opt.step()
            if do_leak:
                r.apply_leak()                       # μ=0 → no-op; μ>0 → forget one cycle

        if t % 2000 == 0 or t == total - 1:
            ka = kept_idx[0]
            aa = abandoned_idx[0] if len(abandoned_idx) else kept_idx[0]
            tag = "P2" if phase2 else "P1"
            print(f"  t{t:>6} {tag}  "
                  f"std[kept={nov[0.0][ka,t]:.2e} aband={nov[0.0][aa,t]:.2e}]  "
                  f"leaky[kept={nov[cfg.mu][ka,t]:.2e} aband={nov[cfg.mu][aa,t]:.2e}]",
                  flush=True)

    # ── 4. save raw series (.npz) + summary (.json) ────────────────────────────
    npz_path = RES_DIR / f"rnd_forget_series_{LEVEL_TAG}.npz"
    np.savez_compressed(
        npz_path,
        nov_std=nov[0.0],
        nov_leaky=nov[cfg.mu],
        kept_idx=kept_idx,
        abandoned_idx=abandoned_idx,
        phase1=cfg.phase1,
        phase2=cfg.phase2,
        mu=cfg.mu,
        leak_every=cfg.leak_every,
        pre_counts=np.array(pre_counts, dtype=np.int64),
    )
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "game": cfg.game, "level_index": cfg.level, "level_tag": LEVEL_TAG,
        "n_states": n_states, "n_kept": n_kept,
        "kept_idx": kept_idx.tolist(), "abandoned_idx": abandoned_idx.tolist(),
        "phase1": cfg.phase1, "phase2": cfg.phase2, "total_steps": total,
        "mu": cfg.mu, "leak_every": cfg.leak_every,
        "rnd_lr": cfg.rnd_lr, "proj_dim": cfg.proj_dim, "seed": cfg.seed,
        "distill_phase1": distill_p1.tolist(), "distill_phase2": distill_p2.tolist(),
        "npz": str(npz_path),
        "end_novelty": {
            "std": {"kept_mean": float(nov[0.0][kept_idx, -1].mean()),
                    "abandoned_mean": float(nov[0.0][abandoned_idx, -1].mean()) if len(abandoned_idx) else None},
            "leaky": {"kept_mean": float(nov[cfg.mu][kept_idx, -1].mean()),
                      "abandoned_mean": float(nov[cfg.mu][abandoned_idx, -1].mean()) if len(abandoned_idx) else None},
        },
    }
    json_path = RES_DIR / f"rnd_forget_summary_{LEVEL_TAG}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[exp_014_5] wrote series -> {npz_path}")
    print(f"[exp_014_5] wrote summary -> {json_path}")
    print("[exp_014_5] end-novelty means:")
    print(f"   standard: kept={summary['end_novelty']['std']['kept_mean']:.3e}  "
          f"abandoned={summary['end_novelty']['std']['abandoned_mean']}")
    print(f"   leaky   : kept={summary['end_novelty']['leaky']['kept_mean']:.3e}  "
          f"abandoned={summary['end_novelty']['leaky']['abandoned_mean']}")

    # ── 5. plot ────────────────────────────────────────────────────────────────
    if not cfg.no_plot:
        from JEPA.experiments.exp_014_figures_and_results.exp_014_5_rnd_forget.plot import (
            make_figures,
        )
        make_figures(npz_path, HERE / "figures", LEVEL_TAG)


if __name__ == "__main__":
    main()
