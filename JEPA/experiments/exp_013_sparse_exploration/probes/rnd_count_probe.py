"""RND-as-count probe: how fast does RND error saturate, and does it track
visit count? Three experiments, all on real LS20-L1 frames + the real exp_012
RND nets:

  (A) GLOBAL saturation curve — pull `intrinsic_reward_raw_mean` vs env-step from
      existing exp_013 RND run metrics (the real training novelty decay).
  (B) error vs VISIT COUNT — train the predictor on a fixed set of distinct
      states whose visit counts span a log range (1..~2000), then plot final
      per-state error against its visit count. Tells us whether error ∝ 1/N-ish
      (count-like) or floors out fast (saturates -> loses resolution at high N).
  (C) single-state decay — SGD on ONE state repeatedly; error vs #updates (the
      isolated decay curve + irreducible floor), plus a never-trained holdout
      state (generalisation / "0 visits" baseline).

Run: uv run python -m JEPA.experiments.exp_013_sparse_exploration.probes.rnd_count_probe
"""
from __future__ import annotations
import glob, json
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RNDTarget, RNDPredictor
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel

DEV = torch.device("cpu")          # small nets; keep off the (contended) GPU
FDIM = 256
LR = 1e-4                           # cfg.rnd_lr
OUT = Path(__file__).resolve().parent
np.random.seed(0); torch.manual_seed(0)


def err(pred_net, target_net, frames_u8):
    """mean_j (pred-target)^2 per frame -> (B,) numpy (the intrinsic raw error)."""
    with torch.no_grad():
        t = target_net(frames_u8.to(DEV))
        p = pred_net(frames_u8.to(DEV))
        return (p - t).pow(2).mean(dim=-1).cpu().numpy()


def collect_distinct_states(n_target=160, max_steps=4000):
    env = VecLS20EnvLevel("ls20", n_envs=8, max_episode_steps=200, seed=1, level_index=0)
    seen = {}
    obs = env.current_obs()
    for _ in range(max_steps):
        a = np.random.randint(0, env.n_actions, size=env.n_envs)
        obs, _, _, _ = env.step(a)
        for f in obs:
            seen.setdefault(f.tobytes(), f.copy())
        if len(seen) >= n_target:
            break
    frames = np.stack(list(seen.values()))[:n_target]
    return torch.from_numpy(frames)            # (M,64,64) uint8


def expA_global_saturation():
    rows = []
    for mj in glob.glob(str(OUT.parent / "runs/exp013_rnd_*/metrics.jsonl")):
        recs = [json.loads(l) for l in open(mj)]
        for r in recs:
            rows.append((r["step"], r.get("intrinsic_reward_raw_mean")))
    rows = [r for r in rows if r[1] is not None]
    rows.sort()
    return rows


def expB_error_vs_visits(frames):
    M = frames.shape[0]
    target = RNDTarget(feature_dim=FDIM).to(DEV)
    pred = RNDPredictor(feature_dim=FDIM).to(DEV)
    opt = torch.optim.Adam(pred.parameters(), lr=LR)

    # Visit counts spanning a log range; a few states get 0 (holdout).
    n_train = M - 12
    counts = np.unique(np.round(np.geomspace(1, 2000, n_train)).astype(int))
    # pad to n_train by repeating the larger buckets
    while len(counts) < n_train:
        counts = np.concatenate([counts, counts[-(n_train - len(counts)):]])
    counts = counts[:n_train]
    holdout_idx = np.arange(n_train, M)              # never trained
    train_idx = np.arange(n_train)

    err0 = err(pred, target, frames)                 # pre-training error (all)

    # Build a shuffled multiset of (state index repeated count_i times).
    stream = np.concatenate([np.full(c, i) for i, c in zip(train_idx, counts)])
    np.random.shuffle(stream)
    B = 64
    for s in range(0, len(stream), B):
        idx = stream[s:s + B]
        fb = frames[idx].to(DEV)
        with torch.no_grad():
            tb = target(fb)
        loss = (pred(fb) - tb).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    err1 = err(pred, target, frames)
    return {
        "counts": counts.tolist(),
        "err_before_mean": float(err0[train_idx].mean()),
        "err_after_per_state": err1[train_idx].tolist(),
        "holdout_err_before": float(err0[holdout_idx].mean()),
        "holdout_err_after": float(err1[holdout_idx].mean()),
    }


def expC_single_state_decay(frames, steps=600):
    target = RNDTarget(feature_dim=FDIM).to(DEV)
    pred = RNDPredictor(feature_dim=FDIM).to(DEV)
    opt = torch.optim.Adam(pred.parameters(), lr=LR)
    one = frames[0:1]
    holdout = frames[1:2]
    curve = []
    for k in range(steps + 1):
        if k % 20 == 0:
            curve.append((k, float(err(pred, target, one)[0]),
                          float(err(pred, target, holdout)[0])))
        with torch.no_grad():
            tb = target(one.to(DEV))
        loss = (pred(one.to(DEV)) - tb).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return curve                                     # (k, err_trained_state, err_holdout)


def main():
    print("collecting distinct LS20-L1 states ...")
    frames = collect_distinct_states()
    print(f"  {frames.shape[0]} distinct frames")

    print("\n=== (A) GLOBAL saturation during real training (raw novelty vs step) ===")
    for step, raw in expA_global_saturation()[::4]:
        print(f"  step {step:>7}  raw_novelty_mean {raw:.4g}")

    print("\n=== (B) error vs VISIT COUNT (final per-state error after n_i visits) ===")
    b = expB_error_vs_visits(frames)
    counts = np.array(b["counts"]); errs = np.array(b["err_after_per_state"])
    print(f"  pre-train mean error (all states): {b['err_before_mean']:.4g}")
    order = np.argsort(counts)
    print(f"  {'visits':>7} {'final_err':>12} {'err/err0':>10}")
    for i in order:
        print(f"  {counts[i]:>7} {errs[i]:>12.4g} {errs[i]/b['err_before_mean']:>10.3f}")
    print(f"  HOLDOUT (0 visits): before {b['holdout_err_before']:.4g} -> "
          f"after {b['holdout_err_after']:.4g}  "
          f"(generalisation drop {1 - b['holdout_err_after']/b['holdout_err_before']:.1%})")

    print("\n=== (C) single-state error vs #SGD updates (decay + floor) ===")
    c = expC_single_state_decay(frames)
    print(f"  {'updates':>7} {'err_state':>12} {'err_holdout':>12}")
    for k, e, h in c[::3]:
        print(f"  {k:>7} {e:>12.4g} {h:>12.4g}")

    (OUT / "rnd_count_results.json").write_text(json.dumps(
        {"B": b, "C": c, "A": expA_global_saturation()}, indent=2))
    print(f"\nsaved {OUT/'rnd_count_results.json'}")


if __name__ == "__main__":
    main()
