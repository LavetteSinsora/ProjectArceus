"""exp_012_2 — multi-step curiosity probe on the EXISTING ICMModule.

Question (from the design review): the proposed fix is to replace ICM's 1-step
forward-prediction-error reward with a k-step discounted rollout error. Before
building anything, answer the cheap empirical question that decides whether the
idea can help on a small deterministic maze (LS20 L1):

    As the ICM forward model learns the dynamics and the 1-step curiosity
    collapses (~1e-5 in ~20 updates, per exp_011_0), does the k-step (open-loop
    rollout) error decay *slower* and retain enough magnitude AND cross-state
    spread to still drive exploration?

If the 5-step error also collapses to flat ~0, multi-step buys nothing here and
we should pivot (disagreement / RND). If it stays meaningfully larger and keeps
state-to-state spread, the multi-step idea is worth implementing.

This probe does TWO things, both on the unmodified `ICMModule` (no policy
retraining, no new model):

  (1) REPLAY  — collect one pool of random-policy transitions on LS20 L1, train a
                FRESH ICMModule with the existing 1-step ICM loss
                (icm.losses_on_batch, beta-weighted), and every update read out,
                on a held-out window set, the per-horizon h=1..K error for two
                rollout modes:
                  * open-loop  (imagination): phi_hat_{t+h} = f(... f(f(phi_t,a_t),a_{t+1})...)
                  * teacher-forced:           one f-step from the TRUE phi_{t+h-1}
                plus the discounted-sum reward sum_h gamma^{h-1} err_ol[h], the
                cross-state std of each horizon's error, and ||phi|| norms.
                This reproduces the collapse and shows each horizon's RELATIVE
                decay — the thing the checkpoints (saved only every 102.4k steps,
                all post-collapse) cannot show.

  (2) SNAPSHOT — the same per-horizon read-out evaluated on the real trained
                exp_011_0 ICM checkpoints (4 steps x 3 seeds), to confirm the
                replay matches reality at the stages that were actually saved.

Caveats (stated honestly in the README): the replay trains on a fixed random
policy pool, a proxy for the near-uniform early-training regime where the real
collapse happens; on a small deterministic maze the held-out envs traverse the
same state space, so generalisation error collapses too — which is itself the
finding. Definitive per-horizon decay timing would need frequent early ICM
checkpoints (a cheap re-run knob), which is the *next* step, not this one.

Run:
    uv run python -m JEPA.experiments.exp_012_ls20_intrinsic_exploration.exp_012_2_multistep_probe.probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"


# ── data collection ─────────────────────────────────────────────────────────

def collect_random_pool(cfg_env: dict, n_envs: int, steps: int, seed: int):
    """Roll out a uniform-random policy on LS20 L1. Returns trajectories as
    obs (steps+1, n_envs, F, F) uint8, actions (steps, n_envs) int64,
    dones (steps, n_envs) bool. obs[t] is the state BEFORE action[t]."""
    envs = VecLS20EnvLevel(
        cfg_env["env_name"], n_envs=n_envs,
        max_episode_steps=cfg_env["max_episode_steps"], seed=seed,
        level_index=cfg_env.get("level_index", 0),
    )
    F = envs.FRAME if hasattr(envs, "FRAME") else 64
    n_actions = cfg_env["n_actions"]
    rng = np.random.default_rng(seed)

    obs = np.zeros((steps + 1, n_envs, F, F), dtype=np.uint8)
    acts = np.zeros((steps, n_envs), dtype=np.int64)
    dones = np.zeros((steps, n_envs), dtype=bool)

    obs[0] = envs.current_obs()
    for t in range(steps):
        a = rng.integers(0, n_actions, size=n_envs).astype(np.int64)
        no, _r, d, _info = envs.step(a)
        obs[t + 1] = no
        acts[t] = a
        dones[t] = d
    return obs, acts, dones, F


def make_windows(obs, acts, dones, K: int, env_slice, max_windows: int):
    """Slice contiguous (K+1)-frame windows with NO internal done.
    Returns seg_obs (B, K+1, F, F) uint8, seg_act (B, K) int64."""
    steps, n_envs = acts.shape
    seg_o, seg_a = [], []
    for n in range(*env_slice.indices(n_envs)):
        t = 0
        while t + K < steps + 1 and len(seg_o) < max_windows:
            # window uses actions t..t+K-1 and frames t..t+K; invalid if any
            # action in the window ended an episode (frame after is a reset).
            if t + K - 1 < steps and not dones[t:t + K, n].any():
                seg_o.append(obs[t:t + K + 1, n])
                seg_a.append(acts[t:t + K, n])
                t += K  # non-overlapping windows -> less correlated samples
            else:
                t += 1
        if len(seg_o) >= max_windows:
            break
    seg_o = np.stack(seg_o).astype(np.uint8)
    seg_a = np.stack(seg_a).astype(np.int64)
    return seg_o, seg_a


def make_transitions(obs, acts, dones, env_slice):
    """Flat (o, o', a) 1-step transitions (excluding episode-ending steps)."""
    steps, n_envs = acts.shape
    lo, hi = env_slice.indices(n_envs)[:2]
    o, no, a = [], [], []
    for n in range(lo, hi):
        for t in range(steps):
            if not dones[t, n]:
                o.append(obs[t, n]); no.append(obs[t + 1, n]); a.append(acts[t, n])
    return (np.stack(o).astype(np.uint8), np.stack(no).astype(np.uint8),
            np.array(a, dtype=np.int64))


# ── per-horizon read-out (the core measurement) ──────────────────────────────

@torch.no_grad()
def horizon_errors(icm: ICMModule, seg_o: torch.Tensor, seg_a: torch.Tensor,
                   K: int, gamma: float, device, chunk: int = 512):
    """Per-horizon squared forward error on windows.

    Returns dict with, for h=1..K (1-indexed): open-loop mean/std, teacher-forced
    mean/std, relative (norm-normalised) open-loop mean; plus discounted-sum
    rewards and mean ||phi||^2. Errors use sum-over-D squared L2 (matches the ICM
    reward def r^i = (eta/2)||phi_hat - phi'||^2)."""
    B = seg_o.shape[0]
    ol = np.zeros((B, K)); tf = np.zeros((B, K)); rel = np.zeros((B, K))
    phisq = np.zeros((B, K + 1))
    for s in range(0, B, chunk):
        o = seg_o[s:s + chunk].to(device)        # (b, K+1, F, F)
        a = seg_a[s:s + chunk].to(device)        # (b, K)
        b = o.shape[0]
        phi = icm.encode(o.reshape(-1, o.shape[-2], o.shape[-1])).reshape(b, K + 1, -1)
        phisq[s:s + chunk] = (phi ** 2).sum(-1).cpu().numpy()
        cur = phi[:, 0]                           # open-loop seed
        for h in range(1, K + 1):
            # open-loop: feed the model's own prediction forward
            pred_ol = icm.predict_next(cur, a[:, h - 1])
            e_ol = (pred_ol - phi[:, h]).pow(2).sum(-1)
            ol[s:s + chunk, h - 1] = e_ol.cpu().numpy()
            rel[s:s + chunk, h - 1] = (e_ol / (phi[:, h].pow(2).sum(-1) + 1e-8)).cpu().numpy()
            cur = pred_ol
            # teacher-forced: one step from the TRUE previous latent
            pred_tf = icm.predict_next(phi[:, h - 1], a[:, h - 1])
            tf[s:s + chunk, h - 1] = (pred_tf - phi[:, h]).pow(2).sum(-1).cpu().numpy()

    disc = np.array([gamma ** h for h in range(K)])     # gamma^0..gamma^{K-1}
    out = {
        "ol_mean": ol.mean(0).tolist(), "ol_std": ol.std(0).tolist(),
        "tf_mean": tf.mean(0).tolist(), "tf_std": tf.std(0).tolist(),
        "rel_ol_mean": rel.mean(0).tolist(),
        "phi_sq_mean": phisq.mean(0).tolist(),
        # the agent's would-be reward (per window) for each definition:
        "reward_1step_mean": float(ol[:, 0].mean()),       # current ICM reward
        "reward_1step_std": float(ol[:, 0].std()),
        "reward_kstep_disc_mean": float((ol * disc).sum(1).mean()),  # proposed
        "reward_kstep_disc_std": float((ol * disc).sum(1).std()),
        # decisive ratios: does deeper horizon retain magnitude / spread?
        "ratio_olK_ol1_mean": float(ol[:, -1].mean() / (ol[:, 0].mean() + 1e-12)),
        "ratio_disc_1step_mean": float((ol * disc).sum(1).mean() / (ol[:, 0].mean() + 1e-12)),
    }
    return out


# ── (1) replay: train fresh ICM, log horizon decay ──────────────────────────

def run_replay(cfg_env, args, seg_o, seg_a, device):
    o_tr, no_tr, a_tr = make_transitions(*args._pool_train)
    o_tr = torch.from_numpy(o_tr); no_tr = torch.from_numpy(no_tr); a_tr = torch.from_numpy(a_tr)
    n = o_tr.shape[0]
    print(f"[replay] {n} train transitions, {seg_o.shape[0]} held-out windows")

    icm = ICMModule(n_actions=cfg_env["n_actions"], n_colors=cfg_env["n_colors"],
                    frame_size=cfg_env["frame_size"], trunk_dim=cfg_env["trunk_dim"],
                    hidden=cfg_env["icm_hidden"]).to(device)
    opt = torch.optim.Adam(icm.parameters(), lr=cfg_env["icm_lr"])
    mb = max(1, n // cfg_env["minibatches"])
    beta = cfg_env["beta"]

    history = []
    # update 0 = untrained read-out
    history.append({"update": 0, "fwd_loss": float("nan"),
                    **horizon_errors(icm, seg_o, seg_a, args.K, args.gamma, device)})
    idx = np.arange(n)
    for update in range(1, args.replay_updates + 1):
        np.random.shuffle(idx)
        fl = 0.0; steps = 0
        for start in range(0, n, mb):
            sel = idx[start:start + mb]
            o = o_tr[sel].to(device); no = no_tr[sel].to(device); a = a_tr[sel].to(device)
            l_inv, l_fwd, _acc, _err = icm.losses_on_batch(o, no, a)
            loss = (1.0 - beta) * l_inv + beta * l_fwd
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(icm.parameters(), cfg_env["grad_clip"])
            opt.step()
            fl += float(l_fwd.item()); steps += 1
        rec = horizon_errors(icm, seg_o, seg_a, args.K, args.gamma, device)
        rec["update"] = update; rec["fwd_loss"] = fl / steps
        history.append(rec)
        if update % max(1, args.replay_updates // 10) == 0 or update == 1:
            print(f"[replay] u{update:3d}  fwd_loss={rec['fwd_loss']:.4g}  "
                  f"r_1step={rec['reward_1step_mean']:.4g}  "
                  f"r_kdisc={rec['reward_kstep_disc_mean']:.4g}  "
                  f"olK/ol1={rec['ratio_olK_ol1_mean']:.2f}")
    return history


# ── (2) snapshot: real trained checkpoints ──────────────────────────────────

def find_checkpoints(repo_root: Path):
    base = (repo_root / "JEPA/experiments/exp_011_ls20_exploration/"
            "exp_011_0_icm_baseline/checkpoints")
    runs = {}
    for run_dir in sorted(base.glob("*")):
        if not run_dir.is_dir():
            continue
        cps = sorted(run_dir.glob("step_*.pt"),
                     key=lambda p: int(p.stem.split("_")[1]))
        if cps:
            runs[run_dir.name] = cps
    return runs


def run_snapshot(args, seg_o, seg_a, device, repo_root):
    runs = find_checkpoints(repo_root)
    out = {}
    for run_name, cps in runs.items():
        out[run_name] = []
        for cp in cps:
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            c = ck["config"]
            icm = ICMModule(n_actions=c["n_actions"], n_colors=c["n_colors"],
                            frame_size=c["frame_size"], trunk_dim=c["trunk_dim"],
                            hidden=c["icm_hidden"]).to(device)
            icm.load_state_dict(ck["icm"]); icm.eval()
            rec = horizon_errors(icm, seg_o, seg_a, args.K, args.gamma, device)
            rec["step"] = int(ck["step"]); rec["eta"] = ck.get("eta")
            out[run_name].append(rec)
            print(f"[snapshot] {run_name} step={ck['step']:>8d}  "
                  f"r_1step={rec['reward_1step_mean']:.4g}  "
                  f"r_kdisc={rec['reward_kstep_disc_mean']:.4g}  "
                  f"olK/ol1={rec['ratio_olK_ol1_mean']:.2f}")
    return out


# ── figures ──────────────────────────────────────────────────────────────────

def make_figures(replay, snapshot, K, gamma):
    FIG.mkdir(parents=True, exist_ok=True)
    u = [r["update"] for r in replay]

    # Fig 1: per-horizon open-loop error decay during replay (log y)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ol = np.array([r["ol_mean"] for r in replay])      # (U, K)
    for h in range(K):
        ax[0].plot(u, ol[:, h] + 1e-12, label=f"h={h+1}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("ICM update"); ax[0].set_ylabel("open-loop sq error (mean)")
    ax[0].set_title("Per-horizon open-loop error decay (replay)"); ax[0].legend()
    ax[0].grid(alpha=0.3)

    r1 = np.array([r["reward_1step_mean"] for r in replay])
    rk = np.array([r["reward_kstep_disc_mean"] for r in replay])
    ax[1].plot(u, r1 + 1e-12, label="1-step (current ICM reward)")
    ax[1].plot(u, rk + 1e-12, label=f"k-step disc (gamma={gamma})")
    ax[1].set_yscale("log"); ax[1].set_xlabel("ICM update"); ax[1].set_ylabel("would-be reward (mean)")
    ax[1].set_title("Current vs proposed reward (replay)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig1_replay_decay.png", dpi=120); plt.close(fig)

    # Fig 2: cross-state spread (std) — a dead signal can't differentiate states
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    olstd = np.array([r["ol_std"] for r in replay])
    for h in range(K):
        ax[0].plot(u, olstd[:, h] + 1e-12, label=f"h={h+1}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("ICM update"); ax[0].set_ylabel("open-loop error std across states")
    ax[0].set_title("Cross-state spread by horizon (replay)"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ratio = np.array([r["ratio_disc_1step_mean"] for r in replay])
    ax[1].plot(u, ratio); ax[1].axhline(1.0, color="k", ls="--", alpha=0.5)
    ax[1].set_xlabel("ICM update"); ax[1].set_ylabel("mean(k-step disc) / mean(1-step)")
    ax[1].set_title("Signal-magnitude gain of k-step over 1-step"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig2_replay_spread.png", dpi=120); plt.close(fig)

    # Fig 3: snapshot — per-horizon open-loop error vs real training step
    if snapshot:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for run_name, recs in snapshot.items():
            steps = [r["step"] for r in recs]
            r1 = [r["reward_1step_mean"] for r in recs]
            rk = [r["reward_kstep_disc_mean"] for r in recs]
            ax[0].plot(steps, np.array(r1) + 1e-12, marker="o", label=f"{run_name[:18]} 1-step")
            ax[1].plot(steps, np.array(rk) + 1e-12, marker="o", label=f"{run_name[:18]} k-disc")
        for a, t in zip(ax, ["1-step reward (real ckpts)", "k-step disc reward (real ckpts)"]):
            a.set_yscale("log"); a.set_xlabel("env step"); a.set_title(t); a.grid(alpha=0.3); a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "fig3_snapshot.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--replay_updates", type=int, default=80)
    ap.add_argument("--max_windows", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_snapshot", action="store_true")
    args = ap.parse_args()

    device = get_device()
    repo_root = HERE.parents[3]
    print(f"[exp_012_2] device={device}")

    # env / model config (faithful to exp_011_0; values match its config_base)
    cfg_env = dict(env_name="ls20", max_episode_steps=200, level_index=0,
                   n_actions=4, n_colors=16, frame_size=64, trunk_dim=256,
                   icm_hidden=256, icm_lr=1e-3, beta=0.2, minibatches=4, grad_clip=0.5)

    print("[exp_012_2] collecting random-policy pool on LS20 L1 ...")
    obs, acts, dones, F = collect_random_pool(cfg_env, args.n_envs, args.steps, args.seed)
    n_envs = acts.shape[1]
    train_envs = slice(0, max(1, int(n_envs * 0.75)))
    held_envs = slice(max(1, int(n_envs * 0.75)), n_envs)
    args._pool_train = (obs, acts, dones, train_envs)

    seg_o_np, seg_a_np = make_windows(obs, acts, dones, args.K, held_envs, args.max_windows)
    seg_o = torch.from_numpy(seg_o_np); seg_a = torch.from_numpy(seg_a_np)
    print(f"[exp_012_2] held-out windows: {seg_o.shape}, actions {seg_a.shape}")

    replay = run_replay(cfg_env, args, seg_o, seg_a, device)
    snapshot = {} if args.skip_snapshot else run_snapshot(args, seg_o, seg_a, device, repo_root)

    make_figures(replay, snapshot, args.K, args.gamma)

    arg_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    results = {"args": arg_dict, "cfg_env": cfg_env,
               "replay": replay, "snapshot": snapshot}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))

    # terminal verdict line
    last = replay[-1]
    print("\n================ VERDICT (replay, final update) ================")
    print(f"  1-step reward (current ICM): {last['reward_1step_mean']:.4g} "
          f"(std across states {last['reward_1step_std']:.4g})")
    print(f"  k-step disc reward (proposed): {last['reward_kstep_disc_mean']:.4g} "
          f"(std {last['reward_kstep_disc_std']:.4g})")
    cv1 = last['reward_1step_std'] / (last['reward_1step_mean'] + 1e-12)
    cvk = last['reward_kstep_disc_std'] / (last['reward_kstep_disc_mean'] + 1e-12)
    print(f"  magnitude gain  mean(k-disc)/mean(1-step) = {last['ratio_disc_1step_mean']:.2f}x")
    print(f"  horizon-K/horizon-1 open-loop error ratio  = {last['ratio_olK_ol1_mean']:.2f}x")
    print(f"  cross-state CV (std/mean): 1-step={cv1:.3f}  k-disc={cvk:.3f}  "
          f"(higher CV => better differentiates states for exploration)")
    print(f"  -> figures in {FIG}")
    print("================================================================")


if __name__ == "__main__":
    main()
