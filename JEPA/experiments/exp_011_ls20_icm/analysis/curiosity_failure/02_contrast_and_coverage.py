"""02 — The decisive test: forward-error CONTRAST (first-seen vs revisited) and
state COVERAGE, from saved checkpoints, on fresh on-policy rollouts.

For each checkpoint of L2 (failed) and L1 (solved, for reference):
  * rebuild policy (ActorCritic) + ICM from the saved state_dicts and saved eta,
  * collect a fresh on-policy rollout at the correct level,
  * hash each next-frame with UI rows 61-62 masked (the built-in distractor) to
    get exact state identity,
  * compute per-transition forward error r^i and split it into FIRST-SEEN vs
    REVISITED transitions within the rollout,
  * count unique masked states reached (coverage), and compare to a uniform
    RANDOM policy under the same budget.

This separates the three hypotheses:
  (a) error -> 0 everywhere (perfect model): max error tiny on first-seen too
  (b) nonzero error but no CONTRAST: first-seen ~ revisited error
  (c) eta makes r^i negligible vs PPO yardsticks (entropy bonus 0.01, +1 reward)

Outputs printed numbers + figures/fig2_contrast.png, figures/fig3_coverage.png.
"""
from __future__ import annotations
import os, glob, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout
from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule, intrinsic_raw_error
from JEPA.experiments.exp_011_ls20_icm.shared.config_base import Config
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel

ROOT = os.path.dirname(__file__)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
DEV = get_device()
print("device:", DEV)

N_ENVS = 8
T = 200            # rollout horizon -> 1600 transitions per rollout
SEED = 12345

L1_CK = "JEPA/experiments/exp_011_ls20_icm/exp_011_0_icm_baseline/checkpoints"
L2_CK = "JEPA/experiments/exp_011_ls20_icm/exp_011_2_icm_ls20_l2/checkpoints"


def mask_ui(frames_uint8: np.ndarray) -> np.ndarray:
    """Zero UI rows 61 and 62 (the built-in per-step distractor) so frame hashes
    reflect game state, not the changing UI counter. frames: (..., 64, 64)."""
    f = frames_uint8.copy()
    f[..., 61:63, :] = 0
    return f


def hash_frames(frames_uint8: np.ndarray):
    """(M,64,64) uint8 -> list of M hashes (UI-masked)."""
    m = mask_ui(frames_uint8)
    return [hash(x.tobytes()) for x in m]


def build_from_ck(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    model = ActorCritic().to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    icm = ICMModule().to(DEV)
    icm.load_state_dict(ck["icm"])
    icm.eval()
    eta = float(ck["eta"])
    return model, icm, eta, cfg, ck["step"]


def analyze_ck(ck_path, level_index):
    model, icm, eta, cfg, step = build_from_ck(ck_path)
    envs = VecLS20EnvLevel(env_name="ls20", n_envs=N_ENVS,
                           max_episode_steps=200, seed=SEED, level_index=level_index)
    roll = collect_rollout(envs, model, DEV, T)
    # per-transition raw forward error (T,N), dones zeroed
    raw, mean_raw = intrinsic_raw_error(icm, roll, DEV)  # raw is (T,N) cpu
    ri = 0.5 * eta * raw                                  # intrinsic reward r^i

    valid = (~roll.dones).numpy()                         # (T,N)
    nxt = roll.next_obs.numpy()                           # (T,N,64,64)

    # first-seen vs revisited: walk in time order PER ENV (env streams are
    # contiguous trajectories); a next-state hash not seen before in that env's
    # stream (and globally across envs) is "first-seen".
    raw_np = raw.numpy()
    ri_np = ri.numpy()
    seen = set()
    fs_err, rv_err = [], []         # forward error
    fs_ri, rv_ri = [], []           # intrinsic reward
    # iterate time-major then env so we approximate the global visitation order
    Tn, Nn = raw_np.shape
    # build per (t,n) hash of next_obs
    flat_next = nxt.reshape(Tn * Nn, 64, 64)
    hashes = hash_frames(flat_next)
    hashes = np.array(hashes, dtype=object).reshape(Tn, Nn)
    coverage_curve = []             # cumulative unique masked states over transitions
    n_trans = 0
    for t in range(Tn):
        for n in range(Nn):
            if not valid[t, n]:
                continue
            h = hashes[t, n]
            novel = h not in seen
            if novel:
                seen.add(h)
                fs_err.append(raw_np[t, n]); fs_ri.append(ri_np[t, n])
            else:
                rv_err.append(raw_np[t, n]); rv_ri.append(ri_np[t, n])
            n_trans += 1
            coverage_curve.append(len(seen))

    fs_err = np.array(fs_err); rv_err = np.array(rv_err)
    fs_ri = np.array(fs_ri); rv_ri = np.array(rv_ri)
    all_err = raw_np[valid]
    all_ri = ri_np[valid]

    res = dict(
        step=int(step), eta=eta,
        n_valid=int(valid.sum()),
        unique_states=len(seen),
        frac_first_seen=len(fs_err) / max(1, n_trans),
        err_mean=float(all_err.mean()), err_p50=float(np.median(all_err)),
        err_p90=float(np.percentile(all_err, 90)), err_max=float(all_err.max()),
        err_cov=float(all_err.std() / max(1e-12, all_err.mean())),
        fs_err_mean=float(fs_err.mean()) if len(fs_err) else float("nan"),
        rv_err_mean=float(rv_err.mean()) if len(rv_err) else float("nan"),
        ri_mean=float(all_ri.mean()),
        fs_ri_mean=float(fs_ri.mean()) if len(fs_ri) else float("nan"),
        rv_ri_mean=float(rv_ri.mean()) if len(rv_ri) else float("nan"),
        ri_p90=float(np.percentile(all_ri, 90)), ri_max=float(all_ri.max()),
        coverage_curve=coverage_curve,
        contrast_ratio=(float(fs_err.mean()) / float(rv_err.mean())) if (len(fs_err) and len(rv_err) and rv_err.mean() > 0) else float("nan"),
    )
    del envs
    return res


def random_coverage(level_index, budget_transitions):
    """Uniform-random policy coverage under the same env budget (UI-masked unique states)."""
    envs = VecLS20EnvLevel(env_name="ls20", n_envs=N_ENVS,
                           max_episode_steps=200, seed=SEED + 1, level_index=level_index)
    seen = set()
    curve = []
    obs = envs.current_obs()
    steps = 0
    rng = np.random.default_rng(SEED + 2)
    while steps < budget_transitions:
        a = rng.integers(0, 4, size=N_ENVS)
        nobs, r, dones, infos = envs.step(a)
        hs = hash_frames(nobs)
        for i in range(N_ENVS):
            if not dones[i]:
                seen.add(hs[i])
            steps += 1
            curve.append(len(seen))
        obs = nobs
    del envs
    return curve[:budget_transitions], len(seen)


def pick_cks(ck_root, seed_glob):
    d = sorted(glob.glob(os.path.join(ck_root, seed_glob)))[0]
    return sorted(glob.glob(os.path.join(d, "step_*.pt")))


def main():
    torch.manual_seed(SEED)
    # representative seed for each level (use seed0)
    l2_cks = pick_cks(L2_CK, "*seed0*")
    l1_cks = pick_cks(L1_CK, "*seed0*")
    print("L2 cks:", [os.path.basename(c) for c in l2_cks])
    print("L1 cks:", [os.path.basename(c) for c in l1_cks])

    l2_res = [analyze_ck(c, level_index=1) for c in l2_cks]
    l1_res = [analyze_ck(c, level_index=0) for c in l1_cks]

    print("\n=== L2 (failed) per checkpoint ===")
    hdr = "step      eta       uniq  %firstseen  err_mean  err_p90   err_max  fs_err   rv_err  contrast  ri_mean   ri_max"
    print(hdr)
    for r in l2_res:
        print(f"{r['step']:>8} {r['eta']:.2e}  {r['unique_states']:>4}  {r['frac_first_seen']*100:8.1f}  "
              f"{r['err_mean']:8.3f}  {r['err_p90']:7.3f}  {r['err_max']:7.2f}  {r['fs_err_mean']:6.3f}  "
              f"{r['rv_err_mean']:6.3f}  {r['contrast_ratio']:7.2f}  {r['ri_mean']:.2e}  {r['ri_max']:.2e}")
    print("\n=== L1 (solved) per checkpoint ===")
    print(hdr)
    for r in l1_res:
        print(f"{r['step']:>8} {r['eta']:.2e}  {r['unique_states']:>4}  {r['frac_first_seen']*100:8.1f}  "
              f"{r['err_mean']:8.3f}  {r['err_p90']:7.3f}  {r['err_max']:7.2f}  {r['fs_err_mean']:6.3f}  "
              f"{r['rv_err_mean']:6.3f}  {r['contrast_ratio']:7.2f}  {r['ri_mean']:.2e}  {r['ri_max']:.2e}")

    # random coverage reference (use the largest budget seen)
    budget = max(len(l2_res[-1]["coverage_curve"]), len(l1_res[-1]["coverage_curve"]))
    rand_l2_curve, rand_l2_uniq = random_coverage(1, budget)
    rand_l1_curve, rand_l1_uniq = random_coverage(0, budget)
    print(f"\nRANDOM coverage in {budget} transitions:  L2={rand_l2_uniq}  L1={rand_l1_uniq}")

    # save numbers
    out = dict(l1=l1_res, l2=l2_res, rand_l2_uniq=rand_l2_uniq, rand_l1_uniq=rand_l1_uniq,
               budget=budget)
    # drop the big curves from json except final
    import copy
    js = copy.deepcopy(out)
    for r in js["l1"] + js["l2"]:
        r["coverage_curve_final"] = r["coverage_curve"][-1]
        del r["coverage_curve"]
    json.dump(js, open(os.path.join(ROOT, "results_02.json"), "w"), indent=2)

    # ---- FIG 2: contrast (first-seen vs revisited forward error) over training ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, res, lbl, color in [(axes[0], l2_res, "L2 (failed)", "tab:red"),
                                (axes[1], l1_res, "L1 (solved)", "tab:green")]:
        steps = [r["step"] for r in res]
        fs = [r["fs_err_mean"] for r in res]
        rv = [r["rv_err_mean"] for r in res]
        ax.plot(steps, fs, "o-", color=color, label="first-seen states")
        ax.plot(steps, rv, "s--", color=color, alpha=0.5, label="revisited states")
        ax.set_title(f"{lbl}: forward error, novel vs familiar")
        ax.set_xlabel("env step of checkpoint")
        ax.set_ylabel("||phi_hat - phi(s')||^2")
        ax.set_yscale("log")
        ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_contrast.png"), dpi=110, bbox_inches="tight")
    print("saved fig2_contrast.png")

    # ---- FIG 3: coverage. unique states reached per fixed budget, last ck vs random ----
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cl2 = l2_res[-1]["coverage_curve"]
    cl1 = l1_res[-1]["coverage_curve"]
    x = np.arange(1, budget + 1)
    ax.plot(np.arange(1, len(cl2)+1), cl2, color="tab:red", lw=2, label=f"L2 ICM policy (final ck): {cl2[-1]} states")
    ax.plot(np.arange(1, len(rand_l2_curve)+1), rand_l2_curve, color="tab:red", ls=":", lw=2, label=f"L2 random: {rand_l2_uniq} states")
    ax.plot(np.arange(1, len(cl1)+1), cl1, color="tab:green", lw=2, label=f"L1 ICM policy (final ck): {cl1[-1]} states")
    ax.plot(np.arange(1, len(rand_l1_curve)+1), rand_l1_curve, color="tab:green", ls=":", lw=2, label=f"L1 random: {rand_l1_uniq} states")
    ax.set_xlabel("valid transitions collected")
    ax.set_ylabel("cumulative unique UI-masked states")
    ax.set_title("State coverage: ICM policy vs uniform-random (same budget)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_coverage.png"), dpi=110, bbox_inches="tight")
    print("saved fig3_coverage.png")


if __name__ == "__main__":
    main()
