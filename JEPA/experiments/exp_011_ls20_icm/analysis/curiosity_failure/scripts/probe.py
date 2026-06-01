"""Curiosity-failure probes for exp_011 ICM (LS20 L1 solved vs L2 failed).

Runs three checkpoint-based probes on the SAVED checkpoints + logs:
  (A) log trajectories: forward error / r^i / inverse_acc  (cheap, from jsonl)
  (B) CONTRAST probe: roll out a saved policy, compute per-transition r^i,
      split by novel (first-seen masked frame) vs revisited; does r^i still
      discriminate?  Run at successive checkpoints for L1 and L2.
  (C) COVERAGE probe: roll out saved policy checkpoints, count unique
      UI-masked states reached per episode + cumulatively; did the goal frame
      ever get reached on L2?

Writes JSON results to analysis/curiosity_failure/probe_results.json.
"""
from __future__ import annotations
import sys, os, json, glob, hashlib
import numpy as np
import torch

ROOT = "/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo"
sys.path.insert(0, ROOT)

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout
from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule, intrinsic_raw_error
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel

DEV = get_device()
print("device", DEV)

EXP = {
    "L1": dict(root=os.path.join(ROOT, "JEPA/experiments/exp_011_ls20_icm/exp_011_0_icm_baseline"),
               level=0),
    "L2": dict(root=os.path.join(ROOT, "JEPA/experiments/exp_011_ls20_icm/exp_011_2_icm_ls20_l2"),
               level=1),
}

# UI rows masked before hashing state (deterministic env; exact-frame counting works)
MASKED_ROWS = list(range(61, 63))


def masked_hash(frame: np.ndarray) -> str:
    f = frame.copy()
    f[MASKED_ROWS, :] = 0
    return hashlib.md5(f.tobytes()).hexdigest()


def first_seed_ckpts(root):
    cdir = sorted(glob.glob(os.path.join(root, "checkpoints", "*seed0*")))[0]
    return sorted(glob.glob(os.path.join(cdir, "step_*.pt")))


def load_models(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = ActorCritic(n_actions=cfg["n_actions"], n_colors=cfg["n_colors"],
                        frame_size=cfg["frame_size"], trunk_dim=cfg["trunk_dim"])
    model.load_state_dict(ck["model"]); model.to(DEV).eval()
    icm = ICMModule(n_actions=cfg["n_actions"], n_colors=cfg["n_colors"],
                    frame_size=cfg["frame_size"], trunk_dim=cfg["trunk_dim"],
                    hidden=cfg.get("icm_hidden", 256))
    icm.load_state_dict(ck["icm"]); icm.to(DEV).eval()
    return model, icm, ck["eta"], ck["step"]


def contrast_probe(level, ckpt_path, T=128, n_envs=8, seed=123):
    """Roll out saved policy, compute per-transition r^i, split novel vs revisited."""
    model, icm, eta, step = load_models(ckpt_path)
    envs = VecLS20EnvLevel(env_name="ls20", n_envs=n_envs, max_episode_steps=200,
                           seed=seed, level_index=level)
    envs.reset_all()
    roll = collect_rollout(envs, model, DEV, T)
    raw, mean_raw = intrinsic_raw_error(icm, roll, DEV)  # (T,N) raw fwd error
    ri = (eta * 0.5 * raw).numpy()                       # intrinsic reward scale

    obs = roll.obs.numpy()       # (T,N,64,64)
    dones = roll.dones.numpy()   # (T,N)
    seen = set()
    novel_vals, fam_vals = [], []
    for t in range(T):
        for n in range(n_envs):
            if dones[t, n]:
                continue
            h = masked_hash(obs[t, n])
            v = float(ri[t, n])
            if h in seen:
                fam_vals.append(v)
            else:
                novel_vals.append(v)
                seen.add(h)
    nv = np.array(novel_vals); fv = np.array(fam_vals)
    out = dict(step=int(step), eta=float(eta), mean_ri=float(ri[~dones].mean()),
               n_novel=int(nv.size), n_fam=int(fv.size),
               novel_mean=float(nv.mean()) if nv.size else None,
               fam_mean=float(fv.mean()) if fv.size else None,
               novel_med=float(np.median(nv)) if nv.size else None,
               fam_med=float(np.median(fv)) if fv.size else None,
               ri_all=ri[~dones].tolist())
    if nv.size and fv.size:
        out["novel_over_fam"] = float(nv.mean() / max(1e-12, fv.mean()))
    return out


def coverage_probe(level, ckpt_path, n_envs=8, max_steps=200, n_resets=4, seed=777):
    """Roll out saved policy; count unique masked states; track goal reached."""
    model, icm, eta, step = load_models(ckpt_path)
    envs = VecLS20EnvLevel(env_name="ls20", n_envs=n_envs, max_episode_steps=max_steps,
                           seed=seed, level_index=level)
    envs.reset_all()
    cum_states = set()
    per_ep_unique = []  # unique states per env-episode chunk
    goal_hits = 0
    total_steps = 0
    ep_state_sets = [set() for _ in range(n_envs)]
    for rep in range(n_resets):
        obs = envs.current_obs()
        for _ in range(max_steps):
            ot = torch.from_numpy(obs).to(DEV)
            with torch.no_grad():
                a, _, _, _ = model.act(ot)
            a = a.cpu().numpy().astype(np.int64)
            obs, rew, dones, infos = envs.step(a)
            total_steps += n_envs
            for n in range(n_envs):
                h = masked_hash(obs[n])
                cum_states.add(h)
                ep_state_sets[n].add(h)
                if rew[n] > 0:
                    goal_hits += 1
                if dones[n]:
                    per_ep_unique.append(len(ep_state_sets[n]))
                    ep_state_sets[n] = set()
    return dict(step=int(step), cum_unique=len(cum_states),
                mean_ep_unique=float(np.mean(per_ep_unique)) if per_ep_unique else None,
                n_episodes=len(per_ep_unique), goal_hits=int(goal_hits),
                total_steps=int(total_steps))


def random_coverage(level, n_envs=8, max_steps=200, n_resets=8, seed=999):
    """Random-policy coverage baseline (behavioral floor)."""
    envs = VecLS20EnvLevel(env_name="ls20", n_envs=n_envs, max_episode_steps=max_steps,
                           seed=seed, level_index=level)
    envs.reset_all()
    cum = set(); goal = 0; total = 0; per_ep = []
    ep_sets = [set() for _ in range(n_envs)]
    rng = np.random.default_rng(seed)
    for rep in range(n_resets):
        obs = envs.current_obs()
        for _ in range(max_steps):
            a = rng.integers(0, envs.n_actions, size=n_envs).astype(np.int64)
            obs, rew, dones, infos = envs.step(a)
            total += n_envs
            for n in range(n_envs):
                h = masked_hash(obs[n]); cum.add(h); ep_sets[n].add(h)
                if rew[n] > 0: goal += 1
                if dones[n]:
                    per_ep.append(len(ep_sets[n])); ep_sets[n] = set()
    return dict(cum_unique=len(cum), goal_hits=int(goal), total_steps=int(total),
                mean_ep_unique=float(np.mean(per_ep)) if per_ep else None)


def main():
    results = {"contrast": {}, "coverage": {}, "random": {}}
    for lab, spec in EXP.items():
        cks = first_seed_ckpts(spec["root"])
        print(f"\n=== {lab}: {len(cks)} checkpoints ===")
        results["contrast"][lab] = []
        results["coverage"][lab] = []
        for ck in cks:
            c = contrast_probe(spec["level"], ck)
            c.pop("ri_all", None)  # keep json small; distribution saved separately for last ck
            results["contrast"][lab].append(c)
            print("  contrast", lab, c["step"], "novel/fam=",
                  c.get("novel_over_fam"), "n_novel", c["n_novel"], "n_fam", c["n_fam"])
            cv = coverage_probe(spec["level"], ck)
            results["coverage"][lab].append(cv)
            print("  coverage", lab, cv["step"], "cum_unique", cv["cum_unique"],
                  "mean_ep_unique", cv["mean_ep_unique"], "goal_hits", cv["goal_hits"])
        # full ri distribution at last checkpoint (for histogram)
        full = contrast_probe(spec["level"], cks[-1])
        results.setdefault("ri_dist", {})[lab] = full["ri_all"]
        results["random"][lab] = random_coverage(spec["level"])
        print("  RANDOM", lab, results["random"][lab])

    out = os.path.join(ROOT, "JEPA/experiments/exp_011_ls20_icm/analysis/curiosity_failure/probe_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
