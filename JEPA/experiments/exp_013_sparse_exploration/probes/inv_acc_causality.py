"""Probe: is the inverse_acc rise the CAUSE of the entropy collapse, or its PRODUCT?

The freeze trigger uses inverse_acc measured on the ON-POLICY rollout. If the policy
narrows, the rollout's transitions become a small repeated set → predicting the action
becomes trivial → inverse_acc can rise BECAUSE the policy collapsed (product), not
because φ learned general controllable features (cause).

This probe replicates the exp_013_1 (c_entropy=0.01, original collapsing config) loop and,
every update, logs BOTH:
  * on_policy_inv_acc  — inverse_acc on the current rollout (what the trainer/freeze uses)
  * fixed_inv_acc      — inverse_acc on a FIXED held-out set of transitions collected once
                         from a UNIFORM-RANDOM policy (policy-independent → φ's TRUE quality)
alongside policy entropy and novelty. Divergence (on_policy >> fixed during the collapse)
⇒ the on-policy inv_acc is a narrow-data artifact and the freeze trigger is being fooled.

Run: uv run python -m JEPA.experiments.exp_013_sparse_exploration.probes.inv_acc_causality
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule, icm_update_from_rollout
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter

from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.config import Config
from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.rnd_phi import RNDPhi
from JEPA.experiments.exp_013_sparse_exploration.exp_013_1_rnd_icm.trainer import (
    _EMAStd, _phi_and_novelty, _rnd_update, _gae_nonepisodic,
)


@torch.no_grad()
def collect_fixed_transitions(game, level, seed, n_target=3000, device=None):
    """Uniform-random rollout → a fixed, policy-INDEPENDENT set of (obs,a,next_obs)
    non-reset transitions. Diverse coverage = a fair test set for φ's controllability."""
    env = VecLS20EnvLevel(env_name=game, n_envs=16, max_episode_steps=200,
                          seed=seed + 999, level_index=level)
    obs = env.current_obs()
    O, A, NO = [], [], []
    while len(A) < n_target:
        a = np.random.randint(0, env.n_actions, size=env.n_envs).astype(np.int64)
        nobs, _r, dones, _i = env.step(a)
        for i in range(env.n_envs):
            if not dones[i]:                      # exclude reset frames
                O.append(obs[i].copy()); A.append(int(a[i])); NO.append(nobs[i].copy())
        obs = nobs
    return (torch.from_numpy(np.stack(O[:n_target])),
            torch.from_numpy(np.array(A[:n_target], dtype=np.int64)),
            torch.from_numpy(np.stack(NO[:n_target])))


@torch.no_grad()
def encode_fixed(icm, obs, device, chunk=512):
    """Encode a fixed set of states with the CURRENT φ → (n, D) on CPU."""
    out = []
    for s in range(0, obs.shape[0], chunk):
        out.append(icm.encode(obs[s:s + chunk].to(device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def eval_fixed_inv_acc(icm, fixed, device, chunk=512):
    obs, acts, nobs = fixed
    correct = 0
    n = obs.shape[0]
    for s in range(0, n, chunk):
        phi_t = icm.encode(obs[s:s + chunk].to(device))
        phi_n = icm.encode(nobs[s:s + chunk].to(device))
        logits = icm.inverse_logits(phi_t, phi_n)
        correct += (logits.argmax(-1).cpu() == acts[s:s + chunk]).sum().item()
    return correct / max(1, n)


def main(updates=80):
    cfg = Config(game="ls20", level_index=0, seed=0)
    cfg.c_entropy = 0.01                          # ORIGINAL config that collapsed
    cfg.leak = 0.01
    device = get_device()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    print("collecting fixed uniform-random transition set ...")
    fixed = collect_fixed_transitions(cfg.game, cfg.level_index, cfg.seed, 3000, device)
    print(f"  {fixed[0].shape[0]} fixed transitions; action hist={np.bincount(fixed[1].numpy())}")

    envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=cfg.seed,
                           level_index=cfg.level_index)
    cfg.n_actions = envs.n_actions
    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)
    icm = ICMModule(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                    frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim, hidden=cfg.icm_hidden).to(device)
    rndphi = RNDPhi(dim=cfg.trunk_dim, hidden=cfg.rnd_hidden, out=cfg.rnd_feature_dim, leak=cfg.leak).to(device)
    ppo_opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    icm_opt = torch.optim.Adam(icm.parameters(), lr=cfg.icm_lr)
    rnd_opt = torch.optim.Adam(rndphi.predictor.parameters(), lr=cfg.rnd_lr)
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps, c_value=cfg.c_value,
                        c_entropy=cfg.c_entropy, grad_clip=cfg.grad_clip, epochs=cfg.epochs,
                        minibatches=cfg.minibatches)
    rff = RewardForwardFilter(cfg.gamma); int_ret_std = _EMAStd(cfg.int_norm_decay)

    phi_frozen = False; inv_streak = 0; freeze_upd = None
    prev_phi = None                                # φ(fixed states) at the previous update
    print(f"\n{'upd':>4}{'entropy':>9}{'onpol_inv':>10}{'FIXED_inv':>10}{'phi_cos':>9}{'nov_raw':>9}{'frozen':>7}")
    for update in range(1, updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)
        phi_cached, nov = _phi_and_novelty(icm, rndphi, rollout, device)
        raw_i = nov.numpy(); T, N = raw_i.shape
        if update <= cfg.norm_warmup_updates:
            norm_i = np.zeros_like(raw_i)
        else:
            rems = np.stack([rff.update(raw_i[t]) for t in range(T)])
            int_ret_std.update(rems)
            norm_i = raw_i / (int_ret_std.std + cfg.int_norm_eps)
        rollout.rewards = torch.from_numpy(norm_i.astype(np.float32))
        _gae_nonepisodic(rollout, cfg.gamma, cfg.gae_lambda)
        ustats = ppo_update(model, ppo_opt, rollout, ppo_cfg, device)

        onpol_inv = float("nan")
        if not phi_frozen:
            icm_stats = icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)
            onpol_inv = icm_stats["inverse_acc"]
            inv_streak = inv_streak + 1 if onpol_inv >= cfg.phi_freeze_inverse_acc else 0
            if inv_streak >= cfg.phi_freeze_patience or update >= cfg.phi_freeze_max_updates:
                for p in icm.phi.parameters(): p.requires_grad_(False)
                icm.phi.eval(); phi_frozen = True; freeze_upd = update
        _rnd_update(rndphi, rnd_opt, phi_cached, rollout.dones, cfg, device)

        fixed_inv = eval_fixed_inv_acc(icm, fixed, device)   # φ's TRUE quality, policy-independent

        # per-update φ-representation drift on the fixed state set: cosine(φ_t, φ_{t-1}).
        # Low cosine = big representation change → expected to coincide with the RND
        # novelty spike (a moving ruler is what makes the predictor error jump).
        cur_phi = encode_fixed(icm, fixed[0], device)
        phi_cos = float(F.cosine_similarity(cur_phi, prev_phi, dim=-1).mean()) if prev_phi is not None else float("nan")
        prev_phi = cur_phi

        print(f"{update:>4}{ustats.entropy:>9.3f}{onpol_inv:>10.3f}{fixed_inv:>10.3f}"
              f"{phi_cos:>9.4f}{float(raw_i.mean()):>9.4g}{str(phi_frozen):>7}"
              + ("   <-- FREEZE" if freeze_upd == update else ""))
    print("\nINTERPRETATION:")
    print(" * FIXED_inv tracks onpol_inv up to ~0.9  → φ genuinely learned (inv_acc rise = CAUSE).")
    print(" * onpol_inv >> FIXED_inv during collapse → narrow-data artifact (PRODUCT); freeze fooled.")
    print(" * phi_cos DIPS (big φ change) exactly when nov_raw spikes → confirms moving-ruler → RND-spike.")


if __name__ == "__main__":
    main()
