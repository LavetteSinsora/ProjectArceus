"""Probe: WHY does the pure-exploration actor's entropy collapse to ~0?

This adapts the exp_013_1 training loop (collect_rollout -> novelty -> normalize ->
_gae_nonepisodic -> ppo_update -> icm_update -> rnd_update) and swaps ONLY the reward
channel, to separate the novelty signal from a generic PPO-with-weak-entropy pathology.

reward_mode:
  * "novelty"  — the real exp_013_1 intrinsic reward (RND novelty in φ-space). BASELINE.
  * "zero"     — reward identically 0 every step (H1a). If entropy still collapses, the
                 collapse is NOT caused by the novelty signal.
  * "noise"    — reward = i.i.d. N(0, sigma) matched to the normalized-novelty scale
                 (H1b). Tests whether *any* nonzero reward drives commitment.

Per update it logs, for H3 (value-lag) and H4 (novelty-informativeness):
  * policy_entropy, approx_kl
  * V_int  = rollout.values.mean()         (critic's predicted intrinsic value)
  * Ret_int= rollout.returns.mean()        (empirical GAE target)
  * adv_mode = advantage of the MOST-PROBABLE action, averaged over states
               (persistent positive bias = value-undershoot feeds commitment)
  * nov_raw  = raw novelty mean (for H4 / H2 correlation)

The ICM/RND machinery still RUNS in every mode (φ trains+freezes, predictor distills)
so the only thing that differs across modes is what is fed to GAE. Method/harness code
is untouched; everything here is read-only re-use of trainer internals.

Run e.g.:
  uv run python -m JEPA.experiments.exp_013_headline_experiment.probes.reward_mode_control \
       --mode zero --c_entropy 0.01 --updates 60 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.device import get_device
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule, icm_update_from_rollout
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter

from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.config import Config
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.trainer import (
    _EMAStd, _phi_and_novelty, _rnd_update, _gae_nonepisodic,
)


@torch.no_grad()
def most_probable_action_advantage(model, rollout, device, chunk=2048):
    """Mean advantage of the argmax-prob action at each visited state.

    H3 signal: if the critic systematically undershoots the intrinsic return, the
    greedy action carries a persistent POSITIVE advantage -> PPO keeps pushing the
    policy toward it -> entropy collapses regardless of where the reward comes from."""
    T, N = rollout.actions.shape
    Fz = rollout.frame
    obs = rollout.obs.reshape(-1, Fz, Fz)
    adv = rollout.advantages.reshape(-1)
    M = obs.shape[0]
    vals = []
    for s in range(0, M, chunk):
        o = obs[s:s + chunk].to(device)
        logits, _v, _f = model.forward(o)
        greedy = logits.argmax(-1).cpu()
        acts = rollout.actions.reshape(-1)[s:s + chunk]
        is_greedy = (greedy == acts)
        # advantage of the taken action WHEN it was the greedy one
        a_slice = adv[s:s + chunk]
        if is_greedy.any():
            vals.append(a_slice[is_greedy])
    if vals:
        return float(torch.cat(vals).mean())
    return float("nan")


def run(mode: str, c_entropy: float, updates: int, seed: int, leak: float,
        intrinsic_episodic: bool, noise_sigma: float, out_path: Path | None):
    cfg = Config(game="ls20", level_index=0, seed=seed)
    cfg.c_entropy = c_entropy
    cfg.leak = leak
    cfg.intrinsic_episodic = intrinsic_episodic
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
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps,
                        c_value=cfg.c_value, c_entropy=cfg.c_entropy,
                        grad_clip=cfg.grad_clip, epochs=cfg.epochs,
                        minibatches=cfg.minibatches)
    rff = RewardForwardFilter(cfg.gamma)
    int_ret_std = _EMAStd(cfg.int_norm_decay)

    phi_frozen = False
    inv_streak = 0
    freeze_upd = None
    rng = np.random.RandomState(seed + 12345)

    print(f"\n=== mode={mode} c_entropy={c_entropy} seed={seed} leak={leak} "
          f"episodic={intrinsic_episodic} sigma={noise_sigma} ===")
    print(f"{'upd':>4}{'entropy':>9}{'kl':>8}{'V_int':>9}{'Ret_int':>9}"
          f"{'adv_grdy':>9}{'nov_raw':>9}{'rew_mean':>9}{'frozen':>7}")

    rows = []
    for update in range(1, updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)
        phi_cached, nov = _phi_and_novelty(icm, rndphi, rollout, device)
        raw_i = nov.numpy()
        T, N = raw_i.shape

        # ---- reward channel swap (the ONLY thing that differs across modes) ----
        if mode == "novelty":
            if update <= cfg.norm_warmup_updates:
                norm_i = np.zeros_like(raw_i)
            else:
                rems = np.stack([rff.update(raw_i[t]) for t in range(T)])
                int_ret_std.update(rems)
                norm_i = raw_i / (int_ret_std.std + cfg.int_norm_eps)
        elif mode == "zero":
            norm_i = np.zeros_like(raw_i)
        elif mode == "noise":
            norm_i = rng.randn(T, N).astype(np.float32) * noise_sigma
        else:
            raise ValueError(mode)

        rollout.rewards = torch.from_numpy(norm_i.astype(np.float32))
        if cfg.intrinsic_episodic:
            from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import compute_gae
            compute_gae(rollout, cfg.gamma, cfg.gae_lambda)
        else:
            _gae_nonepisodic(rollout, cfg.gamma, cfg.gae_lambda)

        # H3 quantities computed on the rollout BEFORE the ppo update consumes it
        V_int = float(rollout.values.mean())
        Ret_int = float(rollout.returns.mean())
        adv_greedy = most_probable_action_advantage(model, rollout, device)

        ustats = ppo_update(model, ppo_opt, rollout, ppo_cfg, device)

        # keep φ/RND machinery identical across all modes
        if not phi_frozen:
            icm_stats = icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)
            ia = icm_stats["inverse_acc"]
            inv_streak = inv_streak + 1 if ia >= cfg.phi_freeze_inverse_acc else 0
            if inv_streak >= cfg.phi_freeze_patience or update >= cfg.phi_freeze_max_updates:
                for p in icm.phi.parameters():
                    p.requires_grad_(False)
                icm.phi.eval()
                phi_frozen = True
                freeze_upd = update
        _rnd_update(rndphi, rnd_opt, phi_cached, rollout.dones, cfg, device)

        row = dict(update=update, entropy=ustats.entropy, approx_kl=ustats.approx_kl,
                   V_int=V_int, Ret_int=Ret_int, adv_greedy=adv_greedy,
                   nov_raw=float(raw_i.mean()), rew_mean=float(norm_i.mean()),
                   phi_frozen=phi_frozen)
        rows.append(row)
        if update % 2 == 0 or update <= 4 or update == updates:
            print(f"{update:>4}{ustats.entropy:>9.3f}{ustats.approx_kl:>8.4f}"
                  f"{V_int:>9.3f}{Ret_int:>9.3f}{adv_greedy:>9.4f}"
                  f"{float(raw_i.mean()):>9.3g}{float(norm_i.mean()):>9.3g}"
                  f"{str(phi_frozen):>7}"
                  + ("  <-FREEZE" if freeze_upd == update else ""))

    ents = [r["entropy"] for r in rows]
    summary = dict(mode=mode, c_entropy=c_entropy, seed=seed, leak=leak,
                   intrinsic_episodic=intrinsic_episodic, noise_sigma=noise_sigma,
                   updates=updates, ent_first=ents[0], ent_min=min(ents),
                   ent_last=ents[-1], ent_lastq_mean=float(np.mean(ents[-max(1, updates // 4):])),
                   freeze_upd=freeze_upd, rows=rows)
    print(f"  --> ent first={ents[0]:.3f} min={min(ents):.3f} last={ents[-1]:.3f} "
          f"lastQ_mean={summary['ent_lastq_mean']:.3f}")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"  wrote {out_path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="novelty", choices=["novelty", "zero", "noise"])
    ap.add_argument("--c_entropy", type=float, default=0.01)
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leak", type=float, default=0.01)
    ap.add_argument("--episodic", action="store_true")
    ap.add_argument("--noise_sigma", type=float, default=0.1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    run(args.mode, args.c_entropy, args.updates, args.seed, args.leak,
        args.episodic, args.noise_sigma, out)
