"""exp_013_1 — RND+ICM ("OCC") training loop. See SYSTEM_CARD.md.

Per update:
  collect rollout → φ(next_obs) via the ICM encoder → RND novelty in φ-space
  → normalise (warm-up + EMA-std of returns, NO centring) → intrinsic reward
  → episodic GAE (single head) → PPO update (policy + value)
  → ICM update (train φ) UNTIL φ is FROZEN (inverse_acc saturates)
  → RND predictor update (+ LEAK shrink-to-init)
  → detect the first extrinsic reward → STOP (the +1 is the stop signal, NOT a reward).
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
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae as _gae_episodic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.metrics import MetricsWriter, mean_feature_cosine
from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule, icm_update_from_rollout
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_012_ls20_rnd.shared.rnd import RewardForwardFilter

from .rnd_phi import RNDPhi
from JEPA.experiments.exp_013_sparse_exploration.exp_013_4_disagreement.disagreement import FrozenPhi


class _EMAStd:
    """EMA of the variance of a scalar stream → a std tracking the CURRENT scale.
    Suited to the LEAKY (non-stationary) novelty, which revives as the predictor
    forgets — a cumulative RMS would lag it."""

    def __init__(self, decay: float):
        self.decay = decay
        self.var: float | None = None

    def update(self, x) -> None:
        v = float(np.asarray(x, dtype=np.float64).var())
        self.var = v if self.var is None else self.decay * self.var + (1.0 - self.decay) * v

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var)) if self.var is not None else 1.0


def _repo_root() -> Path:
    # exp_013_1_rnd_icm -> exp_013_sparse_exploration -> experiments -> JEPA -> Code Repo
    return Path(__file__).resolve().parents[4]


def _gae_nonepisodic(rollout, gamma: float, lam: float):
    """Canonical-RND NON-episodic GAE for the single intrinsic head: identical to
    exp_010's `compute_gae` EXCEPT the bootstrap value and the GAE accumulator are
    NOT masked by `dones`, so the intrinsic return flows ACROSS death/reset (the
    agent isn't deterred from deep exploration that costs a life). Mutates the
    rollout in place (advantages, returns)."""
    T, N = rollout.rewards.shape
    advantages = torch.zeros(T, N, dtype=torch.float32)
    last_gae = torch.zeros(N, dtype=torch.float32)
    next_value = rollout.bootstrap_value.float()
    for t in reversed(range(T)):
        delta = rollout.rewards[t] + gamma * next_value - rollout.values[t]   # no (1-done) mask
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        next_value = rollout.values[t]
    rollout.advantages = advantages
    rollout.returns = advantages + rollout.values
    return rollout


@torch.no_grad()
def _phi_and_novelty(icm: ICMModule, rndphi: RNDPhi, rollout, device, chunk: int = 512):
    """Embed next_obs through the ICM encoder φ and score RND novelty, chunked.
    Returns (phi_cached (T,N,D) cpu, novelty (T,N) cpu). Done-step transitions are
    zeroed (their s' is a reset frame)."""
    T, N = rollout.actions.shape
    Fz = rollout.frame
    D = icm.trunk_dim
    next_obs = rollout.next_obs.reshape(-1, Fz, Fz)
    M = next_obs.shape[0]
    phi = torch.empty(M, D, dtype=torch.float32)
    nov = torch.empty(M, dtype=torch.float32)
    for s in range(0, M, chunk):
        no = next_obs[s:s + chunk].to(device)
        ph = icm.encode(no)                       # φ(s'); no grad (decorator)
        phi[s:s + chunk] = ph.to("cpu")
        nov[s:s + chunk] = rndphi.novelty(ph).to("cpu")
    nov = nov.reshape(T, N) * (~rollout.dones).float()
    return phi.reshape(T, N, D), nov


def _rnd_update(rndphi: RNDPhi, opt, phi_cached: torch.Tensor, dones: torch.Tensor,
                cfg, device) -> float:
    """Distil the predictor toward the target on the rollout's φ(next_obs), then
    apply the LEAK. Excludes done-step transitions."""
    T, N, D = phi_cached.shape
    valid = (~dones).reshape(-1)
    phi = phi_cached.reshape(-1, D)[valid]
    n = phi.shape[0]
    if n == 0:
        return float("nan")
    mb = max(1, n // cfg.minibatches)
    idx = np.arange(n)
    tot = 0.0
    steps = 0
    for _ in range(cfg.rnd_epochs):
        np.random.shuffle(idx)
        for s in range(0, n, mb):
            ph = phi[idx[s:s + mb]].to(device)
            loss = rndphi.distill_loss(ph)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rndphi.predictor.parameters(), cfg.grad_clip)
            opt.step()
            rndphi.apply_leak()                   # forget AFTER the step
            tot += loss.item()
            steps += 1
    return tot / max(1, steps)


@torch.no_grad()
def _collect_holdout(game, level, seed, n_target, device):
    """A FIXED, policy-INDEPENDENT set of (obs,a,next_obs) non-reset transitions from
    a uniform-random policy — the unbiased test set for φ's controllability (so the
    freeze trigger can't be fooled by a narrowing policy)."""
    env = VecLS20EnvLevel(env_name=game, n_envs=16, max_episode_steps=200,
                          seed=seed + 4242, level_index=level)
    obs = env.current_obs()
    O, A, NO = [], [], []
    while len(A) < n_target:
        a = np.random.randint(0, env.n_actions, size=env.n_envs).astype(np.int64)
        nobs, _r, dones, _i = env.step(a)
        for i in range(env.n_envs):
            if not dones[i]:
                O.append(obs[i].copy()); A.append(int(a[i])); NO.append(nobs[i].copy())
        obs = nobs
    return (torch.from_numpy(np.stack(O[:n_target])),
            torch.from_numpy(np.array(A[:n_target], dtype=np.int64)),
            torch.from_numpy(np.stack(NO[:n_target])))


@torch.no_grad()
def _eval_holdout_inv_acc(icm, holdout, device, chunk=512):
    obs, acts, nobs = holdout
    correct = 0
    for s in range(0, obs.shape[0], chunk):
        phi_t = icm.encode(obs[s:s + chunk].to(device))
        phi_n = icm.encode(nobs[s:s + chunk].to(device))
        logits = icm.inverse_logits(phi_t, phi_n)
        correct += (logits.argmax(-1).cpu() == acts[s:s + chunk]).sum().item()
    return correct / max(1, obs.shape[0])


def _save_ckpt(run_dir: Path, model, icm, rndphi, cfg, global_step, frozen):
    ck = run_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    path = ck / f"step_{global_step:08d}.pt"
    torch.save({"step": int(global_step), "config": dataclasses.asdict(cfg),
                "model": model.state_dict(), "icm": icm.state_dict(),
                "rnd_target": rndphi.target.state_dict(),
                "rnd_predictor": rndphi.predictor.state_dict(),
                "phi_frozen": bool(frozen)}, path)
    return path


def _apply_timer_mask(phi_enc, rows):
    """Option-A timer-confound fix: mask the marching step-timer rows in the φ-ENCODER's
    input only. Every novelty/ICM/holdout path routes through `.encode`, so patching it
    makes φ see the TRUE board (43 states, not 1073 timer-stamped frames); the policy's
    SEPARATE encoder is untouched. See probes/frontier_analysis.md."""
    r0, r1 = int(rows[0]), int(rows[1])
    orig = phi_enc.encode

    def masked_encode(obs):
        obs = obs.clone()
        obs[..., r0:r1 + 1, :] = 0          # zero the timer rows → constant → no fake novelty
        return orig(obs)

    phi_enc.encode = masked_encode
    return phi_enc


def train(cfg, smoke: bool = False) -> dict:
    if smoke:
        cfg = cfg.smoke()
    device = get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    exp_dir = _repo_root() / cfg.exp_dir
    run_name = f"{cfg.exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = exp_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = MetricsWriter(run_dir)

    envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps, seed=cfg.seed,
                           level_index=cfg.level_index)
    if cfg.n_actions is None:
        cfg.n_actions = envs.n_actions
    assert cfg.n_actions == envs.n_actions, (
        f"cfg.n_actions={cfg.n_actions} but env reports {envs.n_actions} for {cfg.game}")
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    print(f"[exp013_1] {cfg.exp_name}  device={device}  n_actions={cfg.n_actions}  "
          f"cap={cfg.max_env_steps}  leak={cfg.leak}")

    model = ActorCritic(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim).to(device)

    # φ encoder for the RND ruler: learned ICM features ("icm") or a fixed random
    # encoder ("frozen" = plain RND+leak, no inverse-dynamics).
    if cfg.phi_mode == "frozen":
        icm = None
        phi_enc = FrozenPhi(n_colors=cfg.n_colors, frame_size=cfg.frame_size,
                            trunk_dim=cfg.trunk_dim).to(device)
    else:
        icm = ICMModule(n_actions=cfg.n_actions, n_colors=cfg.n_colors,
                        frame_size=cfg.frame_size, trunk_dim=cfg.trunk_dim,
                        hidden=cfg.icm_hidden).to(device)
        if cfg.init_phi_ckpt:                       # cross-level TRANSFER: init φ from a saved (e.g. L1) run
            ck = torch.load(cfg.init_phi_ckpt, map_location=device, weights_only=False)
            phi_sd = {k[len("phi."):]: v for k, v in ck["icm"].items() if k.startswith("phi.")}
            icm.phi.load_state_dict(phi_sd)
            print(f"[exp013_1]   φ INIT-FROM-CKPT {cfg.init_phi_ckpt} (transfer)")
        phi_enc = icm
    rndphi = RNDPhi(dim=cfg.trunk_dim, hidden=cfg.rnd_hidden, out=cfg.rnd_feature_dim,
                    leak=cfg.leak).to(device)

    if getattr(cfg, "mask_timer", False):
        _apply_timer_mask(phi_enc, cfg.timer_mask_rows)
        print(f"[exp013_1]   TIMER-MASK rows {tuple(cfg.timer_mask_rows)} on φ/novelty path "
              f"(policy input untouched)")

    ppo_opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    icm_opt = torch.optim.Adam(icm.parameters(), lr=cfg.icm_lr) if icm is not None else None
    rnd_opt = torch.optim.Adam(rndphi.predictor.parameters(), lr=cfg.rnd_lr)
    ppo_cfg = PPOConfig(clip_eps=cfg.clip_eps, vf_clip_eps=cfg.vf_clip_eps,
                        c_value=cfg.c_value, c_entropy=cfg.c_entropy,
                        grad_clip=cfg.grad_clip, epochs=cfg.epochs,
                        minibatches=cfg.minibatches)

    rff = RewardForwardFilter(cfg.gamma)          # discounts intrinsic returns at γ
    int_ret_std = _EMAStd(cfg.int_norm_decay)

    # Held-out transition set for an UN-fooled freeze trigger / φ-controllability log.
    # (Only meaningful in icm mode — a frozen random φ has no inverse model.)
    holdout = (_collect_holdout(cfg.game, cfg.level_index, cfg.seed, cfg.holdout_size, device)
               if icm is not None else None)
    print(f"[exp013_1]   phi_mode={cfg.phi_mode}  freeze_metric={cfg.freeze_metric}  "
          f"holdout={holdout[0].shape[0] if holdout else 0}  reward_clip_k={cfg.reward_clip_k}")

    phi_frozen = False
    inv_streak = 0
    freeze_step: int | None = None
    last_inv_acc = float("nan")            # on-policy inverse_acc (logged)
    last_holdout_inv = float("nan")        # held-out inverse_acc (φ's TRUE quality)
    raw_mean_ema: float | None = None      # for the raw-novelty clip
    global_step = 0
    first_reward_step: int | None = None
    t_start = time.time()
    stop_now = False

    for update in range(1, cfg.total_updates + 1):
        rollout = collect_rollout(envs, model, device, cfg.rollout_steps)

        # φ(next_obs) + RND novelty (uses the φ at the START of this update); cache φ.
        # phi_enc is the ICM (.encode, learned) or the FrozenPhi (.encode, fixed random).
        phi_cached, nov = _phi_and_novelty(phi_enc, rndphi, rollout, device)
        raw_i = nov.numpy()                       # (T, N)
        T, N = raw_i.shape
        raw_mean_pre = float(raw_i.mean())        # logged before any clip
        raw_max_pre = float(raw_i.max())

        # normalise → intrinsic reward. warm-up gives 0 (predictor burn-in); else
        # (optionally CLIP raw spikes, then) scale by EMA std of intrinsic returns,
        # NO mean-centring (bonus ≥ 0).
        warming = update <= cfg.norm_warmup_updates
        if warming:
            norm_i = np.zeros_like(raw_i)
        else:
            # raw-novelty clip BEFORE the normalizer (caps spikes; protects ret_std).
            if cfg.reward_clip_k is not None:
                raw_mean_ema = (raw_mean_pre if raw_mean_ema is None
                                else 0.99 * raw_mean_ema + 0.01 * raw_mean_pre)
                if raw_mean_ema > 0:
                    raw_i = np.minimum(raw_i, cfg.reward_clip_k * raw_mean_ema)
            rems = np.stack([rff.update(raw_i[t]) for t in range(T)])
            int_ret_std.update(rems)
            norm_i = raw_i / (int_ret_std.std + cfg.int_norm_eps)

        # The env's +1 is the STOP signal only — never a reward fed to GAE.
        extrinsic = rollout.rewards.clone()
        rollout.rewards = torch.from_numpy(norm_i.astype(np.float32))
        # Intrinsic stream: NON-episodic by default (canonical RND — novelty value
        # bootstraps across death/reset). `intrinsic_episodic=True` for the PPO-style
        # episodic variant.
        if cfg.intrinsic_episodic:
            _gae_episodic(rollout, cfg.gamma, cfg.gae_lambda)
        else:
            _gae_nonepisodic(rollout, cfg.gamma, cfg.gae_lambda)

        ustats = ppo_update(model, ppo_opt, rollout, ppo_cfg, device)

        # train φ (ICM) until frozen; then it is a stationary ruler for RND.
        icm_stats: dict = {}
        if icm is not None and not phi_frozen:      # frozen-φ mode: no ICM training/freeze
            icm_stats = icm_update_from_rollout(icm, icm_opt, rollout, cfg, device)
            last_inv_acc = icm_stats["inverse_acc"]                 # on-policy (can be inflated)
            last_holdout_inv = _eval_holdout_inv_acc(icm, holdout, device)   # φ's TRUE quality
            # The freeze TRIGGER reads the held-out metric by default (on-policy is
            # fooled by a narrowing policy — probes/inv_acc_causality.md).
            trig_inv = last_holdout_inv if cfg.freeze_metric == "holdout" else last_inv_acc
            inv_streak = inv_streak + 1 if trig_inv >= cfg.phi_freeze_inverse_acc else 0
            hit_thresh = inv_streak >= cfg.phi_freeze_patience
            hit_fallback = update >= cfg.phi_freeze_max_updates
            if hit_thresh or hit_fallback:
                for p in icm.phi.parameters():
                    p.requires_grad_(False)
                icm.phi.eval()
                phi_frozen = True
                freeze_step = global_step + cfg.rollout_steps * cfg.n_envs
                reason = f"{cfg.freeze_metric} inv_acc plateau" if hit_thresh else "MAX-UPDATES FALLBACK"
                print(f"[exp013_1] φ FROZEN ({reason}) at update {update} "
                      f"(holdout_inv={last_holdout_inv:.3f}, onpolicy_inv={last_inv_acc:.3f}, "
                      f"~{freeze_step} env steps)")
                # Postmortem clarity: a fallback-freeze with low HELD-OUT inv_acc means φ
                # never became controllable → RND rules over a near-random space (§9).
                if hit_fallback and not hit_thresh and last_holdout_inv < cfg.phi_freeze_inverse_acc:
                    print(f"[exp013_1] WARNING: φ frozen by FALLBACK with held-out inv_acc="
                          f"{last_holdout_inv:.3f} < {cfg.phi_freeze_inverse_acc} — φ is NOT "
                          f"controllable; the RND ruler is near-random (see SYSTEM_CARD §9).")

        # RND predictor update (+ leak) on the cached φ(next_obs).
        rnd_loss = _rnd_update(rndphi, rnd_opt, phi_cached, rollout.dones, cfg, device)

        global_step += cfg.rollout_steps * cfg.n_envs

        # headline: precise env-step of the FIRST extrinsic reward
        if first_reward_step is None and bool((extrinsic > 0).any().item()):
            t_idx = int((extrinsic > 0).float().sum(dim=1).nonzero()[0].item())
            base = global_step - cfg.rollout_steps * cfg.n_envs
            first_reward_step = base + (t_idx + 1) * cfg.n_envs
            print(f"[exp013_1] *** FIRST REWARD at ~{first_reward_step} env steps "
                  f"(update {update}) ***")
            if cfg.stop_on_first_reward:
                stop_now = True

        done_eps = envs.drain_completed_episodes()
        train_succ = (float(np.mean([e.success for e in done_eps]))
                      if done_eps else float("nan"))

        record = {
            "step": global_step, "update": update,
            "policy_loss": ustats.policy_loss,
            "value_loss": ustats.value_loss,
            "policy_entropy": ustats.entropy,
            "approx_kl": ustats.approx_kl,
            "clipfrac": ustats.clipfrac,
            "grad_norm_total": ustats.grad_norm_total,
            "novelty_raw_mean": raw_mean_pre,          # pre-clip
            "novelty_raw_max": raw_max_pre,            # pre-clip (spike detector)
            "intrinsic_reward_norm_mean": float(norm_i.mean()),
            "intrinsic_return_std": int_ret_std.std,
            "v_int_mean": float(rollout.values.mean()),       # for the value-lag question
            "ret_int_mean": float(rollout.returns.mean()),    # empirical intrinsic return
            "rnd_predictor_loss": rnd_loss,
            "inverse_acc": last_inv_acc,                # on-policy (can be inflated)
            "holdout_inv_acc": last_holdout_inv,        # φ's TRUE controllability
            "phi_frozen": bool(phi_frozen),
            "freeze_step": freeze_step,
            "norm_warming": bool(warming),
            "mean_feature_cosine": mean_feature_cosine(rollout.features, rollout.ep_starts),
            "train_success_rate": train_succ,
            "train_episodes": len(done_eps),
            "env_steps_to_first_reward": first_reward_step,
            "sps": global_step / max(1e-6, time.time() - t_start),
            **icm_stats,
        }
        if cfg.log_every > 0 and (update % cfg.log_every == 0 or stop_now):
            writer.write(record)
        if update % 25 == 0 or stop_now:
            print(f"[exp013_1] {cfg.exp_name} update {update}/{cfg.total_updates} "
                  f"step={global_step} frr={first_reward_step} "
                  f"nov_raw={record['novelty_raw_mean']:.4g} "
                  f"r^i_norm={record['intrinsic_reward_norm_mean']:.4g} "
                  f"inv_acc(onpol/holdout)={last_inv_acc:.3f}/{last_holdout_inv:.3f} "
                  f"frozen={phi_frozen} ent={ustats.entropy:.3f}")

        if cfg.save_every > 0 and update % cfg.save_every == 0:
            _save_ckpt(run_dir, model, phi_enc, rndphi, cfg, global_step, phi_frozen)

        if stop_now:
            break

    writer.close()
    _save_ckpt(run_dir, model, phi_enc, rndphi, cfg, global_step, phi_frozen)   # always save final (analysis)
    solved = first_reward_step is not None
    result = {
        "exp_name": cfg.exp_name, "method": "rnd_icm", "game": cfg.game,
        "level_index": cfg.level_index, "seed": cfg.seed,
        "env_steps_to_first_reward": first_reward_step,
        "solved": solved, "censored": not solved,
        "total_env_steps": global_step, "max_env_steps": cfg.max_env_steps,
        "phi_freeze_step": freeze_step, "leak": cfg.leak,
        "freeze_metric": cfg.freeze_metric,
        "holdout_inv_acc_final": last_holdout_inv,   # φ's TRUE controllability at the end
        "onpolicy_inv_acc_final": last_inv_acc,
        "wall_seconds": time.time() - t_start,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[exp013_1] DONE {cfg.exp_name}: "
          f"{'first reward @ ' + str(first_reward_step) if solved else 'CENSORED'} "
          f"env steps; φ_freeze={freeze_step}; total={global_step}")
    return result
