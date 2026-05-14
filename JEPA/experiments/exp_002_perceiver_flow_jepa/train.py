"""
Exp-002: Perceiver-JEPA + Flow Matching Predictor — Training Script.

Key differences from exp-001:
  - No EMA target encoder: encoder trained end-to-end via predictor loss
  - Single-life episodes: training skips the dying transition
  - Flow matching predictor (x0-parameterisation) instead of direct L2
  - Stateless MLP policy instead of cross-attention reasoning token
  - 20 monitoring metrics incl. per-latent loss/grad, time-emb grad, ODE consistency

Run:
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_002_perceiver_flow_jepa.train
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_002_perceiver_flow_jepa.train --resume checkpoints/step_005000.pt
"""

import argparse
import dataclasses
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_002_perceiver_flow_jepa.config import Config
from JEPA.experiments.exp_002_perceiver_flow_jepa.models import load_models
from JEPA.experiments.exp_002_perceiver_flow_jepa.reward_shaping import (
    is_end_of_life, count_energy, count_lives,
)
from JEPA.shared.buffer import ReplayBuffer, PolicyBuffer
from JEPA.shared.env_wrapper import make_env_auto

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_FREQ    = 5_000
LOG_FREQ           = 200
EMBED_METRIC_FREQ  = 25   # compute heavy embedding metrics every N JEPA updates
MAX_EP_STEPS       = 300  # force-flush episode after this many steps (prevents infinite loop
                          # when deterministic policy never triggers a life-end event)
REWARD_CAP         = 50.0 # clip curiosity reward; prevents OOD encoder explosion from
                          # inflating REINFORCE advantages arbitrarily

# Health thresholds
LATENT_NORM_CRITICAL   = 10.0
LATENT_STD_CRITICAL    = 0.01   # per latent std below this for 5 checks → dead latent
LOSS_CV_CRITICAL       = 2.0   # raised from 0.5: CV naturally > 0.5 during rapid descent
TIME_GRAD_WARN         = 1e-4
GRAD_NORM_CRITICAL     = 200.0
ENTROPY_WARN           = 0.30
ODE_COSSIM_WARN        = 0.99


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def effective_rank(matrix: torch.Tensor) -> float:
    """Shannon entropy of singular values (proxy for rank)."""
    try:
        sv = torch.linalg.svdvals(matrix.float().cpu())
        sv = sv / (sv.sum() + 1e-8)
        return float(torch.exp(-(sv * (sv + 1e-8).log()).sum()).item())
    except Exception:
        return float("nan")


def save_checkpoint(
    step: int,
    encoder, predictor, action_embed, policy, cfg: Config, label: str = ""
) -> Path:
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    tag = f"step_{step:06d}" + (f"_{label}" if label else "")
    path = ckpt_dir / f"{tag}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({
        "encoder":     encoder.state_dict(),
        "predictor":   predictor.state_dict(),
        "action_embed": action_embed.state_dict(),
        "policy":      policy.state_dict(),
        "step":        step,
        "config":      dataclasses.asdict(cfg),
    }, tmp)
    tmp.rename(path)
    return path


def load_checkpoint(path: str, encoder, predictor, action_embed, policy, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    action_embed.load_state_dict(ckpt["action_embed"])
    policy.load_state_dict(ckpt["policy"])
    return ckpt.get("step", 0)


# ── Activation monitor ────────────────────────────────────────────────────────

class ActivationMonitor:
    def __init__(self):
        self._hooks = []
        self._buf: dict[str, deque] = {}

    def register(self, module: nn.Module, name: str, window: int = 50) -> None:
        self._buf[name] = deque(maxlen=window)
        def _hook(_mod, _inp, out):
            dead = (out.detach().abs() < 0.01).float().mean().item()
            self._buf[name].append(dead)
        self._hooks.append(module.register_forward_hook(_hook))

    def mean_dead(self, name: str) -> float:
        b = self._buf.get(name, [])
        return float(np.mean(b)) if b else float("nan")

    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()


# ── Health monitor ────────────────────────────────────────────────────────────

class HealthMonitor:
    def __init__(self, window: int = 200):
        self.w = window

        self.flow_loss_total = deque(maxlen=window)
        self.per_latent_loss = [deque(maxlen=window) for _ in range(4)]

        self.grad_enc_s1    = deque(maxlen=window)
        self.grad_enc_s2    = deque(maxlen=window)
        self.grad_pred_mlps = [deque(maxlen=window) for _ in range(4)]
        self.grad_time_emb  = deque(maxlen=window)
        self.grad_policy    = deque(maxlen=window)
        self.clip_model     = deque(maxlen=window)
        self.clip_policy    = deque(maxlen=window)

        self.latent_norms   = [deque(maxlen=50) for _ in range(4)]
        self.latent_pairwise_l2 = deque(maxlen=50)
        self.latent_eff_rank    = deque(maxlen=50)
        self.across_state_std   = deque(maxlen=50)
        self.per_latent_std     = [deque(maxlen=50) for _ in range(4)]

        self.time_emb_grad      = deque(maxlen=window)
        self.ode_step_cossim    = deque(maxlen=50)

        self.pol_loss    = deque(maxlen=window)
        self.entropy     = deque(maxlen=window)
        self.rewards     = deque(maxlen=window)
        self.episodes_done: list = []

        self._low_std_counts = [0] * 4

    def _mean(self, q) -> float:
        return float(np.mean(q)) if q else float("nan")

    def mean_flow_loss(self) -> float:
        return self._mean(self.flow_loss_total)

    def mean_per_latent_loss(self) -> list:
        return [self._mean(q) for q in self.per_latent_loss]

    def mean_latent_norms(self) -> list:
        return [self._mean(q) for q in self.latent_norms]

    def mean_entropy(self) -> float:
        return self._mean(self.entropy)

    def mean_reward(self) -> float:
        return self._mean(self.rewards)

    def completion_rate(self, window: int = 20) -> float:
        recent = self.episodes_done[-window:]
        return float(np.mean(recent)) if recent else 0.0

    def check(self) -> tuple:
        criticals, warnings = [], []

        if self.flow_loss_total and np.isnan(self.flow_loss_total[-1]):
            criticals.append("NaN in flow-matching loss")

        if self.grad_enc_s1 and self._mean(self.grad_enc_s1) > GRAD_NORM_CRITICAL:
            criticals.append(f"Exploding encoder grad: {self._mean(self.grad_enc_s1):.1f}")

        for i, q in enumerate(self.per_latent_std):
            if q and self._mean(q) < LATENT_STD_CRITICAL:
                self._low_std_counts[i] += 1
                if self._low_std_counts[i] >= 5:
                    criticals.append(f"Dead latent {i}: std={self._mean(q):.5f}")
            else:
                self._low_std_counts[i] = 0

        for i, q in enumerate(self.latent_norms):
            if q and self._mean(q) > LATENT_NORM_CRITICAL:
                criticals.append(f"Latent {i} norm explosion: {self._mean(q):.2f}")

        if self.flow_loss_total and len(self.flow_loss_total) > 50:
            recent = list(self.flow_loss_total)[-50:]
            mean_loss = abs(np.mean(recent))
            # Only apply CV check when loss is at a meaningful level (> 0.05).
            # Near zero, occasional buffer-overwrite spikes produce arbitrarily high CV
            # even when training is healthy. CV check is meant for plateau oscillation,
            # not convergence-phase noise.
            if mean_loss > 0.05:
                cv = float(np.std(recent) / (mean_loss + 1e-8))
                if cv > LOSS_CV_CRITICAL:
                    criticals.append(f"Loss CV={cv:.2f} > {LOSS_CV_CRITICAL} (training oscillation)")

        if self.time_emb_grad and self._mean(self.time_emb_grad) < TIME_GRAD_WARN:
            warnings.append(f"Time embedding grad={self._mean(self.time_emb_grad):.2e} (possible time blindness)")

        if self.ode_step_cossim and self._mean(self.ode_step_cossim) > ODE_COSSIM_WARN:
            warnings.append(f"ODE step cos-sim={self._mean(self.ode_step_cossim):.4f} > {ODE_COSSIM_WARN} (ODE collapse?)")

        if self.entropy and self._mean(self.entropy) < ENTROPY_WARN:
            warnings.append(f"Low policy entropy: {self._mean(self.entropy):.3f}")

        return criticals, warnings


# ── ODE consistency monitor (samples every N steps) ──────────────────────────

@torch.no_grad()
def compute_ode_step_cossim(predictor, h_t: torch.Tensor, action_emb: torch.Tensor) -> float:
    """
    Run Euler ODE and measure cosine similarity between consecutive steps.
    Returns mean cos-sim; should be < 0.99 for healthy ODE diversity.
    """
    if h_t.shape[0] == 0:
        return float("nan")
    x_0 = h_t[:1]  # use first sample only
    a_emb = action_emb[:1]
    x_k = x_0.clone()
    N = predictor.n_ode_steps
    steps = [x_k.clone()]
    for k in range(N):
        tau_val = k / N
        tau = torch.full((1,), tau_val, device=h_t.device)
        x1_hat = predictor._predict_clean(x_k, tau, a_emb)
        v_hat = x1_hat - x_0
        x_k = x_k + (1.0 / N) * v_hat
        steps.append(x_k.clone())
    if len(steps) < 2:
        return float("nan")
    sims = []
    for i in range(len(steps) - 1):
        s1 = steps[i].view(-1)
        s2 = steps[i+1].view(-1)
        cos = F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
        sims.append(cos)
    return float(np.mean(sims))


# ── Print stats ───────────────────────────────────────────────────────────────

def print_stats(step: int, health: HealthMonitor, act_mon: ActivationMonitor,
                buf_size: int, ep_count: int, fps: float) -> None:
    per_lat_loss = health.mean_per_latent_loss()
    per_lat_norm = health.mean_latent_norms()
    lat_loss_str = " | ".join(f"L{i}={v:.4f}" for i, v in enumerate(per_lat_loss))
    lat_norm_str = " | ".join(f"L{i}={v:.3f}" for i, v in enumerate(per_lat_norm))
    per_lat_grad = [health._mean(q) for q in health.grad_pred_mlps]
    lat_grad_str = " | ".join(f"L{i}={v:.3f}" for i, v in enumerate(per_lat_grad))

    print(
        f"\n{'─'*72}\n"
        f"  Step {step:7d}  fps={fps:.0f}  buf={buf_size}  ep={ep_count}\n"
        f"  FlowLoss: total={health.mean_flow_loss():.5f}  per-latent: {lat_loss_str}\n"
        f"  Enc grads: SA={health._mean(health.grad_enc_s1):.3f}  "
        f"Perceiver={health._mean(health.grad_enc_s2):.3f}  "
        f"TimeEmb={health._mean(health.time_emb_grad):.4f}\n"
        f"  Pred MLP grads: {lat_grad_str}\n"
        f"  Latent norms: {lat_norm_str}\n"
        f"  Latent pairwise_L2={health._mean(health.latent_pairwise_l2):.4f}  "
        f"eff_rank={health._mean(health.latent_eff_rank):.2f}  "
        f"across_std={health._mean(health.across_state_std):.4f}\n"
        f"  ODE cos-sim={health._mean(health.ode_step_cossim):.4f}  "
        f"(< 0.99 = ODE is evolving)\n"
        f"  Policy: loss={health._mean(health.pol_loss):.5f}  "
        f"gnorm={health._mean(health.grad_policy):.3f}  "
        f"H={health.mean_entropy():.3f}\n"
        f"  Reward: mean={health.mean_reward():.5f}  "
        f"completion={health.completion_rate():.1%}\n"
        f"{'─'*72}"
    )
    dead_parts = [f"{n}={act_mon.mean_dead(n):.2f}" for n in act_mon._buf]
    if dead_parts:
        print(f"  Dead GELU: {' | '.join(dead_parts)}")


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: Config, resume_path: str = None) -> None:
    set_seeds(cfg.seed)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"[exp002] device={device}  max_steps={cfg.max_steps}")

    # ── Models ────────────────────────────────────────────────────────────────
    encoder, predictor, action_embed, policy, baseline = load_models(cfg, device)

    # ── Optional resume ───────────────────────────────────────────────────────
    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(resume_path, encoder, predictor,
                                     action_embed, policy, device)
        print(f"[exp002] Resumed from step {start_step}")

    # ── Activation monitors ───────────────────────────────────────────────────
    act_mon = ActivationMonitor()
    # Encoder SA blocks
    for i, block in enumerate(encoder.sa_blocks):
        act_mon.register(block.ffn.net[1], f"enc_sa{i}_ffn")
    # Perceiver cross-attn FFN
    act_mon.register(encoder.perceiver.cross_attn.ffn.net[1], "perc_cross_ffn")
    # Perceiver self-attn FFN
    act_mon.register(encoder.perceiver.self_attn.ffn.net[1], "perc_self_ffn")
    # Predictor MLPs
    for i, mlp in enumerate(predictor.mlps):
        act_mon.register(mlp.net[1], f"pred_mlp{i}")
    # Policy
    act_mon.register(policy.net[1], "policy_ffn")

    # ── Optimisers ────────────────────────────────────────────────────────────
    # Encoder param groups for separate grad norm monitoring
    enc_s1_params = list(encoder.color_embed.parameters()) + \
                    list(encoder.patch_proj.parameters()) + \
                    list(encoder.sa_blocks.parameters()) + \
                    list(encoder.sa_norm.parameters())
    enc_s2_params = list(encoder.perceiver.parameters())

    model_params = (
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(action_embed.parameters())
    )
    enc_opt  = torch.optim.AdamW(
        list(encoder.parameters()), lr=cfg.encoder_lr, weight_decay=cfg.encoder_wd
    )
    pred_opt = torch.optim.AdamW(
        list(predictor.parameters()) + list(action_embed.parameters()),
        lr=cfg.predictor_lr, weight_decay=cfg.predictor_wd
    )
    pol_opt  = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)

    # ── Buffers ───────────────────────────────────────────────────────────────
    replay_buf = ReplayBuffer(cfg.buffer_size, cfg.recency_fraction, cfg.recent_buffer_size)
    policy_buf = PolicyBuffer(cfg.policy_update_freq)

    # ── Environment ───────────────────────────────────────────────────────────
    env = make_env_auto(cfg.game_id, str(_repo_root / "environment_files"))

    # ── State ─────────────────────────────────────────────────────────────────
    health = HealthMonitor(window=LOG_FREQ)
    frame_np = env.reset()
    ep_count = 0
    step = start_step
    t0 = time.time()

    # Perceiver latent state: None = use placeholders at episode start
    h_t: torch.Tensor | None = None   # (1, n_latents, d_model) on device, or None
    ep_is_first_step = True

    # Episode transition buffer (held until life end)
    ep_transitions: list = []   # list of (frame_np, action, next_frame_np)

    print(f"\n[exp002] Training started  warmup={cfg.warmup_steps}  seed={cfg.seed}\n")

    while step < cfg.max_steps:
        # ── Encode current frame ──────────────────────────────────────────────
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

        with torch.no_grad():
            if h_t is None:
                queries = encoder.perceiver.get_initial_queries(1, device)
            else:
                queries = h_t.detach()

            h_current, sa_out, _ = encoder(frame_t, queries)
            # h_current: (1, n_latents, d_model)

        # ── Select action ─────────────────────────────────────────────────────
        avail = env.available_actions

        if step < cfg.warmup_steps:
            action_idx = int(np.random.randint(0, cfg.n_actions))
            log_prob = None
            entropy_val = None
        else:
            action_idx, log_prob, entropy = policy.act(
                h_current.squeeze(0), avail
            )
            entropy_val = entropy.item()

        # ── Step environment ──────────────────────────────────────────────────
        next_np, is_terminal = env.step(action_idx)
        life_end = is_end_of_life(frame_np, next_np, is_terminal)

        # ── Compute per-step reward (flow-matching prediction error) ──────────
        # This is computed lazily during training steps; here we just record
        # a proxy reward: was the action wall-hit or not?
        # The actual curiosity reward is assigned when we do the policy update.

        # ── Compute intrinsic curiosity reward (flow-matching MSE) ──────────────
        with torch.no_grad():
            next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)
            next_queries = h_current.detach()
            h_next, _, _ = encoder(next_t, next_queries)
            a_idx_t = torch.tensor([action_idx], device=device)
            a_emb_r = action_embed(a_idx_t)
            _, per_lat_r = predictor.predict_with_loss(h_current, h_next, a_emb_r)
            raw_reward = per_lat_r.mean().item()
            # Cap reward: when the episode is very long (encoder overfits to frozen
            # buffer while rollout state drifts OOD), prediction error grows unboundedly.
            # Capping to REWARD_CAP prevents REINFORCE from receiving garbage signal.
            curiosity_reward = min(raw_reward, REWARD_CAP) if np.isfinite(raw_reward) else 0.0

        # ── Store transition in episode buffer (NOT yet in replay) ────────────
        ep_transitions.append((frame_np.copy(), action_idx, next_np.copy()))

        # ── Policy buffer (on-policy, for REINFORCE) ──────────────────────────
        if step >= cfg.warmup_steps and log_prob is not None:
            policy_buf.add(log_prob, curiosity_reward, entropy)
            health.rewards.append(curiosity_reward)
            if entropy_val is not None:
                health.entropy.append(entropy_val)

        # ── Update h_t for next step ──────────────────────────────────────────
        h_t = h_current

        # ── Handle life end, terminal, or episode timeout ─────────────────────
        ep_len = len(ep_transitions)
        force_flush = (ep_len >= MAX_EP_STEPS) and not life_end

        if life_end or force_flush:
            # Flush episode: add all but the last transition to replay buffer.
            # On force-flush (infinite episode guard), exclude last to maintain
            # the training invariant that we never train on a "dying" state.
            for frame_i, action_i, next_i in ep_transitions[:-1]:
                replay_buf.add(frame_i, action_i, next_i)

            if force_flush:
                # Keep the last transition as the start of the next mini-episode
                last = ep_transitions[-1]
                ep_transitions = [last]
                # h_t stays — we continue from current latent state
            else:
                ep_count += 1
                level_done = env.level_completed
                health.episodes_done.append(int(level_done))
                ep_transitions = []
                h_t = None

            if life_end:
                if is_terminal:
                    frame_np = env.reset()
                else:
                    frame_np = next_np
            else:
                frame_np = next_np
        else:
            frame_np = next_np

        step += 1

        # ── JEPA / flow-matching training ─────────────────────────────────────
        if step % cfg.update_freq == 0 and len(replay_buf) >= cfg.min_buffer_size:
            batch = replay_buf.sample(cfg.batch_size, device)

            frames  = batch.frames                 # (B, 64, 64) uint8
            actions = batch.actions                # (B,)
            nframes = batch.next_frames            # (B, 64, 64) uint8

            enc_opt.zero_grad()
            pred_opt.zero_grad()

            # Encode current frames: use placeholder queries (batch, no recurrence)
            B = frames.shape[0]
            q_t = encoder.perceiver.get_initial_queries(B, device)
            h_t_batch, _, _ = encoder(frames, q_t)    # (B, n_latents, d_model)

            # Encode next frames: use h_t_batch as queries (recurrent path)
            h_t1_batch, _, _ = encoder(nframes, h_t_batch.detach())
            # NOTE: h_t1 gradients flow only from nframes encode, NOT through h_t
            # (detach h_t to avoid double-counting gradient from current-frame path)

            # Action embeddings
            a_emb = action_embed(actions)           # (B, d_action)

            # Flow-matching loss
            flow_loss, per_lat = predictor.compute_loss(h_t_batch, h_t1_batch, a_emb)

            if not torch.isfinite(flow_loss):
                print(f"[WARNING step {step}] Non-finite flow loss: {flow_loss.item():.4f} — skipping update")
                enc_opt.zero_grad(); pred_opt.zero_grad()
            else:
                flow_loss.backward()

                # Grad norm monitoring (before clip)
                gn_s1  = grad_norm(enc_s1_params)
                gn_s2  = grad_norm(enc_s2_params)
                gn_mlps = [grad_norm(list(mlp.parameters())) for mlp in predictor.mlps]
                gn_time = grad_norm(list(predictor.time_embed.parameters()))

                # Clip
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters())
                    + list(action_embed.parameters()),
                    cfg.grad_clip_model
                )
                clipped = float(gn_s1 + gn_s2 > cfg.grad_clip_model)

                enc_opt.step(); pred_opt.step()

                # Record metrics
                health.flow_loss_total.append(flow_loss.item())
                for i, v in enumerate(per_lat.tolist()):
                    health.per_latent_loss[i].append(v)
                health.grad_enc_s1.append(gn_s1)
                health.grad_enc_s2.append(gn_s2)
                for i, gn in enumerate(gn_mlps):
                    health.grad_pred_mlps[i].append(gn)
                health.time_emb_grad.append(gn_time)
                health.clip_model.append(clipped)

            # Every EMBED_METRIC_FREQ JEPA updates, compute latent diversity metrics
            if (step // cfg.update_freq) % EMBED_METRIC_FREQ == 0:
                with torch.no_grad():
                    # Sample a fresh small batch for metrics
                    mb = replay_buf.sample(min(16, len(replay_buf)), device)
                    q_m = encoder.perceiver.get_initial_queries(mb.frames.shape[0], device)
                    h_m, _, _ = encoder(mb.frames, q_m)  # (16, 4, d_model)

                    # Per-latent norm and std
                    for i in range(cfg.n_latents):
                        norms_i = h_m[:, i, :].norm(dim=-1).mean().item()
                        std_i   = h_m[:, i, :].std().item()
                        health.latent_norms[i].append(norms_i)
                        health.per_latent_std[i].append(std_i)

                    # Pairwise L2 among 4 latents (within same sample)
                    pairs = []
                    for i in range(cfg.n_latents):
                        for j in range(i+1, cfg.n_latents):
                            d = (h_m[:, i, :] - h_m[:, j, :]).norm(dim=-1).mean().item()
                            pairs.append(d)
                    health.latent_pairwise_l2.append(float(np.mean(pairs)))

                    # Effective rank across the 4 latents
                    h_sample = h_m[0]  # (4, d_model)
                    health.latent_eff_rank.append(effective_rank(h_sample))

                    # Across-state std
                    across_std = h_m.std(dim=0).mean().item()
                    health.across_state_std.append(across_std)

                    # ODE step cos-sim
                    a_m = action_embed(mb.actions[:1])
                    cossim = compute_ode_step_cossim(predictor, h_m[:1], a_m)
                    health.ode_step_cossim.append(cossim)

        # ── Policy training ───────────────────────────────────────────────────
        if step >= cfg.warmup_steps and policy_buf.full():
            log_probs, rewards, entropies = policy_buf.get(device)

            # Normalise advantages
            adv = rewards - baseline.value
            adv = adv / (adv.std().clamp(min=1e-8) + 1e-8)

            baseline.update(rewards.mean().item())

            pol_loss = -(adv * log_probs).mean() - cfg.policy_entropy_lambda * entropies.mean()

            pol_opt.zero_grad()
            pol_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_policy)
            pol_opt.step()
            policy_buf.clear()

            health.pol_loss.append(pol_loss.item())
            health.grad_policy.append(grad_norm(policy.parameters()))

        # ── Logging ───────────────────────────────────────────────────────────
        if step % LOG_FREQ == 0 and step > 0:
            fps = step / (time.time() - t0 + 1e-6)
            print_stats(step, health, act_mon, len(replay_buf), ep_count, fps)

            criticals, warnings = health.check()
            for w in warnings:
                print(f"  [WARN] {w}")
            for c in criticals:
                print(f"  [CRITICAL] {c}")
            if criticals:
                print("[exp002] Critical issue detected — saving checkpoint and stopping")
                save_checkpoint(step, encoder, predictor, action_embed, policy, cfg, "critical")
                break

        # ── Checkpointing ─────────────────────────────────────────────────────
        if step % CHECKPOINT_FREQ == 0 and step > start_step:
            path = save_checkpoint(step, encoder, predictor, action_embed, policy, cfg)
            print(f"  [ckpt] Saved {path.name}")

    # ── End of training ───────────────────────────────────────────────────────
    path = save_checkpoint(step, encoder, predictor, action_embed, policy, cfg, "final")
    print(f"\n[exp002] Training complete at step {step}. Final checkpoint: {path.name}")
    act_mon.remove()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--game-id", type=str, default=None,
                        help="Override game_id, e.g. ls20s for the simplified env")
    args = parser.parse_args()

    cfg = Config()
    # Allow CLI overrides via mutation workaround (frozen dataclass)
    overrides = {}
    if args.max_steps  is not None: overrides["max_steps"]  = args.max_steps
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if args.game_id    is not None: overrides["game_id"]    = args.game_id
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
