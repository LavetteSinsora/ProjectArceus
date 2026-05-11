"""
Exp-003: Normalized Latent JEPA — Training Script.

Key changes from exp_002:
  1. LatentBuffer: stores (frame_t, h_query=h_{t-1}, action_t, h_target=h_{t+1})
     from the recurrent rollout. Training re-encodes frame_t with stored h_query,
     giving encoder gradient while matching the rollout's encoding path exactly.
  2. Stop gradient on h_t1: predictor.compute_loss(h_t_fresh, h_target.detach(), a)
     prevents the collapse attractor (encoder can no longer minimize loss by mapping
     all states to a constant).
  3. Per-component optimizer: Perceiver LR = 5e-5 (vs SA/predictor at 1e-4).
     Rationale: separate per-round weights halve gradient accumulation vs exp_002's
     weight-tied design; lower LR further stabilises Perceiver updates.
  4. Perceiver output_norm: already baked into the encoder architecture (exp_003 encoder).

Run:
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.train
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.train --resume checkpoints/step_050000.pt
"""

import argparse
import dataclasses
import datetime
import io
import json
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

from JEPA.experiments.exp_003_0_normalized_latent_jepa.config import Config
from JEPA.experiments.exp_003_0_normalized_latent_jepa.models import load_models
from JEPA.experiments.exp_003_0_normalized_latent_jepa.reward_shaping import (
    is_end_of_life, count_energy, count_lives,
)
from JEPA.shared.buffer import LatentBuffer, PolicyBuffer
from JEPA.shared.env_wrapper import LS20Env

# ── Run logger ───────────────────────────────────────────────────────────────

class _Tee(io.TextIOBase):
    """Write to both the original stdout and a log file simultaneously."""
    def __init__(self, original: io.TextIOBase, log_path: Path):
        self._orig = original
        self._file = open(log_path, "w", buffering=1, encoding="utf-8")

    def write(self, data: str) -> int:
        self._orig.write(data)
        self._orig.flush()
        self._file.write(data)
        return len(data)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def close(self):
        super().close()
        self._file.close()


def setup_run_logger(resume_path: str | None) -> tuple[Path, _Tee]:
    """
    Create a timestamped run directory under exp_003/runs/ and install a Tee
    that mirrors all print() output to training.log inside that directory.

    Returns (run_dir, tee) so the caller can restore sys.stdout on exit.
    """
    runs_root = Path(__file__).parent / "runs"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "resume" if resume_path else "fresh"
    run_dir = runs_root / f"run_{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "training.log"
    tee = _Tee(sys.stdout, log_path)
    sys.stdout = tee
    print(f"[exp003] Run dir:  {run_dir}")
    print(f"[exp003] Log file: {log_path}")
    return run_dir, tee


# ── Metrics writer ────────────────────────────────────────────────────────────

class MetricsWriter:
    """Appends one JSON line per log step to metrics.jsonl in the run directory."""

    def __init__(self, run_dir: Path):
        self._path = run_dir / "metrics.jsonl"
        self._fh = open(self._path, "w", buffering=1, encoding="utf-8")
        print(f"[exp003] Metrics file: {self._path}")

    def write(self, step: int, fps: float, buf_size: int, ep_count: int,
              health: "HealthMonitor") -> None:
        def _m(q):   return round(float(health._mean(q)), 6) if q else None
        def _ml(qs): return [_m(q) for q in qs]

        record = {
            "step":          step,
            "fps":           round(fps, 1),
            "buf_size":      buf_size,
            "ep_count":      ep_count,
            # Flow-matching loss
            "flow_loss":     _m(health.flow_loss_total),
            "per_lat_loss":  _ml(health.per_latent_loss),
            # Encoder gradients
            "grad_enc_sa":       _m(health.grad_enc_s1),
            "grad_enc_perceiver": _m(health.grad_enc_s2),
            "grad_time_emb":     _m(health.time_emb_grad),
            # Predictor MLP gradients (per latent)
            "grad_pred_mlps":    _ml(health.grad_pred_mlps),
            # Latent quality
            "latent_norms":      _ml(health.latent_norms),
            "latent_pairwise_l2": _m(health.latent_pairwise_l2),
            "latent_eff_rank":   _m(health.latent_eff_rank),
            "across_state_std":  _m(health.across_state_std),
            "per_lat_std":       _ml(health.per_latent_std),
            # ODE consistency
            "ode_cossim":    _m(health.ode_step_cossim),
            # Temporal latent coherence
            "ht_ht1_cossim": _m(health.ht_ht1_cossim),
            # Policy
            "pol_loss":      _m(health.pol_loss),
            "grad_policy":   _m(health.grad_policy),
            "entropy":       _m(health.entropy),
            "mean_reward":   round(health.mean_reward(), 6),
            "completion_rate": round(health.completion_rate(), 4),
        }
        self._fh.write(json.dumps(record) + "\n")

    def close(self):
        self._fh.close()


# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_FREQ   = 5_000
LOG_FREQ          = 200
EMBED_METRIC_FREQ = 25
MAX_EP_STEPS      = 300   # guard against infinite episodes (e.g. deterministic wall-hitting)
REWARD_CAP        = 50.0  # clip curiosity reward

# With output_norm applied, each latent vector has entries with mean=0, var=1.
# Expected L2 norm = sqrt(d_model) = sqrt(128) ≈ 11.31  (by E[||v||²] = d).
# The threshold must be >> sqrt(128); a 3× multiple catches real failures (e.g.
# output_norm bypassed or gradient explosion) while never firing under normal training.
LATENT_NORM_CRITICAL = 34.0   # = ~3 × sqrt(128); fires only if output_norm fails
LATENT_STD_CRITICAL  = 0.01
LOSS_CV_CRITICAL     = 2.0
TIME_GRAD_WARN       = 1e-4
GRAD_NORM_CRITICAL   = 200.0
ENTROPY_WARN         = 0.30
ODE_COSSIM_WARN      = 0.99


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
    try:
        sv = torch.linalg.svdvals(matrix.float().cpu())
        sv = sv / (sv.sum() + 1e-8)
        return float(torch.exp(-(sv * (sv + 1e-8).log()).sum()).item())
    except Exception:
        return float("nan")


def save_checkpoint(step, encoder, predictor, action_embed, policy, cfg, label=""):
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    tag = f"step_{step:06d}" + (f"_{label}" if label else "")
    path = ckpt_dir / f"{tag}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({
        "encoder":      encoder.state_dict(),
        "predictor":    predictor.state_dict(),
        "action_embed": action_embed.state_dict(),
        "policy":       policy.state_dict(),
        "step":         step,
        "config":       dataclasses.asdict(cfg),
    }, tmp)
    tmp.rename(path)
    return path


def load_checkpoint(path, encoder, predictor, action_embed, policy, device):
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
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Health monitor ────────────────────────────────────────────────────────────

class HealthMonitor:
    def __init__(self, window: int = 200):
        self.w = window
        self.flow_loss_total  = deque(maxlen=window)
        self.per_latent_loss  = [deque(maxlen=window) for _ in range(4)]
        self.grad_enc_s1      = deque(maxlen=window)
        self.grad_enc_s2      = deque(maxlen=window)   # Perceiver (all rounds)
        self.grad_pred_mlps   = [deque(maxlen=window) for _ in range(4)]
        self.grad_time_emb    = deque(maxlen=window)
        self.grad_policy      = deque(maxlen=window)
        self.latent_norms     = [deque(maxlen=50) for _ in range(4)]
        self.latent_pairwise_l2 = deque(maxlen=50)
        self.latent_eff_rank  = deque(maxlen=50)
        self.across_state_std = deque(maxlen=50)
        self.per_latent_std   = [deque(maxlen=50) for _ in range(4)]
        self.time_emb_grad    = deque(maxlen=window)
        self.ode_step_cossim  = deque(maxlen=50)
        self.ht_ht1_cossim    = deque(maxlen=window)
        self.pol_loss         = deque(maxlen=window)
        self.entropy          = deque(maxlen=window)
        self.rewards          = deque(maxlen=window)
        self.episodes_done: list = []
        self._low_std_counts  = [0] * 4

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
            if mean_loss > 0.05:
                cv = float(np.std(recent) / (mean_loss + 1e-8))
                if cv > LOSS_CV_CRITICAL:
                    criticals.append(f"Loss CV={cv:.2f} > {LOSS_CV_CRITICAL}")
        if self.time_emb_grad and self._mean(self.time_emb_grad) < TIME_GRAD_WARN:
            warnings.append(f"Time embedding grad={self._mean(self.time_emb_grad):.2e}")
        if self.ode_step_cossim and self._mean(self.ode_step_cossim) > ODE_COSSIM_WARN:
            warnings.append(f"ODE step cos-sim={self._mean(self.ode_step_cossim):.4f} > {ODE_COSSIM_WARN}")
        if self.entropy and self._mean(self.entropy) < ENTROPY_WARN:
            warnings.append(f"Low policy entropy: {self._mean(self.entropy):.3f}")
        return criticals, warnings


# ── ODE consistency monitor ───────────────────────────────────────────────────

@torch.no_grad()
def compute_ode_step_cossim(predictor, h_t: torch.Tensor, action_emb: torch.Tensor) -> float:
    if h_t.shape[0] == 0:
        return float("nan")
    x_0 = h_t[:1]; x_k = x_0.clone()
    a_emb = action_emb[:1]; N = predictor.n_ode_steps
    steps = [x_k.clone()]
    for k in range(N):
        tau = torch.full((1,), k / N, device=h_t.device)
        x1_hat = predictor._predict_clean(x_k, tau, a_emb)
        x_k = x_k + (1.0 / N) * (x1_hat - x_0)
        steps.append(x_k.clone())
    sims = [
        F.cosine_similarity(steps[i].view(-1).unsqueeze(0),
                            steps[i+1].view(-1).unsqueeze(0)).item()
        for i in range(len(steps) - 1)
    ]
    return float(np.mean(sims))


# ── Print stats ───────────────────────────────────────────────────────────────

def print_stats(step, health, act_mon, buf_size, ep_count, fps):
    per_lat_loss = health.mean_per_latent_loss()
    per_lat_norm = health.mean_latent_norms()
    per_lat_grad = [health._mean(q) for q in health.grad_pred_mlps]
    print(
        f"\n{'─'*72}\n"
        f"  Step {step:7d}  fps={fps:.0f}  buf={buf_size}  ep={ep_count}\n"
        f"  FlowLoss: total={health.mean_flow_loss():.5f}  "
        f"per-latent: {' | '.join(f'L{i}={v:.4f}' for i,v in enumerate(per_lat_loss))}\n"
        f"  Enc grads: SA={health._mean(health.grad_enc_s1):.3f}  "
        f"Perceiver(all-rounds)={health._mean(health.grad_enc_s2):.3f}  "
        f"TimeEmb={health._mean(health.time_emb_grad):.4f}\n"
        f"  Pred MLP grads: {' | '.join(f'L{i}={v:.3f}' for i,v in enumerate(per_lat_grad))}\n"
        f"  Latent norms: {' | '.join(f'L{i}={v:.3f}' for i,v in enumerate(per_lat_norm))}\n"
        f"  Latent pairwise_L2={health._mean(health.latent_pairwise_l2):.4f}  "
        f"eff_rank={health._mean(health.latent_eff_rank):.2f}  "
        f"across_std(placeholder)={health._mean(health.across_state_std):.4f}\n"
        f"  ODE cos-sim={health._mean(health.ode_step_cossim):.4f}  "
        f"H_T·H_T+1 cos-sim={health._mean(health.ht_ht1_cossim):.4f}\n"
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

def train(cfg: Config, resume_path: str = None, run_dir: Path = None) -> None:
    set_seeds(cfg.seed)
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[exp003] device={device}  max_steps={cfg.max_steps}")

    encoder, predictor, action_embed, policy, baseline = load_models(cfg, device)

    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(resume_path, encoder, predictor,
                                     action_embed, policy, device)
        print(f"[exp003] Resumed from step {start_step}")

    # ── Activation monitors ───────────────────────────────────────────────────
    act_mon = ActivationMonitor()
    for i, block in enumerate(encoder.sa_blocks):
        act_mon.register(block.ffn.net[1], f"enc_sa{i}_ffn")
    # Perceiver rounds have separate weights — register first round for monitoring
    act_mon.register(encoder.perceiver.rounds[0].cross_attn.ffn.net[1], "perc_r0_cross_ffn")
    act_mon.register(encoder.perceiver.rounds[0].self_attn.ffn.net[1],  "perc_r0_self_ffn")
    if cfg.n_perceiver_rounds > 1:
        act_mon.register(encoder.perceiver.rounds[1].cross_attn.ffn.net[1], "perc_r1_cross_ffn")
        act_mon.register(encoder.perceiver.rounds[1].self_attn.ffn.net[1],  "perc_r1_self_ffn")
    for i, mlp in enumerate(predictor.mlps):
        act_mon.register(mlp.net[1], f"pred_mlp{i}")
    act_mon.register(policy.net[1], "policy_ffn")

    # ── Per-component parameter groups ────────────────────────────────────────
    enc_s1_params = (
        list(encoder.color_embed.parameters()) +
        list(encoder.patch_proj.parameters()) +
        list(encoder.sa_blocks.parameters()) +
        list(encoder.sa_norm.parameters())
    )
    enc_s2_params = list(encoder.perceiver.parameters())  # all Perceiver params

    # Per-component LR: Perceiver at 5e-5, SA+embed at 1e-4
    enc_opt = torch.optim.AdamW([
        {"params": enc_s1_params,  "lr": cfg.sa_lr,         "weight_decay": cfg.encoder_wd},
        {"params": enc_s2_params,  "lr": cfg.perceiver_lr,  "weight_decay": cfg.encoder_wd},
    ])
    pred_opt = torch.optim.AdamW(
        list(predictor.parameters()) + list(action_embed.parameters()),
        lr=cfg.predictor_lr, weight_decay=cfg.predictor_wd,
    )
    pol_opt = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)

    # ── Buffers ───────────────────────────────────────────────────────────────
    latent_buf = LatentBuffer(
        n_latents=cfg.n_latents,
        d_model=cfg.d_model,
        capacity=cfg.buffer_size,
        recency_fraction=cfg.recency_fraction,
        recent_window=cfg.recent_buffer_size,
    )
    policy_buf = PolicyBuffer(cfg.policy_update_freq)

    # ── Environment ───────────────────────────────────────────────────────────
    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)

    # ── Metrics writer ────────────────────────────────────────────────────────
    metrics_writer: MetricsWriter | None = None
    if run_dir is not None:
        metrics_writer = MetricsWriter(run_dir)

    # ── State ─────────────────────────────────────────────────────────────────
    health  = HealthMonitor(window=LOG_FREQ)
    frame_np = env.reset()
    ep_count = 0
    step = start_step
    t0   = time.time()

    h_t: torch.Tensor | None = None   # (1, n_latents, d_model) recurrent state
    # Queries used to produce h_t (= h_{t-1}, or placeholder at t=0)
    # Stored as numpy for the latent buffer
    h_query_np: np.ndarray | None = None  # (n_latents, d_model) float32

    # Episode buffer: (frame_np, h_query_np, action_idx, h_target_np)
    ep_transitions: list = []

    print(f"\n[exp003] Training started  warmup={cfg.warmup_steps}  seed={cfg.seed}\n")

    while step < cfg.max_steps:
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

        # ── Encode current frame (recurrent, no grad) ─────────────────────────
        with torch.no_grad():
            if h_t is None:
                queries = encoder.perceiver.get_initial_queries(1, device)
                # h_query for episode start = placeholder vectors
                h_query_np = queries.squeeze(0).cpu().numpy()
            else:
                queries = h_t.detach()
                # h_query is h_{t-1}, already stored from previous step

            h_current, _, _ = encoder(frame_t, queries)

        # ── Select action ─────────────────────────────────────────────────────
        avail = env.available_actions
        if step < cfg.warmup_steps:
            action_idx = int(np.random.randint(0, cfg.n_actions))
            log_prob = entropy_val = None
        else:
            action_idx, log_prob, entropy = policy.act(h_current.squeeze(0), avail)
            entropy_val = entropy.item()

        # ── Step environment ──────────────────────────────────────────────────
        next_np, is_terminal = env.step(action_idx)
        life_end = is_end_of_life(frame_np, next_np, is_terminal)

        # ── Compute curiosity reward + h_target ───────────────────────────────
        with torch.no_grad():
            next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)
            h_next, _, _ = encoder(next_t, h_current.detach())
            a_emb_r = action_embed(torch.tensor([action_idx], device=device))
            _, per_lat_r = predictor.predict_with_loss(h_current, h_next, a_emb_r)
            raw_reward = per_lat_r.mean().item()
            curiosity_reward = min(raw_reward, REWARD_CAP) if np.isfinite(raw_reward) else 0.0
            h_target_np = h_next.squeeze(0).cpu().numpy()  # (n_latents, d_model)
            ht_ht1_cs = F.cosine_similarity(
                h_current.squeeze(0), h_next.squeeze(0), dim=-1
            ).mean().item()
        health.ht_ht1_cossim.append(ht_ht1_cs)

        # ── Store transition in episode buffer ────────────────────────────────
        # h_query_np is the query used to produce h_current (= h_{t-1} or placeholder)
        ep_transitions.append((
            frame_np.copy(),
            h_query_np.copy(),
            action_idx,
            h_target_np.copy(),
        ))

        # ── Policy buffer ─────────────────────────────────────────────────────
        if step >= cfg.warmup_steps and log_prob is not None:
            policy_buf.add(log_prob, curiosity_reward, entropy)
            health.rewards.append(curiosity_reward)
            if entropy_val is not None:
                health.entropy.append(entropy_val)

        # ── Advance recurrent state ───────────────────────────────────────────
        h_t = h_current
        # The queries for the NEXT step's encoding are h_current
        h_query_np = h_current.squeeze(0).cpu().numpy()

        # ── Handle life end / timeout ─────────────────────────────────────────
        ep_len = len(ep_transitions)
        force_flush = (ep_len >= MAX_EP_STEPS) and not life_end

        if life_end or force_flush:
            for frame_i, hq_i, action_i, ht_i in ep_transitions[:-1]:
                latent_buf.add(frame_i, hq_i, action_i, ht_i)

            if force_flush:
                last = ep_transitions[-1]
                ep_transitions = [last]
            else:
                ep_count += 1
                health.episodes_done.append(int(env.level_completed))
                ep_transitions = []
                h_t = None
                h_query_np = None

            frame_np = env.reset() if (life_end and is_terminal) else next_np
        else:
            frame_np = next_np

        step += 1

        # ── JEPA / flow-matching training ─────────────────────────────────────
        if step % cfg.update_freq == 0 and len(latent_buf) >= cfg.min_buffer_size:
            batch = latent_buf.sample(cfg.batch_size, device)
            # batch.frames:    (B, 64, 64) uint8
            # batch.h_queries: (B, n_latents, d_model) — h_{t-1} from rollout
            # batch.actions:   (B,)
            # batch.h_targets: (B, n_latents, d_model) — h_{t+1} from rollout

            enc_opt.zero_grad()
            pred_opt.zero_grad()

            # Re-encode frame_t with stored h_query as Perceiver queries.
            # This matches the rollout encoding path exactly, giving encoder gradient
            # while using the correct recurrent query (not placeholder).
            h_t_fresh, _, _ = encoder(batch.frames, batch.h_queries.detach())
            # h_t_fresh: (B, n_latents, d_model) — has gradient through encoder params

            a_emb = action_embed(batch.actions)

            # Stop gradient on h_target: it is a fixed rollout-computed target.
            # Prevents collapse attractor (encoder can't minimize loss by h_t ≈ h_t1).
            flow_loss, per_lat = predictor.compute_loss(
                h_t_fresh, batch.h_targets.detach(), a_emb
            )

            if not torch.isfinite(flow_loss):
                print(f"[WARNING step {step}] Non-finite flow loss: {flow_loss.item():.4f}")
                enc_opt.zero_grad(); pred_opt.zero_grad()
            else:
                flow_loss.backward()

                gn_s1  = grad_norm(enc_s1_params)
                gn_s2  = grad_norm(enc_s2_params)
                gn_mlps = [grad_norm(list(mlp.parameters())) for mlp in predictor.mlps]
                gn_time = grad_norm(list(predictor.time_embed.parameters()))

                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters())
                    + list(action_embed.parameters()),
                    cfg.grad_clip_model,
                )
                enc_opt.step(); pred_opt.step()

                health.flow_loss_total.append(flow_loss.item())
                for i, v in enumerate(per_lat.tolist()):
                    health.per_latent_loss[i].append(v)
                health.grad_enc_s1.append(gn_s1)
                health.grad_enc_s2.append(gn_s2)
                for i, gn in enumerate(gn_mlps):
                    health.grad_pred_mlps[i].append(gn)
                health.time_emb_grad.append(gn_time)

            # Embedding diversity metrics (placeholder-path std for comparison)
            if (step // cfg.update_freq) % EMBED_METRIC_FREQ == 0:
                with torch.no_grad():
                    mb = latent_buf.sample(min(16, len(latent_buf)), device)
                    # Use placeholder queries for diversity metric (consistent baseline)
                    B_m = mb.frames.shape[0]
                    q_m = encoder.perceiver.get_initial_queries(B_m, device)
                    h_m, _, _ = encoder(mb.frames, q_m)

                    for i in range(cfg.n_latents):
                        health.latent_norms[i].append(h_m[:, i, :].norm(dim=-1).mean().item())
                        health.per_latent_std[i].append(h_m[:, i, :].std().item())

                    pairs = [
                        (h_m[:, i, :] - h_m[:, j, :]).norm(dim=-1).mean().item()
                        for i in range(cfg.n_latents) for j in range(i+1, cfg.n_latents)
                    ]
                    health.latent_pairwise_l2.append(float(np.mean(pairs)))
                    health.latent_eff_rank.append(effective_rank(h_m[0]))
                    health.across_state_std.append(h_m.std(dim=0).mean().item())

                    a_m = action_embed(mb.actions[:1])
                    health.ode_step_cossim.append(
                        compute_ode_step_cossim(predictor, h_m[:1], a_m)
                    )

        # ── Policy training ───────────────────────────────────────────────────
        if step >= cfg.warmup_steps and policy_buf.full():
            log_probs, rewards, entropies = policy_buf.get(device)
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
            print_stats(step, health, act_mon, len(latent_buf), ep_count, fps)
            if metrics_writer is not None:
                metrics_writer.write(step, fps, len(latent_buf), ep_count, health)
            criticals, warnings = health.check()
            for w in warnings:
                print(f"  [WARN] {w}")
            for c in criticals:
                print(f"  [CRITICAL] {c}")
            if criticals:
                print("[exp003] Critical issue — saving checkpoint and stopping")
                save_checkpoint(step, encoder, predictor, action_embed, policy, cfg, "critical")
                break

        # ── Checkpointing ─────────────────────────────────────────────────────
        if step % CHECKPOINT_FREQ == 0 and step > start_step:
            path = save_checkpoint(step, encoder, predictor, action_embed, policy, cfg)
            print(f"  [ckpt] Saved {path.name}")

    path = save_checkpoint(step, encoder, predictor, action_embed, policy, cfg, "final")
    print(f"\n[exp003] Training complete at step {step}. Final checkpoint: {path.name}")
    act_mon.remove()
    if metrics_writer is not None:
        metrics_writer.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    overrides = {}
    if args.max_steps  is not None: overrides["max_steps"]  = args.max_steps
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    _run_dir, _tee = setup_run_logger(args.resume)
    try:
        train(cfg, resume_path=args.resume, run_dir=_run_dir)
    finally:
        sys.stdout = _tee._orig
        _tee.close()


if __name__ == "__main__":
    main()
