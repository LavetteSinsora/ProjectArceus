"""
Exp-006 training loop — Interleaved freezing on top of exp_003_3.

Two predictors:
  - state_predictor:  h_{t+1} ~ flow_match(h_t, a_t).        L_state targets are detached.
  - action_predictor: a_t     ~ MLP(h_t, h_{t+1}).            No detach — gradient flows
    back into the encoder through both endpoints, preventing the h_t ≈ h_{t+1} collapse.

JEPA loss: L = λ_state·L_state + λ_action·L_action  (default 0.5 / 0.5)

Encoder gradient bookkeeping per JEPA step (system card §4.2):
  path 1: L_state  → h_t_fresh
  path 2: L_action → h_t_fresh
  path 3: L_action → h_tp1_fresh
  (h_t_fresh is detached when reused as the recurrent QUERY of the 2nd encode call.)

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_006_interleaved_freezing.train
    uv run python -m JEPA.experiments.exp_006_interleaved_freezing.train --resume <ckpt>
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import io
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_006_interleaved_freezing.config import Config
from JEPA.experiments.exp_006_interleaved_freezing.models import load_models
from JEPA.experiments.exp_006_interleaved_freezing.reward_shaping import is_end_of_life
from JEPA.experiments.exp_006_interleaved_freezing.monitors.health import HealthMonitor
from JEPA.experiments.exp_006_interleaved_freezing.monitors.writer import MetricsWriter
from JEPA.experiments.exp_006_interleaved_freezing.monitors import (
    attention as attn_mod,
    eval_pass as eval_pass_mod,
    gradients as grad_mod,
    predictors as pred_mod,
    representation as repr_mod,
)
from JEPA.shared.buffer import NextFrameLatentBuffer, PolicyBuffer
from JEPA.shared.env_wrapper import LS20Env


# ── Constants ────────────────────────────────────────────────────────────────

CHECKPOINT_FREQ = 5_000
LOG_FREQ        = 200


# ── Freeze scheduler ─────────────────────────────────────────────────────────
#
# Modes:  "joint"        — no freezing (parent behaviour).
#         "interleaved"  — alternates phases of length cfg.freeze_phase_len.
# Phases (only meaningful in "interleaved"):
#         "encoder_frozen"     — predictors + action_embed train.
#         "predictors_frozen"  — encoder trains.
#
# Transitions are event-driven on running means of (L_action, ht_htp1_cossim).
# See system_card.md §3.

class FreezeScheduler:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not cfg.freeze_enabled:
            self.mode = "joint"
            self.phase = "none"
        else:
            self.mode = cfg.freeze_initial_mode
            self.phase = (cfg.freeze_initial_phase
                          if self.mode == "interleaved" else "none")
        self.phase_step = 0
        self.phases_completed_this_block = 0
        self.n_exits = 0
        self.updates_since_last_exit = 10 ** 9  # large so re-entry is allowed
        self._l_action_win: deque = deque(maxlen=cfg.freeze_threshold_window)
        self._cossim_win:   deque = deque(maxlen=cfg.freeze_threshold_window)

    # Inspectors ─────────────────────────────────────────────────────────────
    def freeze_state(self) -> tuple[bool, bool]:
        """Return (encoder_frozen, predictor_side_frozen) for the current step."""
        if self.mode != "interleaved":
            return (False, False)
        if self.phase == "encoder_frozen":
            return (True, False)
        if self.phase == "predictors_frozen":
            return (False, True)
        return (False, False)

    def mode_code(self) -> int:
        return 1 if self.mode == "interleaved" else 0

    def phase_code(self) -> int:
        if self.mode != "interleaved":
            return -1
        return 0 if self.phase == "encoder_frozen" else 1

    def running_means(self) -> tuple[float, float]:
        la = float(np.mean(self._l_action_win)) if self._l_action_win else float("nan")
        cs = float(np.mean(self._cossim_win))   if self._cossim_win   else float("nan")
        return la, cs

    # Advance one JEPA update ────────────────────────────────────────────────
    def step(self, l_action_val: float, ht_htp1_cossim_val: float | None) -> None:
        if not self.cfg.freeze_enabled:
            return

        if np.isfinite(l_action_val):
            self._l_action_win.append(float(l_action_val))
        if ht_htp1_cossim_val is not None and np.isfinite(ht_htp1_cossim_val):
            self._cossim_win.append(float(ht_htp1_cossim_val))

        self.phase_step += 1
        self.updates_since_last_exit += 1

        if self.mode == "interleaved":
            # Phase flip on phase-length boundary
            if self.phase_step >= self.cfg.freeze_phase_len:
                self.phases_completed_this_block += 1
                self.phase = ("predictors_frozen"
                              if self.phase == "encoder_frozen"
                              else "encoder_frozen")
                self.phase_step = 0

            # INTERLEAVED → JOINT exit
            if self.phases_completed_this_block >= self.cfg.freeze_threshold_min_phases:
                la, cs = self.running_means()
                if (np.isfinite(la) and np.isfinite(cs)
                        and la < self.cfg.freeze_l_action_exit
                        and cs < self.cfg.freeze_cossim_exit):
                    self.mode = "joint"
                    self.phase = "none"
                    self.phase_step = 0
                    self.phases_completed_this_block = 0
                    self.n_exits += 1
                    self.updates_since_last_exit = 0
        else:
            # JOINT → INTERLEAVED re-entry on cossim hysteresis
            if self.updates_since_last_exit >= self.cfg.freeze_reentry_cooldown:
                _, cs = self.running_means()
                if np.isfinite(cs) and cs >= self.cfg.freeze_cossim_reentry:
                    self.mode = "interleaved"
                    self.phase = self.cfg.freeze_initial_phase
                    self.phase_step = 0
                    self.phases_completed_this_block = 0

    # Checkpoint helpers ─────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "phase": self.phase,
            "phase_step": self.phase_step,
            "phases_completed_this_block": self.phases_completed_this_block,
            "n_exits": self.n_exits,
            "updates_since_last_exit": self.updates_since_last_exit,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.mode = sd.get("mode", self.mode)
        self.phase = sd.get("phase", self.phase)
        self.phase_step = sd.get("phase_step", 0)
        self.phases_completed_this_block = sd.get("phases_completed_this_block", 0)
        self.n_exits = sd.get("n_exits", 0)
        self.updates_since_last_exit = sd.get("updates_since_last_exit", 10 ** 9)


def _set_requires_grad(modules, flag: bool) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad = flag


# ── Checkpoint schedule ──────────────────────────────────────────────────────

def _should_checkpoint(step: int, schedule: str) -> bool:
    """
    Whether to save a checkpoint at this step.

    schedule:
      "default" — every CHECKPOINT_FREQ=5000 steps (production setting).
      "staged"  — diagnostic test schedule:
          step <  20_000        : no checkpoints
          20_000 ≤ step ≤ 25_000: every 1_000 steps
          step >  25_000        : every 200 steps
        Used when you want fine-grained checkpoints over a particular window
        for offline analysis. Does not affect the default training run.
    """
    if schedule == "staged":
        if step < 20_000:
            return False
        if step <= 25_000:
            return step % 1_000 == 0
        return step % 200 == 0
    return step % CHECKPOINT_FREQ == 0


# ── Tee logger ───────────────────────────────────────────────────────────────

class _Tee(io.TextIOBase):
    def __init__(self, original: io.TextIOBase, log_path: Path):
        self._orig = original
        self._file = open(log_path, "w", buffering=1, encoding="utf-8")

    def write(self, data: str) -> int:
        self._orig.write(data); self._orig.flush()
        self._file.write(data)
        return len(data)

    def flush(self):
        self._orig.flush(); self._file.flush()

    def close(self):
        super().close()
        self._file.close()


def setup_run_logger(resume_path: str | None,
                     run_label: str | None = None) -> tuple[Path, _Tee]:
    """If run_label is given, use it verbatim as the run-dir name (no timestamp
    prefix). Otherwise fall back to the default 'run_<ts>_<fresh|resume>' name."""
    runs_root = Path(__file__).parent / "runs"
    if run_label:
        run_dir = runs_root / run_label
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = "resume" if resume_path else "fresh"
        run_dir = runs_root / f"run_{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    tee = _Tee(sys.stdout, log_path)
    sys.stdout = tee
    print(f"[exp006] Run dir:  {run_dir}")
    print(f"[exp006] Log file: {log_path}")
    return run_dir, tee


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_checkpoint(step, encoder, state_predictor, action_predictor,
                    action_embed, policy, cfg, label="", ckpt_dir: Path | None = None,
                    freeze_state: dict | None = None):
    if ckpt_dir is None:
        ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"step_{step:06d}" + (f"_{label}" if label else "")
    path = ckpt_dir / f"{tag}.pt"
    tmp = path.with_suffix(".tmp")
    payload = {
        "encoder":          encoder.state_dict(),
        "state_predictor":  state_predictor.state_dict(),
        "action_predictor": action_predictor.state_dict(),
        "action_embed":     action_embed.state_dict(),
        "policy":           policy.state_dict(),
        "step":             step,
        "config":           dataclasses.asdict(cfg),
    }
    if freeze_state is not None:
        payload["freeze_state"] = freeze_state
    torch.save(payload, tmp)
    tmp.rename(path)
    return path


def load_checkpoint(path, encoder, state_predictor, action_predictor,
                    action_embed, policy, device, scheduler: "FreezeScheduler | None" = None):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    state_predictor.load_state_dict(ckpt["state_predictor"])
    action_predictor.load_state_dict(ckpt["action_predictor"])
    action_embed.load_state_dict(ckpt["action_embed"])
    policy.load_state_dict(ckpt["policy"])
    if scheduler is not None and "freeze_state" in ckpt:
        scheduler.load_state_dict(ckpt["freeze_state"])
    return ckpt.get("step", 0)


# ── Activation dead-frac monitor (lifted from exp_003_0) ─────────────────────

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


# ── Print stats ──────────────────────────────────────────────────────────────

def print_stats(step, health: HealthMonitor, act_mon: ActivationMonitor,
                buf_size: int, ep_count: int, fps: float, lambda_state: float,
                lambda_action: float):
    Ls = health.mean_L_state(); La = health.mean_L_action(); Lt = health.mean_L_total()
    per_lat = [health._mean(q) for q in health.sec6_per_latent_state_loss]
    ent = health.mean_entropy()
    a_ent = health._mean(health.sec4["action_pred_entropy"])
    cls_ce = [health._mean(q) for q in health.sec6_action_ce_per_class]
    cls_n  = [health._mean(q) for q in health.sec6_action_count_per_class]
    print(
        f"\n{'─'*78}\n"
        f"  Step {step:7d}  fps={fps:.0f}  buf={buf_size}  ep={ep_count}\n"
        f"  Loss: L_total={Lt:.5f}  = {lambda_state}·L_state({Ls:.5f}) + "
        f"{lambda_action}·L_action({La:.5f})\n"
        f"    L_state per latent: {' | '.join(f'L{i}={v:.4f}' for i, v in enumerate(per_lat))}\n"
        f"    L_action per class: "
        f"{' | '.join(f'a={a} n={int(n) if np.isfinite(n) else 0} ce={c:.4f}' for a, (n, c) in enumerate(zip(cls_n, cls_ce)))}\n"
        f"  Reward: mean={health.mean_reward():.4f}  "
        f"state={health._mean(health.sec6['reward_state_component']):.4f}  "
        f"action={health._mean(health.sec6['reward_action_component']):.4f}\n"
        f"  Policy: H={ent:.3f}/{health.policy_entropy_max:.3f}  "
        f"completion={health.completion_rate():.1%}\n"
        f"  ActionPred H={a_ent:.3f}/{health.action_pred_entropy_max:.3f}  "
        f"ht_htp1_cos={health._mean(health.sec1['ht_htp1_cossim_rollout']):.4f}  "
        f"placeholder_pcos={health._mean(health.sec1['placeholder_pairwise_cossim']):.4f}\n"
        f"  Grad totals: "
        f"patch_sa={health._mean(health.sec7.get('gnorm_patch_sa_total', deque())):.3f}  "
        f"perc_cross_r0={health._mean(health.sec7.get('gnorm_perc_cross_r0_total', deque())):.3f}  "
        f"perc_self_r0={health._mean(health.sec7.get('gnorm_perc_self_r0_total', deque())):.3f}  "
        f"action_pred={health._mean(health.sec7.get('gnorm_action_pred_total', deque())):.3f}\n"
        f"{'─'*78}"
    )
    dead_parts = [f"{n}={act_mon.mean_dead(n):.2f}" for n in act_mon._buf]
    if dead_parts:
        print(f"  Dead GELU: {' | '.join(dead_parts)}")


# ── Main training loop ───────────────────────────────────────────────────────

def train(cfg: Config, resume_path: str = None, run_dir: Path = None,
          checkpoint_schedule: str = "default",
          checkpoint_dir: Path | None = None,
          wandb_run=None) -> None:
    set_seeds(cfg.seed)
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[exp006] device={device}  max_steps={cfg.max_steps}")
    print(f"[exp006] lambdas: state={cfg.lambda_state}  action={cfg.lambda_action}")
    print(f"[exp006] reward weights: state={cfg.reward_w_state}  action={cfg.reward_w_action}  "
          f"clamp={cfg.reward_clamp}")
    print(f"[exp006] checkpoint_schedule={checkpoint_schedule}")
    if checkpoint_dir is not None:
        print(f"[exp006] checkpoint_dir={checkpoint_dir}")

    encoder, state_predictor, action_predictor, action_embed, policy, baseline = \
        load_models(cfg, device)

    scheduler = FreezeScheduler(cfg)
    print(f"[exp006] freeze_enabled={cfg.freeze_enabled}  "
          f"initial mode={scheduler.mode}  phase={scheduler.phase}  "
          f"phase_len={cfg.freeze_phase_len}")

    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(
            resume_path, encoder, state_predictor, action_predictor,
            action_embed, policy, device, scheduler=scheduler,
        )
        print(f"[exp006] Resumed from step {start_step}  "
              f"freeze mode={scheduler.mode} phase={scheduler.phase} "
              f"phase_step={scheduler.phase_step} n_exits={scheduler.n_exits}")

    # Placeholder init snapshot (metric 1.1b)
    if run_dir is not None:
        ph_init_path = run_dir / "placeholder_init.pt"
        if not ph_init_path.exists():
            torch.save(encoder.perceiver.placeholders.detach().cpu().clone(), ph_init_path)
            print(f"[exp006] Cached placeholder init → {ph_init_path}")
        placeholder_init = torch.load(ph_init_path, map_location=device, weights_only=True)
    else:
        placeholder_init = encoder.perceiver.placeholders.detach().clone()

    # ── Activation monitors ─────────────────────────────────────────────────
    act_mon = ActivationMonitor()
    for i, block in enumerate(encoder.sa_blocks):
        act_mon.register(block.ffn.net[1], f"enc_sa{i}_ffn")
    act_mon.register(encoder.perceiver.rounds[0].cross_attn.ffn.net[1], "perc_r0_cross_ffn")
    act_mon.register(encoder.perceiver.rounds[0].self_attn.ffn.net[1],  "perc_r0_self_ffn")
    if cfg.n_perceiver_rounds > 1:
        act_mon.register(encoder.perceiver.rounds[1].cross_attn.ffn.net[1], "perc_r1_cross_ffn")
        act_mon.register(encoder.perceiver.rounds[1].self_attn.ffn.net[1],  "perc_r1_self_ffn")
    for i, mlp in enumerate(state_predictor.mlps):
        act_mon.register(mlp.net[1], f"state_pred_mlp{i}")
    act_mon.register(action_predictor.net[1], "action_pred_ffn")
    act_mon.register(policy.net[1], "policy_ffn")

    # ── Per-component optimizers (system card §4.3) ─────────────────────────
    enc_s1_params = (
        list(encoder.color_embed.parameters())
        + list(encoder.patch_proj.parameters())
        + list(encoder.sa_blocks.parameters())
        + list(encoder.sa_norm.parameters())
    )
    enc_s2_params = list(encoder.perceiver.parameters())

    enc_opt = torch.optim.AdamW([
        {"params": enc_s1_params, "lr": cfg.sa_lr,        "weight_decay": cfg.encoder_wd},
        {"params": enc_s2_params, "lr": cfg.perceiver_lr, "weight_decay": cfg.encoder_wd},
    ])
    state_pred_opt = torch.optim.AdamW(
        list(state_predictor.parameters()) + list(action_embed.parameters()),
        lr=cfg.state_predictor_lr, weight_decay=cfg.state_predictor_wd,
    )
    action_pred_opt = torch.optim.AdamW(
        action_predictor.parameters(),
        lr=cfg.action_predictor_lr, weight_decay=cfg.action_predictor_wd,
    )
    pol_opt = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)

    # Sub-block parameter partition (used by gnorm totals + decomposition + UWR)
    sub_blocks = grad_mod.build_sub_block_params(
        encoder, state_predictor, action_predictor, action_embed, policy
    )
    uwr_snap = grad_mod.UWRSnapshot(sub_blocks)

    # ── Buffers ──────────────────────────────────────────────────────────────
    latent_buf = NextFrameLatentBuffer(
        n_latents=cfg.n_latents, d_model=cfg.d_model, capacity=cfg.buffer_size,
    )
    policy_buf = PolicyBuffer(cfg.policy_update_freq)

    # ── Environment ──────────────────────────────────────────────────────────
    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)

    metrics_writer = MetricsWriter(run_dir, wandb_run=wandb_run) if run_dir else None
    health = HealthMonitor(
        window_fast=LOG_FREQ, window_slow=50, window_eval=5,
        n_latents=cfg.n_latents, n_actions=cfg.n_actions,
        n_perceiver_rounds=cfg.n_perceiver_rounds,
    )

    # ── State ────────────────────────────────────────────────────────────────
    frame_np = env.reset()
    ep_count = 0
    step = start_step
    t0 = time.time()
    jepa_updates = 0

    h_t: torch.Tensor | None = None
    h_query_np: np.ndarray | None = None
    ep_transitions: list = []

    # Step-0 init checkpoint — captures the freshly initialised weights so we
    # can inspect the *untrained* transformer behaviour in the dashboard.
    # Only emitted on fresh runs (start_step == 0); resumes skip it.
    if start_step == 0:
        init_path = save_checkpoint(
            0, encoder, state_predictor, action_predictor,
            action_embed, policy, cfg, "init",
            ckpt_dir=checkpoint_dir,
            freeze_state=scheduler.state_dict(),
        )
        print(f"[exp006] Saved init checkpoint: {init_path.name}")

    print(f"\n[exp006] Training started  warmup={cfg.warmup_steps}  seed={cfg.seed}\n")

    while step < cfg.max_steps:
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

        # ── Encode current frame (rollout, no grad) ─────────────────────────
        with torch.no_grad():
            if h_t is None:
                queries = encoder.perceiver.get_initial_queries(1, device)
                h_query_np = queries.squeeze(0).cpu().numpy()
            else:
                queries = h_t.detach()
            h_current, _, _ = encoder(frame_t, queries)

        # ── Select action ───────────────────────────────────────────────────
        avail = env.available_actions
        if step < cfg.warmup_steps:
            action_idx = int(np.random.randint(0, cfg.n_actions))
            log_prob = entropy = None
            entropy_val = None
        else:
            action_idx, log_prob, entropy = policy.act(h_current.squeeze(0), avail)
            entropy_val = entropy.item()

        # ── Step environment ────────────────────────────────────────────────
        next_np, is_terminal = env.step(action_idx)
        life_end = is_end_of_life(frame_np, next_np, is_terminal)

        # ── Dual-path curiosity reward (system card §5.2) ───────────────────
        with torch.no_grad():
            next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)
            h_next, _, _ = encoder(next_t, h_current.detach())
            a_emb_r = action_embed(torch.tensor([action_idx], device=device))
            h_pred = state_predictor.predict(h_current, a_emb_r)         # full N-step ODE
            state_err  = (h_next - h_pred).pow(2).mean().item()
            action_logits_r = action_predictor(h_current, h_next)
            action_err = F.cross_entropy(
                action_logits_r,
                torch.tensor([action_idx], device=device),
            ).item()
            raw = (cfg.reward_w_state * state_err
                   + cfg.reward_w_action * action_err)
            curiosity_reward = (min(raw, cfg.reward_clamp)
                                if np.isfinite(raw) else 0.0)
            ht_htp1_cs = repr_mod.ht_htp1_cossim(
                h_current.squeeze(0), h_next.squeeze(0)
            )

        health.sec1["ht_htp1_cossim_rollout"].append(ht_htp1_cs)
        health.sec6["reward_state_component"].append(state_err)
        health.sec6["reward_action_component"].append(action_err)

        # ── Store transition (raw next_frame, not h_target) ─────────────────
        ep_transitions.append((
            frame_np.copy(),
            h_query_np.copy(),
            action_idx,
            next_np.copy(),
        ))

        # ── Policy buffer ───────────────────────────────────────────────────
        if step >= cfg.warmup_steps and log_prob is not None:
            policy_buf.add(log_prob, curiosity_reward, entropy)
            health.sec6["reward_total"].append(curiosity_reward)
            if entropy_val is not None:
                health.sec5["policy_entropy"].append(entropy_val)
                # Normalized by ln(|available|)
                hmax = float(np.log(max(len(avail), 1)))
                if hmax > 0:
                    health.sec5["policy_entropy_normalized"].append(
                        entropy_val / hmax
                    )

        # ── Advance recurrent state ─────────────────────────────────────────
        h_t = h_current
        h_query_np = h_current.squeeze(0).cpu().numpy()

        # ── End-of-life flush (no MAX_EP_STEPS guard — energy-bounded) ──────
        if life_end:
            for frame_i, hq_i, action_i, next_i in ep_transitions[:-1]:
                latent_buf.add(frame_i, hq_i, action_i, next_i)
            ep_count += 1
            health.episodes_done.append(int(env.level_completed))
            ep_transitions = []
            h_t = None
            h_query_np = None
            frame_np = env.reset() if is_terminal else next_np
        else:
            frame_np = next_np

        step += 1

        # ── JEPA / dual-predictor training (every update_freq env steps) ────
        if step % cfg.update_freq == 0 and len(latent_buf) >= cfg.min_buffer_size:
            batch = latent_buf.sample(cfg.batch_size, device)
            jepa_updates += 1

            # Freeze schedule — toggle requires_grad on the two sides.
            enc_frozen, pred_frozen = scheduler.freeze_state()
            _set_requires_grad([encoder], not enc_frozen)
            _set_requires_grad(
                [state_predictor, action_predictor, action_embed],
                not pred_frozen,
            )

            # Always zero_grad both sides so dashboard gnorm dips to 0 for the
            # frozen side (system card §6 #8). Weight-decay drift protection
            # comes only from skipping the frozen optimiser's .step() below.
            enc_opt.zero_grad()
            state_pred_opt.zero_grad()
            action_pred_opt.zero_grad()

            # 1. Re-encode h_t fresh. If encoder is frozen, skip grad to save compute.
            if enc_frozen:
                with torch.no_grad():
                    h_t_fresh, _, _ = encoder(batch.frames, batch.h_queries.detach())
                    h_tp1_fresh, _, _ = encoder(batch.next_frames, h_t_fresh.detach())
            else:
                h_t_fresh, _, _ = encoder(batch.frames, batch.h_queries.detach())
                # 2. Re-encode h_{t+1} fresh — detach query so the 2nd forward does
                #    not back-prop into the 1st via the recurrent-query path.
                h_tp1_fresh, _, _ = encoder(batch.next_frames, h_t_fresh.detach())
            a_emb = action_embed(batch.actions)

            # 3. State predictor loss — target detached at the call site
            L_state, per_lat_state = state_predictor.compute_loss(
                h_t_fresh, h_tp1_fresh.detach(), a_emb
            )

            # 4. Action predictor loss — NO detach on either side
            action_logits = action_predictor(h_t_fresh, h_tp1_fresh)
            ce_per_sample = F.cross_entropy(
                action_logits, batch.actions, reduction="none"
            )
            L_action = ce_per_sample.mean()

            # 5. Combined JEPA loss
            L = cfg.lambda_state * L_state + cfg.lambda_action * L_action

            if not torch.isfinite(L):
                print(f"[WARNING step {step}] Non-finite JEPA loss: {L.item():.4f}")
            else:
                # UWR pre-step snapshot
                do_uwr = (jepa_updates % cfg.uwr_freq == 0)
                if do_uwr:
                    uwr_snap.snapshot()

                # Only call backward if at least one side has trainable params.
                # (Both frozen would be a misconfigured scheduler — shouldn't happen.)
                if L.requires_grad:
                    L.backward()

                # Pre-clip per-sub-block totals (sec7 [total] for every key).
                # Frozen sub-blocks surface as 0 gnorm — informative, not an error.
                for sub_name, params in sub_blocks.items():
                    health.push_sec7(
                        f"gnorm_{sub_name}_total",
                        grad_mod.grad_norm(params),
                    )

                # Clip + step on the active side(s) only.
                active_params: list = []
                if not enc_frozen:
                    active_params += list(encoder.parameters())
                if not pred_frozen:
                    active_params += (
                        list(state_predictor.parameters())
                        + list(action_predictor.parameters())
                        + list(action_embed.parameters())
                    )
                if active_params:
                    nn.utils.clip_grad_norm_(active_params, cfg.grad_clip_model)
                if not enc_frozen:
                    enc_opt.step()
                if not pred_frozen:
                    state_pred_opt.step()
                    action_pred_opt.step()

                # Streaming losses + per-class CE
                health.sec6["L_state"].append(L_state.item())
                health.sec6["L_action"].append(L_action.item())
                health.sec6["L_total"].append(L.item())
                for i, v in enumerate(per_lat_state.tolist()):
                    health.sec6_per_latent_state_loss[i].append(v)
                cls = pred_mod.action_pred_ce_per_class(
                    ce_per_sample.detach(), batch.actions, cfg.n_actions
                )
                for a in range(cfg.n_actions):
                    mean_ce, count = cls[a]
                    if np.isfinite(mean_ce):
                        health.sec6_action_ce_per_class[a].append(mean_ce)
                    health.sec6_action_count_per_class[a].append(float(count))

                # Action predictor entropy + input cossim (on training batch)
                with torch.no_grad():
                    health.sec4["action_pred_entropy"].append(
                        pred_mod.action_pred_entropy_from_logits(action_logits)
                    )
                    health.sec4["action_pred_input_cossim"].append(
                        repr_mod.ht_htp1_cossim(h_t_fresh, h_tp1_fresh)
                    )

                # UWR post-step
                if do_uwr:
                    for sub_name, ratio in uwr_snap.ratios().items():
                        health.push_sec7(f"uwr_{sub_name}", ratio)

            # ── Cheap embedding probes (every capture_attn_freq) ────────────
            if jepa_updates % cfg.capture_attn_freq == 0:
                with torch.no_grad():
                    mb = latent_buf.sample(min(16, len(latent_buf)), device)
                    B_m = mb.frames.shape[0]
                    q_m = encoder.perceiver.get_initial_queries(B_m, device)

                    # Eval-mode forward → populates _debug_attn
                    probe = attn_mod.probe_attention(encoder, mb.frames, q_m)

                    # Re-encode under no_grad (training mode again) to get H_m
                    h_m, _, _ = encoder(mb.frames, q_m)

                    # sec1 latent stats
                    norms = repr_mod.latent_norms(h_m)
                    for i, n_ in enumerate(norms):
                        health.sec1_latent_norms[i].append(n_)
                    health.sec1["latent_pairwise_cossim_buf"].append(
                        repr_mod.latent_pairwise_cossim(h_m)
                    )
                    health.sec1["latent_pairwise_l2_buf"].append(
                        repr_mod.latent_pairwise_l2(h_m)
                    )
                    health.sec1["latent_eff_rank"].append(
                        repr_mod.effective_rank(h_m[0])
                    )

                    # Placeholder probes
                    health.sec1["placeholder_pairwise_cossim"].append(
                        repr_mod.placeholder_pairwise_cossim(
                            encoder.perceiver.placeholders
                        )
                    )
                    drift = repr_mod.placeholder_drift_from_init(
                        encoder.perceiver.placeholders, placeholder_init
                    )
                    for i, d in enumerate(drift):
                        if i < len(health.sec1_per_placeholder):
                            health.sec1_per_placeholder[i].append(d)
                    if drift:
                        health.sec1["placeholder_drift_from_init_mean"].append(
                            float(np.mean(drift))
                        )

                    # sec2.1 and sec3.1 — JSD over freshly captured attention
                    health.sec2["patch_sa_row_jsd"].append(
                        attn_mod.patch_sa_row_jsd(probe["sa_blocks"])
                    )
                    perc_jsd = attn_mod.latent_self_attn_row_jsd_per_round(
                        probe["perc_self"]
                    )
                    for r_idx, v in enumerate(perc_jsd):
                        if r_idx < len(health.sec3_latent_self_attn_row_jsd) and np.isfinite(v):
                            health.sec3_latent_self_attn_row_jsd[r_idx].append(v)

                    # sec4 — ODE diagnostics (re-encode → h_m has no grad here)
                    a_m = action_embed(mb.actions[:8])
                    health.sec4["ode_step_cossim"].append(
                        pred_mod.ode_step_cossim(state_predictor, h_m[:8], a_m)
                    )
                    health.sec4["ode_first_vs_final_cossim"].append(
                        pred_mod.ode_first_vs_final_cossim(
                            state_predictor, h_m[:8], a_m
                        )
                    )
                    health.sec4["predictor_velocity_norm"].append(
                        pred_mod.predictor_velocity_norm(
                            state_predictor, h_m[:8], a_m
                        )
                    )

            # ── Gradient-source decomposition (every grad_decomp_freq) ──────
            if jepa_updates % cfg.grad_decomp_freq == 0:
                try:
                    mb = latent_buf.sample(cfg.batch_size, device)
                    decomp = grad_mod.compute_source_decomposition(
                        encoder, state_predictor, action_predictor, action_embed,
                        sub_blocks,
                        mb.frames, mb.h_queries, mb.next_frames, mb.actions,
                    )
                    for k, v in decomp.items():
                        if np.isfinite(v):
                            health.push_sec7(k, v)
                except Exception as e:
                    print(f"[exp006] grad decomposition failed at step {step}: {e}")

            # ── Advance freeze scheduler (uses L_action + running cossim) ──
            l_action_val = L_action.item() if torch.isfinite(L_action) else float("nan")
            cossim_q = health.sec1["ht_htp1_cossim_rollout"]
            running_cs = float(np.mean(cossim_q)) if len(cossim_q) else None
            prev_mode, prev_phase = scheduler.mode, scheduler.phase
            scheduler.step(l_action_val, running_cs)
            if scheduler.mode != prev_mode or scheduler.phase != prev_phase:
                print(f"  [freeze] step={step} jepa_upd={jepa_updates} "
                      f"{prev_mode}/{prev_phase} → {scheduler.mode}/{scheduler.phase} "
                      f"(n_exits={scheduler.n_exits})")

        # ── Policy training ─────────────────────────────────────────────────
        if step >= cfg.warmup_steps and policy_buf.full():
            log_probs, rewards, entropies = policy_buf.get(device)
            adv = rewards - baseline.value
            adv = adv / (adv.std().clamp(min=1e-8) + 1e-8)
            baseline.update(rewards.mean().item())

            pol_loss = (-(adv * log_probs).mean()
                        - cfg.policy_entropy_lambda * entropies.mean())
            pol_opt.zero_grad()
            pol_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_policy)
            pol_opt.step()
            policy_buf.clear()

            health.sec6["pol_loss"].append(pol_loss.item())
            health.push_sec7(
                "gnorm_policy_total",
                grad_mod.grad_norm(policy.parameters()),
            )

        # ── Eval pass (every eval_freq env steps) ───────────────────────────
        if step > start_step and step % cfg.eval_freq == 0:
            try:
                eval_d = eval_pass_mod.run_eval(
                    encoder, state_predictor, action_predictor,
                    action_embed, policy, env, cfg, step, device,
                )
                # Fold values into health (eval-tagged deques)
                for k, v in eval_d.items():
                    if not isinstance(v, (int, float)) or not np.isfinite(v):
                        continue
                    section, name = k.split("/", 1)
                    if section == "sec1" and name in health.sec1:
                        health.sec1[name].append(v)
                    elif section == "sec2" and name in health.sec2:
                        health.sec2[name].append(v)
                    elif section == "sec3":
                        # latent_self_attn_row_jsd_r{i}
                        if name.startswith("latent_self_attn_row_jsd_r"):
                            idx = int(name.split("_r")[-1])
                            if idx < len(health.sec3_latent_self_attn_row_jsd):
                                health.sec3_latent_self_attn_row_jsd[idx].append(v)
                    elif section == "sec4" and name in health.sec4:
                        health.sec4[name].append(v)
                    elif section == "sec6" and name in health.sec6:
                        health.sec6[name].append(v)
                if metrics_writer is not None:
                    metrics_writer.write_eval(step, eval_d)
                # Reset env after eval so training continues from a clean episode
                frame_np = env.reset()
                h_t = None
                h_query_np = None
                ep_transitions = []
            except Exception as e:
                print(f"[exp006] eval pass failed at step {step}: {e}")

        # ── Logging ─────────────────────────────────────────────────────────
        if step % LOG_FREQ == 0 and step > 0:
            fps = step / (time.time() - t0 + 1e-6)
            print_stats(step, health, act_mon, len(latent_buf), ep_count, fps,
                        cfg.lambda_state, cfg.lambda_action)
            if metrics_writer is not None:
                la_mean, cs_mean = scheduler.running_means()
                sec8 = {
                    "sec8/freeze_mode":                 scheduler.mode_code(),
                    "sec8/freeze_phase":                scheduler.phase_code(),
                    "sec8/phase_step":                  scheduler.phase_step,
                    "sec8/phases_completed_this_block": scheduler.phases_completed_this_block,
                    "sec8/n_exits":                     scheduler.n_exits,
                    "sec8/l_action_running_mean":       la_mean,
                    "sec8/cossim_running_mean":         cs_mean,
                }
                metrics_writer.write(step, fps, len(latent_buf), ep_count, health,
                                     extra=sec8)
            criticals, warnings = health.check()
            for w in warnings:
                print(f"  [WARN] {w}")
            for c in criticals:
                print(f"  [CRITICAL] {c}")
            if criticals:
                print("[exp006] Critical issue — saving checkpoint and stopping")
                save_checkpoint(step, encoder, state_predictor, action_predictor,
                                action_embed, policy, cfg, "critical",
                                ckpt_dir=checkpoint_dir,
                                freeze_state=scheduler.state_dict())
                break

        # ── Checkpointing ───────────────────────────────────────────────────
        if step > start_step and _should_checkpoint(step, checkpoint_schedule):
            path = save_checkpoint(step, encoder, state_predictor, action_predictor,
                                   action_embed, policy, cfg,
                                   ckpt_dir=checkpoint_dir,
                                   freeze_state=scheduler.state_dict())
            print(f"  [ckpt] Saved {path.name}")

    path = save_checkpoint(step, encoder, state_predictor, action_predictor,
                           action_embed, policy, cfg, "final",
                           ckpt_dir=checkpoint_dir,
                           freeze_state=scheduler.state_dict())
    print(f"\n[exp006] Training complete at step {step}. Final checkpoint: {path.name}")
    act_mon.remove()
    if metrics_writer is not None:
        metrics_writer.close()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--max-steps",  type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--checkpoint-schedule",
        choices=["default", "staged"],
        default="default",
        help=("'default' = every 5000 steps. "
              "'staged' = none before 20k, every 1000 in [20k, 25k], "
              "every 200 after 25k (diagnostic — does not affect production runs)."),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=("Directory to write checkpoints into. Defaults to "
              "exp_006_interleaved_freezing/checkpoints/. Use this to keep "
              "diagnostic runs (e.g. --checkpoint-schedule staged) isolated "
              "from the main checkpoints directory."),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Mirror metrics.jsonl to Weights & Biases (project/entity/mode from Config).",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help=("Resume an existing W&B run by id. Required to continue logging "
              "into the same dashboard run across a --resume restart."),
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help=("Custom display name for the W&B run (e.g. 'ablation-no-action-reward'). "
              "Defaults to the local run-dir name (run_<timestamp>_<fresh|resume>)."),
    )
    parser.add_argument(
        "--freeze-phase-len",
        type=int,
        default=None,
        help=("Override Config.freeze_phase_len (JEPA updates per freeze phase). "
              "Default config value is 500."),
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help=("Use this string verbatim as the run-directory name instead of "
              "the default 'run_<timestamp>_<fresh|resume>'. Useful for naming "
              "labeled variant runs (e.g. '006_0_interleave=50000')."),
    )
    args = parser.parse_args()

    cfg = Config()
    overrides = {}
    if args.max_steps        is not None: overrides["max_steps"]        = args.max_steps
    if args.batch_size       is not None: overrides["batch_size"]       = args.batch_size
    if args.freeze_phase_len is not None: overrides["freeze_phase_len"] = args.freeze_phase_len
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    ckpt_dir_path = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None

    _run_dir, _tee = setup_run_logger(args.resume, run_label=args.run_label)

    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            mode=cfg.wandb_mode,
            name=args.wandb_name or _run_dir.name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            config=dataclasses.asdict(cfg),
            dir=str(_run_dir),
            settings=wandb.Settings(console="off"),
        )
        (_run_dir / "wandb_run_id.txt").write_text(wandb_run.id)
        if args.resume and not args.wandb_run_id:
            print("[exp006] --resume given without --wandb-run-id; "
                  "starting a NEW wandb run instead of continuing the old one.")

    try:
        train(cfg, resume_path=args.resume, run_dir=_run_dir,
              checkpoint_schedule=args.checkpoint_schedule,
              checkpoint_dir=ckpt_dir_path,
              wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as e:
                print(f"[exp006] wandb.finish() failed: {e}")
        sys.stdout = _tee._orig
        _tee.close()


if __name__ == "__main__":
    main()
