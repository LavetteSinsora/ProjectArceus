"""
JEPA World Model + Policy for LS20 Level 1  —  enhanced training script.

Trains until level-1 completion rate >= 30 % over a 20-episode window,
or until a CRITICAL health event forces a stop-and-diagnose.

Monitors per training step:
  - JEPA loss (total, MSE component, variance-reg component)
  - Gradient norms (encoder/predictor group, policy group)
  - Gradient clip trigger rate (how often grad_norm > clip threshold)
  - Embedding statistics: std, mean-norm, max, NaN presence
  - Dead-activation rate in GELU layers (encoder FFN + predictor MLP)
  - Policy action entropy (exploration health)
  - Reward statistics: mean, std, min/max over rolling window
  - EMA momentum current value
  - Level completion tracking

Stopping conditions:
  CRITICAL (stop + diagnose):
    - NaN in loss OR model parameters
    - Embedding std < 0.02 for 10 consecutive checks  (collapse)
    - JEPA grad norm > 200 (exploding)
  WARNING (log prominently, keep going):
    - Embedding std 0.02–0.05
    - Policy entropy < 0.3
    - Dead activation rate > 60 %
  SUCCESS:
    - Completion rate >= 30 % over last 20 episodes

Checkpoints saved every CHECKPOINT_FREQ steps into experiments/exp_001_vit_jepa_baseline/checkpoints/.

Run:
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.train
    cd "Code Repo" && uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.train --batch-size 128
"""

import argparse
import copy
import dataclasses
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root (Code Repo/) is on sys.path so JEPA package is importable
_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_001_vit_jepa_baseline.config import Config
from JEPA.experiments.exp_001_vit_jepa_baseline.models.encoder import Encoder
from JEPA.experiments.exp_001_vit_jepa_baseline.models.predictor import Predictor
from JEPA.experiments.exp_001_vit_jepa_baseline.models.policy import PolicyNetwork
from JEPA.shared.action_embed import ActionEmbedding
from JEPA.shared.ema import update_ema, ema_momentum
from JEPA.shared.buffer import ReplayBuffer, PolicyBuffer
from JEPA.shared.env_wrapper import LS20Env
from JEPA.experiments.exp_001_vit_jepa_baseline.reward_shaping import compute_reward, PLAYER_START_Y

# ── Constants ────────────────────────────────────────────────────────────────
CHECKPOINT_FREQ = 5_000        # save a checkpoint every N steps
LOG_FREQ        = 200          # print stats every N steps
SUCCESS_RATE    = 0.30         # stop when 30-episode rolling completion rate >= this
SUCCESS_WINDOW  = 20           # rolling window for completion rate

COLLAPSE_PAIRWISE_CRITICAL = 0.05   # within-image pairwise L2 below this for 10 checks → CRITICAL
COLLAPSE_PAIRWISE_WARN     = 0.15   # within-image pairwise L2 below this → WARNING
EFFECTIVE_RANK_WARN        = 2.0    # effective rank below this → WARNING
ACROSS_STD_WARN            = 0.05   # across-state std below this → WARNING
DEAD_ACT_WARN              = 0.60   # GELU dead-frac above this → WARNING
ENTROPY_WARN               = 0.30   # policy action entropy below this → WARNING
GRAD_NORM_CRITICAL         = 200.0  # JEPA grad norm above this → CRITICAL


# ── Gradient norm ─────────────────────────────────────────────────────────────

def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


# ── Activation monitor (GELU hook) ───────────────────────────────────────────

class ActivationMonitor:
    """
    Registers forward hooks on nn.GELU modules to track dead-activation rate.

    A "dead" activation is one with |output| < 0.01.  If most pre-activations
    are strongly negative, the GELU saturates at ≈0, behaving like a dead ReLU.
    """

    def __init__(self):
        self._hooks: list = []
        self._buf: dict[str, deque] = {}

    def register(self, module: nn.Module, name: str, window: int = 50) -> None:
        self._buf[name] = deque(maxlen=window)

        def _hook(_mod, _inp, out):
            dead = (out.detach().abs() < 0.01).float().mean().item()
            self._buf[name].append(dead)

        self._hooks.append(module.register_forward_hook(_hook))

    def mean_dead(self, name: str) -> float:
        buf = self._buf.get(name, [])
        return float(np.mean(buf)) if buf else float("nan")

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Health monitor ────────────────────────────────────────────────────────────

class HealthMonitor:
    """Tracks rolling-window training metrics and flags issues."""

    def __init__(self, window: int = 200):
        self.w = window

        self.jepa_total   = deque(maxlen=window)
        self.jepa_mse     = deque(maxlen=window)
        self.jepa_var_reg = deque(maxlen=window)
        self.grad_jepa    = deque(maxlen=window)
        self.grad_policy  = deque(maxlen=window)
        self.clip_jepa    = deque(maxlen=window)   # 1 if clipped, 0 if not
        self.clip_policy  = deque(maxlen=window)

        self.pairwise     = deque(maxlen=50)   # within-image avg pairwise L2
        self.eff_rank     = deque(maxlen=50)   # average effective rank per image
        self.across_std   = deque(maxlen=50)   # across-state per-dim std

        self.pol_loss     = deque(maxlen=window)
        self.entropy      = deque(maxlen=window)
        self.rewards      = deque(maxlen=window)

        self.episodes_done: list[bool] = []   # True = episode ended with level complete
        self.ep_exploration = deque(maxlen=SUCCESS_WINDOW)  # pct of 64 cells visited per ep
        self._consecutive_low_pairwise = 0

    # ── record methods ──────────────────────────────────────────────────────

    def record_jepa(self, total: float, mse: float, var_reg: float,
                    gnorm: float, clipped: bool) -> None:
        self.jepa_total.append(total)
        self.jepa_mse.append(mse)
        self.jepa_var_reg.append(var_reg)
        self.grad_jepa.append(gnorm)
        self.clip_jepa.append(1.0 if clipped else 0.0)

    def record_emb(self, pairwise: float, eff_rank: float, across_std: float) -> None:
        self.pairwise.append(pairwise)
        self.eff_rank.append(eff_rank)
        self.across_std.append(across_std)
        if pairwise < COLLAPSE_PAIRWISE_CRITICAL:
            self._consecutive_low_pairwise += 1
        else:
            self._consecutive_low_pairwise = 0

    def record_policy(self, loss: float, gnorm: float, clipped: bool,
                      entropy: float) -> None:
        self.pol_loss.append(loss)
        self.grad_policy.append(gnorm)
        self.clip_policy.append(1.0 if clipped else 0.0)
        self.entropy.append(entropy)

    def record_reward(self, r: float) -> None:
        self.rewards.append(r)

    def record_episode(self, completed: bool) -> None:
        self.episodes_done.append(completed)

    def record_exploration(self, pct: float) -> None:
        self.ep_exploration.append(pct)

    # ── derived metrics ─────────────────────────────────────────────────────

    @staticmethod
    def _mean(d) -> float:
        return float(np.mean(d)) if d else float("nan")

    def mean_jepa(self)     -> float: return self._mean(self.jepa_total)
    def mean_grad_j(self)   -> float: return self._mean(self.grad_jepa)
    def mean_pairwise(self) -> float: return self._mean(self.pairwise)
    def mean_eff_rank(self) -> float: return self._mean(self.eff_rank)
    def mean_across_std(self) -> float: return self._mean(self.across_std)
    def mean_entropy(self)      -> float: return self._mean(self.entropy)
    def mean_reward(self)       -> float: return self._mean(self.rewards)
    def mean_exploration(self)  -> float: return self._mean(self.ep_exploration)
    def std_reward(self)    -> float:
        return float(np.std(list(self.rewards))) if len(self.rewards) > 1 else 0.0

    def completion_rate(self, window: int = SUCCESS_WINDOW) -> float:
        recent = self.episodes_done[-window:]
        return float(np.mean(recent)) if recent else 0.0

    def jepa_converged(self, threshold: float = 0.15, window: int = 500) -> bool:
        if len(self.jepa_total) < window:
            return False
        recent = list(self.jepa_total)[-window:]
        return float(np.mean(recent)) < threshold and float(np.std(recent)) < 0.05

    # ── health check ────────────────────────────────────────────────────────

    def check(self, step: int) -> tuple[list[str], list[str]]:
        """
        Returns (critical_issues, warnings).
        critical_issues → stop training immediately.
        warnings        → log prominently, continue training.
        """
        criticals, warnings = [], []

        # NaN in loss
        if self.jepa_total and np.isnan(self.jepa_total[-1]):
            criticals.append("NaN in JEPA loss")

        # Within-image patch collapse
        if self._consecutive_low_pairwise >= 10:
            criticals.append(
                f"Patch collapse: pairwise={self.pairwise[-1]:.4f} < {COLLAPSE_PAIRWISE_CRITICAL} "
                f"for 10 consecutive checks"
            )
        elif self.pairwise and self.pairwise[-1] < COLLAPSE_PAIRWISE_WARN:
            warnings.append(
                f"Low patch diversity: pairwise={self.pairwise[-1]:.4f} (warn < {COLLAPSE_PAIRWISE_WARN})"
            )

        # Low effective rank
        if self.eff_rank and self.eff_rank[-1] < EFFECTIVE_RANK_WARN:
            warnings.append(
                f"Low effective rank: {self.eff_rank[-1]:.2f} (warn < {EFFECTIVE_RANK_WARN})"
            )

        # Low across-state diversity
        if self.across_std and self.across_std[-1] < ACROSS_STD_WARN:
            warnings.append(
                f"Low across-state std: {self.across_std[-1]:.4f} (warn < {ACROSS_STD_WARN})"
            )

        # Exploding gradients
        if self.grad_jepa and self.grad_jepa[-1] > GRAD_NORM_CRITICAL:
            criticals.append(
                f"Exploding JEPA gradients: grad_norm={self.grad_jepa[-1]:.1f} > {GRAD_NORM_CRITICAL}"
            )

        # Policy entropy collapse
        if len(self.entropy) >= 5:
            recent_ent = float(np.mean(list(self.entropy)[-5:]))
            if recent_ent < ENTROPY_WARN:
                warnings.append(f"Policy entropy low: H={recent_ent:.3f} (min {ENTROPY_WARN})")

        # Reward variance: if std is near 0 for 50+ steps, reward signal may be broken
        if len(self.rewards) >= 50 and self.std_reward() < 1e-4:
            warnings.append(
                f"Reward variance degenerate: std={self.std_reward():.6f} "
                "— reward computation may be broken"
            )

        return criticals, warnings


# ── Loss functions ────────────────────────────────────────────────────────────

def compute_patch_weights(
    frames: torch.Tensor, next_frames: torch.Tensor
) -> torch.Tensor:
    """(B,64,64) uint8 → (B,16) weights in [0,1], normalised within each image."""
    diff = (next_frames.float() - frames.float()).abs()
    pw = diff.unfold(1, 16, 16).unfold(2, 16, 16).mean(dim=(-2, -1)).view(frames.shape[0], 16)
    max_pw = pw.amax(dim=1, keepdim=True).clamp(min=1e-8)
    return pw / max_pw


def compute_patch_weights_1d(frame_np: np.ndarray, next_np: np.ndarray) -> np.ndarray:
    """Single pair of (64,64) uint8 numpy frames → (16,) float32 weights in [0,1]."""
    diff = np.abs(next_np.astype(np.float32) - frame_np.astype(np.float32))
    pw = diff.reshape(4, 16, 4, 16).mean(axis=(1, 3)).flatten()  # (16,)
    max_pw = float(pw.max())
    if max_pw < 1e-8:
        return np.zeros(16, dtype=np.float32)
    return pw / max_pw


def detect_moved_cell(frame_np: np.ndarray, next_np: np.ndarray) -> int | None:
    """
    Return the 8×8 fine-grid cell index (0–63) with the most pixel change,
    masking the step-counter rows (61-62) which change every step.
    Returns None if no significant change detected.
    """
    diff = np.abs(next_np.astype(np.float32) - frame_np.astype(np.float32))
    diff[61:63, :] = 0
    cell_diff = diff.reshape(8, 8, 8, 8).sum(axis=(1, 3))   # (8, 8) — sum over 8×8 pixels
    if cell_diff.max() < 2.0:
        return None
    r, c = np.unravel_index(cell_diff.argmax(), cell_diff.shape)
    return int(r * 8 + c)


def jepa_loss_components(
    predicted: torch.Tensor,
    z_target: torch.Tensor,
    weights: torch.Tensor,
    z_online: torch.Tensor,
    variance_reg_lambda: float,
) -> tuple[torch.Tensor, float, float]:
    """
    Returns (total_loss, mse_component_float, var_reg_component_float).

    weights: (B,16) in [0.1, 1.0] — floored patch weights (changed patches weighted more).

    Variance regularisation: one-sided floor at std=0.02.
    With L2-normalised outputs, unit vectors in 128D have per-component std ≈ 0.088.
    The old two-sided (std-1)^2 target of 1.0 is unreachable for unit vectors and fires
    as a large constant penalty.  One-sided relu(0.02 - std) is a collapse safety net only.
    """
    diff_sq = (predicted - z_target).pow(2).sum(dim=-1)   # (B,16) L2-squared
    mse     = (weights * diff_sq).mean()
    std_z   = z_online.std(dim=0).mean()
    var_reg = F.relu(0.02 - std_z)   # fires only if collapse begins
    total   = mse + variance_reg_lambda * var_reg
    return total, mse.item(), (variance_reg_lambda * var_reg).item()


# ── Diagnosis ─────────────────────────────────────────────────────────────────

def diagnose(
    step: int,
    issue: str,
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    policy: nn.Module,
    replay_buf: ReplayBuffer,
    device: torch.device,
    act_mon: ActivationMonitor,
    health: HealthMonitor,
) -> None:
    print("\n" + "=" * 70)
    print(f"  DIAGNOSIS at step {step}")
    print(f"  Issue: {issue}")
    print("=" * 70)

    # ── Parameter stats ──────────────────────────────────────────────────────
    print("\n  Parameter statistics:")
    for name, mod in [("encoder", encoder), ("predictor", predictor),
                      ("policy", policy), ("target_encoder", target_encoder)]:
        all_params = torch.cat([p.detach().flatten() for p in mod.parameters()])
        nan_count = torch.isnan(all_params).sum().item()
        inf_count = torch.isinf(all_params).sum().item()
        print(
            f"    {name:15s}  params={len(all_params):7,}  "
            f"mean={all_params.mean().item():+.4f}  "
            f"std={all_params.std().item():.4f}  "
            f"max={all_params.abs().max().item():.3f}  "
            f"NaN={nan_count}  Inf={inf_count}"
        )

    # ── Embedding sample stats ────────────────────────────────────────────────
    if len(replay_buf) >= 64:
        print("\n  Embedding sample (64 random frames):")
        with torch.no_grad():
            sample = replay_buf.sample(64, device)
            z = encoder(sample.frames)            # (64, 16, 128)
            zt = target_encoder(sample.frames)    # (64, 16, 128)
        for label, zz in [("encoder", z), ("target", zt)]:
            print(
                f"    {label:8s}  std={zz.std().item():.4f}  "
                f"mean={zz.mean().item():+.4f}  "
                f"max={zz.abs().max().item():.3f}  "
                f"NaN={torch.isnan(zz).any().item()}"
            )
        print(f"    encoder–target cosine sim (mean): "
              f"{F.cosine_similarity(z.flatten(1), zt.flatten(1)).mean().item():.4f}")

    # ── Gradient stats (if available) ─────────────────────────────────────────
    print("\n  Gradient norms (last recorded):")
    print(f"    JEPA: {health.mean_grad_j():.4f}  (rolling mean)")
    if health.grad_policy:
        print(f"    Policy: {health._mean(health.grad_policy):.4f}")

    # ── Activation dead-neuron rates ──────────────────────────────────────────
    print("\n  Dead-activation rates (GELU layers, recent mean):")
    for name in act_mon._buf:
        print(f"    {name:30s}: {act_mon.mean_dead(name):.3f}")

    # ── Rolling metric summary ─────────────────────────────────────────────────
    print("\n  Rolling metric summary:")
    print(f"    JEPA loss (mean):   {health.mean_jepa():.4f}")
    print(f"    Patch pairwise L2:  {health.mean_pairwise():.4f}")
    print(f"    Effective rank:     {health.mean_eff_rank():.2f}")
    print(f"    Across-state std:   {health.mean_across_std():.4f}")
    print(f"    Policy entropy:     {health.mean_entropy():.4f}")
    print(f"    Reward mean/std:    {health.mean_reward():.4f} / {health.std_reward():.4f}")
    print(f"    Completion rate:    {health.completion_rate():.3f}")

    # ── Likely causes & remediation ───────────────────────────────────────────
    print("\n  Likely causes & suggested remediation:")
    low = issue.lower()
    if "collapse" in low or "emb_std" in low:
        print("    → Embedding collapse. Options:")
        print("      1. Increase variance_reg_lambda (try 0.05 or 0.1)")
        print("      2. Increase EMA start momentum (e.g. 0.998)")
        print("      3. Reduce JEPA learning rate (try 1e-4)")
    if "nan" in low:
        print("    → NaN detected. Options:")
        print("      1. Reduce learning rate (jepa_lr=1e-4, policy_lr=5e-5)")
        print("      2. Reduce gradient clipping threshold (grad_clip_jepa=0.5)")
        print("      3. Check for extreme pixel values or buffer corruption")
    if "explod" in low or "grad_norm" in low:
        print("    → Exploding gradients. Options:")
        print("      1. Reduce grad_clip_jepa (currently 1.0 → try 0.3)")
        print("      2. Reduce JEPA learning rate")
    if "entropy" in low:
        print("    → Policy entropy collapsed (deterministic policy). Options:")
        print("      1. Add entropy regularisation to policy loss")
        print("      2. Reduce policy learning rate (slower convergence = more exploration)")
    print("=" * 70 + "\n")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(step: int, encoder, target_encoder, predictor, action_embed,
                    policy, cfg, label: str = "") -> Path:
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    tag = f"step_{step:06d}" + (f"_{label}" if label else "")
    path = ckpt_dir / f"{tag}.pt"
    torch.save({
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "action_embed": action_embed.state_dict(),
        "policy": policy.state_dict(),
        "config": dataclasses.asdict(cfg),   # plain dict — no pickle module-path dependency
        "experiment": "exp_001_vit_jepa_baseline",
        "step": step,
    }, path)
    return path


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_stats(step: int, phase: str, health: HealthMonitor,
                act_mon: ActivationMonitor, ema_m: float,
                buf_size: int, ep_count: int, fps: float) -> None:
    c_rate = health.completion_rate(SUCCESS_WINDOW)
    print(
        f"\n{'─'*70}\n"
        f"  Step {step:7d}  [{phase}]  fps={fps:.0f}  buf={buf_size}  "
        f"ep={ep_count}  ema_m={ema_m:.5f}\n"
        f"  JEPA: total={health.mean_jepa():.4f}  "
        f"mse={health._mean(health.jepa_mse):.4f}  "
        f"var_reg={health._mean(health.jepa_var_reg):.4f}  "
        f"gnorm={health.mean_grad_j():.3f}  "
        f"clip%={health._mean(health.clip_jepa)*100:.0f}%\n"
        f"  Embed: pairwise={health.mean_pairwise():.4f}  "
        f"eff_rank={health.mean_eff_rank():.2f}  "
        f"across_std={health.mean_across_std():.4f}\n"
        f"  Policy: loss={health._mean(health.pol_loss):.5f}  "
        f"gnorm={health._mean(health.grad_policy):.3f}  "
        f"clip%={health._mean(health.clip_policy)*100:.0f}%  "
        f"H={health.mean_entropy():.3f}\n"
        f"  Reward: mean={health.mean_reward():.4f}  "
        f"std={health.std_reward():.4f}  "
        f"min={min(health.rewards, default=0):.4f}  "
        f"max={max(health.rewards, default=0):.4f}\n"
        f"  Completion: {c_rate:.1%} over last {SUCCESS_WINDOW} ep  |  "
        f"Grid explored: {health.mean_exploration():.1%} of 64 cells (rolling ep mean)"
    )
    # Dead activation rates (compact one-liner)
    dead_parts = [f"{n}={act_mon.mean_dead(n):.2f}" for n in act_mon._buf]
    if dead_parts:
        print(f"  Dead GELU: {' | '.join(dead_parts)}")
    print(f"{'─'*70}")


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: Config, resume_path: str = None) -> None:
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"[JEPA] device={device}  target=completion_rate>={SUCCESS_RATE:.0%}")
    print(f"[JEPA] warmup={cfg.warmup_steps}  jepa_lr={cfg.jepa_lr}  "
          f"policy_lr={cfg.policy_lr}  batch={cfg.batch_size}")

    # ── Models ────────────────────────────────────────────────────────────────
    encoder = Encoder(
        cfg.d_model, cfg.d_color, cfg.n_heads, cfg.n_blocks, cfg.ffn_dim, cfg.patch_size
    ).to(device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    predictor   = Predictor(cfg.d_model, cfg.d_action).to(device)
    action_emb  = ActionEmbedding(cfg.n_actions, cfg.d_action).to(device)
    policy      = PolicyNetwork(
        cfg.d_model, cfg.n_actions, attn_gain_init=cfg.policy_attn_gain_init
    ).to(device)

    # ── Optional resume from checkpoint ──────────────────────────────────────
    resume_step = 0
    if resume_path is not None:
        resume_step = _load_resume(
            resume_path, encoder, target_encoder, predictor, action_emb, policy, device
        )

    # ── Activation hooks ──────────────────────────────────────────────────────
    act_mon = ActivationMonitor()
    # Encoder: each block has ffn[1] = GELU
    for i, block in enumerate(encoder.blocks):
        act_mon.register(block.ffn[1], f"enc_block{i}_ffn")
    # Predictor: mlp[1] = GELU
    act_mon.register(predictor.mlp[1], "predictor_mlp")
    # Policy FFN: ffn[1] = GELU
    act_mon.register(policy.ffn[1], "policy_ffn")

    # ── JEPA parameter group (encoder trainable — L2 norm prevents scale drift) ─
    jepa_params = (
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(action_emb.parameters())
    )

    # ── Optimisers ────────────────────────────────────────────────────────────
    jepa_opt   = torch.optim.AdamW(jepa_params, lr=cfg.jepa_lr,
                                   weight_decay=cfg.jepa_weight_decay)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)

    # ── Buffers ───────────────────────────────────────────────────────────────
    replay_buf = ReplayBuffer(cfg.buffer_size, cfg.recency_fraction, cfg.recent_buffer_size)
    policy_buf = PolicyBuffer(cfg.policy_update_freq)

    # ── Environment ───────────────────────────────────────────────────────────
    from arc_agi import Arcade, OperationMode
    repo_root = Path(__file__).parent.parent.parent.parent  # exp_001 → experiments → JEPA → Code Repo
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)

    # ── State ─────────────────────────────────────────────────────────────────
    health = HealthMonitor(window=LOG_FREQ)
    frame_np         = env.reset()
    h                = policy.initial_state().to(device)
    step             = resume_step   # picks up from checkpoint step if --resume used
    ep_count         = 0
    ep_visited_cells: set = set()    # fine-grid cells visited in the current episode
    t0               = time.time()
    stop_reason: str = ""

    print(f"\n[JEPA] Training started — will run until completion_rate >= "
          f"{SUCCESS_RATE:.0%} over last {SUCCESS_WINDOW} episodes\n")

    while True:
        # ── Collect one transition ─────────────────────────────────────────
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

        with torch.no_grad():
            z_t = encoder(frame_t).squeeze(0)

        avail = env.available_actions

        if step < cfg.warmup_steps:
            action_idx = int(np.random.randint(0, cfg.n_actions))
            log_prob   = None
            with torch.no_grad():
                h = policy._cross_attn_update(h, z_t)
        else:
            action_idx, log_prob, h, entropy = policy.act(h.detach(), z_t, avail)

        next_np, is_terminal = env.step(action_idx)
        next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)

        # ── Grid exploration tracking ──────────────────────────────────────
        cell = detect_moved_cell(frame_np, next_np)
        if cell is not None:
            ep_visited_cells.add(cell)

        # ── Reward signal for POLICY: intrinsic curiosity via JEPA predictor error ──
        # Reward = weighted MSE between predictor's prediction and target encoder's
        # actual next-state embedding.  Uses the same formula as the JEPA training
        # loss (floored patch weights), so the policy is directly rewarded for being
        # in states that maximise the JEPA objective.  As JEPA trains down the error
        # on visited transitions, the reward there decays, pushing the policy to
        # explore novel states.
        with torch.no_grad():
            a_idx_t      = torch.tensor([action_idx], device=device)
            a_emb_single = action_emb(a_idx_t)                         # (1, d_action)
            z_pred_r     = predictor(z_t.unsqueeze(0), a_emb_single)   # (1, 16, d_model)
            z_tgt_r      = target_encoder(next_t)                      # (1, 16, d_model)

            pw_np    = compute_patch_weights_1d(frame_np, next_np)     # (16,) in [0,1]
            _W_FLOOR = 0.1
            w_r      = torch.from_numpy(
                (_W_FLOOR + (1.0 - _W_FLOOR) * pw_np).astype(np.float32)
            ).to(device)                                               # (16,) in [0.1,1.0]

            err_sq_r = (z_pred_r - z_tgt_r).pow(2).sum(dim=-1).squeeze(0)  # (16,)
            r_intr   = (w_r * err_sq_r).mean().item()

        completion_bonus = 50.0 if (is_terminal and env.level_completed) else 0.0
        reward           = r_intr + completion_bonus

        health.record_reward(reward)   # policy reward (binary + completion bonus)

        if not is_terminal:
            replay_buf.add(frame_np, action_idx, next_np)
            if log_prob is not None:
                policy_buf.add(log_prob, reward, entropy)

        if is_terminal:
            level_done = env.level_completed
            health.record_episode(level_done)
            health.record_exploration(len(ep_visited_cells) / 64.0)
            ep_visited_cells = set()
            ep_count += 1
            frame_np = env.reset()
            h        = policy.initial_state().to(device)
        else:
            frame_np = next_np

        step += 1

        # ── JEPA update ────────────────────────────────────────────────────
        if step % cfg.update_freq == 0 and len(replay_buf) >= cfg.min_buffer_size:
            batch   = replay_buf.sample(cfg.batch_size, device)
            z_on    = encoder(batch.frames)
            with torch.no_grad():
                z_tgt = target_encoder(batch.next_frames)
            a_emb   = action_emb(batch.actions)
            pred    = predictor(z_on, a_emb)
            pixel_w = compute_patch_weights(batch.frames, batch.next_frames)  # (B,16) [0,1]
            _W_FLOOR = 0.1
            w        = _W_FLOOR + (1.0 - _W_FLOOR) * pixel_w                 # (B,16) [0.1,1]
            loss, mse_val, var_val = jepa_loss_components(
                pred, z_tgt, w, z_on, cfg.variance_reg_lambda
            )

            jepa_opt.zero_grad()
            loss.backward()
            pre_clip = grad_norm(jepa_params)
            clipped  = pre_clip > cfg.grad_clip_jepa
            nn.utils.clip_grad_norm_(jepa_params, cfg.grad_clip_jepa)
            jepa_opt.step()

            m = ema_momentum(step, max(step + 1, 50_000), cfg.ema_start, cfg.ema_end)
            update_ema(encoder, target_encoder, m)

            health.record_jepa(loss.item(), mse_val, var_val, pre_clip, clipped)

            # Embedding collapse metrics (every 5 JEPA updates to reduce overhead)
            if (step // cfg.update_freq) % 5 == 0:
                with torch.no_grad():
                    z_check = z_on.detach()
                # 1. Within-image avg pairwise L2
                diff_pw  = z_check.unsqueeze(2) - z_check.unsqueeze(1)  # (B,16,16,d)
                pw_dist  = diff_pw.norm(dim=-1)                          # (B,16,16)
                pairwise_val = (pw_dist.sum(dim=(1, 2)) / (16 * 15)).mean().item()
                # 2. Effective rank (per image, averaged)
                z_cpu  = z_check.cpu()   # svdvals not on MPS; CPU fallback
                ranks  = []
                for b in range(z_cpu.shape[0]):
                    S = torch.linalg.svdvals(z_cpu[b])        # (16,)
                    p = S / S.sum().clamp(min=1e-9)
                    ranks.append(
                        torch.exp(-(p * torch.log(p + 1e-9)).sum()).item()
                    )
                eff_rank_val = sum(ranks) / len(ranks)
                # 3. Across-state diversity
                mean_per_frame = z_check.mean(dim=1)           # (B, d_model)
                across_std_val = mean_per_frame.std(dim=0).mean().item()
                health.record_emb(pairwise_val, eff_rank_val, across_std_val)

        # ── Policy update ──────────────────────────────────────────────────
        if step >= cfg.warmup_steps and policy_buf.full():
            lp_t, rew_t, ent_t = policy_buf.get(device)
            # Normalised advantages with minimum std to prevent zero-gradient
            adv = rew_t - rew_t.mean()
            adv = adv / (adv.std().clamp(min=0.1))
            # Policy gradient loss + entropy regularisation bonus.
            # −lambda * H(π): maximise entropy (= minimise −entropy) → exploration.
            # When entropy drops too low the policy becomes deterministic, repeatedly
            # hitting the same walls and preventing level completion.
            pol_loss = -(lp_t * adv.detach()).mean() - cfg.policy_entropy_lambda * ent_t.mean()

            policy_opt.zero_grad()
            pol_loss.backward()
            pg_pre   = grad_norm(policy.parameters())
            pg_clip  = pg_pre > cfg.grad_clip_policy
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_policy)
            policy_opt.step()

            # Actual H(π) mean over the batch (stored in policy_buf alongside log_probs)
            health.record_policy(pol_loss.item(), pg_pre, pg_clip, ent_t.mean().item())
            policy_buf.clear()

        # ── Logging ────────────────────────────────────────────────────────
        if step % LOG_FREQ == 0:
            fps   = step / max(time.time() - t0, 1e-9)
            phase = "warmup " if step < cfg.warmup_steps else "training"
            ema_m = ema_momentum(step, max(step + 1, 50_000), cfg.ema_start, cfg.ema_end)
            print_stats(step, phase, health, act_mon, ema_m,
                        len(replay_buf), ep_count, fps)

            # Health check
            crits, warns = health.check(step)
            for w in warns:
                print(f"  ⚠  WARNING: {w}")
            if crits:
                for c in crits:
                    print(f"  ✗  CRITICAL: {c}")
                stop_reason = crits[0]
                break

        # ── Checkpoint ─────────────────────────────────────────────────────
        if step % CHECKPOINT_FREQ == 0 and step > 0:
            p = save_checkpoint(step, encoder, target_encoder, predictor,
                                action_emb, policy, cfg)
            print(f"\n[JEPA] ✓ Checkpoint saved → {p.name}")

        # ── Success check ──────────────────────────────────────────────────
        if ep_count >= SUCCESS_WINDOW:
            rate = health.completion_rate(SUCCESS_WINDOW)
            if rate >= SUCCESS_RATE:
                stop_reason = f"SUCCESS — completion_rate={rate:.1%} >= {SUCCESS_RATE:.0%}"
                break

    # ── End of training ─────────────────────────────────────────────────────
    final_path = save_checkpoint(
        step, encoder, target_encoder, predictor, action_emb, policy, cfg,
        label="final"
    )
    print(f"\n[JEPA] Training stopped at step {step}: {stop_reason}")
    print(f"[JEPA] Final checkpoint → {final_path.name}")

    if "CRITICAL" in stop_reason or stop_reason.startswith(
        tuple(["NaN", "Embedding", "Explod"])
    ):
        print("[JEPA] Running post-mortem diagnosis…")
        diagnose(step, stop_reason, encoder, target_encoder, predictor,
                 policy, replay_buf, device, act_mon, health)

    act_mon.remove()
    return stop_reason


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_resume(path: str, encoder, target_encoder, predictor, action_embed,
                 policy, device) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    target_encoder.load_state_dict(ckpt["target_encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    action_embed.load_state_dict(ckpt["action_embed"])
    # strict=False so legacy checkpoints (no attn_gain) keep the constructor's
    # value from cfg.policy_attn_gain_init — i.e. the fix is applied on resume.
    policy.load_state_dict(ckpt["policy"], strict=False)
    step = ckpt.get("step", 0)
    print(f"[JEPA] Resumed from {Path(path).name} at step {step}")
    return step


def _parse_args():
    p = argparse.ArgumentParser(
        description="JEPA world model + policy for LS20 Level 1"
    )
    p.add_argument("--warmup-steps",  type=int,   default=None)
    p.add_argument("--jepa-lr",       type=float, default=None)
    p.add_argument("--policy-lr",     type=float, default=None)
    p.add_argument("--batch-size",    type=int,   default=None)
    p.add_argument("--var-reg",       type=float, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (e.g. checkpoints/step_005000.pt)")
    args = p.parse_args()

    kw: dict = {}
    if args.warmup_steps is not None: kw["warmup_steps"]        = args.warmup_steps
    if args.jepa_lr      is not None: kw["jepa_lr"]             = args.jepa_lr
    if args.policy_lr    is not None: kw["policy_lr"]           = args.policy_lr
    if args.batch_size   is not None: kw["batch_size"]          = args.batch_size
    if args.var_reg      is not None: kw["variance_reg_lambda"] = args.var_reg
    return Config(**kw), args.resume


if __name__ == "__main__":
    cfg, resume_path = _parse_args()
    train(cfg, resume_path=resume_path)
