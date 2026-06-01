"""Shared config for exp_011. Subclasses the exp_010 base (so the entire PPO /
env / model recipe is inherited verbatim) and adds the ICM-specific knobs.

Layout convention is identical to exp_010: checkpoints flat under
`<exp_dir>/checkpoints/step_*.pt`, metrics under
`<exp_dir>/runs/<run>/metrics.jsonl` — the layout the main JEPA dashboard
(port 8787) reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    # Identity
    exp_name: str = "exp_011_base"
    exp_dir: str = "JEPA/experiments/exp_011_ls20_icm"

    # Which LS20 level to drop the agent into, 0-indexed (0 = L1, 1 = L2, ...).
    # 0 reproduces the exp_010 single-level setup exactly.
    level_index: int = 0

    # ── ICM (Pathak et al. 2017) ────────────────────────────────────────────
    # The PPO fields (learning_rate=3e-4, gamma, gae_lambda, clip_eps, c_value,
    # c_entropy, epochs, minibatches, ...) are inherited UNCHANGED from exp_010.

    beta: float = 0.2           # (1-beta)*L_inverse + beta*L_forward      (paper)
    icm_lr: float = 1e-3        # ICM's own optimiser lr                   (paper)
    icm_epochs: int = 1         # passes over each rollout for the ICM update
    icm_hidden: int = 256       # FC width of the inverse/forward heads     (paper)

    # Intrinsic reward r^i = (eta/2)||phi_hat - phi(s')||^2.
    #   eta = None  -> auto-calibrate on the first rollout so the mean per-step
    #                  intrinsic reward equals `intrinsic_target` (then frozen).
    #   eta = float -> use that fixed value (no calibration).
    eta: float | None = None
    intrinsic_target: float = 0.01

    # ── Early stopping: PLATEAU of eval success rate ────────────────────────
    # Stop once eval `success_rate` stops improving — covers both "solved and
    # saturated high" and "stalled and going nowhere". Distinct from exp_010's
    # base stop (sustained >= a fixed threshold), which we override here.
    #   * an eval "improves" if success_rate > best_so_far + early_stop_min_delta
    #   * stop after `early_stop_patience` consecutive non-improving evals
    #   * but never before `early_stop_warmup_evals` evals have happened, so an
    #     exploration run gets a fair shot at finding the first reward before the
    #     all-zero success rate is read as a plateau.
    early_stop_enabled: bool = True
    early_stop_min_delta: float = 0.01
    early_stop_patience: int = 8       # ~8 evals * 50 updates * 1024 = ~400k steps
    early_stop_warmup_evals: int = 8   # ~400k env steps of grace before it can fire
