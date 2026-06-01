"""Config for exp_013_1 — RND+ICM ("OCC"). See SYSTEM_CARD.md.

Single intrinsic value head only: there is NO extrinsic value function because the
first extrinsic reward ENDS the run (stop-on-first-reward), so an extrinsic critic
would never be trained on anything useful. The env's +1 is used solely as the stop
signal / headline metric, never as a reward fed to GAE.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

GAME_N_ACTIONS = {"ls20": 4, "tu93": 4, "re86": 5, "g50t": 5}


@dataclass
class Config:
    # Identity
    game: str = "ls20"
    level_index: int = 0
    seed: int = 0
    exp_dir: str = "JEPA/experiments/exp_013_sparse_exploration/exp_013_1_rnd_icm"

    # Env
    n_envs: int = 16
    max_episode_steps: int = 200
    n_actions: int | None = None        # filled from the env wrapper at build time
    n_colors: int = 16
    frame_size: int = 64
    trunk_dim: int = 256

    # Rollout / budget / stop rule
    rollout_steps: int = 128            # 128*16 = 2048 transitions / update
    max_env_steps: int = 250_000
    stop_on_first_reward: bool = True

    # PPO (single head, episodic GAE — the exp_010 recipe)
    gamma: float = 0.95              # intrinsic horizon shortened 0.99→0.95 to curb the
                                     # non-episodic return INFLATION (V drifted 0.3→8) that
                                     # drove the value-lag / phantom-advantage entropy collapse.
    gae_lambda: float = 0.95
    intrinsic_episodic: bool = False    # False = canonical-RND NON-episodic intrinsic
                                        # value (bootstraps across death/reset); True =
                                        # PPO-style episodic (resets at death).
    clip_eps: float = 0.2
    vf_clip_eps: float = 0.2
    c_value: float = 1.0             # raised 0.5→1.0 = "faster value" (the value-lr knob under
                                     # a shared actor-critic optimiser) to reduce the value lag.
    c_entropy: float = 0.10    # 0.01→0.05→0.10: at 0.01 the actor collapsed to a
                               # deterministic loop (entropy→0, worse-than-random); 0.05
                               # flipped the worst ls20-L1 seed (occ_power_limits.md); raised
                               # to 0.10 because long frontier-length runs still collapsed
                               # onto a frozen-φ degenerate signal (run_diagnosis.md).
    grad_clip: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4

    # ICM (learns the controllable feature space φ via inverse+forward dynamics)
    beta: float = 0.2                   # (1-β)L_inverse + β L_forward
    icm_lr: float = 1e-3
    icm_hidden: int = 256
    icm_epochs: int = 1

    # φ-freeze: freeze the ICM encoder once it separates states (inverse_acc high),
    # giving RND a STATIONARY ruler. Adaptive trigger + a hard fallback.
    phi_freeze_inverse_acc: float = 0.70  # lowered 0.90→0.70: held-out inv_acc maxes ~0.72-0.76
                                          # on ls20/g50t, so 0.90 was unreachable and freeze always
                                          # fell to the update-100 fallback (often on a chance-level
                                          # φ). 0.70 lets it fire ADAPTIVELY near φ's peak (run_diagnosis.md).
    phi_freeze_patience: int = 3        # consecutive updates ≥ threshold
    phi_freeze_max_updates: int = 100   # freeze by here regardless (~200k env steps)
    # Which inverse_acc the freeze trigger reads. "holdout" = a FIXED uniform-random
    # held-out transition set (φ's TRUE controllability); "onpolicy" = the current
    # rollout. On-policy is INFLATED by a narrowing policy and fools the freeze
    # (see probes/inv_acc_causality.md) → default to held-out.
    freeze_metric: str = "holdout"      # "holdout" | "onpolicy"
    holdout_size: int = 2000            # # transitions in the held-out inv_acc set
    # NEVER freeze φ while held-out inv_acc < this × chance (chance = 1/n_actions). Freezing a
    # chance-level φ (re86: holdout ~0.22 vs 0.20 chance) gives a degenerate RND ruler → entropy→0.
    # Below this we keep training φ (no stationary-but-degenerate ruler). See probes/method_improvements.md.
    phi_uncontrollable_factor: float = 1.5

    # φ source for the RND ruler:
    #   "icm"    = ICM inverse-dynamics features (learned, controllable; the default).
    #   "frozen" = a fixed RANDOM encoder (no ICM training/freeze) — "plain RND+leak".
    phi_mode: str = "icm"
    # Optional: initialise the ICM φ-encoder from a saved checkpoint's "icm" state
    # (e.g. an L1-trained run) → cross-level representation TRANSFER. None = random init.
    init_phi_ckpt: str | None = None

    # Timer/observation-confound fix (Option A). The env feeds a marching step-timer in
    # the frame, so every frame is fake-unique (1073 vs 43 TRUE board states) and the
    # novelty signal is anti-informative (probes/frontier_analysis.md, signal_redundancy.md).
    # Mask these rows in the φ/novelty path ONLY (the policy's separate encoder is untouched)
    # so novelty is computed on the true board. Patched onto φ.encode in the trainer.
    mask_timer: bool = True
    timer_mask_rows: tuple = (60, 63)        # inclusive row range zeroed before φ

    # RND-on-φ + leak
    rnd_feature_dim: int = 256
    rnd_hidden: int = 256
    rnd_lr: float = 1e-4
    rnd_epochs: int = 1
    leak: float = 0.05                  # μ: predictor shrink-to-init per update (forget rate). raised
                                        # 0.01→0.05: at 0.01 the novelty floor saturates on long runs
                                        # (exp_014_1 + the 999k frozen_re86 flat floor); 0.05 holds a
                                        # live, μ-dependent floor so the signal isn't spent by ~80k steps.

    # Intrinsic-reward normalisation (warm-up + EMA std of returns, no centring)
    int_norm_decay: float = 0.99
    norm_warmup_updates: int = 2
    int_norm_eps: float = 1e-8
    # Floor-aware normalization: if mean raw novelty < this, the field is "dead" (flat) → skip the
    # std-divide (which amplifies noise as int_return_std shrinks → slow entropy bleed); emit the tiny
    # raw signal instead, so a dead field yields ~0 reward, not amplified noise. (probes/method_improvements.md)
    novelty_dead_eps: float = 0.01
    # Raw-novelty clip (robustness): cap per-step raw novelty at reward_clip_k ×
    # running-mean-raw BEFORE the normalizer, so a transient spike poisons neither
    # ret_std nor the reward. φ-scale-robust (relative to running mean). None = off.
    reward_clip_k: float | None = 5.0

    # Logging / eval / checkpoint cadence (in updates; 0 disables)
    log_every: int = 1
    eval_every: int = 0
    eval_episodes: int = 16
    save_every: int = 0

    @property
    def exp_name(self) -> str:
        tag = self.phi_mode + ("_xfer" if self.init_phi_ckpt else "")
        return f"exp013_1_rndicm_{tag}_{self.game}_L{self.level_index + 1}_seed{self.seed}"

    @property
    def total_updates(self) -> int:
        return self.max_env_steps // (self.rollout_steps * self.n_envs)

    def smoke(self) -> "Config":
        return dataclasses.replace(
            self,
            n_envs=2, rollout_steps=16, max_episode_steps=40,
            max_env_steps=16 * 2 * 8, minibatches=2, epochs=1,
            norm_warmup_updates=1, phi_freeze_max_updates=2, phi_freeze_patience=1,
            stop_on_first_reward=False, eval_every=0, save_every=0, holdout_size=64,
        )
