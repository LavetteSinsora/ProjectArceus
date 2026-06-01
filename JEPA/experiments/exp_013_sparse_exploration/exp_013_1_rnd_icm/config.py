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
    c_entropy: float = 0.05    # raised from 0.01: at 0.01 the actor collapsed to a
                               # deterministic loop on some seeds (entropy→0) and went
                               # worse-than-random; 0.05 flipped the worst seed
                               # censored→solved (see probes/occ_power_limits.md).
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
    phi_freeze_inverse_acc: float = 0.90
    phi_freeze_patience: int = 3        # consecutive updates ≥ threshold
    phi_freeze_max_updates: int = 100   # freeze by here regardless (~200k env steps)
    # Which inverse_acc the freeze trigger reads. "holdout" = a FIXED uniform-random
    # held-out transition set (φ's TRUE controllability); "onpolicy" = the current
    # rollout. On-policy is INFLATED by a narrowing policy and fools the freeze
    # (see probes/inv_acc_causality.md) → default to held-out.
    freeze_metric: str = "holdout"      # "holdout" | "onpolicy"
    holdout_size: int = 2000            # # transitions in the held-out inv_acc set

    # φ source for the RND ruler:
    #   "icm"    = ICM inverse-dynamics features (learned, controllable; the default).
    #   "frozen" = a fixed RANDOM encoder (no ICM training/freeze) — "plain RND+leak".
    phi_mode: str = "icm"
    # Optional: initialise the ICM φ-encoder from a saved checkpoint's "icm" state
    # (e.g. an L1-trained run) → cross-level representation TRANSFER. None = random init.
    init_phi_ckpt: str | None = None

    # RND-on-φ + leak
    rnd_feature_dim: int = 256
    rnd_hidden: int = 256
    rnd_lr: float = 1e-4
    rnd_epochs: int = 1
    leak: float = 0.01                  # μ: predictor shrink-to-init per update (forget rate)

    # Intrinsic-reward normalisation (warm-up + EMA std of returns, no centring)
    int_norm_decay: float = 0.99
    norm_warmup_updates: int = 2
    int_norm_eps: float = 1e-8
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
