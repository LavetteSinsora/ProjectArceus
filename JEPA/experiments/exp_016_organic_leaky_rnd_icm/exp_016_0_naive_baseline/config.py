"""Config for exp_016_0 — naive leaky-RND on IDM features. See SYSTEM_CARD.md §4.

Deliberately minimal: REINFORCE (no value baseline), a continuously-trained IDM
encoder (no freeze), leaky RND over those features, one normalized intrinsic reward.
Every normalization keeps its RAW stats logged so we can later prove it was needed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

GAME_N_ACTIONS = {"ls20": 4, "tu93": 4, "re86": 5, "g50t": 5}


@dataclass
class Config:
    # ── identity ────────────────────────────────────────────────────────────
    game: str = "ls20"
    level_index: int = 0                 # 0 = Level 1
    seed: int = 0
    exp_dir: str = ("JEPA/experiments/exp_016_organic_leaky_rnd_icm/"
                    "exp_016_0_naive_baseline")

    # ── env / rollout / budget ──────────────────────────────────────────────
    n_envs: int = 16
    rollout_steps: int = 128             # 128*16 = 2048 transitions / update
    max_episode_steps: int = 200
    max_env_steps: int = 250_000
    n_colors: int = 16
    frame_size: int = 64
    trunk_dim: int = 256
    n_actions: int | None = None         # filled from the env at build time

    # ── timer/observation mask (applied to BOTH encoder inputs) ─────────────
    timer_mask_rows: tuple = (60, 63)    # inclusive rows zeroed before encoding

    # ── actor (REINFORCE, NO value head) ────────────────────────────────────
    actor_lr: float = 3e-4
    grad_clip: float = 0.5
    ent_coef: float = 0.01
    gamma: float = 0.99                  # reward-to-go discount (episodic)
    # normalization / baseline knobs
    reward_zscore: bool = True           # scale reward by RUNNING std (cross-time scale stability)
    reward_center: bool = False          # also subtract the running MEAN from the reward
                                         # (OFF by default — the lagging cumulative mean caused the
                                         # spurious-negative reward; the batch baseline re-centers anyway)
    use_baseline: bool = True            # (constant baseline) advantage = return − batch-mean return.
                                         # Removes the OFFSET but not per-state/position structure —
                                         # insufficient alone; use a value head for the real fix.
    use_value_head: bool = False         # STATE-DEPENDENT baseline V(s): advantage = return − V(s),
                                         # value head trained by MSE to the returns. The proper fix.
    c_value: float = 0.5                 # weight on the value-head MSE loss
    return_scale_by_std: bool = False    # divide advantage by batch std (OFF: re-amplifies residual
                                         # noise to unit scale and re-injects the drag)
    norm_eps: float = 1e-8

    # ── IDM (inverse-dynamics encoder; trained CONTINUOUSLY, no freeze) ──────
    idm_lr: float = 1e-3
    idm_hidden: int = 256
    idm_layernorm: bool = False        # ablation: LayerNorm h before inverse head + RND
    idm_grad_steps: int = 4              # minibatch grad steps / update
    idm_batch: int = 512
    replay_capacity: int = 50_000
    drop_noops: bool = True             # exclude masked s_t == s_{t+1} from replay

    # ── RND count net + leak ────────────────────────────────────────────────
    rnd_hidden: int = 256
    rnd_out: int = 256
    rnd_lr: float = 1e-4
    rnd_grad_steps: int = 4
    leak: float = 0.1                   # μ: predictor shrink-to-init, ONCE per update

    # ── diagnostics ─────────────────────────────────────────────────────────
    holdout_size: int = 2000           # fixed random transitions for held-out inv_acc
    n_probe_states: int = 5
    probe_roam_steps: int = 2000       # per-env random roam to harvest probe/registry states
    log_state_novelty: bool = True     # full-state novelty landscape each update

    # ── logging / checkpoint cadence ────────────────────────────────────────
    log_every: int = 1
    save_every: int = 10               # save a checkpoint every N updates (0 = off; final always saved)

    @property
    def exp_name(self) -> str:
        return f"exp016_0_naive_{self.game}_L{self.level_index + 1}_seed{self.seed}"

    @property
    def total_updates(self) -> int:
        return self.max_env_steps // (self.rollout_steps * self.n_envs)

    def smoke(self) -> "Config":
        return dataclasses.replace(
            self, n_envs=2, rollout_steps=16, max_episode_steps=40,
            max_env_steps=16 * 2 * 6, idm_grad_steps=1, rnd_grad_steps=1,
            idm_batch=16, holdout_size=64, probe_roam_steps=80, n_probe_states=3,
            save_every=2,
        )
