from dataclasses import dataclass

from JEPA.experiments.exp_012_ls20_rnd.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    exp_name: str = "exp_012_1_rnd_l2"
    exp_dir: str = "JEPA/experiments/exp_012_ls20_intrinsic_exploration/exp_012_1_rnd_l2"

    # Clear L1 then L2 in one episode (incremental +1 reward per level).
    stop_levels: int = 2
    # Two levels need more room than the L1 cap; the game's own per-level step
    # budget still enforces GAME_OVER, so this is a generous outer truncation.
    max_episode_steps: int = 400

    # Warm-start from the L1 RND solution (repo-relative; resolved by the trainer).
    init_ckpt: str = ("JEPA/experiments/exp_012_ls20_intrinsic_exploration/"
                      "exp_012_1_rnd_baseline/checkpoints/step_00204800.pt")

    total_env_steps: int = 3_000_000
