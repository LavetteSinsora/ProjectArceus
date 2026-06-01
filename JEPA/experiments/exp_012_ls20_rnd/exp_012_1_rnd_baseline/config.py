from dataclasses import dataclass

from JEPA.experiments.exp_012_ls20_rnd.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    exp_name: str = "exp_012_1_rnd"
    exp_dir: str = "JEPA/experiments/exp_012_ls20_intrinsic_exploration/exp_012_1_rnd_baseline"
    # Sparse terminal reward + exploration bonus; give it the same budget as the
    # exp_010_0 sparse baseline so the steps-to-first-reward comparison is fair.
    total_env_steps: int = 3_000_000
