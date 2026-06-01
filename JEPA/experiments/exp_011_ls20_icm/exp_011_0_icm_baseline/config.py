from dataclasses import dataclass

from JEPA.experiments.exp_011_ls20_icm.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    exp_name: str = "exp_011_0_icm"
    exp_dir: str = "JEPA/experiments/exp_011_ls20_icm/exp_011_0_icm_baseline"
    # Terminal-only reward on real LS20 is genuinely sparse; give exploration
    # room (same budget as the exp_010_0 baseline it is measured against).
    total_env_steps: int = 3_000_000
