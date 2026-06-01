from dataclasses import dataclass

from JEPA.experiments.exp_011_ls20_icm.shared.config_base import Config as _Base


@dataclass
class Config(_Base):
    exp_name: str = "exp_011_2_icm_l2"
    exp_dir: str = "JEPA/experiments/exp_011_ls20_icm/exp_011_2_icm_ls20_l2"

    # The whole point: start every episode on LS20 Level 2 (0-indexed = 1).
    level_index: int = 1

    # L2 is a much deeper puzzle (~60-action optimal solve vs L1's 13). Give
    # exploration a larger budget; the plateau early-stop will cut it short if
    # success saturates or stalls.
    total_env_steps: int = 6_000_000
