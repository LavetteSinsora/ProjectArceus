"""Dashboard episode runner for exp_011_0. Delegates to the exp_011 shared
runner (ICM-config aware + level aware; the ICM module is unused at playback)."""

from JEPA.experiments.exp_011_ls20_icm.shared.debug_runner import (  # noqa: F401
    run_debug_episode, CAPABILITIES,
)
