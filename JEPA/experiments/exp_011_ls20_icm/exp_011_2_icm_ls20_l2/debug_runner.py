"""Dashboard episode runner for exp_011_2 (LS20 L2). Delegates to the exp_011
shared runner, which drops the agent on cfg.level_index (L2) for playback."""

from JEPA.experiments.exp_011_ls20_icm.shared.debug_runner import (  # noqa: F401
    run_debug_episode, CAPABILITIES,
)
