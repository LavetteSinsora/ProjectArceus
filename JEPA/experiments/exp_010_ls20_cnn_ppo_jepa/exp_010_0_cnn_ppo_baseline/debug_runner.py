"""Dashboard episode runner for exp_010_0. Delegates to the shared runner."""

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.debug_runner import (  # noqa: F401
    run_debug_episode, CAPABILITIES,
)
