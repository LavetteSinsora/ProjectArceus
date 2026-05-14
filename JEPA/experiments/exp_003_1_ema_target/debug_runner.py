"""
Debug runner for exp_003_1_ema_target.

The model architecture is identical to exp_003_0 (exp_003_1 only adds an EMA
target encoder, which is unused at inference time), so the debug runner
delegates to exp_003_0's implementation. Checkpoints in exp_003_1/checkpoints
contain an extra ``target_encoder`` key — exp_003_0's runner only reads the
``encoder``/``predictor``/``action_embed``/``policy`` keys, so the extra key
is harmlessly ignored. exp_003_0's runner also filters unknown Config fields
(e.g. ema_decay_start) when reconstructing the dataclass from the checkpoint.

We force-reload exp_003_0's debug_runner each time this module is reloaded by
the dashboard, so any fixes there are picked up without restarting the server.
"""

import importlib
import sys as _sys

_UPSTREAM = "JEPA.experiments.exp_003_0_normalized_latent_jepa.debug_runner"
if _UPSTREAM in _sys.modules:
    _mod = importlib.reload(_sys.modules[_UPSTREAM])
else:
    _mod = importlib.import_module(_UPSTREAM)

def run_debug_episode(checkpoint_path: str,
                      env_name: str | None = None,
                      max_steps: int = 200) -> dict:
    data = _mod.run_debug_episode(checkpoint_path, env_name=env_name, max_steps=max_steps)
    data["experiment"] = "exp_003_1_ema_target"
    return data


__all__ = ["run_debug_episode"]
