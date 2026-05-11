from dataclasses import dataclass
from JEPA.experiments.exp_003_0_normalized_latent_jepa.config import Config as _Base003


@dataclass(frozen=True)
class Config(_Base003):
    # EMA target encoder decay schedule (cosine ramp from start → end)
    ema_decay_start: float = 0.996
    ema_decay_end:   float = 0.9999
