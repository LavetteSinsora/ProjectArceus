import math
import torch.nn as nn


def update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """
    Update target encoder parameters via EMA.

    θ_target ← m·θ_target + (1−m)·θ_online

    Called once per JEPA gradient step (not every env step).
    Target encoder never receives gradients — EMA is the only update mechanism.
    """
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)


def ema_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 0.9999) -> float:
    """
    Cosine schedule from `start` to `end` over `total_steps`.

    High momentum early (stable targets) → near-1.0 late (nearly frozen target encoder).
    This follows the I-JEPA training schedule.
    """
    progress = min(step / max(total_steps, 1), 1.0)
    cosine_factor = (1.0 - math.cos(math.pi * progress)) / 2.0
    return start + (end - start) * cosine_factor
