"""Device resolution: cuda > mps > cpu."""

from __future__ import annotations

import torch


def resolve_device(spec: str) -> torch.device:
    """Resolve a cfg.device string.

    - "auto" → cuda if available, else mps if available, else cpu
    - any other string is passed straight to torch.device
    """
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
