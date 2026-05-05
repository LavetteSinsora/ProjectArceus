import torch
import torch.nn as nn
from arcengine import GameAction

# LS20 uses exactly ACTION1–4 (no click/coordinate actions)
LS20_ACTIONS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
]
# Maps GameAction enum → 0-indexed integer for embedding lookup
ACTION_TO_IDX = {a: i for i, a in enumerate(LS20_ACTIONS)}


class ActionEmbedding(nn.Module):
    """Learnable embedding for LS20's 4 discrete actions (ACTION1–4)."""

    def __init__(self, n_actions: int = 4, d_action: int = 32):
        super().__init__()
        self.embed = nn.Embedding(n_actions, d_action)

    def forward(self, action_idx: torch.Tensor) -> torch.Tensor:
        """
        action_idx: (B,) long tensor — 0-indexed (0=ACTION1, 1=ACTION2, …)
        returns: (B, d_action)
        """
        return self.embed(action_idx)
