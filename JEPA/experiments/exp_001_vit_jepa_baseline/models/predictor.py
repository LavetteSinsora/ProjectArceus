import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Per-patch MLP predictor for JEPA temporal prediction.

    Maps (current patch embedding, action embedding) → predicted next-state patch embedding.
    Operates independently on each of the 16 patches — no cross-patch attention.
    This is intentional: a per-patch design avoids compatibility issues with masking
    and keeps the predictor lightweight.

    Input per patch: concat([patch_emb (d_model), action_emb (d_action)]) = d_model+d_action
    Output per patch: predicted next-state patch embedding (d_model)
    """

    def __init__(self, d_model: int = 128, d_action: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_action, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, z: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """
        z: (B, 16, d_model) — current state patch embeddings
        action_emb: (B, d_action) — action embedding
        returns: (B, 16, d_model) — predicted next-state embeddings
        """
        # Broadcast action across all 16 patches
        a = action_emb.unsqueeze(1).expand(-1, z.shape[1], -1)  # (B, 16, d_action)
        x = torch.cat([z, a], dim=-1)                            # (B, 16, d_model+d_action)
        return self.mlp(x)                                        # (B, 16, d_model)
