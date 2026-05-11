import torch
import torch.nn as nn
import torch.nn.functional as F


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block with GELU FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Pre-norm FFN with residual
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class Encoder(nn.Module):
    """
    ViT encoder: (B, 64, 64) uint8 → (B, 16, d_model) patch embeddings.

    Pipeline:
      1. Color embed: nn.Embedding(16, d_color) per pixel → (B, 64, 64, d_color)
      2. Unfold into 16×16 patches (4×4 grid = 16 patches) → flatten per patch
      3. Linear projection to d_model
      4. Add learned 2D position embeddings (16 positions)
      5. 2 pre-norm transformer blocks with full self-attention

    Design choices:
      - Pre-norm: stabilises gradients vs. post-norm (Xiong et al. 2020)
      - GELU: smoother gradient than ReLU, standard for modern transformers
      - Learned absolute positional embeddings: sufficient at 16 tokens
    """

    def __init__(
        self,
        d_model: int = 128,
        d_color: int = 4,
        n_heads: int = 4,
        n_blocks: int = 2,
        ffn_dim: int = 512,
        patch_size: int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_side = 64 // patch_size        # 4 patches per side
        self.n_patches = n_side * n_side  # 16 total
        patch_flat_dim = patch_size * patch_size * d_color  # 16×16×4 = 1024

        self.color_embed = nn.Embedding(16, d_color)
        self.patch_proj = nn.Linear(patch_flat_dim, d_model)
        self.pos_embed = nn.Embedding(self.n_patches, d_model)

        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 64, 64) uint8 — color indices 0–15
        returns: (B, 16, d_model) contextualised patch embeddings
        """
        B = x.shape[0]
        p = self.patch_size

        # Color embedding: (B, 64, 64) → (B, 64, 64, d_color)
        x = self.color_embed(x.long())

        # Rearrange into patches: (B, n_patches, p*p*d_color)
        # x shape: (B, H, W, C) → split H and W into grid and patch dims
        x = x.view(B, 64 // p, p, 64 // p, p, -1)   # (B, 4, p, 4, p, d_c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # (B, 4, 4, p, p, d_c)
        x = x.view(B, self.n_patches, -1)             # (B, 16, p*p*d_c)

        # Project to d_model and add positional embeddings
        x = self.patch_proj(x)  # (B, 16, d_model)
        pos = torch.arange(self.n_patches, device=x.device)
        x = x + self.pos_embed(pos)  # broadcast over batch

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        return F.normalize(self.norm(x), dim=-1)  # (B, 16, d_model), unit-norm
