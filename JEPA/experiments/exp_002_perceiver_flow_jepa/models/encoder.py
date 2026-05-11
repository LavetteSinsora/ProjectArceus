"""
Exp-002 Encoder: patch embedding + 2D-RoPE self-attention + weight-tied Perceiver Resampler.

Pipeline:
  1. PatchEmbedding: (B, 64, 64) uint8 → (B, 16, d_model)
     Each 16×16 patch: per-pixel color lookup (nn.Embedding) → flatten → Linear
  2. SelfAttentionBlock × 2: 2D RoPE applied to Q, K; pre-norm; GELU FFN
  3. PerceiverResampler (weight-tied, N rounds):
     - First state: 4 learned placeholder vectors as queries
     - Subsequent states: h_{t-1} as queries
     - Cross-attention (Q=queries, K/V=16 SA outputs) + self-attention among 4 queries
     - Same block weights reused each round
  Output: (B, 4, d_model) latent state vectors h_t  (NOT L2-normalised)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 2D RoPE ───────────────────────────────────────────────────────────────────

def _build_rope_freqs(d_head: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Build inverse frequency table for 1D RoPE.
    Returns shape (d_head // 2,).
    """
    half = d_head // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    return inv_freq  # (d_head/2,)


def _apply_rope_1d(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """
    Apply 1D RoPE to the last dimension of x at given positions.

    x:          (B, n_heads, seq_len, d_per_axis)  — half of d_head
    positions:  (seq_len,)                          — integer positions
    inv_freq:   (d_per_axis // 2,)

    Returns x rotated.  Pairs up dims as (cos, sin) rotations.
    """
    # angles: (seq_len, d_per_axis/2)
    angles = positions.float().unsqueeze(-1) * inv_freq.to(x.device)  # (L, d//2)
    cos = angles.cos()  # (L, d//2)
    sin = angles.sin()  # (L, d//2)

    # Repeat to match d_per_axis by interleaving cos/sin for pairs
    # x has shape (B, H, L, d_per_axis); treat d_per_axis as d//2 complex pairs
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]  # split into two halves

    # broadcast (L, half) → (1, 1, L, half)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot


def apply_rope_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_h: int,
    grid_w: int,
    theta: float = 10000.0,
) -> tuple:
    """
    Apply 2D RoPE to queries and keys for a (grid_h × grid_w) patch grid.

    q, k: (B, n_heads, seq_len, d_head)
    Returns (q_rotated, k_rotated) same shape.

    The d_head is split equally: first half → row-RoPE, second half → col-RoPE.
    Positions are assigned in row-major order: patch (r, c) has
      row_pos = r, col_pos = c.
    """
    B, H, L, d_head = q.shape
    assert L == grid_h * grid_w, f"seq_len {L} != {grid_h}×{grid_w}"
    assert d_head % 2 == 0, "d_head must be even for 2D RoPE"

    half_head = d_head // 2

    # Build row and col position tensors in row-major order
    rows = torch.arange(grid_h, device=q.device).repeat_interleave(grid_w)  # (L,)
    cols = torch.arange(grid_w, device=q.device).repeat(grid_h)             # (L,)

    inv_freq = _build_rope_freqs(half_head, theta)

    q_row = _apply_rope_1d(q[..., :half_head], rows, inv_freq)
    q_col = _apply_rope_1d(q[..., half_head:], cols, inv_freq)
    q_rot = torch.cat([q_row, q_col], dim=-1)

    k_row = _apply_rope_1d(k[..., :half_head], rows, inv_freq)
    k_col = _apply_rope_1d(k[..., half_head:], cols, inv_freq)
    k_rot = torch.cat([k_row, k_col], dim=-1)

    return q_rot, k_rot


# ── Transformer building blocks ───────────────────────────────────────────────

class _FFN(nn.Module):
    """Pre-norm FFN with GELU."""

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class SelfAttentionBlock(nn.Module):
    """
    Pre-norm multi-head self-attention block with 2D RoPE.

    Used for the encoder's 2 SA blocks over 16 patch tokens.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 grid_h: int = 4, grid_w: int = 4, theta: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.theta = theta

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = _FFN(d_model, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 16, d_model)"""
        B, L, D = x.shape
        h = self.norm(x)

        Q = self.q_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Apply 2D RoPE to Q and K
        Q, K = apply_rope_2d(Q, K, self.grid_h, self.grid_w, self.theta)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn = (Q @ K.transpose(-2, -1)) / scale          # (B, H, L, L)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)

        x = x + self.out_proj(out)
        x = self.ffn(x)
        return x


# ── Perceiver Resampler ───────────────────────────────────────────────────────

class _CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention: Q from latents, K/V from context (SA patch outputs).
    Used inside the Perceiver Resampler.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = _FFN(d_model, ffn_dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> tuple:
        """
        queries: (B, n_latents, d_model)
        context: (B, 16, d_model)
        Returns (updated_queries, attn_weights)
          updated_queries: (B, n_latents, d_model)
          attn_weights:    (B, n_heads, n_latents, 16)
        """
        B, Lq, D = queries.shape
        Lc = context.shape[1]

        Q = self.q_proj(self.norm_q(queries)).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(self.norm_kv(context)).view(B, Lc, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(context).view(B, Lc, self.n_heads, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / self.scale   # (B, H, Lq, Lc)
        attn_w = F.softmax(attn, dim=-1)
        out = (attn_w @ V).transpose(1, 2).contiguous().view(B, Lq, D)

        queries = queries + self.out_proj(out)
        queries = self.ffn(queries)
        return queries, attn_w.detach()


class _SelfAttentionAmongLatents(nn.Module):
    """Pre-norm self-attention among the n_latents query vectors."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = _FFN(d_model, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_latents, d_model)"""
        B, L, D = x.shape
        h = self.norm(x)
        Q = self.q_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn_w = F.softmax(attn, dim=-1)
        out = (attn_w @ V).transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(out)
        x = self.ffn(x)
        return x


class PerceiverResampler(nn.Module):
    """
    Weight-tied Perceiver Resampler.

    Compresses 16 SA-enriched patch embeddings → n_latents latent vectors.
    The same cross-attention + self-attention block is reused for each of the
    n_perceiver_rounds rounds (weight-tied, NOT separate blocks per round).

    Episode boundary handling (caller's responsibility):
      - t=0: pass queries = self.placeholders.unsqueeze(0).expand(B, -1, -1)
      - t>0: pass queries = h_{t-1}  (output of previous call)
    """

    def __init__(
        self,
        d_model: int,
        n_latents: int,
        n_placeholders: int,
        n_rounds: int,
        n_heads: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.n_rounds = n_rounds

        # Learned placeholder queries for episode start
        self.placeholders = nn.Parameter(torch.randn(n_placeholders, d_model) * 0.02)

        # Single shared block (weight-tied across rounds)
        self.cross_attn = _CrossAttentionBlock(d_model, n_heads, ffn_dim)
        self.self_attn = _SelfAttentionAmongLatents(d_model, n_heads, ffn_dim)

    def get_initial_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns placeholder queries for episode start: (B, n_latents, d_model)."""
        return self.placeholders.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> tuple:
        """
        queries: (B, n_latents, d_model)  — either placeholders or h_{t-1}
        context: (B, 16, d_model)         — SA-enriched patch embeddings

        Returns:
          latents:      (B, n_latents, d_model)
          attn_weights: list of (B, n_heads, n_latents, 16) — one per round
        """
        attn_weights_all = []
        h = queries
        for _ in range(self.n_rounds):
            h, attn_w = self.cross_attn(h, context)
            h = self.self_attn(h)
            attn_weights_all.append(attn_w)
        return h, attn_weights_all


# ── Full Encoder ──────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Full exp-002 encoder.

    stage1: patch embedding + 2 SA blocks with 2D RoPE → (B, 16, d_model)
    stage2: weight-tied PerceiverResampler              → (B, n_latents, d_model)

    No L2 normalisation — norms are monitored during training instead.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_color: int = 4,
        n_sa_heads: int = 4,
        n_sa_blocks: int = 2,
        sa_ffn_dim: int = 512,
        patch_size: int = 16,
        n_latents: int = 4,
        n_placeholders: int = 4,
        n_perceiver_rounds: int = 2,
        n_perceiver_heads: int = 4,
        perceiver_ffn_dim: int = 512,
        rope_theta: float = 10000.0,
        patch_grid_h: int = 4,
        patch_grid_w: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_grid_h = patch_grid_h
        self.patch_grid_w = patch_grid_w
        n_patches = patch_grid_h * patch_grid_w  # 16
        patch_flat_dim = patch_size * patch_size * d_color  # 16×16×4 = 1024

        # Stage 1: patch embedding
        self.color_embed = nn.Embedding(16, d_color)
        self.patch_proj = nn.Linear(patch_flat_dim, d_model)

        # Stage 1: SA blocks with 2D RoPE
        self.sa_blocks = nn.ModuleList([
            SelfAttentionBlock(d_model, n_sa_heads, sa_ffn_dim,
                               patch_grid_h, patch_grid_w, rope_theta)
            for _ in range(n_sa_blocks)
        ])
        self.sa_norm = nn.LayerNorm(d_model)

        # Stage 2: Perceiver Resampler
        self.perceiver = PerceiverResampler(
            d_model, n_latents, n_placeholders,
            n_perceiver_rounds, n_perceiver_heads, perceiver_ffn_dim,
        )

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 64, 64) uint8
        Returns: (B, 16, d_model) SA-enriched patch embeddings.
        Useful when you need the intermediate SA output (e.g. for dashboard).
        """
        B = x.shape[0]
        p = self.patch_size

        # Color embedding: (B, 64, 64) → (B, 64, 64, d_color)
        emb = self.color_embed(x.long())

        # Rearrange into patches: (B, grid_h, grid_w, p, p, d_color) → (B, 16, p*p*d_color)
        emb = emb.view(B, self.patch_grid_h, p, self.patch_grid_w, p, -1)
        emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous()
        emb = emb.view(B, self.patch_grid_h * self.patch_grid_w, -1)

        # Project + SA blocks
        emb = self.patch_proj(emb)
        for block in self.sa_blocks:
            emb = block(emb)
        emb = self.sa_norm(emb)
        return emb  # (B, 16, d_model)

    def forward(self, x: torch.Tensor, queries: torch.Tensor) -> tuple:
        """
        x:       (B, 64, 64) uint8   — current frame
        queries: (B, n_latents, d_model) — either placeholders (t=0) or h_{t-1}

        Returns:
          latents:      (B, n_latents, d_model)
          sa_out:       (B, 16, d_model)              — for dashboard / targets
          attn_weights: list of (B, n_heads, n_latents, 16)
        """
        sa_out = self.encode_patches(x)
        latents, attn_weights = self.perceiver(queries, sa_out)
        return latents, sa_out, attn_weights
