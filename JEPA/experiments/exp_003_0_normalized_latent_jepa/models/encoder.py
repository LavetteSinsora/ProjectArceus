"""
Exp-003 Encoder: patch embedding + 2D-RoPE self-attention + Perceiver Resampler.

Changes from exp_002:
  1. PerceiverResampler uses SEPARATE weights per round (not weight-tied).
     Rationale: weight-tied rounds accumulate gradient 4× per step (2 encode
     calls × 2 rounds), causing Perceiver grad norm to grow 5→17 over 500K steps.
     Separate weights halve this to 2× (one encode call for h_t, one for h_{t+1}).

  2. PerceiverResampler adds output LayerNorm after all rounds.
     Rationale: pre-norm residual streams grow linearly with depth/time. Each of
     the 8 residual additions per Perceiver call contributes ~0.15 to the L2 norm.
     Over a 42-step life (8 × 42 = 336 residual additions), norms reach ~54.
     Output norm clamps the recurrent boundary to unit scale on every step.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 2D RoPE (unchanged from exp_002) ─────────────────────────────────────────

def _build_rope_freqs(d_head: int, theta: float = 10000.0) -> torch.Tensor:
    half = d_head // 2
    return 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))


def _apply_rope_1d(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    angles = positions.float().unsqueeze(-1) * inv_freq.to(x.device)
    cos = angles.cos().unsqueeze(0).unsqueeze(0)
    sin = angles.sin().unsqueeze(0).unsqueeze(0)
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_rope_2d(q, k, grid_h, grid_w, theta=10000.0):
    B, H, L, d_head = q.shape
    assert L == grid_h * grid_w
    half_head = d_head // 2
    rows = torch.arange(grid_h, device=q.device).repeat_interleave(grid_w)
    cols = torch.arange(grid_w, device=q.device).repeat(grid_h)
    inv_freq = _build_rope_freqs(half_head, theta)
    q_rot = torch.cat([
        _apply_rope_1d(q[..., :half_head], rows, inv_freq),
        _apply_rope_1d(q[..., half_head:], cols, inv_freq),
    ], dim=-1)
    k_rot = torch.cat([
        _apply_rope_1d(k[..., :half_head], rows, inv_freq),
        _apply_rope_1d(k[..., half_head:], cols, inv_freq),
    ], dim=-1)
    return q_rot, k_rot


# ── Transformer building blocks ───────────────────────────────────────────────

class _FFN(nn.Module):
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
    """Pre-norm multi-head self-attention with 2D RoPE over 16 patch tokens."""

    def __init__(self, d_model, n_heads, ffn_dim, grid_h=4, grid_w=4, theta=10000.0):
        super().__init__()
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
        B, L, D = x.shape
        h = self.norm(x)
        Q = self.q_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        Q, K = apply_rope_2d(Q, K, self.grid_h, self.grid_w, self.theta)
        scale = math.sqrt(self.d_head)
        attn = F.softmax((Q @ K.transpose(-2, -1)) / scale, dim=-1)
        if not self.training:
            self._debug_attn = attn.detach().cpu()  # (B, n_heads, 16, 16)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(out)
        return self.ffn(x)


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention: Q from latents, K/V from SA patch outputs."""

    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
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

    def forward(self, queries, context):
        B, Lq, D = queries.shape
        Lc = context.shape[1]
        Q = self.q_proj(self.norm_q(queries)).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(self.norm_kv(context)).view(B, Lc, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(context).view(B, Lc, self.n_heads, self.d_head).transpose(1, 2)
        attn_w = F.softmax((Q @ K.transpose(-2, -1)) / self.scale, dim=-1)
        out = (attn_w @ V).transpose(1, 2).contiguous().view(B, Lq, D)
        queries = queries + self.out_proj(out)
        queries = self.ffn(queries)
        return queries, attn_w.detach()


class _SelfAttentionAmongLatents(nn.Module):
    """Pre-norm self-attention among the n_latents query vectors."""

    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
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
        B, L, D = x.shape
        h = self.norm(x)
        Q = self.q_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        attn_w = F.softmax((Q @ K.transpose(-2, -1)) / self.scale, dim=-1)
        if not self.training:
            self._debug_attn = attn_w.detach().cpu()  # (B, n_heads, 4, 4)
        out = (attn_w @ V).transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(out)
        return self.ffn(x)


class _PerceiverRound(nn.Module):
    """One round of Perceiver: cross-attention followed by self-attention."""

    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.cross_attn = _CrossAttentionBlock(d_model, n_heads, ffn_dim)
        self.self_attn = _SelfAttentionAmongLatents(d_model, n_heads, ffn_dim)

    def forward(self, queries, context):
        queries, attn_w = self.cross_attn(queries, context)
        queries = self.self_attn(queries)
        return queries, attn_w


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler with SEPARATE weights per round and output LayerNorm.

    Compresses 16 SA-enriched patch embeddings → n_latents latent vectors.

    vs exp_002:
      - Each round has its own _PerceiverRound (separate weights, not weight-tied)
      - output_norm: LayerNorm applied to the final h before returning
        This normalizes the recurrent boundary so h_t fed back as next-step queries
        always has controlled scale, preventing linear norm growth over episodes.
    """

    def __init__(self, d_model, n_latents, n_placeholders, n_rounds, n_heads, ffn_dim):
        super().__init__()
        self.n_latents = n_latents
        self.n_rounds = n_rounds

        self.placeholders = nn.Parameter(torch.randn(n_placeholders, d_model) * 0.02)

        # Separate weights per round — not weight-tied
        self.rounds = nn.ModuleList([
            _PerceiverRound(d_model, n_heads, ffn_dim)
            for _ in range(n_rounds)
        ])

        # Output norm: clamps the recurrent state to unit scale on every step
        self.output_norm = nn.LayerNorm(d_model)

    def get_initial_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.placeholders.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> tuple:
        """
        queries: (B, n_latents, d_model)
        context: (B, 16, d_model) — SA-normed patch embeddings

        Returns:
          latents:      (B, n_latents, d_model) — output-normed
          attn_weights: list of (B, n_heads, n_latents, 16), one per round
        """
        h = queries
        attn_weights_all = []
        for round_block in self.rounds:
            h, attn_w = round_block(h, context)
            attn_weights_all.append(attn_w)
        h = self.output_norm(h)
        return h, attn_weights_all


class Encoder(nn.Module):
    """
    Full exp-003 encoder.

    stage1: patch embedding + 2 SA blocks with 2D RoPE → (B, 16, d_model)
    stage2: PerceiverResampler (separate round weights, output_norm) → (B, n_latents, d_model)
    """

    def __init__(
        self,
        d_model=128,
        d_color=4,
        n_sa_heads=4,
        n_sa_blocks=2,
        sa_ffn_dim=512,
        patch_size=16,
        n_latents=4,
        n_placeholders=4,
        n_perceiver_rounds=2,
        n_perceiver_heads=4,
        perceiver_ffn_dim=512,
        rope_theta=10000.0,
        patch_grid_h=4,
        patch_grid_w=4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_grid_h = patch_grid_h
        self.patch_grid_w = patch_grid_w
        patch_flat_dim = patch_size * patch_size * d_color  # 1024

        self.color_embed = nn.Embedding(16, d_color)
        self.patch_proj = nn.Linear(patch_flat_dim, d_model)

        self.sa_blocks = nn.ModuleList([
            SelfAttentionBlock(d_model, n_sa_heads, sa_ffn_dim, patch_grid_h, patch_grid_w, rope_theta)
            for _ in range(n_sa_blocks)
        ])
        self.sa_norm = nn.LayerNorm(d_model)

        self.perceiver = PerceiverResampler(
            d_model, n_latents, n_placeholders,
            n_perceiver_rounds, n_perceiver_heads, perceiver_ffn_dim,
        )

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 64, 64) uint8 → (B, 16, d_model) SA-normed patch embeddings."""
        B = x.shape[0]
        p = self.patch_size
        emb = self.color_embed(x.long())
        emb = emb.view(B, self.patch_grid_h, p, self.patch_grid_w, p, -1)
        emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous()
        emb = emb.view(B, self.patch_grid_h * self.patch_grid_w, -1)
        emb = self.patch_proj(emb)
        for block in self.sa_blocks:
            emb = block(emb)
        return self.sa_norm(emb)

    def forward(self, x: torch.Tensor, queries: torch.Tensor) -> tuple:
        """
        x:       (B, 64, 64) uint8
        queries: (B, n_latents, d_model) — placeholder or h_{t-1}

        Returns:
          latents:      (B, n_latents, d_model) — output-normed, scale-controlled
          sa_out:       (B, 16, d_model)
          attn_weights: list of (B, n_heads, n_latents, 16) per round
        """
        sa_out = self.encode_patches(x)
        latents, attn_weights = self.perceiver(queries, sa_out)
        return latents, sa_out, attn_weights
