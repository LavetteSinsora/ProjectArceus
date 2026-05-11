"""
Exp-003 Flow Matching Predictor — unchanged from exp_002.

Stop gradient on h_t1 is applied at the call site in train.py:
    predictor.compute_loss(h_t_fresh, h_t1_stored.detach(), a_emb)

This ensures:
  - Gradient flows: loss → x̂_1 → MLP → x_tau = (1-τ)h_t + τ·h_t1 → h_t → encoder ✅
  - No gradient through h_t1 (detached target) ✅
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int = 128, time_proj_dim: int = 512):
        super().__init__()
        assert time_emb_dim % 2 == 0
        self.time_emb_dim = time_emb_dim
        self.proj = nn.Sequential(nn.Linear(time_emb_dim, time_proj_dim), nn.GELU())

    def _sinusoidal(self, tau: torch.Tensor) -> torch.Tensor:
        half = self.time_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / half
        )
        angles = tau.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.proj(self._sinusoidal(tau))


class _LatentMLP(nn.Module):
    def __init__(self, d_model: int, d_action: int, time_proj_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + d_action + time_proj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x_tau, action_emb, time_feat):
        return self.net(torch.cat([x_tau, action_emb, time_feat], dim=-1))


class FlowMatchingPredictor(nn.Module):
    """4 separate MLPs (one per latent) with sinusoidal time conditioning."""

    def __init__(self, n_latents=4, d_model=128, d_action=32,
                 time_emb_dim=128, time_proj_dim=512, hidden_dim=512, n_ode_steps=3):
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model
        self.n_ode_steps = n_ode_steps
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim, time_proj_dim)
        self.mlps = nn.ModuleList([
            _LatentMLP(d_model, d_action, time_proj_dim, hidden_dim)
            for _ in range(n_latents)
        ])

    def _predict_clean(self, x_tau, tau, action_emb):
        time_feat = self.time_embed(tau)
        return torch.cat([
            mlp(x_tau[:, i, :], action_emb, time_feat).unsqueeze(1)
            for i, mlp in enumerate(self.mlps)
        ], dim=1)

    def compute_loss(self, h_t: torch.Tensor, h_t1: torch.Tensor,
                     action_emb: torch.Tensor) -> tuple:
        """
        h_t:  (B, n_latents, d_model) — current latents; gradient flows to encoder
        h_t1: (B, n_latents, d_model) — target; MUST be .detach()-ed at call site
        """
        B = h_t.shape[0]
        tau = torch.rand(B, device=h_t.device)
        tau_exp = tau.view(B, 1, 1)
        x_tau = (1.0 - tau_exp) * h_t + tau_exp * h_t1
        x1_hat = self._predict_clean(x_tau, tau, action_emb)
        sq_err = (h_t1 - x1_hat).pow(2)
        per_latent = sq_err.mean(dim=[0, 2])
        return per_latent.mean(), per_latent.detach()

    @torch.no_grad()
    def predict(self, h_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        x_0 = h_t
        x_k = h_t.clone()
        N = self.n_ode_steps
        for k in range(N):
            tau = torch.full((h_t.shape[0],), k / N, device=h_t.device)
            x1_hat = self._predict_clean(x_k, tau, action_emb)
            x_k = x_k + (1.0 / N) * (x1_hat - x_0)
        return x_k

    def predict_with_loss(self, h_t, h_t1, action_emb):
        predicted = self.predict(h_t, action_emb)
        per_latent_mse = (h_t1 - predicted).pow(2).mean(dim=-1).mean(dim=0)
        return predicted, per_latent_mse

    @torch.no_grad()
    def predict_with_trajectory(self, h_t: torch.Tensor, action_emb: torch.Tensor) -> tuple:
        """ODE rollout returning intermediate states.

        Returns (h1_pred, trajectory) where trajectory is a list of N+1 tensors,
        each (B, n_latents, d_model): [x_0, x_{1/N}, x_{2/N}, ..., x_1].
        """
        x_0 = h_t
        x_k = h_t.clone()
        N = self.n_ode_steps
        traj = [x_k.detach().cpu()]
        for k in range(N):
            tau = torch.full((h_t.shape[0],), k / N, device=h_t.device)
            x1_hat = self._predict_clean(x_k, tau, action_emb)
            x_k = x_k + (1.0 / N) * (x1_hat - x_0)
            traj.append(x_k.detach().cpu())
        return x_k, traj
