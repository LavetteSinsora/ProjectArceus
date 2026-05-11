"""
Exp-002 Flow Matching Predictor (x0-parameterisation).

Architecture: 4 separate MLPs, one per latent vector.
Each MLP input: cat([x_tau_i, action_emb, time_feat]) where time_feat is a
sinusoidal time embedding projected to time_proj_dim.

x0-parameterisation:
  MLP outputs x̂_1  (predicted clean next-state latent)
  Velocity: v̂ = x̂_1 − x_0   (subtract SOURCE, not x_tau)
  Training loss: ||x_1 − x̂_1||²  (MSE averaged over d_model)
  This simplifies ||u* − v̂||² = ||(x_1−x_0) − (x̂_1−x_0)||² = ||x_1−x̂_1||²

Euler ODE rollout (n_ode_steps steps):
  x_0 = h_{t,i}  (current latent)
  for k in range(n_ode_steps):
      tau = k / n_ode_steps
      x̂_1 = MLP_i(x_k, tau, a)
      v̂   = x̂_1 − x_0
      x_{k+1} = x_k + (1/n_ode_steps) * v̂
  predicted_h_{t+1,i} = x_N
"""

import math
import torch
import torch.nn as nn


# ── Sinusoidal time embedding ─────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Embeds scalar tau ∈ [0,1] via sinusoidal encoding, then projects to time_proj_dim.
    Output is CONCATENATED (not FiLM-modulated) because the predictor has depth 1
    and concatenation is equivalent in expressiveness at that depth.
    """

    def __init__(self, time_emb_dim: int = 128, time_proj_dim: int = 512):
        super().__init__()
        assert time_emb_dim % 2 == 0
        self.time_emb_dim = time_emb_dim
        self.proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_proj_dim),
            nn.GELU(),
        )

    def _sinusoidal(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (B,) float in [0,1] → (B, time_emb_dim)"""
        half = self.time_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / half
        )
        angles = tau.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)   # (B, time_emb_dim)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (B,) → (B, time_proj_dim)"""
        emb = self._sinusoidal(tau)
        return self.proj(emb)


# ── Single latent MLP ─────────────────────────────────────────────────────────

class _LatentMLP(nn.Module):
    """
    Single MLP for one latent index.
    Input: cat([x_tau_i, action_emb, time_feat]) = d_model + d_action + time_proj_dim
    Output: x̂_1  (predicted clean next-state latent, d_model)
    """

    def __init__(self, d_model: int, d_action: int, time_proj_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = d_model + d_action + time_proj_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x_tau: torch.Tensor, action_emb: torch.Tensor,
                time_feat: torch.Tensor) -> torch.Tensor:
        """
        x_tau:      (B, d_model)
        action_emb: (B, d_action)
        time_feat:  (B, time_proj_dim)
        returns:    (B, d_model) — x̂_1
        """
        inp = torch.cat([x_tau, action_emb, time_feat], dim=-1)
        return self.net(inp)


# ── Full Predictor ────────────────────────────────────────────────────────────

class FlowMatchingPredictor(nn.Module):
    """
    4 separate MLPs (one per latent) with sinusoidal time conditioning.

    Training:
      call compute_loss(h_t, h_t1, action_emb) → scalar loss
      tau is sampled internally from U[0,1]

    Rollout:
      call predict(h_t, action_emb) → predicted h_{t+1}  (n_ode_steps Euler steps)
    """

    def __init__(
        self,
        n_latents: int = 4,
        d_model: int = 128,
        d_action: int = 32,
        time_emb_dim: int = 128,
        time_proj_dim: int = 512,
        hidden_dim: int = 512,
        n_ode_steps: int = 3,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model
        self.n_ode_steps = n_ode_steps

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim, time_proj_dim)
        self.mlps = nn.ModuleList([
            _LatentMLP(d_model, d_action, time_proj_dim, hidden_dim)
            for _ in range(n_latents)
        ])

    def _predict_clean(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_tau:      (B, n_latents, d_model)
        tau:        (B,)
        action_emb: (B, d_action)
        Returns:    (B, n_latents, d_model) — predicted x̂_1 for each latent
        """
        time_feat = self.time_embed(tau)   # (B, time_proj_dim)
        preds = []
        for i, mlp in enumerate(self.mlps):
            pred_i = mlp(x_tau[:, i, :], action_emb, time_feat)  # (B, d_model)
            preds.append(pred_i.unsqueeze(1))
        return torch.cat(preds, dim=1)  # (B, n_latents, d_model)

    def compute_loss(
        self,
        h_t: torch.Tensor,
        h_t1: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> tuple:
        """
        Flow-matching training loss (x0-parameterisation, uniform tau sampling).

        h_t:        (B, n_latents, d_model) — current latents (x_0)
        h_t1:       (B, n_latents, d_model) — target next latents (x_1)
        action_emb: (B, d_action)

        Returns:
          loss:            scalar — mean MSE over all latents and batch
          per_latent_loss: (n_latents,) — per-latent MSE for monitoring
        """
        B = h_t.shape[0]

        # Sample tau ~ U[0, 1]
        tau = torch.rand(B, device=h_t.device)   # (B,)

        # Interpolate: x_tau = (1−tau)·x_0 + tau·x_1
        tau_exp = tau.view(B, 1, 1)
        x_tau = (1.0 - tau_exp) * h_t + tau_exp * h_t1   # (B, n_latents, d_model)

        # Predict x̂_1
        x1_hat = self._predict_clean(x_tau, tau, action_emb)  # (B, n_latents, d_model)

        # Loss = ||x_1 − x̂_1||² averaged over d_model (per-latent)
        sq_err = (h_t1 - x1_hat).pow(2)  # (B, n_latents, d_model)
        per_latent = sq_err.mean(dim=[0, 2])   # (n_latents,)
        loss = per_latent.mean()
        return loss, per_latent.detach()

    @torch.no_grad()
    def predict(self, h_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """
        Euler ODE rollout: predicts h_{t+1} from h_t using n_ode_steps steps.

        h_t:        (B, n_latents, d_model) — current latents
        action_emb: (B, d_action)
        Returns:    (B, n_latents, d_model) — predicted next latents
        """
        x_0 = h_t
        x_k = h_t.clone()
        N = self.n_ode_steps
        for k in range(N):
            tau_val = k / N
            tau = torch.full((h_t.shape[0],), tau_val, device=h_t.device)
            x1_hat = self._predict_clean(x_k, tau, action_emb)
            v_hat = x1_hat - x_0         # velocity = x̂_1 − x_0  (x0-parameterisation)
            x_k = x_k + (1.0 / N) * v_hat
        return x_k

    def predict_with_loss(
        self,
        h_t: torch.Tensor,
        h_t1: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> tuple:
        """
        Euler rollout + compute prediction error for reward signal.
        Returns (predicted_h_t1, per_latent_mse).
        Used in the training loop to get both the reward and the ODE output.
        """
        predicted = self.predict(h_t, action_emb)
        per_latent_mse = (h_t1 - predicted).pow(2).mean(dim=-1).mean(dim=0)  # (n_latents,)
        return predicted, per_latent_mse
