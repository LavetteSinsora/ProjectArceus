"""
Exp-002 Policy Network.

Stateless MLP: flattens 4 latent vectors → 1-hidden-layer MLP → action logits.
No persistent hidden state (unlike exp-001's cross-attention reasoning token).

Training: REINFORCE with EMA running-mean baseline.
  loss = −(R − baseline) · log π(a|h_t) − λ_H · H(π)
  baseline is an exponential moving average of rewards, updated outside this module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Input:  4 latent vectors (B, 4, d_model) → flatten → (B, 4*d_model)
    Hidden: Linear(4*d_model, policy_hidden) → GELU
    Output: Linear(policy_hidden, n_actions) → logits
    """

    def __init__(self, d_model: int = 128, n_actions: int = 4, policy_hidden: int = 512):
        super().__init__()
        self.n_actions = n_actions
        in_dim = 4 * d_model  # 4 latents × 128 = 512

        self.net = nn.Sequential(
            nn.Linear(in_dim, policy_hidden),
            nn.GELU(),
            nn.Linear(policy_hidden, n_actions),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: (B, 4, d_model)
        Returns: (B, n_actions) logits
        """
        B = latents.shape[0]
        x = latents.view(B, -1)  # (B, 4*d_model)
        return self.net(x)

    def act(
        self,
        latents: torch.Tensor,
        available_actions: list = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action for a single environment step.

        latents:           (4, d_model) — single-sample latents (no batch dim)
        available_actions: list of 1-indexed ints, or None for all

        Returns: (action_idx, log_prob, entropy)
          action_idx: int, 0-indexed
          log_prob:   scalar tensor (has gradient for REINFORCE)
          entropy:    scalar tensor H(π) (for entropy regularisation)
        """
        logits = self.forward(latents.unsqueeze(0)).squeeze(0)  # (n_actions,)

        if available_actions is not None and len(available_actions) > 0:
            mask = torch.full_like(logits, float("-inf"))
            for a in available_actions:
                idx = int(a) - 1
                if 0 <= idx < self.n_actions:
                    mask[idx] = 0.0
            logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        return action_idx.item(), dist.log_prob(action_idx), dist.entropy()


class REINFORCEBaseline:
    """
    Exponential moving average baseline for REINFORCE variance reduction.
    Stored outside the policy network (no parameters).
    """

    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self._baseline: float = 0.0
        self._initialised: bool = False

    def update(self, reward: float) -> None:
        if not self._initialised:
            self._baseline = reward
            self._initialised = True
        else:
            self._baseline = self.alpha * self._baseline + (1.0 - self.alpha) * reward

    @property
    def value(self) -> float:
        return self._baseline

    def reset(self) -> None:
        self._baseline = 0.0
        self._initialised = False
