"""Exp-003 Policy — identical to exp_002 (stateless MLP + REINFORCE baseline)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    def __init__(self, d_model: int = 128, n_actions: int = 4, policy_hidden: int = 512):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(4 * d_model, policy_hidden),
            nn.GELU(),
            nn.Linear(policy_hidden, n_actions),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents.view(latents.shape[0], -1))

    def act(self, latents: torch.Tensor, available_actions: list = None
            ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        logits = self.forward(latents.unsqueeze(0)).squeeze(0)
        if available_actions:
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
