"""CNN encoder + actor-critic heads.

Game-agnostic: the only game-dependent quantities are `n_colors` (input
channels) and `n_actions` (policy head width), both passed in explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _orthogonal(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class CNNEncoder(nn.Module):
    """One-hot frame (n_colors, 64, 64) → hidden_dim feature vector."""

    def __init__(self, n_colors: int = 16, hidden_dim: int = 512,
                 frame_size: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            _orthogonal(nn.Conv2d(n_colors, 32, kernel_size=8, stride=4), gain=2 ** 0.5),
            nn.ReLU(inplace=True),
            _orthogonal(nn.Conv2d(32, 64, kernel_size=4, stride=2), gain=2 ** 0.5),
            nn.ReLU(inplace=True),
            _orthogonal(nn.Conv2d(64, 64, kernel_size=3, stride=1), gain=2 ** 0.5),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_colors, frame_size, frame_size)
            conv_out = self.conv(dummy).flatten(1).shape[1]
        self.fc = _orthogonal(nn.Linear(conv_out, hidden_dim), gain=2 ** 0.5)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(self.conv(x).flatten(1)))


class ActorCritic(nn.Module):
    """Shared CNN encoder feeding separate policy-logit and value heads."""

    def __init__(self, n_actions: int, n_colors: int = 16,
                 hidden_dim: int = 512, frame_size: int = 64):
        super().__init__()
        self.encoder = CNNEncoder(n_colors, hidden_dim, frame_size)
        self.policy_head = _orthogonal(nn.Linear(hidden_dim, n_actions), gain=0.01)
        self.value_head = _orthogonal(nn.Linear(hidden_dim, 1), gain=1.0)
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor):
        """Return (logits (B, n_actions), value (B,))."""
        feat = self.encoder(x)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)

    @torch.no_grad()
    def act(self, x: torch.Tensor):
        """Sample an action. Returns (action, log_prob, value), each (B,)."""
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor):
        """For PPO updates: returns (log_probs, entropy, values)."""
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class RecurrentActorCritic(nn.Module):
    """CNN encoder + GRU memory + policy/value heads.

    The GRU gives the policy a hidden state, so it can act differently on two
    *identical* observations depending on where it is in a plan. This is what
    a stateless policy cannot do — and it is required to reproduce a Go-Explore
    solution trajectory, which revisits states (e.g. traversing a corridor out
    and back) with different actions each time.
    """

    def __init__(self, n_actions: int, n_colors: int = 16,
                 hidden_dim: int = 512, frame_size: int = 64,
                 gru_dim: int = 256):
        super().__init__()
        self.encoder = CNNEncoder(n_colors, hidden_dim, frame_size)
        self.gru = nn.GRUCell(hidden_dim, gru_dim)
        self.policy_head = _orthogonal(nn.Linear(gru_dim, n_actions), gain=0.01)
        self.value_head = _orthogonal(nn.Linear(gru_dim, 1), gain=1.0)
        self.n_actions = n_actions
        self.gru_dim = gru_dim

    def initial_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.gru_dim,
                           device=self.policy_head.weight.device)

    def step(self, x: torch.Tensor, h: torch.Tensor):
        """One timestep. x: (B,C,H,W), h: (B,gru_dim).

        Returns (logits (B,n_actions), value (B,), new_h (B,gru_dim))."""
        h = self.gru(self.encoder(x), h)
        return self.policy_head(h), self.value_head(h).squeeze(-1), h

    def forward_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Run the GRU over a whole trajectory. frames: (T,C,H,W).

        Returns per-step policy logits (T, n_actions). Used for sequence
        behavior-cloning."""
        feats = self.encoder(frames)                 # (T, hidden_dim)
        h = torch.zeros(1, self.gru_dim, device=feats.device)
        logits = []
        for t in range(feats.shape[0]):
            h = self.gru(feats[t:t + 1], h)
            logits.append(self.policy_head(h))
        return torch.cat(logits, dim=0)              # (T, n_actions)
