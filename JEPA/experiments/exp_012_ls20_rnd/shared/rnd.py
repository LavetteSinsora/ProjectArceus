"""Random Network Distillation: frozen random target, trainable predictor, and
the two running normalisers RND needs to work.

Faithful to Burda, Edwards, Storkey, Klimov 2018 (arXiv:1810.12894), with the
LS20-forced deviations documented in the system card:

  * the conv backbone is the exp_010 / "7_0" CNNEncoder (not the Nature CNN),
    output dim = trunk_dim (256, paper uses 512);
  * input is the same one-hot (16,64,64) frame the policy sees, so the RND
    networks have their OWN encoders (NOT the shared, non-stationary policy
    encoder); the pixel (x-mu)/sigma normalisation is dropped because one-hot
    input is already bounded.

Kept exactly: intrinsic reward = 1/2 ||f_hat(s') - f(s')||^2, predictor trained
to regress the frozen target, and the intrinsic reward normalised by a running
estimate of the std of the intrinsic *returns* (RewardForwardFilter + RMS).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import (
    CNNEncoder, one_hot_frame, _orth, N_COLORS, FRAME_SIZE, TRUNK_DIM,
)


class RNDTarget(nn.Module):
    """Fixed, randomly-initialised embedding of the frame. Never trained."""

    def __init__(self, n_colors: int = N_COLORS, frame_size: int = FRAME_SIZE,
                 feature_dim: int = TRUNK_DIM):
        super().__init__()
        self.n_colors = n_colors
        self.encoder = CNNEncoder(n_colors=n_colors, frame_size=frame_size,
                                  trunk_dim=feature_dim)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        return self.encoder(one_hot_frame(obs_uint8, self.n_colors))


class RNDPredictor(nn.Module):
    """Trainable network distilling the target. Higher capacity than the target
    (extra FC head) so it *can* fit the random features on visited states."""

    def __init__(self, n_colors: int = N_COLORS, frame_size: int = FRAME_SIZE,
                 feature_dim: int = TRUNK_DIM, hidden: int = TRUNK_DIM):
        super().__init__()
        self.n_colors = n_colors
        self.encoder = CNNEncoder(n_colors=n_colors, frame_size=frame_size,
                                  trunk_dim=feature_dim)
        self.head = nn.Sequential(
            _orth(nn.Linear(feature_dim, hidden), 2 ** 0.5), nn.ReLU(inplace=True),
            _orth(nn.Linear(hidden, hidden), 2 ** 0.5), nn.ReLU(inplace=True),
            _orth(nn.Linear(hidden, feature_dim), 2 ** 0.5),
        )

    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(one_hot_frame(obs_uint8, self.n_colors)))


@torch.no_grad()
def batched_features(net: nn.Module, obs_uint8: torch.Tensor, device,
                     feature_dim: int, chunk: int = 4096) -> torch.Tensor:
    """Run `net` over a flat (M, H, W) frame batch in chunks, no grad. Returns a
    CPU tensor (M, feature_dim). Used to embed next-states with the frozen
    target ONCE per rollout (cached and reused in every PPO minibatch instead
    of recomputing the frozen target 64x/update) and with the predictor for the
    intrinsic reward."""
    M = obs_uint8.shape[0]
    out = torch.empty(M, feature_dim, dtype=torch.float32)
    for s in range(0, M, chunk):
        o = obs_uint8[s:s + chunk].to(device)
        out[s:s + chunk] = net(o).detach().to("cpu")
    return out


def intrinsic_from_features(pred_feats: torch.Tensor, target_feats: torch.Tensor) -> np.ndarray:
    """Raw (un-normalised) intrinsic reward i = 1/2 mean_j (f_hat - f)^2 from
    precomputed predictor/target feature batches (M, D) -> (M,) numpy."""
    return (0.5 * (pred_feats - target_feats).pow(2).mean(dim=-1)).numpy()


class RunningMeanStd:
    """Parallel (Welford / Chan) running mean & variance of a scalar stream."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = x.size
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))


class RewardForwardFilter:
    """Tracks the running discounted intrinsic return per env, used to feed the
    RMS that estimates the std of the intrinsic *returns* (RND's normaliser)."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.rewems: np.ndarray | None = None

    def update(self, rews: np.ndarray) -> np.ndarray:
        rews = np.asarray(rews, dtype=np.float64)
        if self.rewems is None:
            self.rewems = rews.copy()
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
