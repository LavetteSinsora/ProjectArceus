"""Intrinsic Curiosity Module (Pathak, Agrawal, Efros, Darrell, ICML 2017).

Faithful replication of the ICM mechanism, adapted to this project's backbone
(see exp_011_0_icm_baseline/SYSTEM_CARD.md §3-§7 for the deviation table):

  * phi encoder   — a SEPARATE instance of the exp_010 CNN encoder, trained ONLY
                    by the inverse+forward losses (no PPO gradient ever touches
                    it; the policy never touches it). This is the paper's design.
  * inverse model g(phi_t, phi_{t+1}) -> action logits      L_inv = CE        (3)
  * forward model f(phi_t, onehot(a_t)) -> phi_hat_{t+1}     L_fwd = 1/2||.||^2 (5)
  * intrinsic reward  r^i_t = (eta/2)||phi_hat_{t+1} - phi(s_{t+1})||^2         (6)
  * ICM loss          L = (1-beta)*L_inv + beta*L_fwd,  beta=0.2               (7)

The action is fed to the forward model as a one-hot vector (faithful), not a
learned embedding. Inverse/forward heads use ReLU (the codebase activation;
the paper used ELU — a cosmetic deviation, §7 item 3).

Compute notes (the "maximise compute" brief): intrinsic-reward computation is a
single chunked, batched, no_grad pass on the device; the ICM update reuses the
exact rollout tensors PPO already collected (no extra env interaction). Both
exclude episode-ending transitions, whose s_{t+1} is a reset frame.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the exp_010 CNN encoder + one-hot helper verbatim — phi has identical
# capacity to the policy encoder (SYSTEM_CARD §4.2).
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import (
    CNNEncoder, one_hot_frame, N_COLORS, FRAME_SIZE, N_ACTIONS, TRUNK_DIM,
)


def _orth(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)
    return layer


class ICMModule(nn.Module):
    """Separate phi encoder + inverse + forward dynamics models."""

    def __init__(self, n_actions: int = N_ACTIONS, n_colors: int = N_COLORS,
                 frame_size: int = FRAME_SIZE, trunk_dim: int = TRUNK_DIM,
                 hidden: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.n_colors = n_colors
        self.trunk_dim = trunk_dim

        # phi: its own CNN encoder (NOT shared with the policy).
        self.phi = CNNEncoder(n_colors=n_colors, frame_size=frame_size,
                              trunk_dim=trunk_dim)

        # Inverse model g: [phi_t ; phi_{t+1}] -> action logits.
        self.inverse = nn.Sequential(
            _orth(nn.Linear(2 * trunk_dim, hidden), 2 ** 0.5),
            nn.ReLU(inplace=True),
            _orth(nn.Linear(hidden, n_actions), 0.01),
        )
        # Forward model f: [phi_t ; onehot(a_t)] -> phi_hat_{t+1}.
        self.forward_model = nn.Sequential(
            _orth(nn.Linear(trunk_dim + n_actions, hidden), 2 ** 0.5),
            nn.ReLU(inplace=True),
            _orth(nn.Linear(hidden, trunk_dim), 2 ** 0.5),
        )

    # ── encoders / heads ────────────────────────────────────────────────────

    def encode(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        """(B,H,W) palette indices -> (B, trunk_dim) phi feature."""
        return self.phi(one_hot_frame(obs_uint8, self.n_colors))

    def predict_next(self, phi_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        a_oh = F.one_hot(actions.long(), self.n_actions).float()
        return self.forward_model(torch.cat([phi_t, a_oh], dim=-1))

    def inverse_logits(self, phi_t: torch.Tensor, phi_next: torch.Tensor) -> torch.Tensor:
        return self.inverse(torch.cat([phi_t, phi_next], dim=-1))

    # ── losses on a minibatch (for the ICM update) ──────────────────────────

    def losses_on_batch(self, obs, next_obs, actions):
        """obs/next_obs: (B,H,W) uint8 on device; actions: (B,) int64 on device.
        Returns (L_inverse, L_forward, inverse_acc, mean_forward_error)."""
        phi_t = self.encode(obs)
        phi_next = self.encode(next_obs)
        # forward: target phi_next is detached (curiosity error must not train
        # the encoder to make itself trivially predictable through the target).
        phi_hat = self.predict_next(phi_t, actions)
        l_fwd = 0.5 * (phi_hat - phi_next.detach()).pow(2).sum(-1).mean()
        inv_logits = self.inverse_logits(phi_t, phi_next)
        l_inv = F.cross_entropy(inv_logits, actions)
        with torch.no_grad():
            inv_acc = (inv_logits.argmax(-1) == actions).float().mean().item()
            mean_err = (phi_hat - phi_next).pow(2).sum(-1).mean().item()
        return l_inv, l_fwd, inv_acc, mean_err


# ── intrinsic reward over a full rollout (no grad, chunked & batched) ────────

@torch.no_grad()
def intrinsic_raw_error(icm: ICMModule, rollout, device, chunk: int = 256):
    """Per-transition forward-prediction error ||phi_hat - phi(s')||^2.

    Returns (raw (T,N) float32 on CPU, mean_raw_over_valid float). The error on
    episode-ending steps (dones==True) is zeroed — their s_{t+1} is a reset
    frame, so the forward error there is meaningless (SYSTEM_CARD §4.5/§5.2).
    """
    T, N = rollout.actions.shape
    Fz = rollout.frame
    obs = rollout.obs.reshape(-1, Fz, Fz)
    next_obs = rollout.next_obs.reshape(-1, Fz, Fz)
    actions = rollout.actions.reshape(-1)
    M = obs.shape[0]

    raw = torch.zeros(M, dtype=torch.float32)
    for s in range(0, M, chunk):
        o = obs[s:s + chunk].to(device)
        no = next_obs[s:s + chunk].to(device)
        a = actions[s:s + chunk].to(device)
        phi_t = icm.encode(o)
        phi_next = icm.encode(no)
        phi_hat = icm.predict_next(phi_t, a)
        raw[s:s + chunk] = (phi_hat - phi_next).pow(2).sum(-1).cpu()

    raw = raw.reshape(T, N)
    valid = ~rollout.dones                       # (T,N) bool
    raw = raw * valid.float()
    n_valid = int(valid.sum().item())
    mean_raw = float(raw.sum().item() / max(1, n_valid))
    return raw, mean_raw


def icm_update_from_rollout(icm: ICMModule, optimizer, rollout, cfg, device) -> dict:
    """Train the ICM (phi + inverse + forward) on the rollout's transitions with
    its OWN optimiser (PPO is untouched). Mirrors exp_010's
    jepa_update_from_rollout: excludes episode-ending steps, minibatched."""
    T, N = rollout.actions.shape
    Fz = rollout.frame
    valid = (~rollout.dones).reshape(-1)
    obs = rollout.obs.reshape(-1, Fz, Fz)[valid]
    next_obs = rollout.next_obs.reshape(-1, Fz, Fz)[valid]
    actions = rollout.actions.reshape(-1)[valid]
    n = obs.shape[0]
    if n == 0:
        return {"forward_loss": float("nan"), "inverse_loss": float("nan"),
                "inverse_acc": float("nan"), "forward_error_mean": float("nan")}

    mb = max(1, n // cfg.minibatches)
    idx = np.arange(n)
    fl = il = ia = me = 0.0
    steps = 0
    for _ in range(cfg.icm_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, mb):
            sel = idx[start:start + mb]
            o = obs[sel].to(device)
            no = next_obs[sel].to(device)
            a = actions[sel].to(device)
            l_inv, l_fwd, acc, err = icm.losses_on_batch(o, no, a)
            loss = (1.0 - cfg.beta) * l_inv + cfg.beta * l_fwd
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(icm.parameters(), cfg.grad_clip)
            optimizer.step()
            fl += l_fwd.item(); il += l_inv.item(); ia += acc; me += err; steps += 1
    return {"forward_loss": fl / steps, "inverse_loss": il / steps,
            "inverse_acc": ia / steps, "forward_error_mean": me / steps}
