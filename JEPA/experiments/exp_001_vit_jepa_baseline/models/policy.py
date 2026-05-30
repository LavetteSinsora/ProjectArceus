import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Recurrent policy using a cross-attention reasoning token.

    The reasoning token h_t is a d_model-dim hidden state that persists across time steps.
    At each step it attends (via single-head cross-attention) to the current patch embeddings
    from the encoder, producing h_{t+1}. A linear head maps h_{t+1} to action logits.

    Recurrent update rule:
        Q_t  = Linear_Q(h_{t-1})
        K_t, V_t = Linear_K(z_t), Linear_V(z_t)   # z_t from encoder (stop-grad)
        h_t  = LayerNorm(h_{t-1} + CrossAttn(Q_t, K_t, V_t))
        h_t  = LayerNorm(h_t + FFN(h_t))
        a_t  ~ Categorical(softmax(Linear(h_t)))

    Training note:
        h is detached before each policy.act() call in the training loop to prevent
        BPTT through the recurrent chain. Gradients flow only through the single-step
        forward pass (Q/K/V projections, FFN, action head) — equivalent to treating
        the reasoning token as a better state representation rather than a full RNN.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_actions: int = 4,
        ffn_dim: int = 256,
        attn_gain_init: float = 4.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.scale = d_model ** -0.5

        # Learned initial reasoning token (reset at episode start)
        self.h0 = nn.Parameter(torch.zeros(d_model))

        # Cross-attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # Per-dim gate on attn_out to compensate for the L2-norm constraint on z
        # making ‖attn_out‖ ~ O(1) vs ‖h‖ ~ √d_model. Without this, h is stale.
        self.attn_gain = nn.Parameter(torch.ones(d_model) * attn_gain_init)
        self.norm1 = nn.LayerNorm(d_model)

        # Post-attention FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Action readout
        self.action_head = nn.Linear(d_model, n_actions)

    def load_state_dict_with_legacy_gain(self, state_dict: dict) -> None:
        """Load a policy state dict, defaulting attn_gain to 1.0 if absent.

        Pre-fix checkpoints have no `attn_gain` parameter; their effective behavior
        was equivalent to a unit gain. New checkpoints carry the learned vector.
        """
        if "attn_gain" not in state_dict:
            with torch.no_grad():
                self.attn_gain.fill_(1.0)
        self.load_state_dict(state_dict, strict=False)

    def initial_state(self) -> torch.Tensor:
        """Return a fresh reasoning token (detached, ready for a new episode)."""
        return self.h0.detach().clone()

    def _cross_attn_update(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Single cross-attention step + FFN, operating on unbatched tensors.

        h: (d_model,)   — current reasoning token
        z: (16, d_model) — patch embeddings (stop-grad in practice)
        returns: h_new (d_model,)
        """
        # Unsqueeze to (1, 1, d_model) and (1, 16, d_model) for matmul
        Q = self.q_proj(h).unsqueeze(0)          # (1, d_model)
        K = self.k_proj(z)                        # (16, d_model)
        V = self.v_proj(z)                        # (16, d_model)

        # Scaled dot-product: (1, 16) → (1, d_model)
        attn_w = F.softmax(Q @ K.T * self.scale, dim=-1)  # (1, 16)
        attn_out = self.out_proj((attn_w @ V).squeeze(0))  # (d_model,)

        h_new = self.norm1(h + self.attn_gain * attn_out)
        h_new = self.norm2(h_new + self.ffn(h_new))
        return h_new

    def act(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        available_actions: list = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action for one environment step.

        h: (d_model,) — current reasoning token (should be detached before call)
        z: (16, d_model) — encoder output for current frame (stop-grad)
        available_actions: list of ints (1-indexed), or None to allow all
        returns: (action_idx, log_prob, h_new)
          action_idx: int, 0-indexed
          log_prob: scalar tensor with gradient (for REINFORCE)
          h_new: updated reasoning token tensor (caller should detach before next step)
        """
        h_new = self._cross_attn_update(h, z)
        logits = self.action_head(h_new)  # (n_actions,)

        # Mask unavailable actions (ACTION values are 1-indexed; map to 0-indexed)
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
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()   # H(π) — needed for entropy regularisation

        return action_idx.item(), log_prob, h_new, entropy
