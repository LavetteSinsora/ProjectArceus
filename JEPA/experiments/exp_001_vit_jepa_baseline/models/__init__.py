# =============================================================================
# EXPERIMENT-SPECIFIC: Update this file when changing model architecture.
#
# load_models() is called by the dashboard's debug_runner to instantiate models
# for any experiment without knowing its architecture in advance.
#
# CAPABILITIES tells the dashboard which visualization sections to show:
#   has_encoder_attention: True for ViT (attention maps), False for CNN
#   has_policy_attention:  True for cross-attention policy, False for MLP policy
#
# For exp_001: ViT encoder + per-patch MLP predictor + cross-attention policy.
# See system_card.md Section 2 for design rationale.
# =============================================================================

import copy
import torch

from .encoder import Encoder
from .predictor import Predictor
from .policy import PolicyNetwork
from JEPA.shared.action_embed import ActionEmbedding


# Dashboard reads this to decide which visualization sections to show.
CAPABILITIES: dict = {
    "has_encoder_attention": True,   # ViT has self-attention maps (Blocks 1 & 2)
    "has_policy_attention": True,    # Cross-attention reasoning token has attention weights
    "has_patch_embeddings": True,    # Patch-based model: per-patch embedding stats available
    "n_patches": 16,
    "extra": {},                     # Experiment-specific extras (e.g., CNN feature maps)
}


def load_models(cfg, device: torch.device):
    """
    Instantiate all models for this experiment from a Config object.

    Returns (encoder, target_encoder, predictor, action_embed, policy).
    The dashboard calls this dynamically — keep the return signature stable.
    """
    encoder = Encoder(
        cfg.d_model, cfg.d_color, cfg.n_heads, cfg.n_blocks, cfg.ffn_dim, cfg.patch_size
    ).to(device)
    target_encoder = copy.deepcopy(encoder)
    predictor = Predictor(cfg.d_model, cfg.d_action).to(device)
    action_embed = ActionEmbedding(cfg.n_actions, cfg.d_action).to(device)
    policy = PolicyNetwork(cfg.d_model, cfg.n_actions).to(device)
    return encoder, target_encoder, predictor, action_embed, policy


__all__ = ["Encoder", "Predictor", "PolicyNetwork", "load_models", "CAPABILITIES"]
