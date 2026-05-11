import copy
import torch

# All model classes are identical to exp_003 — import directly.
from JEPA.experiments.exp_003_0_normalized_latent_jepa.models import (
    Encoder,
    FlowMatchingPredictor,
    PolicyNetwork,
    REINFORCEBaseline,
    load_models,
    CAPABILITIES,
)
from JEPA.shared.action_embed import ActionEmbedding


def load_models_with_target(cfg, device: torch.device):
    """
    Returns (online_encoder, target_encoder, predictor, action_embed, policy, baseline).

    target_encoder is a deep copy of online_encoder with requires_grad=False.
    It is updated exclusively via EMA from shared/ema.py:update_ema().
    """
    encoder, predictor, action_embed, policy, baseline = load_models(cfg, device)

    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    return encoder, target_encoder, predictor, action_embed, policy, baseline


__all__ = [
    "Encoder", "FlowMatchingPredictor", "PolicyNetwork",
    "REINFORCEBaseline", "ActionEmbedding",
    "load_models", "load_models_with_target", "CAPABILITIES",
]
