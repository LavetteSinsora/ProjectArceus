import torch

from .encoder import Encoder
from .predictor import FlowMatchingPredictor
from .policy import PolicyNetwork, REINFORCEBaseline
from JEPA.shared.action_embed import ActionEmbedding

CAPABILITIES: dict = {
    "has_encoder_attention": False,
    "has_perceiver_attention": True,
    "has_patch_embeddings": False,
    "has_latent_vectors": True,
    "has_flow_matching": True,
    "n_latents": 4,
    "n_patches": 16,
    "n_dims_per_latent": 128,
}


def load_models(cfg, device: torch.device):
    encoder = Encoder(
        d_model=cfg.d_model,
        d_color=cfg.d_color,
        n_sa_heads=cfg.n_sa_heads,
        n_sa_blocks=cfg.n_sa_blocks,
        sa_ffn_dim=cfg.sa_ffn_dim,
        patch_size=cfg.patch_size,
        n_latents=cfg.n_latents,
        n_placeholders=cfg.n_placeholders,
        n_perceiver_rounds=cfg.n_perceiver_rounds,
        n_perceiver_heads=cfg.n_perceiver_heads,
        perceiver_ffn_dim=cfg.perceiver_ffn_dim,
        rope_theta=cfg.rope_theta,
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
    ).to(device)

    predictor = FlowMatchingPredictor(
        n_latents=cfg.n_latents,
        d_model=cfg.d_model,
        d_action=cfg.d_action,
        time_emb_dim=cfg.time_emb_dim,
        time_proj_dim=cfg.time_proj_dim,
        hidden_dim=cfg.predictor_hidden,
        n_ode_steps=cfg.n_ode_steps,
    ).to(device)

    action_embed = ActionEmbedding(cfg.n_actions, cfg.d_action).to(device)
    policy = PolicyNetwork(cfg.d_model, cfg.n_actions, cfg.policy_hidden).to(device)
    baseline = REINFORCEBaseline(cfg.policy_baseline_alpha)

    return encoder, predictor, action_embed, policy, baseline


__all__ = [
    "Encoder", "FlowMatchingPredictor", "PolicyNetwork",
    "REINFORCEBaseline", "ActionEmbedding", "load_models", "CAPABILITIES",
]
