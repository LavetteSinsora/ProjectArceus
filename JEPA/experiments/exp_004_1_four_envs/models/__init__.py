"""
exp_004_1 models — re-exports exp_003_4 architectures (encoder, predictors,
policy) and assembles per-env wiring for joint training on all four envs.

Shared modules: encoder, state_predictor, action_predictor (5-way head).
Per-env modules: action_embed (Embedding(5, 32) each), policy + REINFORCE
baseline. The 4-action envs (LS20, TU93) never sample action index 4 because
the wrappers' `available_actions` mask it out at the policy.
"""

import torch

# Re-export the architecture from exp_003_4 — no changes needed.
from JEPA.experiments.exp_003_4_no_resampler_self_attn.models.encoder import Encoder
from JEPA.experiments.exp_003_4_no_resampler_self_attn.models.state_predictor import StatePredictor
from JEPA.experiments.exp_003_4_no_resampler_self_attn.models.action_predictor import ActionPredictor
from JEPA.experiments.exp_003_4_no_resampler_self_attn.models.policy import (
    PolicyNetwork, REINFORCEBaseline,
)

from JEPA.shared.action_embed import ActionEmbedding

CAPABILITIES: dict = {
    "has_encoder_attention": False,
    "has_perceiver_attention": True,
    "has_patch_embeddings": False,
    "has_latent_vectors": True,
    "has_flow_matching": True,
    "has_action_predictor": True,
    "multi_env": True,
    "n_envs": 4,
    "n_latents": 4,
    "n_patches": 16,
    "n_dims_per_latent": 128,
}


def load_models(cfg, device: torch.device):
    """
    Construct all trainable modules for exp_004_1.

    Returns a 6-tuple:
      encoder, state_predictor, action_predictor,
      action_embeds: dict[env_name -> ActionEmbedding(n_actions, d_action)],
      policies:      dict[env_name -> PolicyNetwork(d_model, n_actions, ...)],
      baselines:     dict[env_name -> REINFORCEBaseline].

    All per-env modules are sized to `cfg.n_actions = 5` (the max over envs).
    `available_actions` masking at policy time prevents 4-action envs from
    ever sampling the extra index.
    """
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

    state_predictor = StatePredictor(
        n_latents=cfg.n_latents,
        d_model=cfg.d_model,
        d_action=cfg.d_action,
        time_emb_dim=cfg.time_emb_dim,
        time_proj_dim=cfg.time_proj_dim,
        hidden_dim=cfg.predictor_hidden,
        n_ode_steps=cfg.n_ode_steps,
    ).to(device)

    action_predictor = ActionPredictor(
        n_latents=cfg.n_latents,
        d_model=cfg.d_model,
        hidden=cfg.action_predictor_hidden,
        n_actions=cfg.n_actions,
    ).to(device)

    action_embeds = {
        name: ActionEmbedding(cfg.n_actions, cfg.d_action).to(device)
        for name in cfg.env_names
    }
    policies = {
        name: PolicyNetwork(cfg.d_model, cfg.n_actions, cfg.policy_hidden).to(device)
        for name in cfg.env_names
    }
    baselines = {
        name: REINFORCEBaseline(cfg.policy_baseline_alpha)
        for name in cfg.env_names
    }

    return encoder, state_predictor, action_predictor, action_embeds, policies, baselines


__all__ = [
    "Encoder",
    "StatePredictor",
    "ActionPredictor",
    "PolicyNetwork",
    "REINFORCEBaseline",
    "ActionEmbedding",
    "load_models",
    "CAPABILITIES",
]
