"""Exp-003-3 encoder — re-export exp_003_0 (architecture unchanged)."""

from JEPA.experiments.exp_003_0_normalized_latent_jepa.models.encoder import (  # noqa: F401
    Encoder,
    PerceiverResampler,
    SelfAttentionBlock,
    _CrossAttentionBlock,
    _PerceiverRound,
    _SelfAttentionAmongLatents,
    apply_rope_2d,
)
