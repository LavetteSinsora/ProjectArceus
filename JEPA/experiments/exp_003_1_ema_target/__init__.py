"""
Exp-003-1: EMA Target Encoder + Placeholder Gradient Fix.

Changes from exp_003:
  1. EMA target encoder (I-JEPA style): H_{T+1} targets are computed by a
     momentum-updated copy of the online encoder, not the online encoder itself.
     This breaks the chicken-and-egg collapse cycle: even if the online encoder
     starts collapsing, the target encoder lags behind and still provides
     non-degenerate targets.

  2. Placeholder gradient fix: the initial-step latent query (placeholder) now
     receives gradient during training. In exp_003, placeholders were frozen because
     the training update used stored numpy arrays (h_query_np) rather than the live
     nn.Parameter, severing the gradient path. Now, transitions that used placeholder
     queries are detected via the is_initial flag in the buffer, and the live
     encoder.perceiver.placeholders parameter is used directly in the forward pass.

  3. Stop-gradient verification: H_{T+1} comes from target_encoder (separate, EMA
     copy, never receives gradient). H_T comes from online encoder (gradient flows).
     This is the correct asymmetric setup from BYOL / I-JEPA.
"""

# The architecture is identical to exp_003_0 (only the target encoder is added,
# unused at inference). Reuse exp_003_0's dashboard panel.js so the diagnostic
# panels render the same way.
PANEL_EXPERIMENT = "exp_003_0_normalized_latent_jepa"
