"""
Exp-003-2 state predictor — flow-matching predictor for h_{t+1} | (h_t, a_t).

Architecturally identical to exp_003_0's FlowMatchingPredictor (§2.4 of the
system card matches that module exactly). We re-export it under the alias
`StatePredictor` to keep the dual-predictor terminology unambiguous inside
this experiment.
"""

from JEPA.experiments.exp_003_0_normalized_latent_jepa.models.predictor import (  # noqa: F401
    FlowMatchingPredictor as StatePredictor,
    SinusoidalTimeEmbedding,
)
