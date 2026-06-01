"""exp_012_1_rnd_l2 — continue the L1-solved RND agent into LS20 Level 2.

Warm-starts the policy + RND target + (L1-distilled) predictor from the
exp_012_1_rnd_baseline L1 checkpoint, then trains with stop_levels=2 so each
episode must clear L1 *then* L2 (incremental +1 reward per level). Because the
predictor already treats L1 states as familiar (intrinsic ~ 0), the exploration
bonus is concentrated on the novel L2 states.
"""
