# exp_005_2_curiosity

Sub-exp C: DV3 sparse extrinsic + Plan2Explore-style curiosity.

In the current trainer, the P2E ensemble + exploration actor are always
trained (across all sub-exps) but the **task** actor sees only extrinsic.
This sub-exp's `p2e_intrinsic_weight > 0` is a forward-compatible hook: once
the trainer mixes intrinsic into the task actor's λ-returns (planned
enhancement), this config drives that ablation.

Today, behaviour is effectively equivalent to `exp_005_0_sparse_goal`, but
the run dir and config remain distinct so comparison plots stay clean once
the trainer change lands.
