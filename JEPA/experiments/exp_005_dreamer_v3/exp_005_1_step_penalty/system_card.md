# exp_005_1_step_penalty

DV3 on LS20 L1 with reward `+1·level_completed − 0.01·step`.

Tests whether a tiny dense penalty (a) accelerates solves once the goal is
discovered, or (b) destabilises the critic's percentile-return scaling (in
which case Per95 ≈ Per5 ≈ small negative number → S small → advantage
amplified).  Both outcomes are informative.

Otherwise identical to `exp_005_0_sparse_goal`. See that system card for
shared spec; only `step_penalty=0.01` differs.
