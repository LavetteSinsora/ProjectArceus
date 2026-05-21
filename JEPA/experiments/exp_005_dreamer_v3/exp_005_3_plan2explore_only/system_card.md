# exp_005_3_plan2explore_only

Pure Plan2Explore on LS20 Level 1. No extrinsic reward at all; the only
training signal for the actor/critic is the K=8 ensemble's per-step
disagreement on next-z.

**Why run this:** it's the cleanest diagnostic for *exploration capacity* on
LS20 L1. If π_e (the P2E exploration actor) never visits the goal cluster
over 500K env steps, then no exploration-augmented DV3 variant we try will
either, and the cause of sub-exp A failing (if it does) is exploration, not
algorithm.  If π_e *does* repeatedly reach the goal but the task actor in
sub-exps A/B/C fails to learn from those visits, the problem is upstream —
the critic / λ-return / percentile-scale machinery isn't propagating the
sparse signal.

## Differences from sub-exp A
- `p2e_acting_steps = 10M` → π_e is the acting policy for the whole run.
- Env reward is always 0; the buffer's extrinsic-reward column is uniformly 0.
  (The world-model reward head still trains, just on a trivial constant.)
- Evaluation uses π_e instead of π_t.

## Stretch use
The trained world model from this run can be a useful checkpoint to *start*
sub-exp A from — i.e. fine-tune the task actor on a pre-explored buffer.
That's not wired up yet but the on-disk checkpoint format supports it.
