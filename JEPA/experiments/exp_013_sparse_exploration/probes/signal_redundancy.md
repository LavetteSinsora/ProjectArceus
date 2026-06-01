# Probe: ICM+RND signal redundancy / informativeness (exp_013_2 kill-or-confirm)

**Question.** Before spending a 500k-step w-sweep on the additive idea
(`r = w·norm(ICM_fwd_err) + (1−w)·norm(RND_on_φ)`, exp_013_2), is that idea even
viable? Kill-or-confirm two failure modes: the two intrinsic signals are (a)
**REDUNDANT** (additive adds nothing) and/or (b) **UNINFORMATIVE** (they don't
track real state-novelty → noise).

**Method.** `probes/signal_redundancy.py` replicates the exp_013_2 additive loop
(w=0.5, c_entropy=0.05) using the audited helpers unchanged — `_phi_and_novelty`
+ `_SignalNorm` (exp_013_2's exact per-signal normaliser) for RND-on-φ, and
`intrinsic_raw_error` (exp_011 icm.py) + `_SignalNorm` for ICM forward error. No
method/harness code was modified. 28 updates × 2048 transitions = **57,344 env
steps** on **ls20 L2 (level_index=1)**, seed 0, MPS, single process.

Per update, over the rollout's **non-done** transitions, I compute `n_icm`,
`n_rnd`, and a ground-truth visitation count from a **global exact-frame hash of
each next-frame**. `novelty_target = −log(count)` (count taken BEFORE
incrementing). A signal that tracks true novelty should be **POSITIVELY**
correlated with `novelty_target` (high reward on rarely-seen states).

**Oracle correction (important).** LS20 frames carry a step-timer on rows 61–62
that marches 1 cell every step **regardless of the agent's action** (verified:
those rows change on 60/60 fixed-action steps). Raw-byte hashing therefore makes
every in-episode frame look unique (timer phase), inflating the visit oracle and
hiding wall-bump no-ops (raw-hash run reported noop_frac = 0.000, 1073 "unique"
states). The probe **masks rows 60–63 before hashing and before the no-op test**,
yielding a clean agent-state oracle. This is a probe-side correction only.

## Results (masked oracle), n = 52,436 pooled non-done transitions

| Metric | Pooled | Mean / update |
|---|---|---|
| **Redundancy** corr(n_icm, n_rnd) | **+0.807** | +0.217 |
| **Inform.** corr(n_icm, −log count) | **−0.456** | −0.161 |
| **Inform.** corr(n_rnd, −log count) | **−0.556** | −0.316 |
| **No-op fraction** (masked wall-bumps) | **0.488** | 0.484 |
| Unique masked next-states reached | — | **43** |

Per-update redundancy trajectory (post-warmup, 25 updates): min −0.37, **max 0.95
(update 11)**, median **0.24**; ≥0.8 on 1/25 updates, ≥0.5 on 6/25. The pooled
+0.807 is inflated by the no-op block structure (both signals are jointly low on
the ~49% no-op transitions, jointly higher on the moves), so the honest
*within-update* redundancy is moderate (~0.24), NOT ~1.0.

Per-update informativeness is negative on the large majority of updates
(corr_rnd_tgt reaches −0.61…−0.75 on updates 18–22) and only turns slightly
positive in the final 2 updates (+0.45 / +0.30), as the policy/φ briefly shift.

## Interpretation

1. **Informativeness is BACKWARDS, not just weak.** Both signals correlate
   **negatively** with true novelty (−0.46 ICM, −0.56 RND): the additive reward
   is, on average, *higher on MORE-visited states*. This is the dominant failure.
   The cause is visible in the other two numbers: **48.8% of transitions are
   wall-bump no-ops** and the agent only ever reaches **43 distinct board states**
   in 52k steps. The reward is keying on something other than state-rarity (φ is
   near-uninformative / not controllable here — consistent with the exp_013
   SYSTEM_CARD §9 "φ not controllable → RND ruler near-random" caveat and the
   inv_acc never clearing the 0.90 freeze threshold; it ends at ~0.76). A signal
   anti-correlated with real novelty will, if anything, push the policy AWAY from
   unexplored states.

2. **Redundancy is moderate within an update (~0.24), not ~1.0.** So the two
   signals are not literally the same number. But this does NOT rescue the idea:
   the two signals being *different* is only useful if at least one of them tracks
   novelty, and **neither does** (both informativeness corrs are negative).
   Combining two non-informative-or-anti-informative signals with a weight `w`
   cannot produce an informative one — there is no `w ∈ [0,1]` that makes
   `w·(−0.46) + (1−w)·(−0.56)` positive.

## VERDICT: additive hypothesis is DEAD — skip the 500k w-sweep.

The kill-criterion is met via the informativeness arm: **both** corr(signal,
novelty_target) are ≈0-or-negative (−0.46 and −0.56), i.e. neither intrinsic
signal tracks true state-novelty on L2 — they are noise w.r.t. the visitation
oracle (driven by the ~49% no-op / 43-state coverage collapse and an
uncontrollable φ). A convex mixture of two non-informative signals stays
non-informative for every weight `w`, so the 500k w-sweep has **no headroom** to
find: it would be sweeping the mixing weight of two rulers that don't measure
novelty. Recommend **not** running it.

**Recommended pivot instead:** the binding constraint is φ-controllability, not
the ICM-vs-RND mix. Both signals live in φ-space, and φ here is near-chance
(holdout inv_acc never reaches the 0.90 freeze gate; ~half of transitions are
no-ops with no agent-state change to predict). Before any directed-exploration
reward can work on L2, φ must become controllable — e.g. fix the no-op /
controllability problem (action-conditioned masking, longer/curriculum φ
pre-training, or a pixel-space count/RND that the timer-mask shows DOES alias to
43 states and so would give a clean −log(count) target). A w-sweep over two
broken rulers is not the experiment to run.

### Reproduce
```
uv run python -m JEPA.experiments.exp_013_sparse_exploration.probes.signal_redundancy 1
# arg = level_index (1 = ls20 L2, default). Writes results/signal_redundancy_L2.json
```
Raw numbers: `probes/results/signal_redundancy_L2.json`.
