# exp_016_0 naive leaky-RND: terminal entropy collapse in LS20-L1 vs no-collapse in TU93-L3

**Question.** What single, isolated force determines terminal entropy collapse (LS20-L1) vs
no-collapse (TU93-L3), and why does the L1 dip at ~u24–40 *recover* while the u46 one is *terminal*?

All numbers below are measured from the two runs' `metrics.jsonl` / `state_novelty.jsonl`
(no checkpoints needed; CPU only). Controlled REINFORCE micro-sims were used to *falsify*
candidate roots, not to assert them.

---

## THE ISOLATED DRIVER

**Coverage-saturation novelty decay that drives the z-scored reward persistently NEGATIVE,
relative to the never-decaying cumulative running mean.**
Concretely: the *sign* of `reward_norm_mean` (= sign of the gap `novelty_raw_mean − run_mean`)
is the single discriminator. When the intrinsic reward turns and *stays* negative, the
no-baseline REINFORCE objective, coupled to the shrinking visited-state distribution, becomes a
"punish every action you sample" field that runs away to a degenerate fixed point. When it stays
positive, the policy is rewarded for what it does and entropy holds.

The root cause of the sign flip is **state-space coverage saturation**, not the normalizer or the
missing baseline (both are aggravators — see "Root isolation" below).

---

## L1-vs-TU93 evidence (the lever)

Gap = `novelty_raw_mean − run_mean`; sign of gap = sign of `reward_norm_mean`.

| run | frac updates with reward<0 | mean gap | last-20 reward_norm_mean | cumulative_unique_states | entropy |
|---|---|---|---|---|---|
| **LS20-L1 (collapses)** | **0.76** | **−14.1** | **−0.42** | **118 (saturated by u41)** | 1.39 → 0.00 |
| **TU93-L3 (no collapse)** | **0.03** | **+39.7** | **+0.63** | 45 (probe cap; true space ≫) | stays 0.11–0.70 |

Why the gap flips negative in L1 but not TU93 — **coverage saturation**:
- `state_novelty.jsonl` shows L1 visits **100 % of its ~118 states from update 1** (101/101 at u1).
  The leaky-RND mean novelty over *all* registered states peaks at **u17 (~279)** then **decays to
  ~50** by u46 and stays there. The leak (μ=0.1) cannot regenerate novelty as fast as the predictor
  re-fits a tiny, fully-covered 118-state space. → `novelty_raw_mean` (35) falls far below the
  lagged cumulative `run_mean` (83) → reward locks negative.
- TU93-L3's true state space is far larger (registry pinned at the 45-state probe cap, but
  `novelty_raw_mean` keeps *growing* 0.15→172 across all 122 updates). Novelty stays ~35 *above*
  `run_mean` forever → reward stays ≈ +0.6 → no collapse.

So the discriminator is exactly as hypothesized: **"novelty decaying below the lagged cumulative
running-mean," and the upstream cause is small-state-space coverage saturation.**

---

## Dip (u24–40, RECOVERS) vs terminal (u46+, FATAL)

Both have negative-leaning reward and elevated grad, so magnitude is *not* the difference. Two
measured differences explain recovery vs death:

1. **Sign oscillation vs sign-lock of the PG multiplier** (`return_norm_mean`, the value REINFORCE
   multiplies log-π by). During the dip it *oscillates*: u24..u35 = +1.27,+1.42,+1.82,+1.56,+1.45,
   +1.31,−0.22,+0.56,+1.17,+1.24,+1.51,+0.37 — sign keeps reverting positive, so each negative push
   is undone. From u44 it **locks negative**: −1.59,−1.60,−1.71,−1.72,−1.73… every update with no
   recovery → one-directional self-amplification → H 1.37→0.58→0.03→0.00 over u45–48.
2. **Within-batch reward spread (escape signal) survives the dip but vanishes at terminal.**
   `reward_norm_std`: dip (u24–40) mean = **0.297**; terminal (u46–55) mean = **0.035**. During the
   dip, states still differ in novelty so there is gradient that can pull the policy back out; at
   terminal all visited states are equally low-novelty, so there is no signal to escape the corner.

The grad_norm spike to 21.9 at u46 is a *consequence* (entropy-bonus + log-π gradients blow up
during the rapid winner-take-all transition, then →0 once deterministic), not a cause.

The collapsed state is a **degenerate stuck fixed point, not reward-seeking**: from u48 on the
policy parks 94 % of visits in **state8**, whose novelty (19.5) is *below* the global mean (41.4),
while the genuinely highest-novelty **state37 (novelty 71.9) gets 0 visits**. Once H=0 and reward
spread≈0, there is neither exploration nor gradient to leave.

---

## Root isolation among (a) cumulative normalizer, (b) no baseline, (c) coverage decay

- **(a) Never-decaying cumulative `run_mean`** — *aggravator, not root.* Recomputing the reward sign
  with an EMA mean (α=0.1–0.3) instead of the cumulative mean only reduces the negative fraction
  over u30–119 from **0.94 → 0.64–0.76**; the reward is *still mostly negative* because
  `novelty_raw_mean` genuinely decayed (mean 35 vs even an EMA ~50+). A faster normalizer delays but
  does not prevent the sign flip.
- **(b) No baseline** — *aggravator/amplifier, not the originating root.* Controlled REINFORCE
  micro-sims show a constant negative *offset* with an **action-independent** reward does **not**
  collapse entropy with or without a baseline (expected PG gradient is ~0 regardless of offset
  sign), even with reward-to-go horizon correlation. The baseline only matters once the reward is
  action/state-correlated *and* the visited distribution is non-stationary — i.e. it removes the
  self-amplification term, so adding `G−G.mean()` would prevent the runaway *given* the negative
  field already exists. It is the trigger that converts a negative field into a collapse, but it
  cannot create the negative field.
- **(c) Coverage-saturation novelty decay** — **THE ROOT.** It is the only factor present in L1 and
  absent in TU93, it is what *flips the reward sign*, and the sign flip is the measured
  discriminator. (a) and (b) determine *how violently* the negative field collapses the policy, not
  *whether* the field appears.

---

## Upstream or downstream of feature-norm inflation?

**Upstream / orthogonal.** The novelty decay is driven by predictor re-fitting of a *fully covered
small state set* (the per-state novelty landscape in `state_novelty.jsonl` peaks then decays
uniformly), not by IDM feature-norm growth. `idm_mean_pairwise_l2` and the IDM/actor drift metrics
do not gate the sign flip, and held-out inverse accuracy reaches ~1.0 in *both* runs. The feature
representation is healthy where it collapses; the collapse is a reward-sign + credit-assignment
phenomenon, with feature-norm effects (if any) downstream of it.
