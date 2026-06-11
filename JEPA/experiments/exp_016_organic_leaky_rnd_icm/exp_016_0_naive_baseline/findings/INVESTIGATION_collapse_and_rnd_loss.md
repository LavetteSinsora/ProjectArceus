# exp_016_0 — Causal investigation: rising RND distill loss + entropy collapse

Run: `runs/exp016_0_naive_ls20_L1_seed0_20260606_193526/` (119 updates, 243,712 env steps).
First extrinsic reward: **33,120 env steps (u17)**. Final: entropy 0.0, noop_fraction 0.914, holdout inverse acc 0.983, coverage 118 states.

All numbers below are measured from `metrics.jsonl` / `state_novelty.jsonl`. Plots: `INVESTIGATION_plots.png`.

---

## Breakpoint summary (exact updates / steps)

| Event | Update | Env step | Evidence |
|---|---|---|---|
| IDM becomes controllable | u13 | ~26.6k | `inverse_acc_holdout` first ≥0.95 |
| First extrinsic reward | u17 | 33,120 | `env_steps_to_first_reward` |
| Feature scale + RND loss peak | u19–20 | ~39k | `idm_mean_pairwise_l2`=158, `novelty_raw_mean`=314, `rnd_distill_loss`=606 |
| **First (transient) entropy dip** | u24–35 | 49k–72k | H 1.32→**0.61**, then *recovers* to 1.39 at u41–43 |
| **Permanent entropy collapse** | **u45→u48** | **92k→98k** | H 1.23→0.58→0.031→5.7e-6; **grad_norm spikes to 21.9 at u46** (90th-pct elsewhere = 2.16) |
| Entropy = 0 (numerically) | u50 | 102k | H=1.0e-20, grad_norm=0 |

The user's "~100k" intuition is right: the irreversible collapse completes at u47–50 (96k–102k). The trigger fires slightly earlier, at **u46 = 94,208 env steps**, as a single actor gradient-norm spike.

---

## Q1 — Why does `rnd_distill_loss` INCREASE instead of →0?

**Dominant mechanism: (b) feature-magnitude growth of the continuously-trained IDM encoder.** Mechanisms (a) drift and (c) leak are secondary/negligible.

Evidence:
- `idm_mean_pairwise_l2` (feature scale proxy) explodes **0.32 (u1) → 158 (u19), ×490**, as the IDM trains up (holdout inv-acc 0.27→0.98 by u13). RND novelty = ½·mean‖P(h)−T(h)‖²; both P and the frozen target T are MLPs over `h`, so when ‖h‖ inflates, ‖T(h)‖ and the absolute MSE inflate **regardless of relative fit**.
- `corr(rnd_distill_loss, idm_mean_pairwise_l2) = 0.94` over the rising phase (u1–20); 0.78 over the full run.
- `corr(rnd_distill_loss, novelty_raw_mean ↔ idm_mean_pairwise_l2) = 0.94` — RND loss and novelty rise/fall **together with feature scale**.
- Scale-squared test: `novelty_raw_mean / pairwise_l2²` is roughly constant (≈0.004–0.015) once the IDM is trained (u10–119), while novelty itself varies ~10×. The absolute MSE is governed by ‖h‖², not by fit quality. (It actually *declines* slightly over time → P fits *better* in relative terms, but absolute loss stays high because the scale stays high.)
- Mechanism (a) moving distribution / drift is **ruled out as the driver**: `corr(rnd_distill_loss, drift_idm_rel_l2) = −0.56` (rising phase) — loss rises while drift *falls* (drift 2.7→0.1 as IDM converges). Drift was largest at the start when loss was lowest.
- Mechanism (c) the μ=0.1 leak adds a constant upward pressure but does **not** explain the ×100 rise; see Q3 — it sets the steady-state *floor*, not the climb.

**Verdict:** The distill loss "increases" because the metric is measured on an unnormalised, exploding feature space. As the IDM trains, ‖h‖ grows ~×490; T(h) and the MSE grow with it. P is in fact fitting better relatively, but absolute MSE is dominated by scale. There is no contradiction with "distilling should →0" — distillation on a *fixed* feature space would; here the ruler itself is stretching.

---

## Q2 — Why does entropy collapse to 0, and what happens at ~100k?

**The user's hypothesis is confirmed: a single actor `grad_norm` spike (u46 = 21.9, vs surrounding ~1–4) is the proximate trigger.** `return_variance` does NOT spike (stays 50–100 across u44–50), so the spike is in the gradient, not in raw return scale.

Lead/lag at the collapse (u44→u46): `grad_norm` 1.25→**21.9** (×17), `reward_norm_std` 0.195→0.090, novelty resolution (`nov_std/nov_mean`) 0.348→0.141, `noop_fraction` 0.279→0.724. The gradient and the action-concentration move first/together; noop follows.

**The deeper mechanism — a running-mean lag that turns REINFORCE into a uniform punisher:**
- The reward is z-scored by **running** (slow-EMA) mean/std (`nov_rms`). By u41+, `run_mean` ≈ 110–118 (memory of the early novelty peak of ~314) while current `novelty_raw_mean` has decayed to ~67. So `reward_norm_mean = (67−112)/std ≈ −0.45` — **every reward in the batch is negative**.
- With γ=0.99 reward-to-go and ÷batch-std, `return_norm_mean` is **−1.5 to −1.7** for u40 onward (u41 −1.53, u46 −1.72). `return_norm_std` is pinned to 1.0 by construction.
- REINFORCE with **no baseline** and uniformly-negative returns: `loss = −E[logπ(a)·G]` with G<0 *pushes every sampled action's probability DOWN*. The action sampled **least** is pushed down least; surviving probability mass concentrates on whatever action is momentarily most frequent → positive feedback. Once one action edges ahead it is sampled more, dominates the (negative) gradient less per-unit-prob, and runs away. The only force resisting this is the entropy bonus (coef 0.01), which is too weak.
- This is exactly why the collapse is **bistable**: a first attempt at u24–35 (H→0.61) self-corrected back to 1.39 (u41–43) when the batch composition shifted. The second attempt at u45 caught and ran away (grad spike u46), and once `reward_norm_std`→0.03 (u47) there is no signal left to re-open the policy → H decays to machine zero.

**It collapses onto action index 2**, with `per_action_prob` → `[~0, ~0.005, 0.994, ~0]` at u47 and `noop_fraction` → 0.913. Action 2 is a **wall-bump / no-op** (board unchanged): the agent learns to sit still because standing still is the least-punished action under the negative-return regime.

**Timing attribution:** the collapse is **not** triggered by the first reward (33k, u17 — 28 updates earlier) nor by IDM controllability (u13). It is triggered by the **decay of current novelty below the lagging running mean** (Q1's scale-driven novelty peaking ~u20 then decaying), which flips the reward sign around u40 and detonates at u45–46.

---

## Q3 — The paradox: collapsed policy revisits the same states, so why doesn't novelty/distill loss fall?

**Resolved directly from `state_novelty.jsonl`: the high loss is absolute feature-scale inflation, NOT a few never-revisited states.**

- Post-collapse the policy hammers **one** state (id 8, the wall-bump board) ~**1,900 times per update**, every update u50–119. With ~1,900 distill samples of this single state per update, its novelty *should* go to ~0. It does **not** — it plateaus at **~20–35** (u50: 30.8, u60: 21.1, u90: 31.8, u110: 34.7).
- Visited-this-update states (n≈4) have novelty mean ≈40–47; abandoned-but-seen states (n≈114) ≈45–61. The gap is only ~1.3×, **not** the orders-of-magnitude difference you'd see if abandoned states were the culprit. Even the maximally-distilled state can't be driven down.

Why the floor exists: the **μ=0.1 leak** pulls the predictor 10% back toward init **once per update**, regenerating error faster than the 4 RND grad steps (lr 1e-4) can remove it — exactly the "visitation-rate signal that never permanently saturates" the leak was *designed* to produce. But because it operates on the **inflated** feature space (‖h‖²≈75²), the floor it holds is ~30 in absolute MSE, not ~0. So the leak sets the floor; the IDM feature scale sets the height of that floor.

**Distinguishing (i) vs (ii):** the dominant cause is **(ii) absolute feature-scale inflation** — proven because the ×1900-visited state itself sits at ~30. The leak (i-adjacent) only explains why it doesn't reach 0 at all; it does not explain the magnitude.

**Is Q1 causally upstream of Q2? YES.** The scale-driven novelty *peak* (Q1, ~u19–20) is what loads the slow `run_mean` to ~112. The subsequent decay of novelty (as the IDM scale partially settles and the policy narrows) drops current novelty below that stale mean, flipping `reward_norm_mean` negative (u40+), which is the precondition for the no-baseline REINFORCE runaway (Q2). The transient, mis-scaled novelty injected by the inflating feature space is therefore the **upstream cause** of the entropy collapse, not an innocent bystander. This is "the cost of the initial RND-loss increase."

---

## Causal chain (Q1 → Q2 → Q3)

1. **IDM trained continuously** → encoder output norm inflates ×490 (`pairwise_l2` 0.32→158 by u19).
2. **(Q1)** RND novelty = ½‖P(h)−T(h)‖² lives on that inflating space → `rnd_distill_loss`/`novelty_raw_mean` spike to ~314/606 (u20), *not because of poor fit* (corr 0.94 with scale, −0.56 with drift).
3. The novelty **peak loads the slow running-mean** to ~112; as novelty then decays, `reward_norm_mean` flips **negative** (−0.45 by u41) → `return_norm_mean` ≈ −1.7.
4. **(Q2)** REINFORCE **with no baseline** + uniformly-negative z-scored returns = "punish every action"; weakest entropy bonus loses; one self-corrected attempt (u24–35) then a runaway at **u45→46 (grad_norm 21.9)** → commit to action 2 (wall-bump) → H=0, noop=0.913.
5. **(Q3)** Collapsed policy revisits one state ×1900/update, but novelty stays ~30 because the **μ=0.1 leak regenerates error on the inflated scale** each update — the floor is set by leak, its height by feature scale. So `rnd_distill_loss` stays high forever even with maximal revisitation.

---

## Recommended fixes / next experiments (ranked)

1. **Add a baseline to the policy gradient (highest leverage).** The no-baseline REINFORCE is the actual collapse engine: with a value/advantage baseline, a uniformly-negative reward batch produces ~zero-mean advantages and cannot punish-all. Either restore a value head (advantage = G − V) or, minimally, subtract the per-batch return mean (`G − G.mean()`) before ÷std. This alone should prevent the u46 runaway. *(Test: re-run with `G − G.mean()`; expect entropy to stay >1 through 100k.)*

2. **Normalize the RND feature space / fix the z-score centering.** Two coupled bugs: (i) novelty is measured on an unbounded ‖h‖ → LayerNorm/`RunningMeanStd` the IDM features *before* T/P, or freeze the IDM after warm-up (it's already controllable by u13) so the ruler stops stretching; (ii) the running-mean z-score lags badly — use a faster EMA, or recenter per-batch, so a decaying novelty distribution doesn't produce a persistently-negative reward. Fixing (i) removes the spurious novelty peak that loads the stale mean (kills the Q1→Q2 link at the source).

3. **Decouple/anneal the leak, or apply it post-normalization.** μ=0.1 on the inflated space holds a ~30 novelty floor on a state visited 1900×. With normalized features the same leak yields a sensible O(1) floor. Consider a smaller μ or μ scaled to the feature norm so the "never-saturates" property survives without inflating the count signal. Lower priority than (1)/(2) since the leak only sets the floor height.

Secondary: raise `ent_coef` (0.01 is too weak to resist a ×17 grad spike) as a cheap stopgap, but it treats the symptom, not the negative-return cause.
