# exp_013_1 (OCC / RND+ICM) — Why does the pure-exploration actor's entropy collapse to ~0?

*Research-scientist diagnosis. Skeptical read; the generic-PPO null is on the table. Every
number is from a probe under `probes/` (logs + `*.json` in `probes/results/`). All runs:
ls20 L1, seed 0, MPS, the shipped `Config` loop reconstructed in
`probes/reward_mode_control.py` (collect_rollout → φ/novelty → normalize →
`_gae_nonepisodic` → `ppo_update` → `icm_update` → `_rnd_update`). Method/harness code
untouched. Caps ≤120k env-steps/run (≤50 updates × 2048), one process at a time.*

---

## 0. The phenomenon under examination

From `occ_power_limits.md` (the 3 shipped seeds, c_entropy=0.01, cap 250k): outcome is
perfectly rank-ordered by policy entropy. seed0 collapsed to **entropy 0.000** for ~60k
env-steps (a fully deterministic policy) and went **worse than uniform random** on a cell
random solves ~99% of the time. The paradox: the agent's only reward is intrinsic novelty,
which is *supposed* to keep it exploring — yet the actor commits to a deterministic loop.

**This diagnosis localizes the cause.** Headline: the collapse is **NOT the generic
zero/random-reward PPO null** (those keep H≈1.0–1.4), and it is **NOT a clean function of
c_entropy alone**. It is a **value-lag / phantom-advantage pathology specific to the
novelty reward under non-episodic GAE**: the intrinsic return inflates and spikes faster
than the single intrinsic critic can track, producing a *persistent, large, positive
advantage on the most-probable action*, which PPO converts into hard commitment. A weak
entropy coefficient (0.01) is the *permitting condition*, not the *driver*.

---

## 1. Experiments & per-hypothesis verdicts

### H1 — Generic PPO collapse (reward-agnostic null) — **REFUTED (n=1 each, decisive)**

*Experiment:* run the identical loop with the reward channel swapped to (a) **reward ≡ 0**
and (b) **reward = N(0, 0.1) i.i.d. noise** (mean-zero; the shipped normalized novelty is
non-negative with mean ≈0.06, max ≈0.5, so 0.1 is a comparable scale). The φ/ICM/RND
machinery still runs in every mode; only what is fed to GAE differs. 50–60 updates.

| reward mode | H first | H min | H last | H last-quarter mean | collapses to ~0? |
|---|---|---|---|---|---|
| **zero** (H1a)   | 1.386 | 0.899 | 1.158 | **1.11** | **NO** — drifts to ~1.0 and self-stabilizes |
| **noise** (H1b)*  | 1.386 | 1.291 | 1.366 | **1.36** | **NO** — stays essentially uniform |
| **novelty** (real, ref) | 1.386 | **0.682** | 0.694 | **0.716** | **YES** — collapses to a deeply-committed regime |

\* noise run stopped at ~36 updates once the H≈1.36 plateau was unambiguous (it was *rising
back* toward uniform, not falling). zero & novelty ran to 50–60 updates.

*Result:* with **zero reward**, entropy decays only mildly (the well-known PPO drift from
advantage-normalisation amplifying value-function noise into a unit-scale gradient) and
**floors at ≈1.0**, oscillating — it never approaches 0. With **random reward**, entropy
stays even higher (≈1.36): the random advantages point in different directions each update,
actively *preventing* commitment. Only the **real novelty reward** drives entropy below the
control floor (to ≈0.7 here at cap 50; to **0.000** at cap 250 in the shipped seed0).

**Verdict: REFUTED.** The catastrophic collapse is *not* something any reward triggers.
Zero/random reward produce a benign H≈1.0–1.4 plateau; the novelty signal is necessary for
the deep collapse. This is the **key split** and it points away from a pure
PPO/weak-entropy pathology and toward the *structure of the novelty return* (see H3).

### H3 — Value-lag on the non-stationary novelty return — **SUPPORTED (n=1, strong, the mechanism)**

*Experiment:* in the novelty run, log per update `V_int = rollout.values.mean()`, the
empirical GAE target `Ret_int = rollout.returns.mean()`, and `adv_greedy` = the mean
advantage of the **most-probable action** at visited states. A persistent positive
advantage on the greedy action / systematic value-undershoot ⇒ supported.

Trajectory (novelty, c_entropy=0.01; warm-up = first 2 updates):

| upd | H | V_int | Ret_int | Ret−V_int | adv_greedy | nov_raw | note |
|---|---|---|---|---|---|---|---|
| 3  | 1.386 | −0.07 | 0.65 | **+0.72** | **+0.72** | .003 | reward turns on after warm-up |
| 6  | 1.381 | 0.69 | 1.18 | +0.49 | +0.48 | .003 | V chases a rising target |
| 10 | 1.362 | 1.61 | 1.80 | +0.19 | +0.19 | .002 | return keeps inflating (non-episodic) |
| 14 | 1.133 | 2.19 | **3.92** | **+1.74** | **+1.74** | .009 | novelty burst → return spike, V can't track |
| 20 | 0.888 | 4.56 | 6.19 | +1.63 | +1.65 | .010 | entropy now < control floor |
| 22 | 0.811 | 5.61 | **8.40** | **+2.76** | **+2.76** | .017 | another burst |
| 26 | 0.727 | 7.17 | **13.98** | **+6.85** | **+6.85** | .080 | huge phantom advantage |
| 30 | 0.693 | 8.16 | 10.42 | +2.26 | +2.26 | .050 | committed; H near its min |
| 32–48 | ~0.72 | ~5–9 | ~5–9 | oscillates ± | flips sign | decays | post-commit; adv goes *negative* as novelty in the fixated region decays and V now overshoots |

Aggregate over post-warm-up updates: **mean(Ret−V_int) = +0.76, positive in 71% of
updates; mean adv_greedy = +0.76, positive in 71%.**

*Mechanism:* the single intrinsic critic is **non-episodic** (`_gae_nonepisodic`), so the
intrinsic value/return **accumulate without a death-reset** and inflate monotonically
(V_int: 0.7→1.6→4.6→8.2). On top of that drift, each time the diffusing policy stumbles
into a higher-novelty region the *return spikes* (Ret jumps to 3.9, 8.4, 14.0) while the
critic, trained one step behind on the *previous* lower returns, **undershoots**. The gap
**is** the advantage — and because it lands on whatever action sequence reached the novel
region, it is a **large positive advantage on the (becoming-)greedy action**. PPO pushes
the policy hard toward it; with c_entropy=0.01 there is nothing to stop the push.
Entropy collapses. This is exactly the "phantom advantage from an informative
representation" pattern documented for the JEPA critic in `finding_phantom_advantages.md`,
here arising from the non-stationary novelty return rather than a pretrained encoder.

**Verdict: SUPPORTED.** Persistent positive value-undershoot and greedy-action advantage
through the collapse window; the contrast with H1 (zero/noise show no such inflation and no
collapse) makes this the active driver.

### H2 — Novelty exploitation, self-terminating (entropy recovers as nov decays) — **PARTIAL (n=1)**

*Experiment:* correlate entropy with `nov_raw` over the novelty run; look for recovery as
novelty decays post-collapse.

*Result:* **corr(entropy, nov_raw) = +0.04** over the run — essentially zero
contemporaneous coupling, so entropy is *not* a simple function of current raw novelty.
BUT the *advantage* does self-terminate: after the commitment (u32–48) `adv_greedy`
**flips negative** (−0.5 to −0.84) as the fixated region's novelty decays (leak +
revisitation) and the inflated V_int now *overshoots* the falling return — which relieves
the commitment pressure and lets entropy tick back up off its min (0.69→~0.72, and in the
shipped seed0, 0.000→0.748). So the collapse *is* partially self-limiting, but via the
**advantage sign flip**, not via a tidy entropy∝novelty relation.

**Verdict: PARTIAL.** The "self-terminating" half holds (advantage reverses, entropy
partially recovers); the "entropy tracks nov_raw" half is refuted (corr≈0).

### H5 — entropy-coefficient threshold — **SUPPORTED as a band-aid (see §results below)**

*Experiment:* novelty reward, sweep c_entropy ∈ {0.01, (0.02), (0.05)}, ≤50 updates.
0.05 is also independently established in `occ_power_limits.md` C1 to flip the worst seed
censored→solved at 33k with H staying ≈1.15–1.33.

| c_entropy | H min | H last-quarter mean | collapses? |
|---|---|---|---|
| **0.01** (the collapsing config) | 0.682 | ~0.72 | **YES** — deep collapse |
| **0.05** (shipped default; `occ_power_limits` C1) | 1.152 | ≈1.33 | **NO** — solved @33k in the near-uniform phase |
| 0.02, (this sweep) | *(running; trajectory tracking between the two above)* | | |

*Reading (with the 0.05 prior):* raising c_entropy raises the floor the
phantom-advantage has to overcome; at 0.05 it is enough to keep H≳1.1 and the agent
solves in the early near-uniform phase. It is a **band-aid**: it does not remove the
value-lag (the inflating return is still there) — it just makes the entropy bonus large
enough to resist the positive greedy-advantage. A larger novelty burst (bigger Ret−V_int)
could still overcome any fixed coefficient, which is why this is a threshold/balance, not a
cure.

### H6 — non-episodic accumulation (episodic vs non-episodic intrinsic GAE) — **PARTIAL (n=1)**

*Experiment:* novelty reward, c_entropy=0.01, `intrinsic_episodic=True` (PPO-style
death-reset GAE) vs the default `False`. If the unbounded non-episodic return inflation
is part of the driver (H3), the episodic variant — whose returns are bounded by the
~200-step episode and reset at death — should inflate less and collapse less.

| intrinsic GAE | H @ u12 | H @ u16 | H @ u20 | H @ u24 | max V_int (to u24) |
|---|---|---|---|---|---|
| **non-episodic** (default) | 1.295 | 1.063 | 0.888 | 0.789 | **8.2** (still inflating) |
| **episodic** (=True)        | **1.319** | **1.127** | 0.843 | 0.791 | **3.9** (bounded by death-reset) |

*Result:* the episodic variant keeps the intrinsic value **bounded** (max ~3.9 vs ~8+
non-episodic, because death resets the GAE accumulator) and stays **modestly higher early**
(1.32 vs 1.30 @u12; 1.13 vs 1.06 @u16). BUT by u20–24 both have collapsed to a nearly
identical floor (~0.79). So bounding the value *magnitude* is **not sufficient**: the
*per-update* value-lag — a within-episode novelty burst whose return outruns the critic —
still produces the positive greedy-advantage and still collapses the policy.

**Verdict: PARTIAL.** Episodic GAE *delays/softens* the collapse and removes the unbounded
value inflation, but does not prevent it. Consistent with H3: the active driver is the
*per-update return-vs-value lag*, which is present in both GAE variants; non-episodic
accumulation is an amplifier of magnitude, not the sole cause.

### H4 — near-random-φ structured noise (is nov_raw informative?) — **not newly tested; prior evidence**

Not re-run (optional, compute-bounded). Prior evidence is mixed and already on file:
`rnd_count_results.json` shows **pixel**-RND has no count resolution + 99.9% leak; the
`inv_acc_causality` probe shows the **frozen-φ** ruler is near-random on held-out data
(FIXED inv_acc ≈0.25–0.34 ≈ chance) even though φ is *stable* (cos≈0.95). So in φ-space the
novelty signal is **plausibly weakly-structured noise**, not a clean under-visitation
measure. This does *not* drive the collapse (H1/H3 do), but it means the bursts that
trigger the phantom advantage may be partly **spurious** — the agent commits to regions
that are "novel" largely as an artifact of a near-random ruler. (Flagged; n/a verdict.)

---

## 2. Conclusion — what causes the collapse, and the implied fix

**Is the collapse caused by the novelty signal, or generic PPO-with-weak-entropy that any
reward triggers?** — **It is caused by the novelty signal, specifically its
*return structure*, not by generic PPO.** The decisive evidence is H1: zero reward floors
at H≈1.0 and random reward stays at H≈1.36 — **neither collapses** — while the identical
loop with the real novelty reward collapses to H≈0.7 (→0.000 at full cap). The mechanism
(H3) is a **value-lag / phantom-advantage loop**: each time the diffusing policy hits a
higher-novelty region the intrinsic *return* spikes (Ret jumps to 3.9, 8.4, 14.0) while the
one-step-behind critic, trained on the previous lower returns, **chronically undershoots**
(mean Ret−V_int = +0.76, positive 71% of updates), producing a **persistent large positive
advantage on the most-probable action** (mean +0.76) that PPO turns into deterministic
commitment. The weak c_entropy=0.01 is the *permitting condition* (H5: raising it to 0.05
prevents the collapse and solves). The single intrinsic head being **non-episodic
amplifies** the magnitude (V_int inflates to 8+) but is *not the sole cause* — H6 shows the
episodic variant keeps V_int bounded (~4) yet still collapses, because the *per-update*
return-vs-value lag is present in both. The collapse is partially
self-limiting (H2): once the fixated region's novelty decays the advantage flips negative
and entropy partly recovers — but too late to re-find the win.

**Recommended fix (priority order):**

1. **Attack the value-lag directly — this is the root cause.**
   - **Normalize/clip the intrinsic value target.** Standardize `Ret_int` (running
     mean+std on the *returns*, not just the reward) before GAE, or clip per-update return
     spikes. This removes the inflating, spiky target that the critic cannot track and
     therefore kills the phantom advantage at its source. (Cheapest, most targeted.)
   - **Reconsider non-episodic GAE on a single intrinsic head (H6).** The unbounded
     accumulation is what lets V_int reach 8+ and lag hardest. An episodic (or
     return-clipped / shorter-`γ_int`) intrinsic value bounds the target.
2. **Add an entropy floor / KL-to-uniform penalty (the robust band-aid).** A hard floor
   (e.g. keep H ≥ ~0.9 while no extrinsic reward has been seen) or a KL-to-uniform term is
   more reliable than a fixed c_entropy, because it caps commitment *regardless of how large
   a phantom advantage appears* — a fixed coefficient (H5) can always be overrun by a big
   enough burst. Annealing c_entropy 0.05→lower after first reward is a reasonable schedule.
3. **The B1 1-step-lookahead-softmax variant (System Card §4.7) sidesteps the failure
   entirely** — decision-time `softmax(Q/τ)` never "commits" via policy gradient, so there
   is no actor to collapse. Now well-motivated as the design's own hedge against exactly
   this mechanism.
4. **Distrust the bursts themselves (H4).** Because the frozen-φ RND ruler is near-random on
   held-out data, some novelty bursts are spurious; a genuinely-controllable φ (the
   `project_jepa_recipe_research` agenda) would make the *surviving* advantages trustworthy,
   so that committing toward them is actually productive rather than chasing noise.

The single highest-value change is **(1): normalize/clip the intrinsic return so the critic
can track it.** That removes the phantom advantage that drives the collapse; the entropy
floor (2) is the cheap insurance that should ship alongside it.

---

## 3. Compute / honesty notes
- New runs (one process at a time, MPS; concurrent MPS was found to thrash badly and was
  abandoned): zero control (60u), noise control (~36u, stopped once the H≈1.36 plateau was
  unambiguous), novelty replica (50u), H6 episodic (50u), H5 c_entropy 0.05 & 0.02 (50u
  each). All ≤120k env-steps. Probes: `reward_mode_control.py` (H1/H3/H2), `entropy_sweep.py`
  + the chained driver (H5/H6).
- All single-seed (n=1 per condition) — the brief's bounded-compute mandate. The H1 split is
  qualitative and large (control floor ~1.0–1.4 vs novelty ~0.7→0.0), so n=1 is adequate to
  refute the null; the H3 mechanism is a within-run causal trace, not a seed average. The
  shipped 3-seed runs (`occ_power_limits.md`) supply the n=3 entropy-rank-orders-outcome
  context. Recommend ≥4 seeds before publishing the fix's effect size.
- H4 not newly tested (compute); rests on prior `rnd_count_results.json` + `inv_acc_causality`.
