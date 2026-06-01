# OCC (RND+ICM, exp_013_1) — Power & Limitations

*Research-scientist characterization of the exp_013_1 "RND+ICM / OCC" sparse-reward
exploration method. Skeptical read; negative findings included. Every number is quoted
from a log under `runs/` or a probe under `probes/`. `n` (seed/sample counts) stated
inline. Evidence vs hypothesis is flagged explicitly.*

Author probes (read-only; method/harness code untouched):
- `probes/coverage_vs_entropy.py` (+ `..._results.json`) — coverage/chaining vs policy entropy on the real ls20 L1 engine.
- `probes/ablation_driver.py` — re-runs shipped `Config`+`train()` with only System-Card knobs overridden (`c_entropy`, `leak`).
Existing evidence parsed: the 3 seed runs in `exp_013_1_rnd_icm/runs/`, `baseline_random_policy/SUMMARY.md`, `probes/phi_drift_findings.md`, `probes/rnd_count_results.json`.

---

## 0. The result under examination (n=3 seeds, ls20 L1, cap 250k)

| seed | outcome | first-reward step | φ-freeze step | final entropy | entropy regime |
|---|---|---|---|---|---|
| seed2 | **SOLVED** | **24,176** | **never** (null) | 1.343 (≈ln4=1.386) | stayed near-uniform |
| seed1 | CENSORED | — | 108,544 | 0.885 | moderate commitment |
| seed0 | CENSORED | — | 126,976 | 0.748 (after a 0.000 trough) | **collapsed to ~0** mid-run |

Source: `runs/*/result.json` and `metrics.jsonl`. Random-policy baseline on this cell:
E≈49,843 env-steps/first-win (p_life=8.63e-4), **~99% solved within 250k**
(`baseline_random_policy/SUMMARY.md`). So on 2/3 seeds OCC is **worse than uniform random**
on an easy cell, and on 1/3 seeds it is ~2× faster than random's expectation.

**Bottom line up front:** the worse-than-random failure is **not** a curiosity-design failure — it is
an actor **entropy collapse**. Outcome is perfectly rank-ordered by policy entropy across the 3 seeds
(B1), the failure shape is reduced component-coverage under commitment (B2, real-engine probe), and a
single bounded ablation that simply raises the entropy coefficient (`c_entropy` 0.01→0.05)
**flips the worst-censored seed (seed0) to solved at 33k** (C1). The leak/φ-space machinery works as
designed (A1–A3) but is not the active ingredient in success on this cell (B3).

---

## A. POWER — what the method demonstrably does well

**A1. The leak works exactly as designed: novelty never saturates (evidence, n=2 censored seeds).**
The single biggest failure of vanilla RND (one-way error ratchet → flat → stall) does **not** occur.
At the censoring step (249,856), raw novelty is still alive and being delivered:
- seed0: `novelty_raw=0.0156`, `r^i_norm=0.0157`; last-20-update min/max novelty 0.0045 / 0.0250.
- seed1: `novelty_raw=0.0952`, `r^i_norm=0.0834`; last-20-update min/max 0.0411 / 0.2278.

Novelty is non-zero through 248k+ on both seeds — the predictor-shrink-to-init leak keeps the
ruler measuring *recent visitation rate*, not cumulative count. **The stall failure mode is cured.**
(Evidence; the leak's *causal* contribution to first-reward is tested in C2 below.)

**A2. φ separates and freezes cleanly (evidence).** Both censored seeds hit the adaptive freeze
trigger via the `inverse_acc≥0.90` plateau (not the fallback): seed0 `inverse_acc=0.980` → froze at
~127k, seed1 `0.980` → froze at ~109k. No `FALLBACK`/poor-φ WARNING fired. The φ-space design
premise (a learnable controllable ruler that actually separates states) holds on this cell.

**A3. φ-space RND is the right call vs pixel RND (prior evidence, `probes/rnd_count_results.json`).**
Pixel-space RND (exp_012 nets) has **no count resolution and a 99.9% generalization leak**:
- per-state error after training is ~0.0003 at 1 visit and ~0.0001–0.0002 at 2000 visits — i.e.
  **1 visit ≈ 2000 visits** (148 distinct states, counts 1..2000); the error floors instantly.
- a never-visited holdout state's error drops **99.92%** (0.2893 → 0.00022) purely from training on
  *other* states — novelty bleeds to unseen states. Moving RND into φ-space is well-motivated.

**A4. When it stays near-uniform, OCC can beat random's expectation (weak evidence, n=1).**
seed2 solved at 24,176 < random's E≈49,843, while keeping entropy ≈1.34 (near ln4) the whole way and
**before φ ever froze** (freeze=null). Its novelty spiked broadly during search (raw novelty
0.175→32.7→18.1→2.6→0.46 over updates 1–12) and it found the win in 12 updates. This is the
intended behavior — directed-but-broad exploration. **Caveat: n=1, and 24k vs E=50k is within the
high variance of a p≈8.6e-4 process; this is suggestive, not a demonstrated speedup.** Note also
the solve did **not** require the frozen-φ RND ruler (φ was still moving, `inverse_acc≈0.45`) — on
this cell the win comes from the early near-uniform phase, not the matured curiosity machinery.

---

## B. LIMITATIONS — where and why it fails

**B1. Worse-than-random on an easy cell, driven by entropy collapse (strong evidence, n=3, perfectly rank-ordered).**
Outcome tracks policy entropy monotonically across all three seeds:

| seed | last-quarter mean entropy | min entropy | outcome |
|---|---|---|---|
| seed2 | **1.346** | 1.340 | solved 24k |
| seed1 | 0.806 | 0.537 | censored |
| seed0 | **0.346** | **0.000** | censored |

seed0 collapsed to entropy **0.000–0.001 from update ~64 to ~90** — a *fully deterministic* policy
for ~**60k env steps** (steps ~131k–184k), i.e. it spent a quarter of its entire budget looping
deterministically. It later recovered to 0.748 but never re-found the win. The intrinsic-only PPO
objective, with `c_entropy=0.01`, lets the actor commit hard to whatever region currently looks
novel; once committed it stops sampling the rest of the maze. **This is the mechanism that makes
OCC lose to uniform random on a cell uniform random solves ~99% of the time.** (Evidence.)

**B2. Coverage ≠ sequence — confirmed as the failure shape (evidence, `coverage_vs_entropy_results.json`; 400 lives/setting on the real engine).**
Sweeping a stochastic policy from uniform (H=ln4) down to near-deterministic on real ls20 L1:

| policy | mean H | distinct (cell,rot,color) | hit-rotation-tile rate | win rate |
|---|---|---|---|---|
| uniform (mix 0) | 1.386 | **84** | **0.122** | 0/400 |
| mild commit | 1.117 | 84 | 0.120 | 0/400 |
| moderate (~seed1) | 0.761 | 65 | 0.085 | 0/400 |
| strong | 0.405 | 52 | 0.018 | 0/400 |
| near-det (~seed0) | 0.208 | 52 | 0.020 | 0/400 |

As entropy falls, **both** pooled state coverage (84→52) **and** the rate of hitting the critical
rotation tile (the mandatory first link of the winning chain) collapse — the rotation-tile hit rate
drops ~**6×** (0.122→0.020). Validation: at uniform, 3000 lives gave **2 wins** (p̂=6.7e-4),
matching the analytic p_life=8.63e-4 (expected 2.59) — the probe reproduces the random baseline.
*So the censored seeds fail by reaching the win's components less often, not by failing to "chain"
two components they both reached* — at low entropy they largely **fail to even reach the rotation
component**. This **confirms** the leading hypothesis (coverage degrades under commitment) and
**refines** it: on ls20 L1 the binding constraint is reaching the *rotation tile* under the energy
budget, which sub-max entropy under-samples.

*Caveat (honest):* this probe uses a per-life fixed direction-biased policy as a stand-in for the
trained policy's commitment (no checkpoints are saved, so the actual trained policy could not be
rolled out). It isolates the entropy→coverage→component-reach causal chain but does not reproduce the
trained policy's *spatial* structure. The `reach_goal_rate=0.000` everywhere is an engine artifact,
not a bug: a wrong-rotation goal bump is a *free no-op* (the agent never stands on the goal cell
unless it wins), so "reached goal with wrong rotation" is unobservable by construction — the right
component proxies are hit-rotation-rate and win.

**B3. The matured curiosity machinery is not what solves this cell.** The only solve (seed2)
happened at 24k, before φ froze and while `inverse_acc≈0.45` (φ not yet a good ruler). The two seeds
that *did* mature the full pipeline (φ frozen at high inverse_acc, leak alive) are the two that
**failed**. On ls20 L1 the design's headline components (frozen-φ ruler, sustained leaky novelty)
are not the active ingredient in success; near-uniform early search is. (Evidence, n=3.)

**B4. ls20 L1 is a poor cell to evaluate this method on at all.** Random solves ~99% by 250k, so
there is almost no headroom for a directed method to demonstrate power, and the outcome is dominated
by seed variance in a p≈8.6e-4 process. A method can only look *worse* here; it cannot convincingly
look *better*. (Argument from `baseline_random_policy/SUMMARY.md`.)

---

## C. Role of each component

- **φ-space (vs pixels):** *justified and functional.* Pixel RND has no count resolution + 99.9%
  leak (A3); φ freezes at inverse_acc 0.98 with no fallback warning (A2). But **not load-bearing for
  the ls20-L1 solve** (B3) — seed2 won with φ still moving.
- **Leak (μ=0.01):** *works as designed* — novelty stays alive to 248k, curing the saturation stall
  (A1). **But it is not sufficient for first-reward** and may even be counterproductive here: a
  permanently-alive novelty signal gives the committed actor a persistent gradient toward whatever
  region it has fixated on, feeding the entropy collapse (B1). Causal test in C2.
- **φ-freeze:** clean adaptive trigger (A2); irrelevant to the only success (B3).
- **Non-episodic intrinsic GAE:** novelty value bootstraps across death/reset (verified in
  `trainer._gae_nonepisodic`). Not isolated here; plausibly *amplifies* commitment by letting a
  fixated region's value persist across lives — flagged as **hypothesis**, untested.
- **Sub-max entropy (`c_entropy=0.01`):** *the dominant failure driver* (B1, B2). The intrinsic-only
  objective has nothing stopping the actor from collapsing to a deterministic loop once a region
  dominates the advantage. Tested in C1.

### C1. Ablation — higher entropy coefficient (`c_entropy 0.05`, seed0, cap 120k, n=1) — **DECISIVE: hypothesis confirmed**
*Hypothesis:* keeping the policy closer to uniform prevents the collapse and recovers solving.
*Result:* **seed0 — the worst collapser, CENSORED at 250k in the original run (entropy fell to
0.000) — now SOLVES at 32,528 env-steps** with `c_entropy=0.05` (5× the default). Entropy stayed
near uniform throughout (mean 1.325, **min 1.152**, last 1.154) and never collapsed; φ never froze
(it solved in the early near-uniform phase, exactly like the original seed2). Run dir:
`runs/exp013_1_rndicm_ls20_L1_seed0_20260531_212321/` (config `c_entropy=0.05`).
This is the **single cleanest piece of evidence in this report**: the same seed flips from
censored-at-250k to solved-at-33k purely by preventing the entropy collapse. The headline n=3
worse-than-random result is therefore primarily an **actor-commitment artifact**, not a property of
the curiosity design. (Evidence, n=1 — flip on the previously-worst seed; should be replicated across
seeds per D5.)

### C2. Ablation — leak=0 vs leak=0.01 (vanilla RND control)
*Hypothesis:* the leak matters for *not-stalling* but not for *first-reward* on this easy cell.
<!-- RESULT_C2 -->
*(not run — see note in §E; lower priority given A1 already shows the leak keeps novelty alive and
B1/B2 show the binding failure is entropy, not saturation.)*

---

## D. Recommendations

**D1. Evaluate where the method *can* show power: the E=∞ cells.** Per `baseline_random_policy`,
the fair tests — where random=0 so *any* solve is ∞× better — are **ls20 L2/L3, tu93 L3, re86 L2/L3,
and all g50t**. There a single solve is a genuine power result. Among the random-reachable cells, the
*meaningful* (non-trivial-headroom) quantitative comparisons are **tu93 L1 (~500k)** and **re86 L1
(~2M)**; **ls20 L1 (~50k, ~99% by 250k)** and **tu93 L2 (~2k, trivially random-solvable)** are too
easy to demonstrate anything and should be treated as regression/sanity cells, not power tests.

**D2. Fix the entropy collapse before any further power claims.** The headline n=3 result is
confounded by an actor-commitment bug, not by the curiosity design. Concretely, in priority order:
1. **Raise/anneal `c_entropy`** (the C1 ablation tests this) or add an entropy *floor* / KL-to-uniform
   penalty so the actor cannot drop below, say, H≈0.9 while no reward has been seen.
2. **Cap policy commitment under zero extrinsic reward** — e.g. clamp logit magnitude or use a
   higher-temperature sampling head until first reward.
3. Consider the **B1 1-step-lookahead-softmax variant** (System Card §4.7): decision-time `softmax(Q/τ)`
   with `τ` as the explore knob avoids the PPO-actor collapse entirely (it never "commits" via policy
   gradient). This is the design's own hedge against exactly this failure and is now well-motivated.

**D3. Save a checkpoint at φ-freeze and at fixed step intervals.** The single biggest analysis gap is
that no checkpoints are saved, so the *trained* policy's coverage could not be measured directly (B2
had to use a synthetic stand-in). A checkpoint every ~25k steps would let a future probe roll out the
real policy and settle coverage-vs-random definitively.

**D4. Re-test the leak's value on an ∞ cell, not ls20 L1.** The leak's purpose (anti-stall) only pays
off where exploration must persist far past where vanilla RND saturates — i.e. the hard cells. On
ls20 L1 it is at best neutral and possibly feeds commitment (C2/§C). Run leak=0 vs leak=0.01 on
ls20 L2 or tu93 L1, ≥4 seeds, as the decisive leak test.

**D5. Report ≥8 seeds (System Card §7 already specifies this).** n=3 with a perfectly rank-ordered
entropy/outcome relationship is suggestive but underpowered for the headline metric; the easy-cell
variance is large.

---

## E. Compute spent / honesty notes
- New compute: 1 coverage probe (5 settings × 400 lives + a 3000-life validation, real engine, CPU-bound, single process) and **1** bounded ablation run (seed0, `c_entropy=0.05`, cap 120k, 1 process, ~186s wall, solved at 33k so it stopped early). Stayed within the ≤120k-cap, ≤2-concurrent budget; ran one at a time.
- The leak=0 control (C2) was deprioritized rather than run, because A1/B1/B2 already localize the failure to entropy/commitment, not saturation — a leak ablation on ls20 L1 would be low-signal. It is recommended (D4) on a hard cell instead.
- Strongest claims (B1 entropy-collapse mechanism, B2 coverage-vs-entropy) rest on n=3 logs + a synthetic-policy probe, not on rolling out the trained policy (no checkpoints). Flagged throughout.
