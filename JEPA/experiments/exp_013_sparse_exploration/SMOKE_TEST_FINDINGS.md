# exp_013 Sparse-Exploration Harness — Behavioral Smoke Test

**Date:** 2026-05-31  **Hardware:** Apple M3 Pro (MPS)  **Tester:** automated behavioral smoke test (pre-calibration gate)

**Scope:** Verify the exp_013 ICM/RND harness *behaves* correctly before a large multi-seed
calibration. Primary probe = ls20 **L1** (`--level 0`), the cheap, known-solvable case
(uniform-random baseline ≈ 50,000 env steps to first reward; 13-move minimum solution).
Plumbing was already verified by the owner; this run targets *behavior*, and especially the
known **ICM curiosity-collapse** failure mode.

Runs: ICM & RND, ls20 L1, `--max-env-steps 200000`, `--eval-every 25`, stop-on-first-reward ON.
Cap chosen < the requested 600k purely for smoke-test wall-clock under MPS contention; 200k still
comfortably exceeds the ~50k random baseline, so first-reward is observable. Throughput on this
M3 Pro was ~50-100 sps under 2-4-way contention with a background exp_012 job also on MPS.

**What actually ran to completion:** ICM seed 0 and RND seed 0 are the two clean, fully
characterised runs and are the basis of the verdict. RND s0 ran to its natural stop (solved).
ICM s0 was **manually censored at 92,160 env steps** (I stopped it rather than burn ~25 more
min to the 200k cap once the behaviour was unambiguous; no result.json was written for it).
ICM seed 1 and RND seed 1 were started but stopped early (≈4-6k steps each) to free MPS so the
seed-0 pair could finish faster; their first ~2 updates reproduce the seed-0 collapse signature
exactly (ICM raw 44.6→1.36, norm 0.0487→0.0017 by update 2) and are cited as a second-seed
sanity check on the *curiosity* behaviour only — not as independent first-reward data points.
Stale run dirs from the killed seed-1 batches were left in `runs/` (see Cleanup note).

---

## 1. First-reward results (test 1 + stop-rule test 3)

| method | seed | env_steps_to_first_reward | solved | censored | total_env_steps | wall (s) |
|--------|------|---------------------------|--------|----------|-----------------|----------|
| **RND**| 0    | **46,448**                | yes    | no       | 47,104          | 686      |
| **ICM**| 0    | none (censored)           | no     | yes\*    | 92,160\*        | manual\* |
| RND    | 1    | (not run to completion — stopped ~6k for MPS)  | — | — | ~6,144 | — |
| ICM    | 1    | (not run to completion — stopped ~6k for MPS)  | — | — | ~6,144 | — |

\* ICM s0 was manually stopped at 92,160 steps (see header note); it had found **no** reward by
then — i.e. censored at ≥92k, vs the cap of 200k it never reached.

Random-policy reference (ls20 L1, from project memory / Markov-chain analysis):
≈ 50,000 env steps to first reward (p ≈ 8.6e-4 per life). A working intrinsic method should
reach first reward in the same order of magnitude or faster.

**Headline:** RND **beat** the random baseline (46.4k < ~50k) and solved cleanly. ICM was
**worse than random** — no reward by 92k (1.8× the random expectation, 2.0× RND's solve) when I
censored it. On the easy, known-solvable level this is a clear behavioural red flag for the ICM arm.

_Stop-rule sanity (test 3): PASS._
- Solved case (RND s0): `result.json` has `solved=true, censored=false,
  env_steps_to_first_reward=46448`, and the run stopped promptly the same update the reward
  appeared (`total_env_steps=47104`, i.e. it halted within one rollout of the first reward). The
  log printed `*** FIRST REWARD at ~46448 env steps (update 23) ***` then `DONE`.
- Censored case: ICM s0 reached 92k with `env_steps_to_first_reward=None` throughout (would have
  been written `censored=true` had it hit the 200k cap; I stopped it manually before then, so no
  result.json — but the in-flight metric confirms the censoring logic, frr stayed `None`).
  The mechanism (`solved = first_reward_step is not None`, `censored = not solved`) is correct.

---

## 2. Curiosity-collapse diagnosis (the key check)

**Verdict: the "normalized ICM" fix does NOT keep curiosity alive in the way intended. It does not
*permanently* zero the bonus (the bonus revives later — see below), but the cumulative normalizer
is poisoned by ICM's enormous startup transient and crushes the bonus to ~1e-4 for the ENTIRE
first ~43k env steps. That dead-zone is precisely the window in which the agent should be
exploring toward its first reward, and it wastes it. RND, by contrast, keeps a healthy O(0.003-0.05)
bonus throughout and solves at 46k. So on the headline metric the ICM normalizer behaves like a
bug, even though the bonus is not literally dead forever.**

### Full ICM s0 trajectory (45 updates / 92k steps; raw / norm / retstd / entropy / v_int)

```
 u  1 step  2048  raw=61.3   norm=0.0488    retstd=1.26e3  ent=1.386  v_int=-0.06
 u  5 step 10240  raw=0.401  norm=3.05e-4   retstd=1.32e3  ent=1.383  v_int= 0.21
 u  9 step 18432  raw=0.102  norm=8.74e-5   retstd=1.16e3  ent=1.316  v_int= 0.16   <- bonus ~dead
 u 13 step 26624  raw=0.335  norm=3.30e-4   retstd=1.02e3  ent=1.317  v_int= 0.14
 u 17 step 34816  raw=0.122  norm=1.34e-4   retstd= 914    ent=1.386  v_int= 0.10   <- ent back to MAX
 u 21 step 43008  raw=0.166  norm=1.98e-4   retstd= 837    ent=1.385  v_int= 0.06
 u 25 step 51200  raw=14.7   norm=0.0184    retstd= 797    ent=1.153  v_int= 0.38   <- REVIVAL
 u 29 step 59392  raw=6.64   norm=0.00886   retstd= 750    ent=1.212  v_int= 0.56
 u 33 step 67584  raw=9.44   norm=0.0133    retstd= 709    ent=1.104  v_int= 0.75
 u 37 step 75776  raw=24.2   norm=0.0330    retstd= 733    ent=0.985  v_int= 1.13
 u 41 step 83968  raw=16.2   norm=0.0198    retstd= 822    ent=1.033  v_int= 1.78
 u 45 step 92160  raw=13.9   norm=0.0172    retstd= 806    ent=1.144  v_int= 1.65   <- still NO reward
```
ICM raw: init=61.3, min=0.068, last=13.9. ICM norm: init=0.0488, **min=7.9e-5**, max=0.0488.

### Full RND s0 trajectory (23 updates / 47k steps; solved)

```
 u  1 step  2048  raw=0.0856  norm=0.0488   retstd=1.75  ent=1.386  v_int=-0.06
 u  5 step 10240  raw=0.0251  norm=0.0220   retstd=1.14  ent=1.372  v_int= 1.21
 u  9 step 18432  raw=0.0183  norm=0.0166   retstd=1.10  ent=1.373  v_int= 1.52
 u 13 step 26624  raw=0.0127  norm=0.0112   retstd=1.13  ent=1.374  v_int= 1.47
 u 17 step 34816  raw=0.00832 norm=0.00712  retstd=1.17  ent=1.316  v_int= 1.31
 u 21 step 43008  raw=0.00498 norm=0.00412  retstd=1.21  ent=1.366  v_int= 0.97
 u 23 step 47104  raw=0.00374 norm=0.00305  retstd=1.23  ent=1.370  v_int= 0.78   <- SOLVED @46448
```
RND raw: 0.086 → 0.0037 (smooth). RND norm: 0.0488 → 0.00305 (never below ~3e-3). retstd stable ~1.1-1.2.

### Interpretation

- **ICM phase 1 (u1-21, 0-43k) — the poisoned-normalizer dead-zone.** Raw forward error collapses
  61.3 → ~0.1 within ~5 updates (ls20 is tiny and near-deterministic, so the forward model nails
  it almost instantly — same dynamic as exp_011). But `intrinsic_return_std` stays pinned at
  ~0.8-1.3e3, because the *cumulative* `RunningMeanStd` baked in the giant update-1 transient and
  cannot down-weight it. So `norm = raw / (sqrt(var)+eps)` is divided to **~1e-4** and the
  intrinsic signal is effectively off for the entire first 43k steps. Entropy even returns to the
  ln(4)=1.386 MAX at u17 — the policy is purely random in exactly the window it should be exploring.
- **ICM phase 2 (u25+, 51k+) — partial revival.** Once the (random) policy wanders into states it
  hasn't modelled, raw spikes to O(10-24), and because the cumulative std has *slowly* decayed
  (1300→~750 as the startup sample ages out of the running mean), the normalized bonus recovers to
  O(0.02-0.03) and entropy finally drops (→~1.0). So the bonus is not dead forever — but the damage
  (a wasted 43k-step head start) is done, and **ICM still had not found any reward by 92k.**
- **RND is textbook healthy throughout.** Raw is small and quasi-stationary from step 1, so retstd
  tracks it (~1.1-1.2), the normalized bonus holds at O(0.003-0.05) the whole run, entropy is
  gently shaped, v_int rises then falls as the agent homes in, and it solves at 46k — *faster than
  random.* This is exactly the behaviour the shared normalizer was supposed to give both methods.
- **Net:** the normalizer keeps RND alive but mis-scales ICM. Does normalized-ICM keep r^i alive?
  **Not during the critical early-exploration window — no.** It is alive only after a long dead
  period, by which point ICM is already underperforming random on the headline metric.

### ROOT CAUSE — a real harness bug / design flaw

`shared/trainer.py` builds the normalizer from a **non-decaying cumulative** estimator:

```python
rff = RewardForwardFilter(cfg.gamma_int)   # discounted intrinsic RETURN per env
int_ret_rms = RunningMeanStd()             # Welford: count grows forever, NO decay
...
rems = np.stack([rff.update(raw_i[t]) for t in range(T)])
int_ret_rms.update(rems)
norm_i = raw_i / (np.sqrt(int_ret_rms.var) + cfg.int_norm_eps)
```

`RunningMeanStd` (exp_012/shared/rnd.py) is a Chan/Welford accumulator whose `count` only ever
grows. For RND this is fine: raw novelty is stationary, so the variance estimate is meaningful and
stable. For **ICM it is pathological**: the update-1 raw error is ~700× larger than the converged
error (61 vs ~0.1). That giant transient is baked permanently into `int_ret_rms.var` (≈1.3e3) and,
because the estimator never down-weights old samples, it never recovers as raw shrinks. The
denominator is frozen huge while the numerator decays → the normalized bonus is divided into
oblivion. This is **the exp_011 collapse wearing a different hat**: the README claims the shared
normalizer is "the 2018 large-scale-curiosity fix for the frozen-η collapse," but the fix as
implemented does not hold for a non-stationary raw signal fed through a non-decaying RMS.

(Note: this same cumulative-RMS would also slowly drift for RND on a non-stationary game, but RND's
raw signal is benign here so it is not triggered in this smoke test.)

---

## 3. Bugs / scale problems found

1. **[MAJOR — behavioral, the headline finding] The normalizer mis-scales ICM (startup-transient
   poisoning).** See §2. ICM's update-1 raw forward error (~61) is ~700× its converged value
   (~0.1). The shared `int_ret_rms = RunningMeanStd()` is a *cumulative* (never-decaying) Welford
   estimator; that one giant transient pins `intrinsic_return_std` at ~1e3 and suppresses the
   normalized ICM bonus to ~1e-4 for the entire first ~43k env steps — the exact window the agent
   needs for first-reward exploration. Consequence: ICM was **worse than random** (no reward by
   92k vs ~50k random / 46k RND). This is the exp_011 collapse relocated into the normalizer, not
   cured by it. The bonus does revive after ~50k (the transient slowly ages out and novel states
   re-spike raw), so it is a *scaling/transient* bug, not a permanent zero — but it is decisive on
   the headline metric. RND is unaffected because its raw signal is small and stationary from t=0.

2. **[MINOR — observed scale fact, may bite other games] ICM raw is wildly non-stationary at
   startup; RND raw is benign.** Any normalizer choice for the calibration must be robust to a
   large, short-lived initial spike on the ICM side. The current cumulative RMS is not.

3. **[INFRA — throughput]** At 4 concurrent runs on this M3 Pro (with a background exp_012 job
   also on MPS) per-run throughput is ~50 sps; ~100 sps at 2-way; nominally 150-400 sps solo.
   600k-step censored runs are multi-hour at 4-way. Budget concurrency accordingly.

No crashes, NaNs, or plumbing faults were observed in any run. The dual-stream wiring, done-step
zeroing, stop-on-first-reward, result.json/metrics.jsonl, and level-0(=L1) routing all behaved
correctly. The harness code itself was **not modified** (the only issue is the normalizer design
choice in `trainer.py`, documented above as a fix recommendation, not silently changed).

---

## 4. GO / NO-GO

- **RND arm: GO.** Behaves correctly, beats the random baseline (46.4k < ~50k), curiosity stays
  alive and well-scaled, stop rule works. Ready for the multi-seed calibration.
- **ICM arm: NO-GO as currently implemented.** It is *worse than random* on the easiest,
  known-solvable level because the cumulative normalizer suppresses the bonus to ~1e-4 through the
  whole first-reward exploration window. Running the full calibration on this ICM as-is would burn
  a lot of (multi-hour, multi-seed, multi-game) compute to mostly produce censored ICM runs and a
  misleading "ICM ≪ RND" conclusion that is really a normalizer artifact, not a property of ICM.
  **Fix the normalizer, re-run this exact smoke, then GO.**

**Overall recommendation: HOLD the full calibration until the ICM normalizer is fixed.** The fix
is small and the smoke is cheap (~15-30 min for the ICM+RND ls20-L1 pair at 2-way). Do not launch
the large multi-seed/multi-game sweep with the current ICM.

### Suggested fixes (ICM normalizer — pick one; all are small, local edits to `trainer.py`)
1. **Decaying / EMA normalizer (preferred):** replace the cumulative `RunningMeanStd` used for the
   intrinsic-return std with an exponential-moving variance (e.g. decay 0.99) or a fixed-window
   RMS, so the denominator tracks the *current* novelty scale instead of being pinned by the
   startup spike. This also future-proofs RND on non-stationary games.
2. **Warmup-skip:** exclude the first K updates (e.g. K=2-5) from `int_ret_rms` updates so the
   untrained-forward-model transient never enters the variance estimate. Cheapest one-liner.
3. **Clip raw before the forward filter:** clip `raw_i` to a running percentile before
   `RewardForwardFilter`, so a single huge transient cannot dominate the variance.
- **Re-test gate:** after the fix, re-run ICM ls20 L1 (2 seeds) and require
  `intrinsic_reward_norm_mean` to stay O(0.01-1) for the first ~40k steps (NOT decay to ~1e-4),
  and `env_steps_to_first_reward` to land at or below the ~50k random baseline. Only then GO.

### Suggested per-game step caps (calibration guidance)
- ls20 L1 (`--level 0`): a working method should solve in ≲50-60k; a **150-200k cap** is plenty
  and cleanly separates "beats random" from "censored". (RND used 47k; the requested 600k is
  overkill for L1 and just lengthens censored runs.)
- Harder levels (ls20 L2 = `--level 1`, and re86/tu93/g50t early levels): expect censoring; a
  **400k cap** is a reasonable smoke budget to confirm graceful degradation, but is not enough to
  claim a solve. Scale the cap with the random baseline per game×level once
  `baseline_random_policy/` numbers are available (that Monte-Carlo job was still running during
  this smoke — wire its outputs into the cap selection).
- Concurrency: prefer **2 concurrent** runs on this M3 Pro for clean ~100 sps turnaround; 4-way
  works but halves per-run throughput.

## 5. Cleanup note

Left in `runs/` (safe to delete): the smoke-plumbing run `exp013_icm_ls20_L1_seed0_20260531_170556`
and the four short/killed seed-1 + first-batch dirs (`*_170618`, `*_170620`, `*_170622`,
`*_170624`, `*_171228`, `*_171230`). The two runs backing this report are
`exp013_rnd_ls20_L1_seed0_20260531_170938` (solved, has result.json) and
`exp013_icm_ls20_L1_seed0_20260531_170937` (manually censored at 92k, metrics.jsonl only).
