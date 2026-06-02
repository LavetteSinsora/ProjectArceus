# Correctness + faithfulness review — exp_013_5 (proposal D / B1, actor-free lookahead-softmax)

Reviewer pass: bug hunt + faithfulness vs SYSTEM_CARD §4.7 "B1" invariants (i)–(v).
Scope: `lookahead.py`, `trainer.py`, `config.py`, `run.py`, and the reused exp_013_1 / exp_011 / exp_010 helpers.
Method: code read + targeted `uv run python` checks + a full `--smoke` run (passes, 8 updates, metrics round-trip verified).
**No code was modified.**

## Verdict: FAITHFUL. No correctness bug found that would silently corrupt a multi-hour run.

All five accuracy invariants hold. The implementation is a clean, correct realization of B1. Findings
below are (a) a few low-severity nits and (b) ML-dynamics concerns that are *inherent to B1* (not bugs),
which the reviewer was asked to flag.

---

## Invariants (i)–(v) — verdict

**(i) Model used ONLY for the decision; targets learn from reality — HOLDS.**
- Decision: `_lookahead_Q` (lookahead.py:46-54) scores `rndphi.novelty(phi_hat) + γ·value(phi_hat)` where
  `phi_hat = icm.predict_next(phi_s, a)` — the predicted next latent. Used only to build `Q`, then sampled
  (lookahead.py:58-63). No gradient (whole fn under `@torch.no_grad()`).
- Learning target: the intrinsic reward comes from `_phi_and_novelty` (trainer.py:105) = RND novelty at the
  **real** `φ(next_obs)` (exp_013_1/trainer.py:80-98, `icm.encode(next_obs)`), never at `phi_hat`. GAE returns
  are built from this real reward (trainer.py:124-128). Confirmed: no model-predicted state enters the
  value/novelty *training* target.

**(ii) No policy gradient — HOLDS.**
- `grep` over the experiment dir shows no `ppo_update`, no advantage-based policy step, no actor net. The only
  learners are `value_update` (regress to returns, lookahead.py:97-129), `icm_update_from_rollout` (φ/inverse/
  forward), and `_rnd_update` (predictor + leak). `rollout.advantages` is computed by GAE but **only
  `rollout.returns` is consumed** (lookahead.py:103) — advantages are dead, which is harmless.
- `log_probs`/`entropy` from `lookahead_act` are stored/logged only (lookahead.py:86, trainer.py:164); no loss
  reads them.

**(iii) Value trained by TD/GAE on real normalized novelty; φ detached in the value update — HOLDS.**
- `value_update` regresses `V_int(φ(s))` → `rollout.returns` (lookahead.py:103,114-122). φ is recomputed under
  `torch.no_grad()` (lookahead.py:113-114), so the value loss never touches φ; φ trains only via the ICM update.
- Shapes correct: `B = T·N` flatten (lookahead.py:101-103), value-clip implemented (lookahead.py:117-120).
- Stored `rollout.values` during collection = `value(φ(s))` (lookahead.py:81,86), exactly what `_gae_nonepisodic`
  expects as `V(s_t)`; bootstrap `v_last = value(φ(last_obs))` (lookahead.py:90-92) is consistent. Verified.

**(iv) Softmax (not argmax), τ, per-state standardise — HOLDS.**
- `lookahead_act` standardises `Q` over the A actions per state `(Q−mean)/(std+1e-6)`, divides by τ, and samples
  from `Categorical(logits=Qn/τ)` (lookahead.py:60-62). Sampling, not argmax. Confirmed loop-free by design.

**(v) φ frozen after the held-out gate, but ICM heads keep training — HOLDS.**
- Freeze sets `icm.phi.requires_grad_(False)` + `icm.phi.eval()` (trainer.py:142-145). `icm_update_from_rollout`
  runs **every** update (trainer.py:135, no `if not phi_frozen` guard), so the inverse/forward heads keep training
  post-freeze → the lookahead's `predict_next` keeps improving. CNNEncoder has **no BatchNorm/Dropout**
  (model.py:17), so `.eval()` is behaviorally a no-op and frozen φ is bit-stable. `predict_next` reads
  `forward_model`, not φ, so it is unaffected by the freeze. Confirmed.
- Reward-clip (trainer.py:114-118), warm-up (110-112), EMA-std return normalisation (119-121) match the other
  variants exactly (same code path as exp_013_1).

---

## Findings

### Correctness nits (LOW severity — none affect run correctness)

1. **`int_norm_eps` lives in the denominator with EMA-std, not RMS — cosmetic divergence, harmless.**
   `config.py:71` `int_norm_eps=1e-8`; used at `trainer.py:121` `raw_i/(int_ret_std.std + eps)`. `_EMAStd.std`
   floors at 1.0 until first update, so eps is immaterial. No bug; noted only because it differs from canonical
   RND's RMS+epsilon convention. **No action needed.**

2. **`std` Bessel correction → NaN only in the A==1 degenerate case (not reachable here).** `Q.std(-1)`
   (lookahead.py:60) uses the n−1 denominator; with A=1 it returns NaN (verified). All configured games have
   A∈{4,5} (`config.py:13`), so unreachable. The `+1e-6` already makes the std==0 case safe (verified: yields a
   uniform softmax). **No action needed**, but if a 1-action env is ever added this would NaN the policy.

3. **No "fallback-freeze with low holdout_inv" WARNING (present in exp_013_1, absent here).** exp_013_1 prints a
   loud warning when φ is frozen by the max-updates fallback while held-out inverse-acc is still low (poor/
   uncontrollable φ → the RND ruler and, here, the *forward model* are near-random). exp_013_5's freeze block
   (trainer.py:141-147) drops that warning. The smoke run froze at u2 via the fallback with `holdout_inv=0.234`
   silently. **LOW**: logging-only, but for B1 a poor forward model degrades the *decision* (Q ranking), so this
   warning is arguably *more* important here than in exp_013_1. Recommend re-adding for postmortem clarity.

4. **`extrinsic` sign / stop-signal handling — correct.** `extrinsic = rollout.rewards.clone()` is captured
   before `rollout.rewards` is overwritten with normalized novelty (trainer.py:123-124); first-reward detection
   reads `extrinsic` (trainer.py:152-154). The env +1 never enters GAE. Verified, no double-counting.

5. **`RewardForwardFilter` / `_EMAStd` / `raw_mean_ema` state — correctly persisted across updates, never reset
   mid-run** (trainer.py:84-85,93,115-121). `rff` is non-episodic (never reset on done), matching canonical RND
   and the non-episodic GAE. No incorrect cross-update mutation found.

### ML-dynamics concerns specific to B1 (NOT bugs — design risks to watch)

A. **Value trained on real φ(s) but QUERIED at predicted φ̂' — off-distribution extrapolation (the central B1
   risk).** `value_update` only ever sees `φ(s)` (real encodings), but `_lookahead_Q` evaluates `value(phi_hat)`
   where `phi_hat` is the *forward-model output* (lookahead.py:53). Early on, `phi_hat` need not lie on the φ(s)
   manifold the value MLP was fit on, so `V_int(φ̂')` is an extrapolation and can be arbitrary. Concrete evidence
   from the smoke metrics: `v_int_mean` is **negative** (−0.05 … −0.10) while every true return is **non-negative**
   (novelty ≥ 0); the value head freely outputs negatives off-distribution. This only perturbs the *ranking*
   (invariant (i) protects the target), but a systematically biased `V_int(φ̂')` can bias action choice. Worth
   monitoring: forward-model error vs. the spread of `Q` columns.

B. **τ is fixed at 1.0 over *standardised* Q, but the standardisation hides the V_int-vs-novelty scale problem,
   not the cross-term scale.** `Q = nov(φ̂') + γ·V_int(φ̂')` mixes two terms whose relative magnitudes drift as
   `V_int` grows (smoke shows `ret_int_mean ≈ 0.5–1.7`, `v_int_mean` drifting upward from −0.10 toward +0.02 over
   8 updates). Per-state standardisation (lookahead.py:60) makes τ robust to the *overall* Q scale per state, but
   if one term dominates the other across actions the effective exploration temperature shifts even with τ fixed.
   The card says τ is "swept; replaces entropy coef" — recommend sweeping τ (run.py exposes `--tau`).

C. **Lookahead can over-commit when the forward model collapses to action-invariance.** If `predict_next`
   produces nearly action-independent `phi_hat` (a known ICM-forward failure mode), all A columns of `Q` collapse,
   standardisation → ~0, and the policy → uniform (verified behavior). That is *safe* (degrades to random, not to
   a deterministic loop), so the failure mode is benign for the stop-on-first-reward metric. Conversely, a
   confidently-wrong forward model with high inter-action Q spread + low τ could over-commit; the softmax with
   τ=1 mitigates this.

D. **Forward-model quality gates the decision, but φ is frozen on the *inverse* metric only.** The freeze trigger
   reads inverse-acc (holdout) (trainer.py:138-140); the forward model (which the decision actually uses) has no
   gate. If φ freezes early with a still-poor forward model, the lookahead acts on a weak ranker until the heads
   catch up. Not a bug (heads keep training post-freeze, invariant (v)), but it is the B1-specific reason finding
   #3's warning matters.

---

## Items explicitly checked and cleared
- `rollout.policy_entropy_mean` dataclass attribute round-trips into `metrics.jsonl` (verified in smoke output).
- `features=phi_s` stored as float32 (lookahead.py:87); `mean_feature_cosine` consumes it for logging only — fine.
- Device/dtype of the per-action loop: `a_vec` built on `phi_s.device` (lookahead.py:51); `F.one_hot` inside
  `predict_next` (icm.py:80) on-device. No host/device mismatch.
- `done`-masking: novelty zeroed on done-steps in `_phi_and_novelty` (exp_013_1/trainer.py:97); RND update
  excludes done-steps (exp_013_1/trainer.py:106-107). Decision-time novelty at φ̂' is intentionally unmasked
  (decision only) — correct.
- Non-episodic GAE bootstrap/accumulator unmasked (exp_013_1/trainer.py:60-77), matching the intrinsic-stream
  convention; returns = advantages + values are correct.
