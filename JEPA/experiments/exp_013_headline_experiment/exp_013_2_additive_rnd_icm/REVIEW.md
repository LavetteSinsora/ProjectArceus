# exp_013_2 (ADDITIVE RND+ICM) — Code + ML-Dynamics Review

Read-only review. No code modified. Scope: `exp_013_2_additive_rnd_icm/{config,trainer,run}.py`,
reusing `exp_013_1b_leaky_rnd_on_icm_phi/{trainer,rnd_phi,config}.py` and `exp_011_ls20_icm/shared/icm.py`.
Bounded by: `SYSTEM_CARD.md`, `probes/occ_power_limits.md`, `probes/inv_acc_causality` result,
`baseline_random_policy/SUMMARY.md`.

---

## (a) Correctness verdict — NO BUGS FOUND

The implementation does what its docstring claims. Specific checks:

**1. Per-signal normalisation — CORRECT.**
Two independent `_SignalNorm` instances (`norm_icm`, `norm_rnd`, trainer.py:113–114), each a
`RewardForwardFilter` + `_EMAStd` + raw clip, divide-not-centre (`raw / (ema.std + eps)`,
trainer.py:70). Mixed as `r = w*n_icm + (1-w)*n_rnd` (trainer.py:148). This matches the design.

**2. Raw signals computed on CURRENT (pre-update) models — CORRECT.**
`_phi_and_novelty` (RND-on-φ) and `intrinsic_raw_error` (ICM forward error) both run at
trainer.py:135–138, *before* `ppo_update` (156), `icm_update_from_rollout` (160) and `_rnd_update`
(182). Both read the models as they were at rollout time. Good. (Minor: φ(next_obs) is encoded
twice — once in `_phi_and_novelty`, once inside `intrinsic_raw_error` — redundant compute, not a
bug.)

**3. The deliberate change — ICM update runs EVERY update, φ frozen via `requires_grad_(False)` —
VERIFIED CORRECT.**
- Post-freeze, `icm.phi.parameters()` get `requires_grad_(False)` and `icm.phi.eval()`
  (trainer.py:169–171). `icm_update_from_rollout` still calls `icm.encode` (φ forward), gets
  finite values, and `loss.backward()` populates grads only for the inverse/forward heads. φ
  params have `grad=None`, so `clip_grad_norm_(icm.parameters())` skips them and Adam's `step()`
  skips them. So post-freeze the update trains ONLY the heads; φ stays bit-exact fixed. ✓
- `icm.phi.eval()` is harmless: `CNNEncoder` is strided convs + ReLU, **no BatchNorm/Dropout/
  LayerNorm** (confirmed in `exp_010/shared/model.py`), so train/eval mode is identical and φ is
  never put back into `.train()`. No correctness impact. ✓
- `intrinsic_raw_error` and `_phi_and_novelty` are both `@torch.no_grad()`, so frozen φ does not
  break the reward computation. ✓
- Net effect as intended: RND's "ruler" (φ) is stationary after freeze while the ICM-forward
  reward keeps evolving as the heads keep training. ✓

**4. done-step zeroing / shapes / devices — CORRECT.**
`intrinsic_raw_error` zeroes done-step error (icm.py:133–134) and returns `(T,N)` CPU float32;
`_phi_and_novelty` masks done steps (`* (~dones).float()`, exp_013_1/trainer.py:96). Both reshaped
to `(T,N)`, `.numpy()`'d, mixed, cast to float32, wrapped back to a tensor for GAE
(trainer.py:151). Shapes consistent.

**5. Reward wiring / GAE — CORRECT.**
`extrinsic = rollout.rewards.clone()` is isolated for stop-detection only (trainer.py:150,
185–190); the intrinsic mix overwrites `rollout.rewards`. Non-episodic GAE on the summed reward by
default (`intrinsic_episodic=False` inherited). `_gae_nonepisodic` does not mask `dones` — the
intended canonical-RND behavior.

**6. Warmup / checkpoint / holdout / freeze reuse — CORRECT.**
During `norm_warmup_updates` the reward is all-zero (trainer.py:142–144). Freeze trigger, holdout
inv-acc, fallback WARNING, and `_save_ckpt` are reused verbatim from the audited exp_013_1.

**Inherited config note (not a bug, but load-bearing):** `phi_freeze_max_updates=100` is inherited.
At `max_env_steps=500_000`, `total_updates = 500000/(128*16) ≈ 244`. So φ will almost certainly hit
the freeze (adaptive or the u100 fallback) by ~200k env steps; **on ls20 L2 the held-out inv-acc is
~chance, so the fallback path with the "φ NOT controllable" WARNING is the expected branch.** Code
handles it; the implication is a dynamics limitation (below), not a bug.

---

## (b) ML-dynamics limitations — ranked by threat to validity

### THREAT 1 (fatal-to-the-stated-question): Both signals live in a φ that probes show is ~chance-controllable on LS20 → the experiment cannot isolate "additive RND+ICM" from "two noise signals over a near-random φ."
The `inv_acc_causality` result is unambiguous: **held-out inv_acc ≈ 0.25–0.34 (chance for 4 actions)
while on-policy inflates to 0.98**, and φ is otherwise stable (cos≈0.95). So φ does NOT encode
controllable structure on LS20; on-policy 0.98 is the narrowing-policy artifact. Both 13_2 rewards
are functions of this same φ:
- `norm(ICM_forward)` = error of a forward model predicting a near-random φ(s') from φ(s),a. If φ
  isn't action-controllable, the forward model's residual is dominated by the irreducible/aliased
  part of φ, i.e. it rewards "φ-values the forward head hasn't fit yet," not "states the agent
  controllably reached." Closer to a stochastic-φ novelty than to curiosity.
- `RND-on-φ` = predictor error on the same φ. With a near-random φ, this is novelty of φ-vectors,
  which on a near-random embedding need not align with state novelty.
Summing two functions of the same weak φ does **not** add an independent view; it inherits and
compounds the φ-quality limitation. **No "additive" verdict drawn here can be attributed to the
RND+ICM combination rather than to φ.** This is the single biggest validity threat.

### THREAT 2 (very high): Entropy collapse is the dominant, documented LS20 failure mode, and 13_2 does NOTHING new about it.
`occ_power_limits.md` is decisive: on ls20 **L1**, outcome is perfectly rank-ordered by policy
entropy across 3 seeds; the worse-than-random failures are entropy collapse, and a single
`c_entropy` 0.01→0.05 bump flipped the worst seed censored→solved. 13_2 inherits `c_entropy=0.05`
(already the "fix") but adds no entropy floor/KL-to-uniform/temperature cap. On the **harder** L2
(E=∞, deeper required chain) the commitment pressure is at least as strong, and the reward mix does
not touch the actor objective. **Expect 13_2 to be governed by the same entropy dynamics as 13_1
regardless of the RND/ICM mix.** If it collapses, the additive design will be falsely blamed (or
falsely credited if it happens not to collapse on a given seed). At n=1 seed (default run) this is
pure variance.

### THREAT 3 (high): Redundancy — on a deterministic maze the two signals are largely the same "unvisited-φ" signal, so the sum averages two correlated noisy estimators rather than fusing complementary information.
LS20 is deterministic. "ICM forward-prediction error" and "RND predictor error" are both
*prediction errors in φ-space that shrink with visitation*; once the forward/predictor models have
seen a region, both drop. They differ only in functional form (a,φ-conditioned regression vs.
fixed-random-target distillation) and in that RND has the leak (revives) while the ICM forward head
does not. On a deterministic env these are strongly positively correlated. Summing two correlated
noisy signals mostly **averages noise**; it does not obviously add directed information. The design
asserts complementarity but provides no mechanism that would make ICM-forward reward a region RND
misses (or vice versa) on a deterministic maze.

### THREAT 4 (high): ICM-forward collapse degenerates the mix into a down-weighted exp_013_1, and `norm(ICM)` then amplifies noise.
exp_011 showed the ICM forward error collapses. Here only RND has a leak; the ICM forward head has
none, and it keeps training every update (the deliberate change), so its raw error is **driven
toward small**. Two consequences:
- As `icm_raw → small`, `r ≈ (1-w)·norm(RND)`: 13_2 becomes a **0.5×-weighted exp_013_1**. The
  "additive" experiment asymptotically tests a *scaled* version of the thing it's supposed to be
  compared against. (And because both signals are normalised to unit return-std, the down-weight is
  partly undone by normalisation — see THREAT 5 — so the asymptote is closer to "exp_013_1 with a
  noisier reward" than "0.5× exp_013_1.")
- `norm(ICM) = icm_raw / (ema.std+eps)` divides a collapsing signal by a *shrinking* std. As both
  numerator and denominator → 0, the ratio's variance is dominated by sampling noise and the EMA
  lag. So `w·norm(ICM)` injects roughly mean-1 **noise** into the reward — a persistent distractor
  on top of RND. This is a way the sum can be **strictly worse than RND-alone**.

### THREAT 5 (medium-high): Normalisation equalises SCALE, not INFORMATIVENESS, so w=0.5 can let the less-informative signal drag the more-informative one.
Per-signal divide-by-return-std guarantees each signal contributes ~unit magnitude regardless of
how informative it is. If (say) RND-on-φ carries the directed signal and ICM-forward is mostly
collapsed noise (THREAT 4), then 50/50 forces an equal-magnitude noise channel into the reward the
value head must fit. There is no reason 0.5 is optimal, and the worst case (sum < best single) is
exactly when one signal is much weaker — which the priors suggest is the case here. Normalisation
makes this worse, not better: it *guarantees* the weak signal gets equal influence.

### THREAT 6 (medium): Non-stationary reward + single value head + value-lag, now with TWO decaying clocks.
The ICM-forward reward is non-stationary (decays as heads learn) and RND is non-stationary (leak
revives). The single intrinsic value head chases a moving target; with two superimposed
non-stationary signals on different schedules (one monotone-ish decay, one leak-revived oscillation)
the value target is noisier than in 13_1. Combined with non-episodic GAE (value persists across
death), this can sustain commitment to a fixated region (feeds THREAT 2). Not catastrophic alone but
compounds 1–2.

### THREAT 7 (design-of-experiment): Cell choice ls20 L2 (E=∞) gives a binary "solve / not-solve" outcome with no quantitative baseline — at n=1 it cannot answer "is additive useful?"
L2 is E=∞ for random, so any solve is ∞× better than random — good for showing *power*, useless for
*comparison*: there is no random rate to beat quantitatively, and the only readout is solve-or-not.
With one seed, "13_2 solved L2" vs "13_1 didn't" is indistinguishable from seed variance (recall the
L1 result hinged entirely on whether entropy happened to collapse on a given seed). The experiment
as configured (single run, E=∞ cell, no matched 13_1/RND-only/ICM-only arms) **cannot** answer
whether the additive combination is the cause of any outcome.

---

## (c) Bottom line

**As set up, this experiment cannot answer its question ("is it useful to combine RND+ICM?").**
Three independent reasons, each sufficient:
1. **Confounded by φ.** Both summands are functions of a φ that is ~chance-controllable on LS20
   (held-out inv_acc ≈ chance). Any result is attributable to φ quality, not to the RND+ICM
   combination (THREAT 1).
2. **Confounded by entropy collapse.** The documented dominant LS20 failure is actor commitment;
   13_2 adds nothing to address it, so outcomes will be governed by whether entropy collapses on the
   seed, not by the reward mix (THREAT 2).
3. **No discriminating comparison.** E=∞ cell + binary outcome + (default) n=1 + no matched ablation
   arms ⇒ no way to attribute a solve/no-solve to "additive" vs. variance (THREAT 7).

Additionally the mix is likely to **degenerate** toward a noisy, down-weighted exp_013_1 as
ICM-forward collapses (THREAT 4), and 50/50 scale-normalisation can make the sum worse than the
better single signal (THREAT 5). So even a clean run risks measuring "RND-alone plus injected noise."

### Cheaper / sharper way to actually answer "is additive useful?"
Do a controlled ablation on a cell with quantitative headroom, NOT a single L2 run:

1. **Pick a finite-E cell with headroom: tu93 L1 (~500k) or re86 L1 (~2M)** (per
   `baseline_random_policy/SUMMARY.md`), so "beats random's E" is a real, quantitative target. Use
   ls20 L1 only as a regression/sanity cell, never as the power test. Reserve E=∞ cells for a
   secondary "can it ever solve" check.
2. **Run the 4-arm matched ablation, same seeds, same budget:** (i) RND-on-φ only (= exp_013_1),
   (ii) ICM-forward only, (iii) additive w=0.5 (this exp), (iv) ideally a **w-sweep**
   {0, 0.25, 0.5, 0.75, 1.0} — note w=0 and w=1 already recover arms (i)/(ii), so the sweep IS the
   ablation. "Additive useful" ⇔ some interior w beats both endpoints on first-reward step at
   matched seeds. This is the single sharpest, cheapest control and 13_2 already exposes `--w-icm`.
3. **≥8 seeds** (SYSTEM_CARD §7) — the L1 evidence shows easy-cell/seed variance dominates;
   n=1 on an E=∞ cell is uninterpretable.
4. **Control the confounds before claiming anything about the mix:** (a) add the entropy floor / KL-
   to-uniform / temperature-cap from `occ_power_limits.md` D2 so outcomes aren't dictated by
   collapse; (b) log/report the **correlation between `n_icm` and `n_rnd` per update** — if they are
   highly correlated (expected on a deterministic maze, THREAT 3), the additive hypothesis is dead
   on arrival and no full run is needed.
5. **Cheapest pre-run probe (no training):** on a fixed rollout, compute `corr(norm(ICM_forward),
   norm(RND-on-φ))` and each signal's correlation with true visitation count. If the two are ~1:1
   correlated and/or neither tracks visitation (likely, given φ≈chance), that *alone* answers
   "additive isn't useful here" for a few seconds of compute, before spending a 500k-step run.

**One-line verdict:** code is correct and faithful to its spec; the *learning setup* cannot
isolate the additive combination from φ-quality and entropy-collapse confounds, and the chosen
single-seed E=∞ cell yields no quantitative discriminator — replace with a w-sweep (w=0/1 = the
single-signal controls) on a finite-E headroom cell with an entropy floor and a cheap
signal-correlation probe first.
