# Method improvements A/B/C/D — findings + prioritized fixes

*Resumes the (dead) method-improvement agent. Built on `run_diagnosis.md`, `collapse_verdict.md`,
the τ-sweep runs on disk, and a read of the method code. Config changes ALREADY applied: D τ
1.0→0.25; B/C φ-freeze gate 0.90→0.70; c_entropy 0.05→0.10.*

## D (lookahead) — τ tuning is NOT enough; the Q signal is the problem
- **τ-sweep on tu93 L2** (cheap cell, random solves ~2k): τ=0.15 censored, τ=0.25 censored,
  **τ=0.40 solved at 14,640** — but that's still ~7× slower than random (~2k) and ~19× slower than
  A/B/C (784). **Monotone: the more you trust D's Q (lower τ), the worse it does** → D's
  lookahead Q (novelty + γV̂) is *anti-informative* on this cell. τ is a band-aid.
- Sanity (earlier): at τ=0.25, D was 0/1 on tu93 L2 while A/B/C all solved in 784.
- **Action:** don't spend frontier budget on D as-is. Either (a) set τ≈0.4 AND drop the per-state
  Q-standardization (it strips magnitude, amplifying noise across only n_actions logits), or
  (b) ablate novelty-only vs novelty+V̂ to find which term mis-ranks actions. D needs a structural
  rethink, not a temperature.

## C (additive) — normalizer self-suppression on long runs (concrete code fix)
- Mechanism (from `exp_013_2/trainer.py::_SignalNorm`): each signal is divided by the **EMA-std of
  its discounted RETURN** (`RewardForwardFilter` → `_EMAStd`). On long non-episodic runs that
  return-std **inflates** (diagnosis saw `icm_ret_std` → 54–98), so `n_icm = raw/std` is **divided
  down to ~0** → the ICM half of the reward self-suppresses → field goes dead → collapse. C solved
  g50t L1 because it finished fast (updates 16–57) *before* the std inflated; on long re86 it died.
- **Proposed fix (not yet applied — needs a decision):** stop the return-std from running away —
  options: (i) cap/clip `ema.std` to a max, (ii) normalize by the std of the **reward** rather than
  the discounted return, or (iii) periodically reset the `RewardForwardFilter`. (i) is the smallest,
  safest change. Also reconsider fixed w=0.5 vs an adaptive mix.

## B (RND-on-ICM-φ) — φ-controllability is the ceiling, and it's cell-dependent
- Held-out inv_acc reaches ~0.72–0.76 on ls20/g50t (the 0.70 gate now fires adaptively there) but
  is **stuck at chance (~0.19) on re86** → on re86 the RND ruler is computed on a near-random φ, so
  B can't work there regardless of the gate. **Action:** on re86 prefer A (frozen-φ); don't expect
  B to beat A where φ won't become controllable. Consider a longer φ-warmup / higher icm_lr probe on
  re86 to test whether φ is fundamentally un-inverse-dynamics-controllable there.

## A (frozen-φ RND+leak) — keep; most stable
- Diagnosis flagged A as the most stable run (no φ-freeze gate to misfire, no controllability
  dependence). It's the natural frontier bet on cells where ICM-φ fails (ls20/re86). Lever sweep
  (leak μ, rnd_lr, feature dim) not yet run — see `exp_014_1_rnd_saturation_diagnosis` for how μ
  trades off saturation vs floor.

## Cross-cutting (ties to `collapse_verdict.md` + leaky RND)
- The dominant collapse mode is **reward-field starvation** (signal → flat → slow entropy bleed),
  NOT value-lag (that was an ls20-probe-only artifact). The **leak** directly addresses this by
  keeping the field alive (`exp_014_1`), and c_entropy 0.10 resists the bleed. Both are the right
  levers; the C self-suppression fix above is the third.

## Priority before the frontier sweep
1. Apply the C return-std cap (cheap, fixes a clear self-suppression bug).
2. Decide D: τ=0.4 + drop Q-standardization, or bench it.
3. Run A/B/C(fixed)/D(fixed) on the frontier; expect A strongest on ls20/re86, B on g50t.
