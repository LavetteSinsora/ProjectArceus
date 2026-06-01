# exp_013 — Run Diagnosis: new exploration methods vs icm/rnd baselines

> ⚠️ **CORRECTION (see `collapse_verdict.md`).** Two claims in this doc were overturned by a
> later evidence-grounded review:
> 1. The **"separate ext/int value heads protect icm from collapse"** framing is **wrong** — the
>    baseline `rnd` runs (which *have* dual heads, c_entropy=0.01, no φ) collapse *harder* (6/8 → ~0)
>    than our methods. What separates collapse from non-collapse is **time spent on a dead reward
>    field**, not architecture: runs that solve fast (<~60 updates) exit before the entropy bleed.
> 2. "mean(ret−V)≈0 ⇒ not value-lag" is correct *for re86*, but value-lag **is** real on ls20
>    (in a reconstructed-loop probe). Collapse has **two distinct mechanisms on two cells** — see verdict.

**Scope:** analysis of EXISTING `metrics.jsonl` / `result.json` only (no method code modified, no new training runs). Real runs filtered by `total_env_steps > 1000` (tiny 256-step smokes excluded). Method ↔ label map:
- **A** = `exp_013_1_rnd_icm`, `phi_mode=frozen` (frozen-φ + RND-on-φ)
- **B** = `exp_013_1_rnd_icm`, `phi_mode=icm` (trainable φ via ICM inverse model + RND-on-φ)
- **C** = `exp_013_2_rnd_icm_additive` (`rnd_icm_additive`, ICM-novelty + RND-on-φ summed, w_icm=0.5)
- **D** = `exp_013_5_lookahead` (`lookahead_mcts`, actor-free depth-1 Boltzmann over Q = novelty + γV̂)
- **E** = `exp_013_4_disagreement` — **no real runs on disk** (only a 256-step smoke; see below).

All of A–D share: `gamma=0.95, gae_lambda=0.95, c_value=1.0, c_entropy=0.05, rollout=128×16, epochs=4, grad_clip=0.5, phi_freeze_max_updates=100, freeze_metric=holdout(≥0.9), reward_clip_k=5`. The baseline `icm`/`rnd` use a **different, RND-style architecture**: **separate ext/int value heads**, `gamma_ext=0.999, gamma_int=0.99, ext_coef=2.0, int_coef=1.0, c_entropy=0.01`, **no φ-freeze**.

---

## Solve-rate table (real runs)

| method | g50t L1 | re86 L1 | median env-steps-to-first-reward |
|---|---|---|---|
| baseline **icm** (n=8) | 3/8 | **7/8** | g50t 177k / re86 450k |
| baseline **rnd** (n=8) | 4/8 | 3/8 | g50t 122k / re86 424k |
| **A** frozen-φ (n=2) | 1/2 | 1/2 | g50t 147k / re86 120k |
| **B** rnd_icm φ=icm (n=2) | **2/2** | **0/2** | g50t 102k |
| **C** additive (n=2) | **2/2** | **0/2** | g50t 73k |
| **D** lookahead (n=2) | **0/2** | 1/2 | re86 626k |

**Sample sizes are tiny (n=2 per new-method cell).** The "regression" is real but on a single cell (re86 L1) and at n=2; on **g50t L1 the new methods actually BEAT both baselines** (B/C 2/2 vs icm 3/8, rnd 4/8). State n in any conclusion.

Random-policy reachability (from `baseline_random_policy/SUMMARY.md`): **re86 L1 E≈2.0M steps (p≈5e-5), finite**; **g50t L1 E=∞ (0 clears in 30k lives, p≤1e-4)**. This is load-bearing for puzzle 1.

---

## Puzzle 1 — D (lookahead): 0/2 g50t vs 1/2 re86. Why?

**Finding: D's softmax policy is too FLAT (τ=1.0 over per-state-standardized Q ⇒ ~uniform-random). It is essentially a random walk, so it only solves the game that random can reach.**

Evidence (`lookahead.py:58-63`): `π = softmax( standardize_A(Q) / τ )`, where Q is standardized across the **A=5 actions per state** (mean 0, std 1) and τ=1.0. A softmax of unit-variance logits over 5 actions has **mean entropy ≈ 1.22** and **mean max-prob ≈ 0.51** (Monte-Carlo, 20k draws). Observed `policy_entropy` in every D run sits at **~1.30–1.39** (g50t s0 t50=1.33, re86 s0 t50=1.31; max possible ln5=1.61) — i.e. the policy never departs meaningfully from uniform. The "best" lookahead action is picked barely above chance. The lookahead Q exists but cannot steer the policy at τ=1.0.

Because the policy ≈ random:
- **g50t L1 is unreachable by random (E=∞)** → D solves 0/2 (g50t s0/s1 both censored at 299k).
- **re86 L1 is reachable by random (E≈2.0M)** → D stumbles into a solve: re86 s0 solved at **626k steps** (consistent with ~2M expectation × luck), s1 censored at 999k. The 1/2 vs 0/2 split is explained by **base-game random reachability, not by D doing anything informative**.

Secondary D issues:
- **φ-freeze fires by the FALLBACK (`update≥100` ⇒ step 204800) in ALL four runs**, never by the holdout≥0.9 gate. At freeze, `holdout_inv_acc` = 0.72 (g50t s0), 0.50 (re86 s0), **0.21≈chance (re86 s1)** — φ frozen while NOT controllable.
- **re86 s1 (failed):** after freeze, `novelty_raw_mean` collapses to ~5e-4, `intrinsic_return_std`→0.023, holdout stuck 0.21 (1/5=chance). Dead novelty on a dead φ → no exploration signal at all.
- **value_loss spikes** to 507 (g50t s0), 226 (re86 s0): V̂_int target on raw novelty is spiky; transient, recovers, no NaN.

τ tuning math: at **τ=0.25**, the same standardized Q gives entropy ≈ 0.36 and max-prob ≈ 0.85 — a genuinely directed greedy-ish policy. **D's τ=1.0 is the primary bug.**

---

## Puzzle 2 — B (re86 0/2) & C (re86 0/2) regress vs icm (7/8). Why?

**Finding: the failure is ENTROPY COLLAPSE onto a degenerate RND-on-φ novelty signal computed on an uncontrollable (chance-holdout) frozen φ — a self-reinforcing loop. The baseline icm avoids it via a different architecture (separate ext/int critics + high entropy), NOT via a better intrinsic reward.**

**B, re86 s0 (the smoking gun):**
- `policy_entropy`: 1.61 → **9e-4 by t25, min 9e-6** — policy becomes deterministic (1 action). (`/tmp/scan.py` flag `ENT_COLLAPSE=9e-06`.)
- `holdout_inv_acc` **stuck at 0.19 = chance (1/5)** the entire run, while `inverse_acc` (train) = 0.96. → **the inverse model memorized training transitions but generalizes at chance: φ is NOT controllable.**
- `novelty_raw_mean`: 0.157 → 0.001 (RND-on-φ novelty dies as φ stops carrying action-relevant structure); `intrinsic_return_std`→0.004.
- φ-frozen by FALLBACK at step 204800 with **holdout=0.19** — exactly the "fallback-with-low-holdout, φ not controllable" warning case. φ then never recovers.

**B, re86 s1:** softer version — entropy 1.61→0.72, holdout plateaus 0.35, never solves.

**C, re86 (both seeds):** entropy 1.61→0.66/0.84 (collapsing), holdout plateaus ~0.55 (mediocre), and the **ICM return-normalizer blows up: `icm_ret_std` → 50 (s0) / 98 (s1)** while `rnd_ret_std` stays ~0.3. Under fixed `w_icm=0.5`, the post-normalization ICM contribution is squashed by its own exploding std; the combined signal still drives a slow entropy collapse. Never solves.

**Baseline icm, re86 s0 (SOLVED 22k):** `policy_entropy` stays **1.45–1.61 throughout**; separate `v_ext`/`v_int` heads; `intrinsic_return_std` rises to ~15 cleanly. Even the failed icm s3 keeps entropy ~1.1. The two-head critic + `c_entropy=0.01` + dual-gamma keeps the policy exploratory long enough to hit the sparse reward.

**Value-lag check:** mean(ret_int − v_int) ≈ +0.05 / +0.08 / +0.02 (last-10 ≈ 0 or slightly negative) for B/C re86 — **the critic is NOT lagging; it fits its returns fine.** So this is *not* classic critic value-lag. It is a **degenerate-reward + single-head-policy collapse**: a single intrinsic critic with `c_value=1.0` produces a confident advantage that pushes the policy deterministic toward whatever the (now-meaningless) novelty gradient points at, and a frozen non-controllable φ makes that signal a fixed artifact rather than a real exploration bonus.

**Why icm wins re86 but not because its bonus is better:** icm never freezes φ, keeps separate ext/int value channels, and runs at low entropy-pressure-but-high-actual-entropy. The new methods' single-head + forced-freeze-at-u100 + RND-on-φ pipeline is the regression source, not "RND-on-φ is intrinsically worse." On the **g50t** cell (random-unreachable, so a much harder credit problem) the new methods' richer early signal actually beats icm 2/2 vs 3/8.

---

## Puzzle 3 — C solves g50t L1 2/2 (fast) but a prior probe called it "dead" on ls20. Reconcile.

**Finding: signal informativeness is environment-dependent, AND the solves happen BEFORE the degeneration sets in.** Both C g50t solves finish **fast** (32k @ update 16, 116k @ update 57) — **well under the freeze fallback at update 100** (φ-freeze NEVER fired in either g50t run). During those updates:
- `icm_raw_mean` is large and informative early (98 / 21), `mean_feature_cosine`≈1.0, `holdout_inv_acc` rising (0.22→0.68 s0), entropy healthy (1.6→1.1).
- The env's forward-prediction error provides a usable, well-scaled bonus on g50t that it does **not** on ls20 (the prior "dead" probe was ls20-specific — ls20's color/grid structure makes the ICM forward error uninformative). 

So: **C is alive wherever (a) the ICM/RND-on-φ signal is informative for that env AND (b) the task solves quickly, before update-100 freeze + normalizer blowup.** re86 L1 fails both: it is a long-horizon solve, so the run runs *past* the degeneration. The g50t-vs-ls20 contrast confirms environment-dependence; the g50t-vs-re86 contrast confirms time-to-solve dependence.

---

## Puzzle 4 — Entropy collapse / value-lag / phantom-advantage; did γ=0.95 + c_value=1.0 help? Did φ-freeze fire adaptively?

- **Entropy collapse: YES, on the long runs.** B re86 s0 → 9e-6; B re86 s1 → 0.72; C re86 → 0.66/0.84; C/B g50t solved before collapsing (entropy ≥1.1 at solve). D never collapses (it's stuck near uniform for the opposite reason — τ too high). So collapse appears whenever a single-head intrinsic critic runs long on a degenerating φ.
- **ret ≫ V (phantom advantage)? NO here.** mean(ret_int − V_int) ≈ 0 for the failing B/C runs — the intrinsic critic tracks its returns. The earlier ls20 "phantom-advantage" diagnosis (critic hallucinating advantages from an informative rep) does **not** match these logs; the mechanism here is *degenerate reward on uncontrollable φ + confident single-head advantage*, not a lagging/hallucinating critic.
- **Did γ=0.95 + c_value=1.0 mitigate vs earlier ls20?** Not on re86. Entropy still collapsed to ~1e-3 (B s0). The mitigation did not prevent the new failure mode. (It may help on shorter solves but those didn't need it.)
- **φ-freeze fired ADAPTIVELY only when it didn't matter; otherwise via the bad FALLBACK.** Every long run (B/C/D on re86, D on g50t) froze at exactly **update 100 / step 204800** because `holdout_inv_acc` never reached 0.9 — i.e. the `update≥phi_freeze_max_updates` fallback (`trainer.py:141`). At those freezes holdout ranged **0.19 (chance) → 0.72**. **This is the low-holdout fallback warning case: φ frozen while not controllable.** The fast solves (B/C g50t) never reached update 100, so they never froze — which is *why they worked*.

---

## Puzzle 5 — Outright bugs / instabilities

- **No NaNs / Infs anywhere** (`/tmp/scan.py` over all real runs). No normalizer poisoning to NaN.
- **BUG (design): freeze fallback `update≥100` freezes φ unconditionally** even at chance holdout (`trainer.py:141`). On every long run φ froze at step 204800 with holdout 0.19–0.72. After freeze, RND-on-φ runs on a fixed, possibly-degenerate latent → dead/degenerate novelty (B re86 s0 novelty→0.001; D re86 s1 novelty→5e-4).
- **BUG (D): τ=1.0 over per-state-standardized Q ⇒ near-uniform policy** (entropy ~1.3/1.61). The controller is effectively random (`lookahead.py:58-63`).
- **Entropy collapse to ~0** (B re86 s0: 9e-6) — deterministic policy, exploration dead.
- **ICM return-normalizer blowup** (C re86: `icm_ret_std` 54/98) — fixed `w_icm=0.5` doesn't account for the two streams' wildly different return scales; the bigger-std stream is self-suppressed after normalization.
- **value_loss spikes** to 507/226/120 (D) and 83 (B g50t s0): transient V_int explosions on raw novelty targets; recover, but grad_norm pre-clip spikes to 78–142 (clipped at 0.5).
- **holdout_inv_acc near chance (≈0.19–0.21 = 1/5)**: B re86 s0, D re86 s1 — φ not controllable; the freeze gate's 0.9 threshold is never met on g50t/re86 (best seen ~0.72), so the gate is effectively dead and only the fallback ever fires.

---

## What is BROKEN vs WORKING

**BROKEN**
1. **D's τ=1.0** — policy ≈ uniform random; lookahead Q has no influence. (highest priority)
2. **φ-freeze fallback `update≥100`** freezes φ at chance/mediocre holdout → degenerate RND-on-φ novelty on long runs. The holdout≥0.9 gate is unreachable on g50t/re86, so the freeze is never *adaptive* — only the bad fallback fires.
3. **Single-head intrinsic critic + c_value=1.0** collapses entropy to ~0 on long runs once novelty degenerates (B/C re86).
4. **Additive C: fixed w_icm=0.5 with un-coupled per-stream normalizers** → ICM return-std blows up (50–98) and self-suppresses.
5. **E (disagreement) was never run for real** — only a 256-step smoke exists on disk. No data to evaluate.

**WORKING**
- No numerical instability (no NaN/Inf).
- B and C **beat both baselines on g50t L1 (2/2)** and solve fast — the RND-on-φ / additive bonus is genuinely informative on g50t and works *when the solve precedes the update-100 freeze*.
- A (frozen-φ) is the most stable new variant: entropy stays healthy (~1.5), no collapse, 1/2 on both cells; it avoids the φ-degeneration because φ is fixed from the start.
- Baseline icm/rnd dual-head + high-entropy recipe is robust on re86 L1 (icm 7/8).

---

## CONCRETE proposed improvements

**D (lookahead) — fix the temperature first (cheapest, highest-value):**
- **Lower τ to 0.2–0.3** (or temperature-anneal 0.5→0.15). A `--smoke` at τ∈{0.15,0.25,0.5} on g50t L1 will confirm entropy drops to ~0.4–0.9 and the policy becomes directed. Predicted: g50t goes from 0/2 toward solving if the Q is at all informative.
- **Do NOT standardize Q across only A=5 actions before τ** (it forces unit-scale logits and kills selectivity). Either skip standardization and tune τ on the raw V_int scale, or standardize with a *running* scale across states/time, not per-state.
- Add a `lookahead_Q_gap` / argmax-action-prob metric to logs so future runs show whether Q is informative.

**φ-controllability / freeze (B, C, D):**
- **Raise `phi_freeze_max_updates` far above the run length (or disable the update-cap fallback).** The fallback is the proximate cause of dead φ on long runs. Freeze should be *gated only* by holdout improvement.
- **Lower the freeze holdout gate from 0.9 to a reachable, env-relative target** (e.g. freeze when holdout has plateaued for `patience` updates above ~0.5, not an absolute 0.9 that g50t/re86 never hit).
- **Add a hard guard: never freeze φ while `holdout_inv_acc < 1.5×chance` (chance=1/n_actions).** If φ is uncontrollable, keep training it or fall back to **frozen-φ-from-init (variant A)** rather than freezing a half-trained degenerate φ.
- **Prefer A (frozen-φ) as the default for long-horizon cells** (re86, L2/L3): it sidesteps the entire φ-degeneration failure mode and was the most stable.

**Entropy collapse (B, C):**
- **Adopt the baseline's separate ext/int value heads** (or at least add an extrinsic value channel) instead of the single intrinsic head — the dual-head recipe is empirically what keeps icm at high entropy and 7/8 on re86.
- **Raise/anneal `c_entropy`** for the single-head variants, or add an **entropy floor / KL-target early-stop** in PPO; current `c_entropy=0.05` did not prevent the 9e-6 collapse.
- Consider **lower `c_value` (e.g. 0.5 like baseline)** so the intrinsic critic's confident advantage stops dominating the policy update once the bonus degenerates.

**Additive C:**
- **Replace fixed w_icm=0.5 + independent normalizers** with a coupled scheme: normalize each stream to unit return-std *then* sum (so neither stream self-suppresses), or weight by inverse running-std. The `icm_ret_std`→98 blowup shows the current scheme is unstable on long runs.
- **Keep C** — it is *not* dead; it wins g50t 2/2. Its re86 failure is the shared freeze/entropy bug, not C-specific. Re-run re86 with the freeze fix.

**Keep / drop:**
- **Keep A (frozen-φ)** — most robust new variant; promote to default for long cells.
- **Keep B and C** — they beat baselines on g50t; re-run after the freeze + entropy fixes before judging the re86 regression. Current re86 verdict is confounded by the bug *and* n=2.
- **D**: keep but re-run after τ fix; current results are uninformative (random-policy proxy).
- **E (disagreement)**: actually run it — no real data exists.

**Frontier (L2/L3) guidance:**
- Per baseline grid, **icm/rnd already solve ls20/g50t/re86 L2/L3 at 0/8** (only tu93 L1/L2/L3 and the L1 cells are tractable). These are E=∞ random cells: a single solve is the result. **Do not spend the new-method budget on g50t/re86/ls20 L2/L3 until the L1 freeze+entropy+τ fixes are validated** — they will inherit the same collapse on the (necessarily long) L2/L3 runs. **tu93 L2** is the cheapest tractable cell (baseline median ~2k steps) and a good fast sanity check for any fixed method before committing compute to the harder frontier.

---

### Appendix — key quoted numbers
- D entropy (uniform): g50t s0 1.31–1.34 / re86 s0 1.31 (ln5=1.61); MC τ=1.0 ⇒ E[H]=1.22, E[maxp]=0.51; τ=0.25 ⇒ 0.36 / 0.85.
- D freeze step = 204800 (=100×2048) in all 4 runs; holdout@freeze: g50t s0 0.72, re86 s0 0.50, **re86 s1 0.21**.
- B re86 s0: entropy 1.61→9e-6; holdout flat 0.19 (chance); inverse_acc 0.96 (train); novelty 0.157→0.001.
- C re86: entropy →0.66/0.84; icm_ret_std →54/98; rnd_ret_std ~0.3; holdout ~0.55.
- C g50t solves @ update 16 / 57 (steps 32768 / 116736), φ never frozen, entropy ≥1.1.
- mean(ret_int−V_int) ≈ 0 for failing B/C re86 → no critic value-lag.
- Baseline icm re86 s0 (solved 22k): entropy 1.45–1.61, separate v_ext/v_int.
- No NaN/Inf in any real run.
- Solve rates: B g50t 2/2, re86 0/2; C g50t 2/2, re86 0/2; D g50t 0/2, re86 1/2; A 1/2 both; icm re86 7/8, g50t 3/8 (n as noted).
