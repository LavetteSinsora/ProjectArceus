# exp_013 — Per-method IMPROVEMENTS (evidence-backed, prioritized)

Goes beyond `run_diagnosis.md`. Merges the prior method-improvement notes (τ-sweep, C self-
suppression mechanism, B controllability ceiling) with deeper per-method log evidence + new
probes. Every claim cites real per-update logs (run names + numbers). Config changes ALREADY
applied and built upon (not re-proposed): D τ 1.0→0.25; B/C φ-freeze gate 0.90→0.70; c_entropy
0.05→0.10. n is tiny everywhere (n=1–2 per cell) — directional only. New test runs here are
capped ≤30k env steps. **No shipped method/config files were modified — diffs below are proposals.**

Chance inv_acc = 1/n_actions: ls20/tu93 = 0.25, re86/g50t = 0.20.

---

## A — frozen-φ RND+leak (`exp_013_1b_leaky_rnd_on_icm_phi --phi-mode frozen`)

### Evidence: RND novelty on a frozen-random φ is INFORMATIVE early, then collapses to a near-flat floor and stays there for the rest of a long run.
`exp013_1_rndicm_frozen_re86_L1_seed1` (FAILED, 999k steps, 488 rows):

| update | nov_raw_mean | rnd_pred_loss | policy_entropy | intrinsic_return_std |
|---|---|---|---|---|
| 1 | 0.0978 | 0.1883 | 1.609 | 1.000 |
| 41 | 0.0089 | 0.0178 | 1.582 | 0.288 |
| 81 | 0.0054 | 0.0109 | 1.378 | 0.236 |
| 161 | 0.0054 | 0.0107 | 0.966 | 0.158 |
| 281 | 0.0054 | 0.0107 | 0.591 | 0.087 |
| 481 | 0.0048 | 0.0097 | 1.006 | 0.033 |

novelty falls **~20×** (0.098 → ~0.005) by update ~40–80, then is **dead-flat at ~0.005 for the
remaining ~900k steps**. μ=0.01 leak does NOT revive it: the predictor re-fits the fixed-random
target on the small reachable φ-set faster than μ forgets → visitation signal saturates.
`intrinsic_return_std` shrinks 1.0→0.033 → late-run the normalizer divides a flat signal by ~0.03
(noise amplification → slow entropy bleed 1.61→0.59 at u281). The solved A runs worked only because
the solve preceded the floor: `frozen_re86_L1_seed0` solved @ **119,568** (nov 0.009 at u46);
`frozen_g50t_L1_seed1` solved @ **147,104**. So A is "most stable" only because it never melts down,
but its exploration signal is effectively spent by ~80k steps → A is a good frontier bet ONLY on
cells solvable inside that window.

Leak probe (NEW, μ∈{0.01,0.05}, tu93 L2, 30k): both solve @ **2528 steps (update 2)** — tu93 L2
solves before leak matters, so it can't discriminate μ. The 999k re86 floor above is the operative
datum motivating a stronger leak. (See also `exp_014_1_rnd_saturation` for the μ
saturation/floor trade-off.)

### Priority changes for A
1. **[HIGH] Raise leak μ 0.01 → 0.05** (`config.py:94`). The 999k floor shows μ=0.01 saturates;
   5× faster forget keeps `rnd_pred_loss` off the ~0.01 floor and sustains a real visitation signal
   past ~80k steps. Cheapest lever; the leak is A's whole value proposition.
2. **[HIGH] Floor-aware normalization.** When novelty is flat, dividing by `int_return_std`→0.033
   amplifies noise. Floor the denominator: `std = max(ema.std, raw_running_mean)` (or skip the
   divide once raw < ε≈0.01) so a dead signal yields ~0 reward, not amplified noise driving the
   entropy bleed.
3. **[MED] Bump `rnd_lr` 1e-4 → 3e-4** to let the predictor re-fit between the (now larger) leaks.
   Keep `rnd_feature_dim=256` (already gives ~0.01 loss headroom).
4. **[MED] Promote A to default for long/E=∞ cells — but ONLY with (1)+(2)**, else it inherits the
   flat-floor problem on every long L2/L3 run.

---

## B — RND on ICM-φ (`exp_013_1b_leaky_rnd_on_icm_phi`, phi_mode=icm)

### Evidence: holdout_inv_acc per game — controllable on g50t, AT CHANCE on re86.
| run | solved | holdout max | train_inv max | ent min/last |
|---|---|---|---|---|
| `icm_g50t_L1_seed0` | yes (174k) | **0.706** | 0.929 | 0.381 / 0.958 |
| `icm_g50t_L1_seed1` | yes (30k) | 0.402 | 0.617 | 1.038 / 1.038 |
| `icm_re86_L1_seed0` | **no** | **0.222** | 0.959 | **0.000 / 0.0004** |
| `icm_re86_L1_seed1` | **no** | 0.352 | 0.946 | 0.541 / 0.719 |

g50t holdout 0.71/0.40 (>> 0.20 chance → controllable); re86 holdout **0.222/0.352** vs chance 0.20
with train_inv 0.95+ → **re86 φ is essentially NOT inverse-dynamics-controllable** (memorize-train /
chance-holdout). `icm_re86_L1_seed0` is the smoking gun: chance holdout + entropy → **0.000**.

### Does the new 0.70 gate fire adaptively? — Largely NO on these cells.
- g50t s0 reaches holdout 0.706 (at the gate) but **solves @174k before the patience-3 streak
  freezes** → `phi_freeze_step=None`.
- re86 both seeds: holdout ≤0.352, gate can't fire → **freeze falls to the u100 step-fallback at
  step 204800**, freezing a chance-level φ. So the 0.70 gate is reachable only on g50t and only as
  the run ends → it never prevents the bad fallback on the cell that needs it (re86).

### Is re86 fundamentally uncontrollable? — Evidence says YES at this budget.
Two seeds plateau at holdout 0.22/0.35 over 200k+ steps, train 0.95+. A longer warmup / higher
icm_lr will inflate *train* acc (already 0.95) but the holdout gap is structural. Don't chase it.

### Priority changes for B
1. **[HIGH] Hard guard: never freeze φ while `holdout_inv_acc < 1.5×chance`.** re86 froze a 0.22 φ
   at the fallback → dead RND ruler → entropy 0.000. Diff:
   ```python
   # exp_013_1b_leaky_rnd_on_icm_phi/trainer.py, in the freeze block
   chance = 1.0 / cfg.n_actions
   uncontrollable = last_holdout_inv < 1.5 * chance
   if (hit_thresh or hit_fallback) and not uncontrollable:
       ... freeze ...
   else:
       pass  # keep training φ; if fallback fired on an uncontrollable φ, behave as phi_mode='frozen'
             # (use the random-init φ as the RND ruler) rather than freezing a degenerate learned φ.
   ```
2. **[HIGH] Route re86-like cells to A (frozen-φ), not B.** With chance holdout, a learned-φ ruler
   is strictly worse than fixed-random; B's only edge (controllable φ) is absent. Router decision.
3. **[MED] Decouple the freeze fallback from run length** — `phi_freeze_max_updates=100`
   (=204800 steps) is the proximate cause of every degenerate freeze. Raise to ~1e9 / disable;
   rely on holdout-plateau gate + guard (1). `config.py:73`.
4. **[LOW] Do NOT chase re86 controllability with longer warmup / higher icm_lr / different β** —
   train inv_acc already 0.95; the holdout gap won't close.

---

## C — additive (`exp_013_2_additive_rnd_icm`)

### Evidence: the std-blowup is REAL and monotonic; plus the same B-style freeze collapse.
`additive_re86_L1_seed1` (FAILED):

| update | ent | holdout | icm_ret_std | icm_norm | rnd_norm | v_int | phi_frozen |
|---|---|---|---|---|---|---|---|
| 41 | 1.570 | 0.190 | 3.7 | 0.001 | 0.018 | 0.21 | F |
| 81 | 1.276 | 0.429 | 5.6 | 1.289 | 0.731 | 14.40 | F |
| 121 | 0.932 | 0.549 | 20.0 | 0.604 | 0.367 | 11.23 | **T** |
| 201 | 0.818 | 0.549 | 41.9 | 0.265 | 0.278 | 5.86 | T |
| 361 | 0.689 | 0.549 | 85.1 | 0.153 | 0.183 | 3.68 | T |
| 481 | 0.849 | 0.549 | 96.3 | 0.125 | 0.231 | 3.17 | T |

Three real facts (refining the prior note's "divided to ~0" — it's a *drift*, not full zeroing):
- **`icm_ret_std` climbs MONOTONICALLY 1 → 96 and never stabilizes** (54 on s0; 18–24 on ls20). The
  ICM-forward-error *return* inflates because φ is frozen but the forward HEAD keeps training on a
  long run; the EMA-of-return-variance normalizer (`_SignalNorm`) tracks but never bounds it.
- **icm_norm DOES drift down vs rnd** (1.29→0.13 while rnd_norm holds ~0.18–0.37) → ICM is roughly
  half-weighted vs RND late, under nominal w=0.5. Partial self-suppression, not the full zeroing
  the diagnosis implied.
- **Dominant failure = same as B:** φ frozen at holdout 0.549 by the u100 fallback (pinned there
  forever), entropy bleeds 1.61→0.69, single intrinsic head. C's g50t solves (s0 @115k, s1 @
  30,752) all finish BEFORE the u100 freeze (`phi_frozen` never True at solve) with entropy ≥1.1.

(Note: a mean-only replay of `_SignalNorm` from logged per-update means diverges from the logged
`icm_norm_mean` — replay 0.82 vs logged 0.16 — because per-step within-block variance drives the
EMA-std. The logged numbers above are ground truth.)

### Priority changes for C
1. **[HIGH] Stop the `icm_ret_std`→96 ramp.** Smallest/safest = clamp each normalizer std to a
   slow ratchet ceiling; better = normalize by the **raw signal std**, not the discounted-RETURN
   std (the return inflation is the artifact):
   ```python
   # exp_013_2/trainer.py  _SignalNorm: replace RewardForwardFilter→EMAStd(returns) with EMAStd(raw)
   #   per stream, OR after self.ema.update(rems):
   self.ceil = self.ema.std if self.ceil is None else max(self.ceil*0.999, self.ema.std)
   return raw / (min(self.ema.std, self.ceil) + self.eps)
   ```
2. **[HIGH] Make w adaptive — w=0.5 is NOT right.** The two streams' scales diverge ~100× over a
   run, so no fixed convex weight stays balanced. Inverse-running-std weighting:
   ```python
   s_i, s_r = norm_icm.ema.std, norm_rnd.ema.std
   w_eff = (1/(s_i+eps)) / ((1/(s_i+eps)) + (1/(s_r+eps)))   # down-weight the noisier stream
   r = w_eff*n_icm + (1-w_eff)*n_rnd                          # log w_eff
   ```
3. **[HIGH] Same φ-freeze guard as B** (never freeze at holdout<1.5×chance; raise/disable u100
   fallback). C inherits the exact B collapse on re86 — confirmed above.
4. **[MED] Keep C.** It beats both baselines on g50t (2/2; s1 @30,752). Its failures are the shared
   freeze/entropy bug + std-ramp, not a C-specific dead signal. Re-run re86 after (1)+(3).

---

## D — lookahead (`exp_013_3_mcts_lookahead`)

### V_int IS learned and useful — and it dominates Q (so the bonus is NOT "all novelty").
`lookahead_re86_L1_seed0` (SOLVED 626k): `v_int_mean` tracks `ret_int_mean` tightly throughout
(u305 **v_int=4.395 vs ret_int=4.201**). With nov~0.03 and V_int~4, raw Q = nov + 0.95·V_int is
**V_int-dominated ~100:1**; per-state standardization (over the A=5 actions) then strips the
cross-state magnitude, leaving only the within-state ranking and unit-variance logits for τ.

### Mini-probe (this session): τ ∈ {0.15, 0.25, 0.40}, tu93 L2, 30k cap, seed 0.
Runs `exp013_5_lookahead_tu93_L2_seed0_*`. tu93 L2 random E≈2k; A/B/C reference solve ≈784 steps.

| τ | solved | env-steps-to-first-reward | policy_entropy first/min/last |
|---|---|---|---|
| 0.15 | **NO** (cens 28.7k) | — | 0.030 / **0.000** / 0.000 |
| 0.25 | **NO** (cens 28.7k) | — | 0.161 / **0.009** / 0.009 |
| 0.40 | **YES** | **14,640** | 0.411 / 0.132 / 0.474 |

**Two conclusions, both important:**
- **The shipped τ=0.25 is too greedy** — D self-locks to a deterministic loop (entropy 0.009) and
  censors; **τ=0.40** (entropy ~0.45) is the only one that solves. Lower-τ-is-worse: D has NO
  entropy bonus and NO policy gradient, so once V_int prefers a (wrong) action everywhere it locks.
- **But even τ=0.40's 14,640 is ~7× slower than random (~2k) and ~19× slower than A/B/C (~784) on
  the same cell.** So D's lookahead Q is **anti-informative / near-random-at-best** here — τ is a
  band-aid, not a fix. The monotone "trust the Q less → do better" pattern says the Q mis-ranks
  actions.

### Priority changes for D
1. **[HIGH] τ 0.25 → 0.40** (`config.py:37`) — the only viable point in the probe (τ=0.25 censors).
2. **[HIGH] Add an entropy/temperature floor** (no c_entropy exists): anneal τ 0.6→0.4, or
   ε-uniform mixture `π=(1-ε)softmax(Qn/τ)+ε·U`, ε≈0.05. τ=0.15→entropy 0.000 shows the actor-free
   controller will fully self-lock without one.
3. **[HIGH] Replace per-state Q-standardization** — z-scoring over only A=5 actions forces unit
   variance, so a no-real-choice state is amplified as sharply as a decisive one → spurious commit.
   Use per-state mean-subtract + a GLOBAL running std (`lookahead.py:60`), preserving relative
   action magnitude so a flat state stays flat under τ.
4. **[MED] Ablate novelty-only vs novelty+γV̂** to find which term mis-ranks (Q is anti-informative
   on tu93 L2; need to know if it's the novelty floor or V̂). Add `lookahead_Q_gap`
   (within-state max−mean of raw Q) + argmax-action-prob to logs — D currently logs neither.
5. **[HIGH] Same φ-freeze guard** — D froze at holdout 0.21–0.50 via the u100 fallback in all real
   runs; `lookahead_re86_L1_seed1` then had nov→5e-4 (dead forward model → useless Q).
6. **[verdict] Do NOT spend frontier budget on D as-is.** Even best-τ is slower than random on the
   cheapest tractable cell. D needs a structural Q fix (3)+(4), not just a temperature.

---

## RANKED: what to change BEFORE the frontier sweep

1. **φ-freeze hard guard across B/C/D: never freeze while `holdout_inv_acc < 1.5×chance`; disable
   the u100 step-fallback.** One fix addresses the dominant failure in B (re86 ent→0.000), C (re86
   ent→0.69), D (dead Q). Evidence: every long run froze a chance-/mediocre-holdout φ at step
   204800.
2. **D τ=0.25 → 0.40 + entropy floor.** Direct probe: τ=0.25 censors (ent 0.009), τ=0.40 solves
   tu93 L2 @14,640 (ent ~0.45). Reverses a shipped value that is actively breaking D. (But also
   bench D honestly — even τ=0.40 is ~7× slower than random; see D-verdict.)
3. **A leak μ 0.01 → 0.05 + floor-aware normalization.** A is the only collapse-free variant but
   its novelty floors at ~0.005 by ~80k steps (re86 999k); the leak is the one lever targeting A's
   real weakness. Promote A to default for E=∞/long cells only after this.
4. **C: adaptive inverse-std w + normalize raw-std not return-std (or ratchet-clamp std).** Kills
   the `icm_ret_std`→96 ramp; w=0.5 is wrong because the streams diverge ~100×.
5. **Router: send re86-like cells to A, not B/C.** re86 φ is at chance holdout across 2 seeds
   (0.22/0.35, train 0.95+) → structurally not inverse-controllable; learned-φ ruler is strictly
   worse than frozen-random there. Don't burn budget making re86 φ controllable.
6. **Add diagnostics before the sweep:** D `lookahead_Q_gap` + argmax-prob; C `w_eff`. Without them
   the frontier runs aren't diagnosable.

**Cheap post-fix sanity cells (do these first):** tu93 L2 (~2k random; A solved @2528, D-τ0.4
@14,640) is the fastest gate; then g50t L1 (new methods already 2/2) and one re86 L1 seed under A.
Defer all g50t/re86/ls20 L2/L3 (E=∞, baseline 0/8) until the freeze-guard + τ + leak fixes pass on
L1/tu93 — they will otherwise inherit the same collapse on the necessarily-long runs.
