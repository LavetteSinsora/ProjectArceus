# Frontier sweep analysis — why no method broke a frontier cell, and what to run next

*Research-scientist analysis of the just-completed FRONTIER sweep (Colab/CUDA, methods A/B/C/D,
games ls20 & g50t, levels L2 & L3, seeds 0/1, cap 400k). Grounded in the 18 per-run logs in
`~/Downloads/*.log` plus the prior probes (`signal_redundancy.md`, `method_improvements.md`,
`transfer_analysis.md`, `EXPERIMENTS_OVERVIEW.md` §4, `exp_014_1` leaky-RND, the random-policy
SUMMARY). No method/env code was modified.*

**Logs parsed (n = 18 runs):**
A: `A_frozen_{ls20,g50t}_L2_s{0,1}`, `A_frozen_ls20_L3_s{0,1}`, `A_frozen_g50t_L3_s0` (crashed, header only).
B: `B_icm_ls20_L2_s{0,1}`, `B_icm_ls20_L3_s{0,1}`, `B_icm_g50t_L2_s1`.
C: `C_additive_g50t_L2_s{0,1}`, `C_additive_ls20_L2_s0`, `C_additive_g50t_L3_s0` (crashed, header only).
D: `D_lookahead_g50t_L2_s0`, `D_lookahead_ls20_L3_s1`.

**Headline confirmed:** every completed run ends `solved: False, censored: True,
env_steps_to_first_reward: None, total_env_steps: 399360`. **0/14 completed runs solved any
frontier cell at 400k. The frontier is unbroken.** This is consistent with the random baseline:
ls20 L2/L3 and all g50t are E=∞ cells (random never solves them in-budget).

---

## 1. Per-(method, cell) final-state dynamics

`decay = nov_raw(last logged u175) / nov_raw(first logged u25)`. Entropy min/max over the 7 logged
updates. `freeze` = φ-freeze step (None = never froze). `holdout` = final held-out inv_acc.

| method | cell | seed | nov u25 | nov u175 | decay | ent min–max | freeze step | holdout final | solved |
|---|---|---|---|---|---|---|---|---|---|
| A frozen-φ | ls20 L2 | 0 | 0.0371 | 0.0033 | **0.09** | 1.12–1.32 | None (n/a) | nan | ✗ |
| A frozen-φ | ls20 L2 | 1 | 0.0103 | 0.0016 | 0.15 | 1.00–1.38 | None | nan | ✗ |
| A frozen-φ | ls20 L3 | 0 | 0.0337 | 0.0054 | 0.16 | 1.22–1.34 | None | nan | ✗ |
| A frozen-φ | ls20 L3 | 1 | 0.0127 | 0.0013 | 0.10 | 1.20–1.32 | None | nan | ✗ |
| A frozen-φ | g50t L2 | 0 | 0.0347 | 0.0042 | 0.12 | 1.44–1.56 | None | nan | ✗ |
| A frozen-φ | g50t L2 | 1 | 0.0116 | 0.0027 | 0.24 | 1.31–1.43 | None | nan | ✗ |
| A frozen-φ | g50t L3 | 0 | — | — | — | — | — | — | **CRASH** (header only) |
| B rnd_icm | ls20 L2 | 0 | 0.0785 | 0.0181 | 0.23 | 1.09–1.26 | **65536** (u32) | 0.731 | ✗ |
| B rnd_icm | ls20 L2 | 1 | 0.0425 | 0.0277 | 0.65 | 0.94–1.35 | 75776 (u37) | 0.734 | ✗ |
| B rnd_icm | ls20 L3 | 0 | 0.0597 | 0.0153 | 0.26 | 0.99–1.23 | 96256 (u47) | 0.718 | ✗ |
| B rnd_icm | ls20 L3 | 1 | 0.0288 | 0.0315 | **1.09** | 0.87–1.24 | 116736 (u57) | 0.718 | ✗ |
| B rnd_icm | g50t L2 | 1 | 0.0234 | 0.1629 | **6.97** | 0.87–1.47 | 159744 (u78) | 0.716 | ✗ |
| C additive | g50t L2 | 0 | icm 0.208 → 0.070 | rnd 0.199→0.532 | icm 0.34 | 1.30–1.50 | 112640 (u55) | 0.701 | ✗ |
| C additive | g50t L2 | 1 | icm 0.988 → 0.060 | rnd 1.15→0.488 | icm 0.06 | 1.13–1.57 | 77824 (u38) | 0.720 | ✗ |
| C additive | ls20 L2 | 0 | icm 0.71 → 0.068 | rnd 1.28→0.725 | icm 0.10 | 0.99–1.35 | 55296 (u27) | 0.725 | ✗ |
| C additive | g50t L3 | 0 | — | — | — | — | — | — | **CRASH** (header only) |
| D lookahead | g50t L2 | 0 | 0.0098 | 0.0117 | 1.19 | π_ent 0.04–0.73 | 204800 (u100) | **0.355** | ✗ |
| D lookahead | ls20 L3 | 1 | 0.0025 | 0.0020 | 0.80 | π_ent 0.02–0.59 | 204800 (u100) | **0.299** | ✗ |

(`D_lookahead_ls20_L3_s1 (1).log` is a byte-identical duplicate of `D_lookahead_ls20_L3_s1.log`;
counted once.)

### Refining the "no collapse, just no coverage" conclusion

**Confirmed for A, B, C — with one nuance.** Entropy stays healthy across A/B/C: every A/B run sits
in **1.0–1.56**; C policies sit **0.99–1.57**. No run shows the runaway-to-zero collapse that
plagued exp_010 / early exp_013 (the c_entropy 0.10 + φ-freeze-gate 0.70 fixes hold). So the
classic entropy-collapse failure mode **is solved on these cells**. The binding failure is
**coverage / chaining a directed sequence**, not collapse.

**Two corrections the logs force:**

1. **Method A is mislabeled / signal-dead, not just "frozen-φ RND."** Every A log shows
   `inv_acc(onpol/holdout)=nan/nan`, `frozen=False` for the entire run, and `holdout=0` in the
   config line. With holdout=0 the freeze gate can never read a held-out inv_acc, so φ never freezes
   and the ICM inverse head is never even evaluated — A is running as a **pure leaky-RND on a
   never-frozen random φ**. Its novelty decays hardest of any method (decay 0.09–0.24; ls20 L2 s0
   = **0.088**, the steepest in the sweep). So for A the story is *stronger* than "healthy
   exploration that misses the reward": its **novelty field is the closest to dead** of all methods,
   which means by late training A's intrinsic pull toward unexplored states is weakest. A's entropy
   staying ~1.3 is the **uniform-prior floor** (c_entropy 0.10 holding the policy near-uniform once
   the reward field flattens), not evidence of active directed exploration.

2. **B is the only method whose field does NOT die, and that is the key positive signal.** B's
   novelty decay is shallow-to-rising: ls20 L3 s1 = **1.09** (novelty *grew*), g50t L2 s1 = **6.97**
   (grew 7×). B's φ is genuinely controllable on these cells — holdout inv_acc **0.716–0.734**, and
   the 0.70 freeze gate fires **adaptively** (u32–u78). So on ls20/g50t the RND-on-φ ruler is being
   computed in a controllable space, the field stays alive, entropy stays healthy — **and it still
   never solves.** That is the cleanest possible demonstration that the bottleneck is **not** field
   death and **not** φ-quality on these cells: it is the inability to *chain* the specific long
   action sequence. (Contrast re86 in `method_improvements.md`, where φ is stuck at chance ~0.19 —
   a *different* failure that this sweep did not test.)

**D is a separate, worse failure (collapse-ish + dead φ).** D's π_ent swings violently
(0.02–0.73) and its v_loss spikes erratically (5e-4 → 0.52 → 5e-4), and its φ holdout is **stuck at
chance** (g50t 0.355, ls20 L3 0.299 — barely above 1/n_actions). This matches
`method_improvements.md`: D's lookahead-Q is anti-informative; τ is a band-aid. D adds nothing on
the frontier and should be excluded from the next sweep (per the task).

---

## 2. WHY no solve on the frontier — diagnosis

### (a) Horizon (400k) is plausibly too short for a *directed* searcher, and definitely for an undirected one

- ls20 **L1** needs a **13-move minimum** solution and random takes **E≈49,843 steps** (p_life
  8.63e-4, ~43 win-chances/life) — `finding_random_policy_ls20_l1`, baseline SUMMARY.
- ls20 **L2/L3** are **E=∞ for random**: the solution is *longer than the per-life energy budget*
  (only ~22 win-opportunities/life vs a solution exceeding the budget — SUMMARY §ls20). So the
  task is not "find a 13-step sequence faster" — it is "execute a sequence that random literally
  cannot complete within one life's energy." That is a **credit-assignment-over-a-long-combo**
  problem, not a horizon-length problem alone.
- Rough directed-searcher estimate: if L2's solution is ~2–3× L1's 13 moves (≈30–40 ordered
  moves) and a *perfectly* directed novelty searcher needed even ~10× the random per-life cost
  amortized over many lives, 400k (≈195 PPO updates × 2048) is **marginal**. A 1M–2M cap is the
  cheapest lever that could flip a marginal cell, but on a pure E=∞ cell more steps of *undirected*
  novelty will not help (see (b)).

### (b) Coverage breadth — the policy covers but cannot CHAIN; novelty pull is weakening

- The decisive prior measurement (`signal_redundancy.md`, n=52,436 pooled transitions, ls20 L2):
  the agent reaches only **43 distinct masked board states in 52k steps**, and **48.8% of
  transitions are wall-bump no-ops**. So even an exploration method is bouncing around a tiny
  reachable set; the goal requires a specific long rotation→goal combo that undirected novelty
  cannot assemble. This is the textbook *coverage ≠ sequence* gap.
- The frontier logs corroborate the "pull is weakening" half: for A the novelty field thins to
  **0.09–0.24× its u25 value** (e.g. A ls20 L2 s0: nov 0.0371→0.0033). Once the field is that flat,
  the intrinsic gradient toward the frontier vanishes and the policy drifts on the entropy prior —
  it is no longer *directed* even though entropy is "healthy."
- B is the counter-example that proves the point: B keeps its field alive (decay 0.23→6.97) **and
  still cannot chain the sequence** — so adding more/longer novelty does not, by itself, produce the
  goal combo. The missing ingredient is **directed return to the frontier** (Go-Explore), not more
  novelty magnitude.

### (c) Signal decay / field death — real for A, NOT for B; and it is on the CONFOUNDED full board

- A's field genuinely thins toward death (steepest decay 0.088). Even *with* the μ=0.01 leak, A's
  novelty floor on these cells is low and falling. `exp_014_1` shows the leak raises a positive
  floor, but A's leak (μ=0.01) is evidently too small to hold the field up over 400k on a 43-state
  support being re-distilled every update.
- Crucially, **all of this novelty is measured on the UNMASKED full board.** The timer confound
  (issue 5 in OVERVIEW; `signal_redundancy.md`): rows 60–63 carry a step-timer that marches every
  step regardless of action, so the model sees **1073 "unique" raw frames vs 43 true masked board
  states**. The nov_raw in these logs is therefore partly *timer-phase novelty* — a signal that is
  ~uniform across all transitions and carries **no information about board-state rarity**. This both
  (i) inflates early novelty (every frame looks new) and (ii) explains the prior finding that the
  ICM/RND signals correlate **negatively** with true masked-state novelty (−0.46, −0.56). The field
  isn't just thinning — a large fraction of what remains is **timer noise, not state-novelty.**

### (d) The timer confound is the single most damning unaddressed defect

`signal_redundancy.md` is unambiguous: on the masked oracle, **both intrinsic signals are
anti-informative** (reward higher on *more*-visited states). The frontier sweep ran on the
**unmasked** board (the env still feeds rows 60–63 to the models — the mask was only ever applied
probe-side). So every A/B/C novelty number in these logs is computed against a confounded input.
Until the timer is masked in the model input, no φ-space or RND-space novelty method is measuring
what we think it measures. This is the highest-leverage bug to fix before re-running anything.

---

## 3. Ranked improvement designs (grounded in the evidence)

### #1 — Masked-frame count-based intrinsic reward (build it; strongest lever)
- **Change:** intrinsic reward = `1/√N(hash(masked_board))`, where `masked_board` zeroes rows 60–63
  before hashing. Replaces (or seeds) the φ/RND novelty path entirely. No φ, no inverse head, no
  freeze gate.
- **Why (evidence):** directly kills both (4) anti-informativeness and (5) the timer confound from
  OVERVIEW. The masked space is exactly **43 states on ls20 L2** (`signal_redundancy.md`), so a
  count gives a *stationary, monotone, provably-correct* −log(count) novelty target — the thing the
  current rulers fail to track (corr −0.46/−0.56). `exp_014_1` already shows the masked hash aliases
  cleanly to the true board.
- **Expected effect:** removes the noise floor that is currently swamping A's signal and inverting
  B/C's. Should at minimum make novelty *informative*; necessary (not sufficient) for a solve.
- **Cost:** low. Hashing is a dict lookup; OVERVIEW §4 lists it "not built" but trivial. ~1 day.

### #2 — Go-Explore frontier-return (the actual cure for coverage ≠ sequence)
- **Change:** maintain an archive of visited masked states with the trajectory that reached each;
  periodically *reset the agent to a promising frontier state* (highest-novelty / least-visited
  cell, or deepest-in-energy state) and explore from there, instead of always exploring from the
  episode start.
- **Why (evidence):** B proves the field can stay alive and φ can be controllable yet the agent
  *still cannot chain the 30–40-move combo* — because every life it re-explores from S and burns
  energy before reaching the frontier. Returning *to* the frontier is the only method here that
  attacks chaining directly. Historical precedent: `claude_automate` (count-based + frontier return)
  **solved ls20 L1 at the 13-step optimum**; the project memory notes it solved L1 at 100%. The
  baseline SUMMARY's "solution longer than one life's energy budget" is exactly what frontier-return
  defeats (you arrive at the frontier with full energy).
- **Expected effect:** the **single most likely lever to break an E=∞ cell.** Converts a
  global-search problem into a sequence of short local searches from a saved frontier.
- **Cost:** medium — needs an archive + a reset-to-state hook. Check whether the env exposes
  `set_state`/save-restore; if not, replay-by-action from the start (cheaper than it sounds given
  deterministic dynamics). ~2–4 days.

### #3 — Longer cap (1M–2M) on the *marginal* cells only, combined with #1/#2
- **Change:** raise cap to 1M on ls20 L2 and tu93 L1 (the cells where a directed searcher is
  plausibly within reach); keep 400k for diagnosis only.
- **Why (evidence):** 400k = ~195 updates; with the field thinning by u175 the effective directed
  budget is much less. ls20 L2's solution exceeds one life's energy → needs many directed lives.
- **Expected effect:** flips *marginal* cells; **wasted on pure E=∞ cells without #1/#2** (more
  undirected novelty ≠ chaining).
- **Cost:** linear compute (~2.5× per cell). Caffeinated machine can absorb it.

### #4 — Raise A's leak μ (cheap field-death fix for the frozen-φ path)
- **Change:** A's μ from 0.01 → sweep {0.02, 0.05}. (`exp_014_1`: floor rises with μ.)
- **Why (evidence):** A's novelty decays hardest (0.088–0.24); its field is closest to dead. A
  higher μ holds a higher positive floor so re-visited states keep producing reward.
- **Expected effect:** keeps A's field alive longer; modest. Pair with #1 so the floor is over a
  *correct* (masked) ruler — raising μ on the confounded full board just raises a timer-noise floor.
- **Cost:** trivial (one hyperparameter sweep).

### #5 — Fix C's normalizer self-suppression (only if C is kept)
- **Change:** cap `_EMAStd` (the discounted-return std) at a max, OR normalize by reward-std not
  return-std (`method_improvements.md`).
- **Why (evidence):** in the frontier logs C's ICM half collapses every run — icm_n
  0.208→0.070 (s0), 0.988→0.060 (s1), 0.71→0.068 (ls20) — while rnd_n stays ~0.5; the return-std
  inflates and divides the ICM signal to ~0 (self-suppression confirmed live).
- **Expected effect:** keeps C's ICM half contributing. **But** `signal_redundancy.md` already
  declared the additive hypothesis DEAD (no convex `w` of two anti-informative signals is
  informative). So this fix is *low priority* — only worth it if C is re-run, and only after #1
  makes the underlying signals informative. **Recommend dropping C, not fixing it.**

### #6 — NGU episodic memory (complementary, second-wave)
- **Change:** add a within-episode kNN-count bonus on masked embeddings on top of lifelong novelty.
- **Why:** gives a per-life "haven't been here *this life*" pull that resists the 48.8% no-op /
  re-visit churn. Complementary timescale to #1.
- **Expected effect:** moderate; best layered on #1+#2. **Cost:** medium. Defer to after the
  #1+#2 sweep result.

*(PopArt / ensemble-disagreement from OVERVIEW §4 are de-prioritized: PopArt attacks the critic /
value-lag, which these logs show is NOT the active failure here — entropy is healthy, no collapse.
Disagreement is still a non-redundant signal but, like B, would face the chaining wall; build it
only on masked frames if #1/#2 stall.)*

---

## 4. Recommended next sweep (most likely to break a frontier cell; excludes D)

**Primary sweep — masked-count + Go-Explore frontier-return, on the reachable-margin and the
hardest E=∞ cells:**

| method | cells | seeds | cap | rationale |
|---|---|---|---|---|
| **M = masked-frame count** (new, #1) | ls20 L1 (sanity, must beat 50k), ls20 L2, ls20 L3, g50t L2 | 0,1,2 | **1M** | establishes an *informative* baseline; ls20 L1 is the go/no-go sanity (if M can't beat random's 50k there, the build is wrong) |
| **GE = Go-Explore frontier-return** (new, #2) + masked count | ls20 L2, ls20 L3, g50t L2 | 0,1,2 | **1M** | the directed-chaining lever; the bet to actually break an E=∞ cell |
| **B (rnd_icm) + masked input** (existing, with timer mask applied to model input) | ls20 L2, ls20 L3 | 0,1 | 1M | tests whether B's healthy-field + controllable-φ converts to a solve once its ruler stops measuring timer noise; cheap to re-run |

**Excluded:** D (anti-informative Q, dead φ — `method_improvements.md`); C additive (dead per
`signal_redundancy.md`); A on its current confounded full-board signal (re-run A only with masked
input + μ∈{0.02,0.05} as a cheap secondary if compute remains).

**Pre-flight (blocking):** apply the **timer mask (rows 60–63) to the model input**, not just
probe-side, for *all* novelty paths. Without this every novelty number stays confounded
(diagnosis §2d). This is the single change that gates whether the next sweep is interpretable.

**Decision gate:** the sanity cell is ls20 L1 under M — it must solve well under 50k (random's E).
If GE+M breaks **any** ls20 L2/L3 or g50t L2 cell, that is the headline result (∞× better than
random). If GE+M still censors at 1M on a true E=∞ cell with a *correct* (masked) novelty signal,
the conclusion flips from "our exploration signal is broken" to "the cell needs planning/curriculum
beyond model-free intrinsic exploration" — itself a publishable negative result, now cleanly
attributable (not confounded by the timer or by field death).
