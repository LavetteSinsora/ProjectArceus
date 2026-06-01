# exp_013 — Collapse Verdict: what actually causes the intrinsic-only entropy collapse?

*Skeptical reconciliation of the two prior diagnoses. Every number below is read directly
from on-disk `metrics.jsonl` (the real training runs) or, where flagged, from the
reconstructed-loop probe `probes/results/h1_novelty_ce01_s0.json`. No method/config code
was modified; no new training runs. Analysis scripts: `/tmp/analyze_collapse.py`,
`/tmp/timeline.py`, `/tmp/make_fig.py`.*

---

## TL;DR verdict

**There is not one collapse mechanism — there are (at least) two, and they live on
different cells. Our "value-lag / phantom-advantage" theory is REAL but cell-specific (it is
the mechanism in the reconstructed *ls20* loop), and it is essentially ABSENT in the real
on-disk re86 runs that motivated the regression. On re86 the collapse is a different
animal: a degenerate, decaying, near-constant novelty field on an uncontrollable φ, driving
a slow PPO entropy bleed that has nothing to do with a lagging critic.** The two writeups
disagree because they were looking at two different cells; both are individually correct on
their own cell, and the `entropy_collapse_diagnosis.md` over-generalized one cell's
mechanism into "the" mechanism.

Crucially, the **novelty-spike-at-φ-drift** theory is **FALSE as the trigger**: on the
cleanest hard-collapse run (B re86 s0) entropy is already at H=0.10 by update 99 and the
φ-freeze fires at update 100 — **the collapse precedes the freeze by ~56 updates** and there
is **no novelty spike at the freeze** (novelty is a flat ~0.001–0.012 throughout).

And the single most uncomfortable fact for *both* prior writeups: the **baseline `rnd`
runs — dual-head ext/int critics, c_entropy=0.01, no φ-freeze, the architecture
`run_diagnosis.md` called "protective" — collapse HARDER than our new methods** (Hmin
0.005–0.13, 6/8 seeds → ~0). So the dual-head architecture is *not* what prevents collapse,
and a single-head critic is *not* a necessary condition for it.

---

## The collapse census (n stated; from `metrics.jsonl`)

`collapse` = policy entropy floor; YES = Hmin<0.3 (near-deterministic), soft = 0.3–0.7.

| method (arch) | cell | n | Hmin range | collapse |
|---|---|---|---|---|
| **B** rnd_icm φ=icm (single int-head) | re86 L1 | 2 | 0.000 (s0), 0.54 (s1) | s0 **YES**, s1 soft |
| **C** additive (single int-head) | re86 L1 | 2 | 0.41, 0.54 | both soft |
| **A** frozen-φ (single int-head) | re86 L1 | 2 | 1.46, 0.47 | s0 no, s1 soft |
| B / C / A | g50t L1 | 2 ea | 0.38–1.06 | solved before any deep collapse |
| **rnd baseline** (dual-head) | re86 L1 | 8 | **0.005, 0.023, 0.044, 0.089, 0.115, 0.134**, 0.69, 1.42 | **6/8 YES** |
| **icm baseline** (dual-head) | re86 L1 | 8 | 0.58–1.47 | 0/8 hard, ~4/8 soft |
| (ls20 L1 real runs B/C/A) | ls20 L1 | several | all ≥1.10 | **none collapsed on disk** — runs are 8–14 updates, far too short |

Two things jump out:
1. **The hard collapses on disk are: B re86 s0 (H→0.0) and the rnd baseline (6/8 → ~0).**
   The new single-head methods are *not* uniquely afflicted; the dual-head baseline is worse.
2. **No ls20 L1 run on disk actually collapses** — they are all tiny (≤14 updates). The
   entire ls20 "phantom-advantage" story rests on the *reconstructed loop probe*
   (`reward_mode_control.py`), not on a shipped run. That's legitimate but must be stated:
   the value-lag evidence is from a probe, the re86 evidence is from real runs.

---

## Theory 1 — Value-lag / phantom advantage (Ret−V persistently +)

**Quantitative test: mean(Ret−V_int) and frac-positive in the updates before/at collapse onset.**

**On ls20 (reconstructed loop, `h1_novelty_ce01_s0.json`, n=1) — CONFIRMED, strong:**
- post-warmup mean(Ret−V) = **+0.78**, positive in **69%** of updates (matches the original
  "+0.76 / 71%" claim — verified).
- In the **6 updates before/at collapse onset (u29): mean(Ret−V) = +2.59, 71% positive**;
  `adv_greedy` rides up to **+2.76 / +3.27 / +2.96** exactly as entropy falls 1.06→0.74→0.68.
- V_int inflates monotonically (0.3→1.6→4.1→8.1) under non-episodic GAE while Ret_int spikes
  ahead of it. This is a genuine value-undershoot → positive greedy advantage → commitment.

**On re86 (real on-disk runs B s0, C s1) — REFUTED:**
- B re86 s0: overall mean(Ret−V) = **+0.05**; in the 8 updates before onset (u44) = **−0.08,
  44% positive**. The critic tracks its returns (e.g. u43: V=2.51, Ret=2.64; u71: V=3.88,
  Ret=3.91; u151: V=1.62, Ret=1.63). No undershoot during the collapse.
- C re86 s1: overall mean(Ret−V) = **+0.01**; 8 updates before onset = **−0.29, 0% positive**
  (the critic *over*-shoots). V_int is large (16→7→3) but it is *matched* by Ret_int.

**Verdict: PARTIAL — TRUE on ls20 (probe), FALSE on re86 (real runs).** Our theory holds on
exactly one cell, the one it was derived from, and breaks on the cell where the regression
actually happened. The `run_diagnosis.md` "Ret−V≈0, NOT classic value-lag" reading is correct
for re86; the `entropy_collapse_diagnosis.md` value-lag reading is correct for ls20. They are
not contradictory — they are different cells. **Do not present value-lag as the universal
cause.**

## Theory 2 — Novelty-spike at φ-drift / φ-freeze resets the RND counter

**Test: does a novelty_raw spike coincide with the freeze step, and does entropy fall after it?**

- B re86 s0: φ-freeze fires at **update 100** (the `update≥100` fallback, since
  holdout_inv_acc never reaches the 0.9 gate). Entropy at u99 is already **0.104**; at u44 it
  was 0.80; the collapse began ~u30. **The collapse precedes the freeze by ~56–70 updates.**
- novelty_raw_mean across the whole run is a flat **0.0067 → peak 0.158(u1) → ~0.001–0.012**
  with **no spike at u100**. There is a tiny bump at the freeze (nov 0.0073→0.0120,
  irn 0.27→0.43 for one update) — but entropy is already 0.08 and keeps falling smoothly to
  ~0 afterward; the bump changes nothing.
- The novelty field is *not* drifting-then-resetting; holdout_inv_acc is pinned at
  **0.19–0.20 (chance, 1/5) for the entire 488 updates** — φ never carries action-relevant
  structure, so there is no "feature drift" to reset.

**Verdict: FALSE (as the trigger), on every cell with data.** Freeze is downstream of, and
later than, the collapse. No spike initiates it. This theory is not supported.

## Theory 3 — Degenerate frozen-φ signal (collapse tracks a near-constant reward field)

**Test: does collapse track φ being a chance-level / near-constant novelty field?**

- B re86 s0: holdout_inv_acc **0.19 (chance) start to finish**, while train inverse_acc =
  0.96 → φ has memorized transitions but generalizes at chance; the RND-on-φ "novelty" it
  produces is a near-constant artifact (novelty_raw 0.001–0.007, std collapses). Entropy
  bleeds to 0 as this field flattens. This matches `run_diagnosis.md`'s "degenerate reward on
  uncontrollable φ."
- **But the same flat-field-driven entropy bleed happens in the rnd baseline, which has no φ
  at all and no freeze.** rnd re86 s3: intrinsic_reward_raw decays **0.117 → 0.001** by u41
  and stays ~0.001; entropy then bleeds **1.61 → 0.85 (u180) → 0.025**. Same outcome, no φ,
  no freeze. So the *frozen-φ* part is not necessary; what is shared is **a near-constant /
  vanishing intrinsic reward field feeding a confident critic + weak/normal entropy bonus**.

**Verdict: PARTIAL → mostly TRUE on re86 as a *mechanism* but mis-named.** The operative
cause is **"vanishing / near-constant intrinsic reward → PPO entropy bleed,"** of which a
degenerate frozen-φ is one instance (B/C) and a simply-saturated RND predictor is another
(baseline rnd). Calling it specifically "frozen-φ" over-narrows it; the freeze is an
aggravator, not the root.

---

## What the data actually supports (the honest synthesis)

There are **two distinct collapse regimes**, distinguished by whether the intrinsic *return*
out-runs the critic:

- **Regime V (value-lag), seen on ls20 (probe, n=1):** a still-live, spiky, *inflating*
  novelty return under non-episodic GAE that the single critic undershoots →
  large positive greedy advantage → fast commitment. Ret−V ≈ +2.6 at onset. This is the
  phantom-advantage mechanism, and it is real *here*.
- **Regime D (dead-field bleed), seen on re86 (real runs, B/C + baseline rnd):** the
  intrinsic reward *decays to a near-constant trickle* (raw 0.001), the critic tracks it fine
  (Ret−V≈0), and entropy bleeds down slowly over 100–600 updates via the well-known
  PPO/advantage-normalization drift acting on an effectively-rewardless objective. With a
  long-horizon cell (re86 needs ~100k+ steps; 488–976 updates) the bleed has time to reach
  ~0. The g50t/B/C runs *solve in <60 updates* and exit before the bleed deepens — which is
  why they look fine. **Time-on-a-dead-field, not architecture, separates collapse from
  non-collapse.**

The common denominator across *both* regimes is **a confident single/seen advantage with no
adequate entropy floor**; what differs is the *source* of that advantage (a lagging critic vs
a flat field). The proposed fixes that target value-lag (normalize/clip the return) only help
Regime V; Regime D needs an entropy floor / KL-to-uniform or a non-vanishing exploration
signal, because there is no lag to fix.

### Per-theory verdict table

| theory | ls20 (probe) | re86 (real runs) | g50t (real runs) | overall |
|---|---|---|---|---|
| 1. Value-lag / phantom advantage | **TRUE** (+0.78/+2.6 at onset) | **FALSE** (Ret−V≈0/−0.08) | n/a (solves first) | **PARTIAL — cell-specific** |
| 2. Novelty-spike at φ-drift/freeze | no data | **FALSE** (collapse precedes freeze by ~56u; no spike) | FALSE | **FALSE** |
| 3. Degenerate / vanishing reward field | (consistent) | **TRUE** but mis-named "frozen-φ" (baseline rnd does it w/o φ) | avoided (solves first) | **PARTIAL/TRUE — rename to "dead-field bleed"** |

### Where our stated theory is wrong, precisely
1. It claims value-lag is **the** mechanism for the exp_013 regression. The regression is on
   **re86**, where the on-disk runs show **Ret−V≈0** — value-lag is absent. We generalized an
   ls20-probe result onto a re86 failure it does not explain.
2. The headline "+0.76 mean(Ret−V), 71% positive" is **true only for the reconstructed ls20
   loop**, not for any collapsing on-disk run. State the source.
3. The framing "the novelty signal is necessary; zero/noise reward don't collapse" is
   contradicted by the **baseline rnd runs collapsing to ~0 once their reward decays to a
   ~constant trickle** — i.e. an *effectively-zero* reward DOES collapse entropy here, given
   enough updates. The H1 control "zero floors at ~1.0" was only run to 50–60 updates; the
   real long runs show the floor keeps falling.

---

## The single cleanest figure (generated: `probes/collapse_verdict_fig.png`)

**Two panels, shared structure, twin y-axes — entropy (blue, left) vs Ret−V_int (red,
right), over PPO updates:**
- **(a) ls20 reconstructed loop:** Ret−V is **persistently positive and growing to +3** while
  entropy falls — value-lag visibly *drives* the drop. Reader sees: red rides up, blue rides
  down, in lock-step.
- **(b) re86 B s0 (real run):** Ret−V **oscillates around 0** the whole time while entropy
  still collapses to 0; the **φ-freeze marker (u100) sits well to the right of the collapse
  onset (u~30–44)**; novelty (green, scaled) is flat-near-zero with no spike. Reader sees:
  red flat at 0, blue collapses anyway, freeze is too late to be causal.

Put side by side, the figure makes the verdict unmissable: **same symptom (entropy→0), two
different mechanisms; value-lag is the left panel only.** This is the one figure to ship.

### One-paragraph summary to present the whole story
> Intrinsic-only PPO collapses to a deterministic policy by two distinct routes. On ls20
> (reconstructed-loop probe, n=1) it is genuine **value-lag**: under non-episodic GAE the
> still-live novelty return inflates and spikes faster than the single critic can track,
> leaving a large positive advantage on the greedy action (mean Ret−V = +0.78, +2.6 in the
> updates just before entropy falls) that PPO commits to. On re86 (real runs: new methods B/C
> and the icm/rnd baselines) the critic instead **tracks its returns** (Ret−V ≈ 0), and
> collapse is a slow **entropy bleed on a vanished, near-constant intrinsic reward field**
> (RND/ICM signal decays to ~0.001) that simply has time to run to zero on a long-horizon
> cell — the dual-head baseline `rnd` does this *worst* (6/8 seeds → H≈0), so architecture is
> not the cause. The φ-freeze and any "novelty spike" are **not** triggers: on the cleanest
> hard collapse, entropy is already ~0.1 fifty-plus updates before φ freezes, with no spike.
> The real cure is cell-dependent: clip/normalize the intrinsic return for the value-lag
> regime, and add an entropy floor / non-vanishing exploration signal for the dead-field
> regime.

---

## Honesty / scope notes
- n: hard collapses analyzed in depth = B re86 s0 (n=1), C re86 s1 (n=1), rnd-baseline re86
  s3 (n=1, representative of 6/8); ls20 value-lag = reconstructed probe (n=1). Census table
  spans all on-disk re86 runs (B/C n=2 each, A n=2, baselines n=8 each).
- The ls20 value-lag evidence is from a **probe loop**, not a shipped run; no on-disk ls20 run
  is long enough to collapse. Treat Regime V as "demonstrated in a faithful reconstruction,"
  not "observed in a release run."
- `intrinsic_episodic`, c_entropy, c_value etc. were read from configs, not varied here.
- Run names quoted: `exp013_1_rndicm_icm_re86_L1_seed0_20260601_045045` (B s0),
  `exp013_2_additive_re86_L1_seed1_20260601_070110` (C s1),
  `exp013_rnd_re86_L1_seed3_20260601_043612` (rnd baseline),
  `exp013_icm_re86_L1_seed1_20260601_040331` (icm baseline),
  probe `probes/results/h1_novelty_ce01_s0.json` (ls20 value-lag).
