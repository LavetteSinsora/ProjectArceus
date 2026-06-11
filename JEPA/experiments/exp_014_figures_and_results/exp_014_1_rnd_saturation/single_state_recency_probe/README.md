# single-state recency probe — an abandoned state's RND error climbs back up

**Figures:**
- `figures/single_state_recency_ls20_L2.png` — annotated two-phase view.
- `figures/single_state_saturate_then_revive_ls20_L2.png` — **overlay-style**: same
  colors / log-y novelty / single state as exp_014_1's `rnd_saturation_vs_visits.png`,
  so it sits beside (or overlays) a saturation panel. Shows the novelty **saturate
  down** while the state is visited, then — where standard RND just stays flat and
  dead — the **leaky curve turning back up** toward the state's original novelty.

A minimal, fully-controlled companion to the exp_014_1 saturation figure. exp_014_1
shows that a **re-visited** state's standard-RND novelty ratchets to machine-zero
and the leak holds a floor. This probe shows the complementary half: what happens
to a state that is **memorized and then abandoned** (not visited recently).

## The claim

Take one real masked board state **A** and:

1. **MEMORIZE** — distill the predictor on A until its RND error is driven to
   ~machine-zero (A is "known").
2. **ABANDON** — never touch A again; each update distill one *other* random state
   (the real loop's `rnd_epochs=1`) and apply one leak, while re-measuring A.

- **Standard RND (μ=0):** A stays "known" forever — flat at the bottom. A revisit
  would yield no intrinsic reward.
- **Leaky RND (μ>0):** A's error **bounces back up** as the predictor leaks toward
  its random init, climbing toward A's original (un-trained) novelty. The rebound
  rate grows with μ. *That* recovered error is the renewable, recency-based
  exploration signal standard RND lacks.

## Key numbers (LS20 L2, seed 0)

Anchor error: deepest value during MEMORIZE → value at end of ABANDON (200 updates).
A's original un-trained novelty ≈ **0.15** (the ceiling the leak climbs back toward).

| method | memorize floor | end of abandon | rebound |
|---|---|---|---|
| standard RND (μ=0) | 4.5e-16 | **6.3e-6** | stays dead (interference-limited) |
| leaky μ=0.001 | 2.1e-7 | 6.4e-3 | climbs |
| leaky μ=0.01 | 2.2e-5 | 1.2e-1 | climbs to ~original |
| leaky μ=0.05 | 6.1e-4 | 1.5e-1 | climbs to ~original, fastest |

The leaky lines are cleanly **μ-ordered** at every update and never cross; μ=0 is
the lowest line by ~3+ orders of magnitude throughout.

## Method (all real, no synthetic states)

- Env / state identity / RND engine / frozen projection: **identical** to
  `exp_014_1/diagnose.py` (imported read-only) — `VecLS20EnvLevel(level_index=1)`,
  timer rows 60–63 masked, one-hot board → frozen random Linear → `RNDPhi`.
- Anchor A = the most-visited distinct masked state from a 600-step pre-roam.
- One `RNDPhi` per μ ∈ {0, 0.001, 0.01, 0.05}, **identical init**, fed the
  **identical** anchor and the **identical** pre-drawn sequence of "other" states —
  the only difference across lines is the leak.
- Cadence = the real loop's: a block of distill steps, then **one** `apply_leak()`
  per update. MEMORIZE = `MEMORIZE_INNER=50` steps/update × `MEMORIZE_UPDATES=40`;
  ABANDON = `ABANDON_INNER=1` step/update × `ABANDON_UPDATES=200`.

### Isolating the leak from interference (two deliberate choices)

A single shared predictor over a tiny (~40-state) support has a cross-state
**interference** floor: distilling other states perturbs A's prediction even with
**no** leak. We keep interference below the leak signal so the figure attributes
the regeneration to the leak:

1. **Well-separated "other" pool** — greedy farthest-point selection in feature
   space (seeded with A), so fitting them barely touches A directly. Still real,
   frequently-visited states; "random" = a random draw from that pool each update.
2. **`ABANDON_LR=1e-7` ≪ `RND_LR=1e-4`** — the leak is learning-rate-**independent**
   (a multiplicative shrink toward init), while interference scales with the distill
   lr. A small abandon lr drives residual interference toward zero (μ=0 stays flat)
   **without changing the leaky lines**.

> **Honest caveat / robustness:** at full abandon lr (`ABANDON_LR=RND_LR`) the
> leaky lines are essentially unchanged, but μ=0 also creeps up to ~1e-3 purely via
> interference — the same regime as `exp_014_2`. The low abandon lr is a
> *mechanism-isolation* knob, not a result-changing one for the leak: it cleans up
> the μ=0 control, it does not manufacture the leaky rebound.

## Files

- `probe.py` — self-contained script. All knobs are constants at the top.
  Run: `uv run python -m JEPA.experiments.exp_014_figures_and_results.exp_014_1_rnd_saturation.single_state_recency_probe.probe`
  (optional `--level 2` for L3, `--seed N`).
- `figures/single_state_recency_<game>_L<n>.png` — the figure.
- `results/single_state_recency_<game>_L<n>.json` — config, per-μ key numbers, and
  the full `err_hist` series.
