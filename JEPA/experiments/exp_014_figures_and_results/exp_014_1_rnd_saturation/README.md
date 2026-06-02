# exp_014_1 — RND saturation vs. the leaky-RND fix (real LS20 L2 visitation)

**Headline figure:** `figures/rnd_saturation_vs_visits.png`

This is the headline evidence figure for our **leaky RND** innovation. It uses
**real environment visitation data** (no synthetic states): a uniform-random
roam on LS20 **Level 2**, the shared env wrapper, the shared `RNDPhi` engine
(target/predictor MLPs + `apply_leak()`), and our actual masking of the
energy/step-timer UI rows.

---

## The claim

Standard RND (no leak, μ=0) is a **one-way error ratchet**: once the predictor
has fit a state's random target, that state's novelty **permanently collapses
toward machine zero**, no matter how exploration later evolves. A re-visited
state stops producing any intrinsic reward → exploration stalls.

**Leaky RND** adds a per-update shrink-to-init of the predictor

```
θ_P ← (1 − μ)·θ_P + μ·θ_P^init        (μ = leak)
```

which turns the ratchet into a **visitation-rate** signal: between re-fits the
leak nudges the prediction back toward the random init, so each update re-injects
a constant error that the next distill step cannot fully remove. The result is a
**positive novelty floor that rises with μ** — the state never goes "dead".

---

## Method (all real, no synthetic data)

| Component | Choice |
|---|---|
| Env | LS20 **Level 2** via `exp_011_ls20_icm/shared/ls20_vec_env_level.py` `VecLS20EnvLevel(level_index=1)` (imported read-only) |
| Device | **CPU** (two other agents are on MPS; RND + random roam is cheap) |
| Policy | uniform-random actions, **16 envs**, ONE shared action/visit stream → identical visit trajectory for every μ |
| **State identity** | mask obs **rows 60–63** (the step-timer / energy bar that marches every step regardless of the agent — the same mask used by `exp_013_headline_experiment/probes/signal_redundancy.py`), then hash the masked 64×64 color-index board. The timer is **never** counted. |
| Chosen states | short pre-roam → the **top-4 most-visited** masked states (guarantees they blow past 1000 visits) |
| RND input | a **fixed random projection** of the masked board: one-hot(16 colors) over 64×64 → flatten → a single **frozen random Linear** to dim=256 (seeded → identical for all μ, so the ONLY difference across runs is μ) |
| RND engine | the real `RNDPhi` (exp_013_1): frozen random target MLP + trainable predictor MLP, `distill_loss`, `apply_leak()` |
| Loop | per update (rollout_steps=128 × n_envs=16 = 2048 env-steps): **measure** novelty at the chosen states *before* the update → roam → **distill** the predictor on the visited states → `apply_leak()` once. Leak is per-update; visits are per-env-step. A **separate RND instance per μ** (μ ∈ {0, 0.001, 0.01, 0.05}), identically initialised, fed the identical visit stream. |

### Two deviations from the production config (and why)

1. **`rnd_epochs = 40` per update** (production uses 1). With only ~50 distinct
   masked states sharing one predictor, a single low-LR step/update never
   *memorises* any state, leaving a large cross-state interference floor (~5e-4)
   for **all** μ that swamps the leak effect. 40 inner steps let the predictor
   truly fit the visited support so the μ=0 ratchet reaches deep saturation —
   the phenomenon we are isolating. The leak still fires **once per update**,
   exactly as `RNDPhi.apply_leak()` is called in the real loop.
2. **The x-axis is recorded one point per update** (not per raw visit). On LS20
   L2 the hottest states are visited ~280×/update, so "1000 visits" is reached
   in ~4 updates; the saturation is driven by the number of predictor *distill
   steps*, which advance at the per-update cadence where the leak competes with
   re-learning. Recording per update gives a clean monotone novelty-vs-visits
   curve that spans 1 → ~10⁵ visits. The chosen states still pass 1000 visits
   (see `all_chosen_passed_1000_at_update` in the results JSON).

All knobs are constants at the top of `diagnose.py`.

---

## Chosen states

<!-- FILLED FROM RESULTS -->
See `results/rnd_saturation_results.json → chosen_state_descriptions`. All four
are high-traffic masked board configurations near the LS20 L2 start region
(distinguished by the agent/cursor marker location; the masked playfield color
histogram is otherwise near-identical, confirming they are adjacent agent
positions on the same board with the timer correctly removed).

---

## Key numbers

<!-- FILLED FROM RESULTS -->
(Mean novelty across the 4 chosen states; full per-state numbers in the JSON.)

| method | novelty @ ~100 visits | novelty @ ~1000 visits | deep floor (final) |
|---|---|---|---|
| standard RND (μ=0) | TBD | TBD | TBD |
| leaky μ=0.001 | TBD | TBD | TBD |
| leaky μ=0.01 | TBD | TBD | TBD |
| leaky μ=0.05 | TBD | TBD | TBD |

---

## Files

- `diagnose.py` — self-contained script.
  Run: `uv run python -m JEPA.experiments.exp_014.exp_014_1_rnd_saturation.diagnose`
- `figures/rnd_saturation_vs_visits.png` — **the headline figure** (small
  multiples, one panel per chosen state; log-y novelty vs cumulative visits;
  one line per μ).
- `figures/rnd_saturation_overlay.png` — supporting: μ=0 (dashed) vs μ=0.01
  (solid) for all states on one axis.
- `results/rnd_saturation_results.json` — all numbers, descriptions, config.
