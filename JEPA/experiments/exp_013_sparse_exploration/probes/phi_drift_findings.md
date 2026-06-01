# φ-Drift Probe — how fast does the ICM encoder stabilize? (LS20 L1)

**Question (for a future method).** RND / pseudo-counts need a *stationary*
embedding to count against. ICM's encoder φ is trained by inverse dynamics, so
the same state `s` maps to a different `φ(s)` over training — the "ruler" moves,
smearing any count. **When (at what training step) does φ stop moving enough,
relative to how far apart distinct states sit, that we can trust RND-on-φ?**

Status: **READ-ONLY** probe. Loads `exp_011_0` ICM checkpoints, rolls a random
policy to collect probe frames. Modifies nothing under `environment_files/`,
the shared env wrappers, or `exp_011`.

---

## Method

- **Fixed probe set:** 1500 distinct LS20 L1 frames collected by a uniform-random
  policy (seed 12345, 8 envs), deduped by raw frame bytes. Held fixed across all
  checkpoints and all 3 seeds. (1500 distinct states were reached in only 273
  random steps — L1 is a small, near-deterministic maze, consistent with the
  prior random-policy analysis.)
- **Encode:** for each checkpoint (per seed, ordered by step) load `icm.phi`
  (a `CNNEncoder`, trunk_dim=256) and encode the fixed set → `Φ_t (1500, 256)`.
- **Drift on the SAME states**, consecutive checkpoints:
  - cosine: `mean cos(φ_t(s), φ_{t+1}(s))` (raw features)
  - L2: `mean ‖ φ̂_t(s) − φ̂_{t+1}(s) ‖` on **unit-normalized** features.
- **Resolution:** at each `t`, mean pairwise L2 among distinct states (unit-normed,
  subsample 800) — the gap φ must preserve to tell states apart.
- **Key ratio:** `drift_L2(t→t+1) / inter_state_L2(t+1)`. `≪ 1` ⇒ "stable enough
  to count on."
- Contrast: same metrics for the **policy `encoder`** (the separate exp_010 CNN
  encoder stored under the `encoder` key).

Checkpoints: 4 per seed at steps **102400, 204800, 307200, 409600** → 3 drift
windows of ~102.4k steps each. Device: MPS.

> **Caveat (important):** checkpoints start at **102400**. The first ~100k env
> steps of φ drift — almost certainly the *largest* — are **not captured**. All
> conclusions below are about drift **after** ~100k steps.

---

## Results — ICM φ

Per-seed (cos = mean cosine on same states; L2 = unit-norm drift; res =
inter-state L2 at interval end; ratio = L2/res):

| window (step) | seed0 cos / L2 / ratio | seed1 cos / L2 / ratio | seed2 cos / L2 / ratio |
|---|---|---|---|
| 102.4k→204.8k | 0.987 / 0.145 / 0.136 | 0.965 / 0.247 / 0.263 | 0.917 / 0.382 / 0.342 |
| 204.8k→307.2k | 0.966 / 0.231 / 0.216 | 0.952 / 0.291 / 0.299 | 0.971 / 0.228 / 0.202 |
| 307.2k→409.6k | 0.959 / 0.254 / 0.249 | 0.918 / 0.385 / 0.402 | 1.000 / 0.014 / 0.012 |

**Averaged across seeds:**

| window | cos | drift L2 | inter-state L2 | **drift/resolution** |
|---|---|---|---|---|
| 102.4k→204.8k | 0.956 | 0.258 | 1.041 | **0.247** |
| 204.8k→307.2k | 0.963 | 0.250 | 1.056 | **0.239** |
| 307.2k→409.6k | 0.959 | 0.218 | 1.035 | **0.221** |

## Results — policy `encoder` (contrast)

| window | cos | drift L2 | inter-state L2 | **drift/resolution** |
|---|---|---|---|---|
| 102.4k→204.8k | 0.849 | 0.430 | 0.898 | **0.463** |
| 204.8k→307.2k | 0.880 | 0.411 | 0.882 | **0.460** |
| 307.2k→409.6k | 0.955 | 0.201 | 0.847 | **0.251** |

---

## Interpretation

1. **φ is partially but not fully stable after 100k steps, and it does NOT keep
   tightening.** The averaged drift/resolution ratio sits at **~0.22–0.25 for
   every window** out to 409.6k — essentially flat, not decaying toward 0. Mean
   cosine on the same states stays ~0.92–1.00 (raw features barely rotate), but
   the unit-norm L2 drift per ~100k-step window is ~0.22–0.26, i.e. **roughly a
   quarter of the typical inter-state gap**. A count placed in φ-space at step
   `t` is displaced by ~25% of the state-to-state distance one window later.

2. **High per-seed, non-monotonic variance.** Seed2 nearly freezes in the last
   window (ratio 0.012, cos 0.9999) while seed1 *worsens* (ratio 0.40, cos 0.92).
   So there is **no single seed-robust step where φ "locks"** within the captured
   range. The "inverse_acc saturates to ~1.0 fast" note from prior work does
   **not** imply a frozen encoder: inverse-dynamics accuracy can be near-perfect
   while φ continues to drift/re-scale (only the *relative* geometry the inverse
   head needs is pinned, not the absolute embedding).

3. **The ICM φ is ~2× more stable than the policy encoder** in the first two
   windows (ratio ~0.24 vs ~0.46), as expected — PPO churns its encoder under a
   moving value target, whereas φ's inverse-dynamics objective saturates early.
   Both converge to ratio ~0.25 by the last window, but neither reaches `≪ 1`.

---

## Conclusion (the design answer)

- **φ never becomes rigorously "stable enough to count on" within the captured
  range.** The drift-per-100k-window / inter-state-resolution ratio plateaus at
  **~0.22–0.25 (averaged)** from 102.4k through 409.6k and is **not trending to
  0**; one seed even rises to 0.40. There is **no step at which the ratio falls
  to ≪ 1** robustly across seeds.
- **Practical read for RND/pseudo-counts on φ:** treating ICM-φ as a fixed ruler
  on LS20 L1 will smear counts by ~¼ of the inter-state gap per ~100k steps —
  marginal-to-unsafe. If you need RND-on-φ, **don't trust a static φ snapshot**;
  instead use one of:
  - a **slow EMA / periodically-frozen target φ** and recompute counts when φ has
    moved < a small fraction of the inter-state distance (this probe is the exact
    monitor for that gate: trigger a count reset when window drift/res > ~0.1);
  - a **separately-frozen random target encoder** for the counting space (true
    RND) rather than the moving ICM φ;
  - count in the **policy encoder only very late** (its ratio also only reaches
    ~0.25 by 409.6k — worse than φ early on, so φ is the better of the two but
    still not ideal).
- **Caveats:** (i) first ~100k steps uncaptured — early drift is larger, so the
  "stable enough" point is *at best* ≥102.4k and this probe cannot place it
  earlier; (ii) only 4 checkpoints / 3 seeds, ~100k-step granularity — finer
  checkpoints would sharpen the curve; (iii) L1 is tiny and near-deterministic,
  so this is an optimistic case (harder levels likely drift more — see exp_011_2
  L2 checkpoints for a follow-up).

---

## Files

- Script: `JEPA/experiments/exp_013_sparse_exploration/probes/phi_drift_probe.py`
- Raw numbers: `JEPA/experiments/exp_013_sparse_exploration/probes/phi_drift_results.json`
- Figure: `JEPA/experiments/exp_013_sparse_exploration/probes/phi_drift.png`
- Re-run: `uv run python JEPA/experiments/exp_013_sparse_exploration/probes/phi_drift_probe.py`
