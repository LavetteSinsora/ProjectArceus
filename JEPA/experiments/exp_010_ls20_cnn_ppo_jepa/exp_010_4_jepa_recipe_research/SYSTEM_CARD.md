# exp_010_4 — Research map: training a *high-quality* JEPA world model for LS20 L1

**Type:** research proposal / map (initial thoughts, to be executed adaptively).
**Author role:** ML researcher. Later I switch to implementer and run these, branching on results.
**Hard constraint:** the **encoder architecture is frozen** to the exp_007/exp_010 CNN trunk
(4 strided convs → Linear → ReLU, one-hot 16-colour 64×64 input, **trunk_dim 256**). Everything
else — target mechanism, predictor, loss, data, optimiser schedule, early-stop — is in scope.
**Compute:** Apple M3 Pro (MPS), ~10× slower than a 3090. **No brute-force grids.** OFAT along a
dependency DAG, cheap proxy metrics gate expensive runs.

---

## 0. Why this experiment exists (what we already know)

From `RESEARCH_LESSONS.md` and the exp_010_2 post-mortem (`debug/pretrain_diagnosis.html`):

- **JEPA loss lies.** Falling loss is necessary-but-insufficient; collapse / shortcut / predict-the-mean
  all drive it down. Judge representations by *frozen-transfer + probes*, never by the loss. (lesson 1,2)
- **No-EMA shared-encoder JEPA collapses.** exp_003_0→003_1 showed an EMA target + stop-grad restores
  effective rank and probe accuracy; VICReg only *delays* collapse. exp_010_2 had stop-grad but **no EMA**.
  (lesson 2.1, 3)
- **exp_010_2 concretely failed three ways** (verified from `encoder_final.pt` + 5-row metrics):
  1. **Early-stop on raw forward-MSE killed training at epoch 1.** `val_jepa` *rose* (1.35→2.23→2.31)
     — a moving stop-grad target, no EMA — so patience-4 fired and the shipped `encoder_final.pt` is the
     epoch-1 model (`epoch=1`, `best_val_jepa=1.3513`). It sits ~29% (direction-only) from random init;
     `fc.bias`≈0.045. **The "pretrained" encoder was ≈ random.**
  2. **The IDM (action head) saturates instantly** (val acc 0.915 at epoch 1, dead flat) and is ~10× smaller
     than the forward term — it is not a reliable shaping signal here; suspect a shortcut / near no-op.
  3. **Data is goal-starved**: random policy hits the goal <1%, so the world model models wandering and is
     clueless about the one task-relevant event. (lesson 3, 5)

**Thesis.** The biggest wins, in order of leverage, are: **(A) a trustworthy eval harness**, then
**(B) a non-collapsing target mechanism (EMA)**, then **(C) controllable/action-aware latents**, then
**(D) goal-aware data**. Architecture and predictor capacity matter least (lesson 1.2) and are constrained
anyway. We attack in that dependency order so each result is attributable.

---

## 1. Definition of "high-quality" (success criteria, fixed up front)

A good LS20-L1 world-model representation should, on a held-out frame/transition set:

1. **Not be collapsed** — effective rank a healthy fraction of 256; per-dim std bounded away from 0;
   low anisotropy (mean pairwise cosine not →1).
2. **Linearly expose task state** — frozen linear probes recover **agent (x,y)** and **goal (x,y)**
   (and agent→goal relative offset) at high R², approaching the *supervised oracle ceiling* (§3, E0.3).
3. **Be a real forward model** — k-step latent rollout beats both `predict-the-mean` and
   `predict-identity (h_t)` baselines (normalized R²), and is **action-conditioned**: counterfactual
   actions yield distinct predictions in the correct direction.
4. **North-star — transfer.** A **frozen** encoder + PPO matches or beats the from-scratch CNN
   (exp_010_0) in sample-efficiency / success on LS20 L1. This is the metric exp_010_2 failed and the
   only one that ultimately counts. Run sparingly (decision gates only).

Target outcome: a frozen JEPA encoder that gives a *measurable* sample-efficiency win over scratch —
the highest quality LS20 L1 admits (bounded by the oracle in E0.3).

---

## 2. The dependency DAG (attack order)

```
 Layer 0  EVAL HARNESS + PROXY CALIBRATION   ← everything depends on this
     │            (E0.1 metrics, E0.2 proxy↔transfer, E0.3 oracle ceiling, E0.4 infra/early-stop fix)
     ▼
 Layer 1  ANTI-COLLAPSE TARGET  (EMA vs stop-grad vs VICReg)   ← single biggest lever; gates the rest
     │            decision gate G1: freeze the target mechanism
     ▼
 Layer 2  CONTROLLABILITY  (IDM as loss vs probe; action-conditioning strength)
     │            decision gate G2: freeze the loss composition
     ▼
 Layer 3  DATA  (random vs solver/goal-mixed vs exploration; coverage analysis)
     │            decision gate G3: freeze the data recipe → this is where task-relevant gains live
     ▼
 Layer 4/5  REFINEMENTS  (augmentation; latent-norm; predictor capacity)  ← low priority, cheap checks
     ▼
 FINAL     best recipe, 3-seed, full frozen-PPO transfer vs scratch + oracle
```

We resolve the top of the DAG first because its choice changes the meaning of everything below it
(e.g. testing data recipes on a collapsing encoder is uninterpretable).

---

## 3. Layer 0 — Evaluation harness & proxy calibration  *(do first)*

**Rationale.** Lesson 1/2: without a metric that tracks representation *quality*, every later experiment
is noise. This layer is mostly infra + a small calibration study, and it pays for itself immediately.

**E0.1 — Build the frozen eval suite** (cheap, reused everywhere; logged every epoch):
- *Feature health:* mean ‖h‖, per-dim std histogram, **effective rank** (participation ratio of the
  feature covariance), mean pairwise cosine.
- *Linear probes (frozen h):* ridge/logistic decode of **agent (x,y)**, **goal (x,y)**, **agent→goal
  offset**, local wall occupancy. Report val R²/accuracy.
  - *Probe labels come from the palette frame itself* — locate the agent / goal / wall colour-index cells
    in the 64×64 frame (no internal env state needed; robust and env-agnostic). First task of E0.1 is to
    confirm the colour indices for agent/goal/wall on real frames.
- *Forward quality:* **normalized latent R²** (predict deviation-from-mean) and **k-step rollout** R² vs
  `predict-mean` and `predict-identity` baselines.
- *Controllability:* (i) prediction spread across the 4 counterfactual actions for a fixed h_t;
  (ii) a **freshly-trained linear IDM on frozen features** (a *probe*, not a training loss) → "how much
  controllable info is linearly present."
- *Sanity gate:* the suite must rank `random-init < exp_010_2-epoch1 < a-quick-EMA-encoder < oracle`
  correctly, else the metric is wrong. **Falsifiable check on the metric itself.**

**E0.2 — Which cheap metric predicts transfer?** (the key calibration; ~5–6 frozen-PPO runs)
- Take a ladder of encoders of varying quality (random-init, exp_010_2 epoch-1, an EMA quick-train,
  a VICReg quick-train, the oracle from E0.3). Run **frozen-encoder PPO** on each (short budget).
- Correlate each cheap metric (effective rank, agent/goal probe R², controllability) against the
  frozen-PPO transfer score. **Pick the single best proxy** (hypothesis: goal-probe R² or effective rank).
- *Consequence:* from here on, only configs that clear a proxy threshold earn an expensive PPO run.
  This is how we avoid grids.

**E0.3 — Oracle ceiling** (defines "as high quality as LS20 L1 allows"):
- Train the *same fixed encoder* **supervised** to predict ground-truth state (agent/goal pos, derived in
  E0.1) from a single frame. This is the best linearly-probeable representation the architecture admits.
- Its probe scores and its frozen-PPO transfer are the **upper bound** every JEPA recipe is measured against.

**E0.4 — Infra / bugfixes carried as prerequisites** (fold into Layer-1 trainer):
- **Replace raw-MSE early-stop** with a *scale-invariant* criterion (normalized R² or the E0.2 proxy on
  val) — the exp_010_2 bug that shipped an epoch-1 model.
- **Checkpoint every epoch** + keep the best-by-proxy, so we never silently ship an untrained encoder again.
- Fixed held-out eval set + fixed seeds; single shared `train_jepa`/eval entrypoint reused by all layers.

*Deliverable of Layer 0:* a one-command eval that emits the full metric vector for any encoder, a chosen
proxy metric with a known transfer threshold, and the oracle ceiling numbers.

---

## 4. Layer 1 — Anti-collapse target mechanism  *(biggest single lever)*

**H1.** A slow **EMA target encoder + stop-grad** prevents the rank collapse and the rising-loss
instability of exp_010_2, beating shared-encoder stop-grad on effective rank and probe accuracy.

**E1.1 — Target ablation, forward-loss-only** (IDM OFF, to isolate the target effect; 3–4 configs):
| cfg | target | note |
|-----|--------|------|
| A (baseline) | shared encoder, stop-grad | should reproduce the rising-loss / low-rank failure |
| B | **EMA target**, momentum ∈ {0.99, 0.996, 0.999} | coarse 3-point, not a grid |
| C | shared encoder + **VICReg** var+cov | tests "VICReg only delays collapse" |
- Metrics: full E0.1 suite per epoch; compare effective-rank trajectory + agent/goal probe R².
- **Falsifiable:** if EMA does *not* raise effective rank / probe over baseline, collapse is not
  target-driven → re-open the hypothesis (look at LR, ReLU-nonnegativity geometry, init).

**E1.2 — EMA + VICReg vs EMA alone:** does VICReg add value *on top of* EMA (lesson: complement)?
Keep VICReg only if it measurably helps rank/probe at equal compute.

**Decision gate G1:** freeze the target mechanism (expectation: EMA, momentum from E1.1, ± light VICReg).
Run **one** frozen-PPO confirmation on the winner vs baseline to verify the proxy↔transfer link holds.

*Why first:* predictor, loss weighting, and data all behave differently on a collapsing vs healthy target;
fixing this makes Layers 2–3 interpretable.

---

## 5. Layer 2 — Controllability / action-conditioning  *(needs a healthy target)*

**H2.** With a non-collapsing target, the latent can be made *controllable* (encodes action-effects),
but the IDM-as-loss is a shortcut that doesn't help — consistent with exp_010_2's instantly-saturating IDM.

**E2.1 — IDM: loss vs probe** (3 configs on the G1 target):
- forward-only · forward + IDM(grad into encoder) · forward + IDM(**detached** from encoder, probe-only).
- Compare controllability + agent/goal probe + rank. **Falsifiable:** if IDM-as-loss beats IDM-as-probe on
  controllability *without* hurting rank, keep it; else drop it to a probe (my prior: drop it).

**E2.2 — Explicit action-conditioning pressure** (only if E2.1 shows weak controllability):
- Encourage counterfactual-action divergence (e.g. predict distinct next-latents per action; small
  contrastive-over-actions term), measure whether controllability rises without rank loss.
- Watch the "consecutive frames barely differ → action ignorable" trap (lesson 4): report the fraction of
  transitions where the agent actually moved; weight/curate toward moving transitions if needed.

**Decision gate G2:** freeze the loss composition.

---

## 6. Layer 3 — Data generation / sampling  *(where task-relevant gains live)*

**H3.** Goal-aware data measurably improves goal-related probe accuracy and transfer over pure random data
— because random data is <1% goal and a world model can't model an event it never sees.

**E3.1 — Data-mix ablation** (coarse, ~3 sources; **we have a cheap goal-rich source**):
- *random* (baseline) · *random + solver trajectories* (the `claude_automate` policy solved L1 in 13
  steps — a near-free supply of goal-reaching transitions) · *count-based-exploration policy data*.
- Mix ratios at 2–3 coarse points (e.g. 100/0, 80/20, 50/50), not a grid.
- Metrics: **goal-position probe R²** + goal-transition modeling + transfer at G3.
- **Falsifiable:** if goal-mixed data does *not* lift goal-probe R²/transfer over random, goal-starvation
  isn't the bottleneck (then coverage/representation is) — pivot to E3.2 coverage.

**E3.2 — Coverage analysis** (cheap, explanatory): state-visitation entropy / goal-distance histogram per
source; correlate coverage with probe accuracy to explain *why* a data source wins.

**Decision gate G3:** freeze the data recipe. Expect the largest *task-relevant* jump here.

---

## 7. Layers 4–5 — Refinements  *(low priority, cheap checks only)*

- **E4 Augmentation** (open Q4): palette-categorical input limits options; test small spatial
  translations / random crops *iff* the env semantics tolerate them. Likely small on a synthetic env —
  one cheap check, keep only if it clearly helps.
- **E5 Latent normalization / predictor capacity** (lesson 1.2: predictor matters least): quick test of
  unit-sphere latent normalization (à la exp_003_0) as an alt anti-collapse, and confirm a 2-layer MLP
  predictor suffices. Single checks, not a study.

---

## 8. Final consolidation

Best recipe from G1·G2·G3 (+ any kept refinement) → **3-seed** run → full **frozen-encoder PPO transfer**
vs the from-scratch CNN (exp_010_0) and vs the oracle (E0.3). Also report unfrozen warm-start to compare
with exp_010_2's null result. Success = a frozen JEPA encoder with a real sample-efficiency win, probe
scores approaching oracle, and a documented healthy-representation profile.

---

## 9. Compute budget & methodology guardrails

- **OFAT down the DAG**, 3–4 configs per layer, freeze winners. No Cartesian products.
- **Proxy-gated PPO:** expensive frozen-PPO only at G1/G2/G3 and final (~5–8 PPO runs total beyond E0.2).
- **Free curves:** log the full eval vector every epoch on a fixed held-out set → most comparisons need no
  extra runs.
- **Single seed for screening; 3 seeds only at gates/final.** Keep epochs modest (MPS); the encoder is small.
- **Pre-register** the proxy threshold (E0.2) and falsification conditions per experiment *before* running,
  so we don't rationalize collapse as success.

---

## 10. Carried-forward lessons that shaped this plan (from exp_010_2)

- **L-A.** Never early-stop on raw MSE; use a scale-invariant metric + per-epoch checkpoints. (We shipped a
  1-epoch encoder.)  → E0.4.
- **L-B.** The IDM head saturated instantly and may be a shortcut; treat it as a *probe* by default, prove
  it as a loss before keeping it. → Layer 2.
- **L-C.** Rising forward loss with no EMA = moving-target instability → EMA is the first real lever. → Layer 1.
- **L-D.** Representation quality must be measured by frozen-transfer/probes, with an oracle ceiling for
  context — not by the JEPA loss. → Layer 0.

## 11. Open questions this map should close
- Which cheap proxy best predicts frozen-PPO transfer? (E0.2)
- Does EMA alone fix exp_010_2 collapse, or is goal-aware data also required? (G1 vs G3)
- Can we get a controllable latent without an IDM shortcut? (E2.1)
- How much does augmentation help on a synthetic palette env? (E4)

---
*Status:* **proposal.** Next action when executing: implement Layer 0 (E0.1 eval suite + E0.3 oracle), since
all downstream decisions depend on it; then run E1.1 and branch on the effective-rank result.
