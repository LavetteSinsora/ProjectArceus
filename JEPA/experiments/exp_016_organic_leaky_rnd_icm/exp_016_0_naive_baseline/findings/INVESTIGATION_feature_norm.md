# exp_016_0 — The IDM feature-norm inflation: isolated driver, plateau mechanism, and downstream chain

**Run measured:** `checkpoints/exp016_0_naive_tu93_L3_seed0_20260606_212547` (13 checkpoints, 20 480 → 249 856 env steps), all measurements on `torch.device('cpu')` against a fixed 45-state probe set harvested with `diagnostics.harvest_states`. Cross-checked against `runs/.../metrics.jsonl` and the LS20-L1 proxy run.

## TL;DR

The single fundamental driver of the ‖h‖ inflation is the **inverse-dynamics cross-entropy applied to UN-normalized encoder features**. It is the *only* gradient that reaches the encoder (verified in code). The plateau is set **exactly** when inverse accuracy hits 1.0 and the CE gradient to the encoder vanishes (~790× decay measured). This one mechanism — unnormalized inverse-dynamics representation learning — is the **root** upstream of the RND novelty/distill-loss inflation, the controllability/separability dashboard, and the early reward-sign flip / entropy collapse, because the frozen RND maps novelty ∝ ‖h‖².

## 1. The encoder gradient comes ONLY from inverse CE (code-verified)

- `tracker.idm_update` (lines 86–92): the encoder gets grad from `F.cross_entropy(inverse_logits(h,hp), a)` and nothing else.
- `tracker.rnd_update` clips/steps **`rnd.predictor.parameters()` only** (line 117); `RNDPhi.target` is `requires_grad_(False)` and `distill_loss` detaches the target. RND never touches the encoder.
- The policy uses a **separate** `Actor` encoder (`actor_enc`); the IDM encoder is not in the actor optimizer.

→ There is exactly one force shaping ‖h‖: the inverse CE. Everything else is downstream.

## 2. The plateau coincides with CE→0 / acc→1.0 (direct measurement)

Direct ‖h‖ on the fixed probe set, overlaid with metrics.jsonl:

| step | inv_loss | inv_acc | ‖h‖ (direct) | idm_pairwise_l2 | novelty_raw |
|---|---|---|---|---|---|
| 20 480 | 0.416 | 0.83 | **92.6** | 37.6 | 36 |
| 40 960 | 0.0017 | **1.00** | **166.7** | 138.5 | 83 |
| 61 440 | 0.0000 | 1.00 | 153.7 | 145.8 | 140 |
| 102 400 | 0.0000 | 1.00 | 170.3 | 152.9 | 133 |
| 163 840–249 856 | 0.0000 | 1.00 | **~165 (flat)** | ~161 (flat) | 134–172 |

‖h‖ doubles between 20k and 40k — the exact window where CE falls from 0.42 to 0.0017 and accuracy saturates at 1.0 — then is **flat to 3 significant figures** (165.28 → 165.09) for the remaining 200k steps. The plateau is not a property of the data or the architecture; it is the CE gradient switching off.

## 3. The growth force is the CE gradient, and it vanishes at saturation (causal, not just correlational)

Re-running the inverse CE on a fixed held-out 1 024-transition set and measuring the **gradient norm w.r.t. the encoder parameters** at each checkpoint:

| step | CE | acc | ‖∂CE/∂enc‖ | ‖h‖ |
|---|---|---|---|---|
| 20 480 | 0.308 | 0.86 | **1.030** | 97.5 |
| 40 960 | 0.005 | 1.00 | 0.056 (18×↓) | 194.0 |
| 143 360 → 249 856 | 0.00003 | 1.00 | **0.0013 (790×↓)** | ~207 (flat) |

The growth pressure on the encoder collapses by ~1000× exactly as accuracy reaches 1.0. **No gradient → no growth → plateau.** This is the isolated plateau mechanism.

## 4. WHERE the norm grows: conv activations, not weight norms

Decomposing the encoder (`conv → flatten → fc(Linear) → ReLU`):

| step | conv-act ‖·‖ | trunk pre-ReLU ‖·‖ | conv weight ‖·‖ | trunk W ‖·‖ | trunk b ‖·‖ | inv-head W0 / W2 |
|---|---|---|---|---|---|---|
| 20 480 | 77.9 | 170.4 | 21.73 | 23.18 | 0.07 | 22.76 / 0.33 |
| 40 960 | 131.1 | 283.1 | 22.16 | 23.69 | 0.08 | 22.85 / 0.44 |
| 249 856 | 122.9 | 256.8 | 22.25 | 23.92 | 0.08 | 22.91 / 0.51 |

- **Weight norms barely move** (conv +2%, trunk W +3%, inverse head +0.7%/+55% but W2 is tiny in absolute terms). The bias is ~0.
- **Conv ACTIVATION norm grows ~60%** (78 → 123) over the inflation window and then plateaus.

So the inflation is the conv learning a **higher-norm, better-separated activation configuration**, not weight blow-up. The trunk Linear is essentially a fixed projection; the work is done in the conv stack producing larger, more separable feature vectors. This is why naively scaling an early checkpoint's `h` does **not** reduce CE (a control test: scaling early-ckpt features by c∈[0.5,5] gives a CE *minimum* near c=1, rising on both sides) — the inflation is directional (along separating axes), not isotropic gain.

## 5. CE-sharpening confirmed: logits grow via ‖h‖, with the inverse head ~frozen

Inverse-head logit magnitude grows 4.2 → 15.6 across training while the inverse-head weights are essentially frozen (W0: 22.76 → 22.91). The logit growth is therefore carried by ‖h‖ (97 → 207). Larger separable features → larger logit margins → sharper softmax → lower CE. The CE objective on unnormalized features has a built-in incentive to grow margins by growing feature magnitude, and it does so until the data is perfectly classified.

## 6. Feature inflation is UPSTREAM of the other dashboard phenomena

The RND target/predictor are MLPs with **no input normalization**, so for inputs in the locally-linear regime novelty = ½·mean(P(h)−T(h))² scales ≈ ‖h‖². Measured:

- **nov / ‖h‖² ≈ 0.005, constant** across all 13 checkpoints (0.0060 → 0.0053). RND target output norm tracks ‖h‖ (139 → 245).
- **corr(pairwise_l2², novelty_raw) = 0.90**; **corr(novelty_raw, rnd_distill_loss) = 0.9998**. The distill loss "rise & high plateau" is just the RND error operating on inflated inputs — it inherits ‖h‖²'s inflate-then-plateau shape. RND is not independently saturating; it is being fed a growing-then-frozen ruler.
- **Reward-sign flip / entropy collapse:** the reward is a z-score by a **cumulative** Welford running mean (`nov_rms`, never reset; `trainer.py` 170–173). During the inflation transient the raw novelty spikes faster than the cumulative `run_mean` can catch up, so once novelty stops rising the z-scored reward goes **negative** (measured reward_norm_mean at updates 3–5 = −0.48, −0.26, −0.86; raw novelty < run_mean in early updates). Entropy collapses 1.386 → ~0.6 (min 0.108) over the same window. The magnitude that loads and then under-shoots the cumulative mean is ‖h‖²-driven novelty.

**The causal chain (single root):**
```
unnormalized inverse-dynamics CE  →  conv grows separable high-norm h  →  ‖h‖ inflates till acc=1 (CE grad→0) then plateaus
        └─► novelty = ½‖P(h)−T(h)‖² ∝ ‖h‖²  →  novelty + rnd_distill_loss inflate-then-plateau
        └─► cumulative-mean z-score lags the ‖h‖² spike  →  reward sign-flips negative  →  entropy collapse
```

## 7. Contrast control (LS20-L1): no CE saturation → no plateau

On the LS20-L1 proxy run, inverse accuracy plateaus at ~0.99 (never reaches 0.999), so CE never fully floors. There the norm proxy **inflates then RELAXES** (pairwise_l2 peaks 158 at ~39k, decays to 75) rather than plateauing — consistent with the φ-drift finding (the encoder keeps moving because CE keeps providing gradient). This contrast is itself evidence: the plateau exists iff CE saturates. Whether you get a flat plateau (tu93, acc=1.0) or an inflate-then-relax (ls20, acc≈0.99) is governed entirely by whether the inverse CE gradient dies.

## 8. The single recommended fix

**Normalize the encoder output before it is used by both the inverse head and RND** — i.e. append a `LayerNorm(trunk_dim)` (no affine, or affine with weight≈1) after the trunk Linear in `CNNEncoder.forward`, or L2-normalize `h` to a fixed radius. This removes the scale degree of freedom that CE exploits:

- CE can no longer reduce loss by growing ‖h‖; it must learn the *direction* that separates actions. Logit margins are then controlled by the (small) inverse-head weights, not by runaway feature magnitude.
- RND novelty becomes scale-invariant, so it measures genuine **directional** novelty rather than ‖h‖² — eliminating the inflate-then-plateau of novelty_raw and rnd_distill_loss.
- The cumulative-mean z-score no longer sees a one-time magnitude spike, eliminating the early reward-sign flip and the associated entropy collapse.

Lower-priority alternatives that attack the same root: (a) **freeze the IDM encoder after a warm-up** (the exp_013 recipe) so the RND ruler is stationary; (b) re-center/whiten `h` before RND. But LayerNorm is the most direct, single-line cure for the *driver* because it removes the magnitude lever the CE pulls.

**What it prevents downstream:** the ‖h‖² loading of novelty and rnd_distill_loss, the run_mean-lag reward-sign flip, and the entropy collapse — i.e. four dashboard panels traced to one cause.
