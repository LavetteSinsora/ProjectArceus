# exp_016_0 — Build spec: naive leaky-RND on IDM features

**Intent.** The simplest version that runs, so failure points are *exposed, not
pre-patched*. Every safeguard from the lineage (value head, φ-freeze, the tuning
ladder) is left out on purpose. We log enough to (a) watch each failure happen and
(b) prove whether each normalization we *did* keep is actually necessary.

This section is the authoritative spec we build against. §6–§7 are reference
(failure points to watch, literature best-practice). Locked decisions are **bold**.

---

## 1. Task & protocol
- Game **LS20**, **Level 1** (`level_index=0`). Terminal-only reward.
- Metric: **`env_steps_to_first_reward`** (summed over actors). We **run the full
  budget** (don't stop at first reward) so we can watch the post-reward dynamics
  (entropy collapse etc.); the first-reward step is recorded when it happens.
- Budget cap **`max_env_steps = 250_000`**; **4 seeds**; **run locally (MPS)**.
- `n_envs=16`, `rollout_steps=128` → **2048 transitions/update**; `max_episode_steps=200`.

## 2. Architecture — two independent modules

**Actor** (acts): `CNNEncoder` (exp_010, `trunk_dim=256`) → **policy head only, 4
actions. NO value head.** Trained by REINFORCE.

**Tracker** (rewards): its **own** `CNNEncoder` + an inverse-dynamics head + a leaky
RND count net.
- **IDM encoder** `h = enc(s)`, shaped *continuously* by an inverse-dynamics loss
  `CE(inverse(h_t,h_{t+1}), a_t)`. **No freeze** — we measure the drift.
- **Count net = leaky RND** over `h` (`RNDPhi`, `dim=hidden=out=256`): frozen random
  target `T`, trainable predictor `P`; `novelty(h)=½·mean‖P(h)−T(h)‖²`; leak
  `θ_P ← (1−μ)θ_P + μ·θ_init` **once per update**.

Encoders are separate (different weights/optimizers); PPO/REINFORCE never touches the
tracker and vice-versa. **Both encoder inputs are timer-masked** (rows 60–63 zeroed).

## 3. The update loop (exact order)

```
1. COLLECT rollout (on-policy): (s_t, a_t, s_{t+1}, done) for T×N    [policy acts on masked s]
2. REWARD  (no grad, tracker encoder at its CURRENT weights):
     h'      = tracker_enc(mask(s_{t+1}))
     raw_t   = ½·mean‖P(h') − T(h')‖²
     raw_t   = raw_t · (1 − done_t)                                 # reset frames carry no novelty
     update running (mean,std) of raw over non-done transitions
     r'_t    = (raw_t − run_mean) / (run_std + ε)                   # Z-SCORE (can be negative)
3. RETURNS (REINFORCE, no baseline, no bootstrap):
     G_t     = Σ_{k≥t, same episode} γ^{k−t} · r'_k     (γ=0.99, reset at done, 0 at rollout end)
     G_t    /= (batch_std(G) + ε)                                   # SCALE-ONLY (no mean subtract)
4. ACTOR update (REINFORCE):
     loss = −mean_t[ logπ(a_t|s_t) · G_t ]  − ent_coef·mean_t[entropy_t]
5. TRACKER-ENCODER update (IDM): sample batch from the cross-update replay buffer
     loss = CE( inverse(enc(s), enc(s')), a )                       # grad into the encoder
6. COUNT-NET update (RND): distill P→T on this rollout's h' (non-done), then apply_leak() ONCE
7. global_step += T·N ;  record first env-step where extrinsic > 0
```

Buffers: IDM uses a **cross-update replay** (off-policy OK — it's representation
learning), **excluding no-op transitions** (masked `s_t == s_{t+1}`); fills for one
update before training starts. REINFORCE + RND use the **current rollout only**.

> Known v0 approximation: fixed-T rollouts truncate the last partial episode (no value
> → bootstrap with 0), under-crediting tail actions. Accepted for the naive baseline.

## 4. Locked hyperparameters
| group | values |
|---|---|
| env | game=ls20, level_index=0, n_envs=16, rollout_steps=128, max_episode_steps=200, max_env_steps=250_000 |
| actor | CNNEncoder trunk_dim=256, policy head 4-way (init gain 0.01), Adam lr 3e-4, grad_clip 0.5, **ent_coef 0.01** |
| reward | reward = z-score(novelty) running stats; **γ=0.99** reward-to-go episodic; returns ÷ batch-std (scale only) |
| IDM | own CNNEncoder + inverse MLP(2·256→256→4); Adam lr 1e-3; replay capacity 50_000, batch 512, **4 grad-steps/update**; drop no-ops |
| RND | RNDPhi(dim=256, hidden=256, out=256); Adam lr 1e-4; **4 grad-steps/update**; **leak μ=0.1** once/update |
| mask | timer_mask_rows=(60,63), applied to BOTH encoder inputs |
| protocol | seeds {0,1,2,3}; run full budget; record first-reward step |

## 5. Logging — and the normalization-ablation stats

We deliberately keep z-score + return-scaling, **but log the raw quantities so we can
later check whether they were needed** (same principle for every other choice).

**Per-update scalars → `metrics.jsonl`:**
- *Normalization ablation:* `novelty_raw_mean/std` (pre-norm), `run_mean`, `run_std`,
  `reward_norm_mean/std` (expect ≈0/≈1), `return_raw_mean/std`, `return_norm_mean/std`,
  `return_std_divisor`. → tells us if z-score / ÷std actually changed anything.
- *Policy / REINFORCE variance (F1, F3):* `entropy`, `per_action_prob` (4-vector),
  `grad_norm`, `return_variance`.
- *RND / count (F4, leak):* `rnd_distill_loss`, `novelty_floor` (min over visited),
  `novelty_raw_mean`.
- *IDM / encoder (F2, F8):* `idm_inverse_loss`, `inverse_acc_onpolicy`,
  `inverse_acc_holdout` (fixed random set), **`noop_fraction`** (wall-bump rate).
- *Coverage (F9):* `unique_states_this_update`, `cumulative_unique_states`,
  `unique_per_episode` (looping), `new_states_this_update`.
- *Drift* (5 probe states, **relative-L2 not cosine** — ReLU makes cosine ≈1):
  per-update `drift_rel_l2` for **both** encoders, `drift_over_pairdist` ratio,
  `mean_pairwise_l2` (collapse detector), `drift_from_init`.
- Headline: `env_steps_to_first_reward`, `train_success_rate`.

**Full-state novelty landscape → `state_novelty.jsonl` (the key diagnostic):**
LS20-L1 has ~110 distinct masked states, so maintain a global registry
`state_key → (exemplar, cumulative_visits)`. **After each update, re-encode every known
state with the current tracker encoder and log `{state_id: novelty, visits, visits_this_update}`.**
This one artifact shows the leak (do unvisited states drop?), saturation vs μ-floor,
recency regeneration, and novelty-vs-true-count — directly, on real training data.

**Probe states:** harvested once at init (random roam), 5 chosen for diversity
(agent position; bottom-left pattern matched/not), held fixed for the drift plots.

## 6. Failure points we expect to see (watch these)
- **F1** REINFORCE variance (no baseline) → noisy/slow; watch `grad_norm`,
  `return_variance`. *Fix later: value baseline V(s).*
- **F2** Encoder drift (IDM continuous) → moving ruler; watch `drift_rel_l2`,
  `drift_over_pairdist`, and whether `state_novelty` of un-revisited states moves.
  *Fix later: warm-up→freeze.*
- **F3** Entropy collapse → deterministic loop; watch `entropy`, `per_action_prob`,
  `unique_per_episode`. *Fix later: entropy floor / KL-to-uniform.*
- **F4** Novelty saturation; watch `novelty_floor` (→ machine-zero = bad, μ-floor = good).
- **F8** Timer confound — mitigated by the mask; sanity: `cumulative_unique_states`
  should track ~110 (masked), not thousands.
- **F9** Coverage wall — the real ceiling; if `v_ext≡0` it's coverage, not the tracker.

## 7. Best-practice reference (condensed)
- **RND (Burda 2018):** normalize intrinsic reward by return-std; score the *next*
  state; (paper uses dual value heads + non-episodic returns — we're intrinsic-only).
- **ICM (Pathak 2017):** inverse dynamics for *controllable* features; watch **held-out**
  inverse accuracy (on-policy inflates as the policy narrows).
- **Large-Scale Curiosity (Burda 2018b):** random features are a strong baseline; our
  probe showed frozen-random φ gives ~99% RND leak on LS20 — hence the IDM.
- **PG hygiene:** value baseline, return/advantage normalization, entropy reg, reward
  normalization are the four cheap moves that make PG trainable; PPO is the standard
  upgrade from REINFORCE.
- **Ensembles ("more random nets"):** RND-target ensemble (lower-variance novelty) or
  predictor disagreement / Plan2Explore (no stationary-ruler dependence) — exp_016_2.

## Appendix — file map
- `config.py` — all knobs (the §4 table).
- `actor.py` — policy-only actor (CNNEncoder + policy head).
- `tracker.py` — IDM encoder + inverse head + replay buffer (RND imported from exp_013_1b).
- `diagnostics.py` — state registry, probe states, drift, full-state novelty.
- `trainer.py` — the §3 loop + §5 logging.
- `run.py` — CLI.
- `probes/frozen_encoder_resolution.py` — the resolution probe (motivates the IDM).
