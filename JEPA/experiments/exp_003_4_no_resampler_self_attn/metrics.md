# Training-Run Monitoring Metrics Glossary

A reference index of metrics used to inspect JEPA-style training runs (exp_003 and the upcoming exp_003_2 with a paired action predictor). Each entry: **what it is**, **how it is computed**, **why we watch it**, and **soundness notes** (assumptions, hook points, gotchas observed in the current code).

Conventions used below:
- `H_t ∈ R^{L × D}` — latent tensor at env step `t`, where `L = n_latents = 4`, `D = d_model = 128`.
- `H_t[i] ∈ R^D` — the i-th latent token at step t.
- An **eval pass** = freeze weights, `model.eval()`, run **N** independent eval episodes (default N = 5) **until end-of-life** (detected via `is_end_of_life` in [reward_shaping.py](JEPA/experiments/exp_003_1_ema_target/reward_shaping.py)). Because LS20 is deterministic and has no environment-side randomness, **`T_episode` is the same across all N rollouts at a given checkpoint** — so to get N different trajectories we **sample from the policy's output distribution exactly as during training** (`Categorical.sample()` in `policy.act`), rather than acting greedily. Greedy action selection would give five identical episodes and make the N-fold averaging meaningless. Use a different sampling seed per episode in the eval pass so the N trajectories are decorrelated.
- Online encoder is used everywhere unless the metric is explicitly about the EMA target.
- "Pairwise cosine sim" over a set `{v_1, …, v_K}` means `mean_{i<j} cos(v_i, v_j)`.

Where the current code already computes the same thing (or something near it), the existing identifier is cited.

---

## Section 1 — Representation Health (no-collapse monitors)

### 1.1 `placeholder_pairwise_cossim`
- **What:** Mean pairwise cosine similarity between the `n_placeholders = 4` learned placeholder vectors in `PerceiverResampler.placeholders`.
- **How:** `P = encoder.perceiver.placeholders` (shape `(4, D)`). Compute `mean_{i<j} cos(P[i], P[j])`. No env episode needed — pure parameter probe; can be logged every training log step.
- **Why:** If the placeholders converge to a single direction, every cross-attn query becomes the same query, and the resampler can only emit one direction. Healthy target ≈ ~0 (orthogonal); a steady upward drift toward 1 is a leading indicator of imminent collapse.
- **Soundness:** Cheap, deterministic, no assumptions. The plot already produced by [plot_placeholder_cossim.py](JEPA/experiments/exp_003_0_normalized_latent_jepa/plot_placeholder_cossim.py) is the right reference.

### 1.1b `placeholder_drift_from_init_cossim`
- **What:** Per-token cosine similarity between each placeholder's **current** value and its value at **step 0 of training** (initialization). Four scalars (one per placeholder); typically also report their mean.
- **How:** At the start of training (or on first load), cache `P_init = encoder.perceiver.placeholders.detach().clone()` and persist it to the run dir (e.g., `runs/.../placeholder_init.pt`). On resume, load it from disk rather than re-snapshotting. At each log step compute `cos(P_init[i], P_current[i])` for `i = 0..3`. Display all four lines plus their mean.
- **Why:** Complements 1.1. 1.1 says how distinct the placeholders are *from each other*; 1.1b says how much they have **moved from initialization at all**. If 1.1b drops fast while 1.1 stays high, the placeholders are learning meaningful but still distinct directions (good). If 1.1b stays ≈ 1 (no movement) the placeholder gradient path is broken (e.g., the exp_003_1 `is_initial` flag fix is regressing). If 1.1b drops while 1.1 also drops to 1 (all collapsing toward each other) the placeholders are merging into a single direction.
- **Soundness:** Must persist `P_init` so resumes still reference the original `step=0` init, not the post-resume value. For checkpoints with no cached `P_init` (i.e., resuming an old run), either re-cache and accept the resulting drift baseline shift (annotate the dashboard), or attempt to reconstruct from the very first checkpoint in `checkpoints/`.

### 1.2 `latent_pairwise_cossim @ t∈{1, 10, 20}`
- **What:** Mean pairwise cosine similarity between the 4 latent tokens of `H_t` at fixed env step indices `t = 1, 10, 20` within an eval episode. Three separate scalars per episode, averaged across N eval episodes. (If at a given checkpoint `T_episode < 20`, the `t = 20` slot is simply unavailable for that checkpoint and reported as `n/a`.)
- **How:** During an eval rollout, after the perceiver returns `H_t = encoder(x_t, queries_t)`, capture `H_t` at the requested t values and compute `mean_{i<j} cos(H_t[i], H_t[j])`. Use a fresh placeholder query at `t=1`; at `t≥2`, feed back `H_{t−1}` as queries (the standard recurrent rollout).
- **Why:** This is the user-facing collapse signal. `t=1` measures the freshness of the first cross-attn read; later t's tell us whether the recurrent boundary degrades over time. Watch all three together to distinguish "born-narrow" from "narrows-with-rollout".
- **Soundness:** Existing [`health.latent_pairwise_l2`](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py) measures L2 between latents, not cosine — these are NOT equivalent once `output_norm` clamps the latents to unit-ish scale. Add cosine alongside L2 rather than replacing it.

### 1.3 `round0_postCA_pairwise_cossim @ t=1`  *(primary collapse probe in exp_003_4)*
- **What:** Mean pairwise cosine similarity between the 4 latent tokens **immediately after the cross-attention block of round 0** in the Perceiver Resampler, at `t=1` (placeholder queries).
- **How:** Hook the output of `encoder.perceiver.rounds[0].cross_attn` (a `_CrossAttentionBlock`) during an eval forward pass. The hook target returns a `(queries, attn_w)` tuple — take `queries`, shape `(B, 4, D)`. Compute pairwise cosine on each batch element, average.
- **Why:** Previously this probe targeted the post-SA-among-latents output. exp_003_4 removed `_SelfAttentionAmongLatents`, so round 0 is now `ffn(cross_attn(...))` and its cross-attn output is the closest analogue. Watching this point captures whether the 4 latents collapse despite no SA mixing them — the central question of this experiment.

### 1.4 `Ht_Htp1_cossim` (single-episode average)
- **What:** Average cosine similarity between successive latents within an episode: `mean_t cos_avg(H_t, H_{t+1})`, where `cos_avg` averages the cosine across the 4 latent tokens (token-i to token-i).
- **How:** During an eval episode, for each step t with both `H_t` and `H_{t+1}` available, compute `mean_i cos(H_t[i], H_{t+1}[i])`. Average across t, then across N episodes.
- **Why:** Measures whether the representation changes meaningfully step-to-step. Near-1 = sluggish (state-blind); near-0 = healthy variability.
- **Soundness:** Already collected during training in [train.py:340-343](JEPA/experiments/exp_003_1_ema_target/train.py#L340-L343) as `health.ht_ht1_cossim`, but that is computed at training-time on a single recurrent rollout, not a dedicated eval. Keep that as a streaming proxy; add an eval-time companion with the same name suffixed `@eval` for cleaner trend lines.

### 1.5 `H1_HT_cossim`
- **What:** Cosine similarity between `H_1` (first latent of the episode) and `H_T` (latent at the **final step of the life**, i.e. the last step before `is_end_of_life` fires).
- **How:** Per eval episode, capture `H_1` and `H_T` where `T = T_episode`. Compute `mean_i cos(H_1[i], H_T[i])`. Average across the N episodes.
- **Why:** Together with 1.4 this tells us whether the trajectory in latent space is exploring or pacing in place. 1.4 measures local change; 1.5 measures global displacement.
- **Soundness:** `T_episode` is identical across the N rollouts at a given checkpoint (deterministic env) but can change between checkpoints as the policy improves. Log `T_episode` next to the metric so length-driven shifts are not misread as representation shifts.

---

## Section 2 — Image-Patch Self-Attention

The SA blocks in `encoder.sa_blocks` produce attention matrices `A ∈ R^{H × 16 × 16}` (H = `n_sa_heads = 4`, 16 patch tokens). Each row `A[h, i, :]` is a probability distribution after softmax — the weights with which patch `i` reads from all 16 patches. The two metrics below need an **interpretable, distribution-aware similarity**. We use the **mean pairwise Jensen–Shannon divergence (JSD)**, bounded in `[0, ln 2]`. Implementation:

```python
def pairwise_jsd(P):  # P: (K, V) — K distributions over V
    M = 0.5 * (P[:, None, :] + P[None, :, :])
    kl = (P[:, None, :] * (P[:, None, :].log() - M.log())).sum(-1)
    jsd = 0.5 * (kl + kl.transpose(0, 1))
    K = P.shape[0]
    iu = torch.triu_indices(K, K, offset=1)
    return jsd[iu[0], iu[1]].mean()
```

### 2.1 `patch_sa_row_jsd` *(distinctness of what each patch attends to, within a step)*
- **What:** Within a single forward pass, how different are the 16 attention distributions (one per row) from each other.
- **How:** For each block `b` and head `h`, `pairwise_jsd(A[h, :, :])` — pairwise JSD across the 16 rows. Average across heads, blocks, and (during eval) across time steps within the episode, then across N episodes.
- **Why:** If all 16 patches end up attending the same way, the SA layer is performing a constant weighted average — degenerate. A healthy block has distinct, position-aware row patterns; JSD should be well above 0.
- **Soundness:** Need access to the post-softmax matrix. Currently stored only in eval mode at [encoder.py:101](JEPA/experiments/exp_003_0_normalized_latent_jepa/models/encoder.py#L101). For lightweight training-time logging, store it under a flag (e.g., `block._capture_attn = True`) so we are not always paying the host-copy cost.

### 2.2 `patch_sa_temporal_jsd` *(per-patch attention drift across time)*
- **What:** For a fixed patch index `i` and head `h`, how different are the rows `A_t[h, i, :]` across time steps within an episode.
- **How:** Collect `A_t[h, i, :]` for `t = 1..T_eval`, compute `pairwise_jsd` of that stack (K = T_eval distributions of length 16). Average across `i`, then heads, then blocks, then episodes.
- **Why:** Complements 2.1. Even if rows are distinct within a step, the attention might be identical across all states — meaning the SA pattern is state-independent, a softer collapse mode. We want this strictly > 0.
- **Soundness:** Compute on the same eval pass as 2.1 to amortize cost. The 16-dim distributions are small, so the K×K JSD is cheap even with K = 20.

### 2.3 `grad_norm_patch_sa`
- **What:** L2 norm of the gradient flowing into the SA-block + patch-embedding parameter group.
- **How:** Already computed in the loop as `gn_s1 = grad_norm(enc_s1_params)` ([train.py:435](JEPA/experiments/exp_003_1_ema_target/train.py#L435)), logged to `health.grad_enc_s1`.
- **Why:** Watching for exploding grads (the system_card warns about norm growth) and for grads collapsing to near-zero (vanishing signal). Pair with weight-update ratio if available.
- **Soundness:** Already exists. Keep.

---

## Section 3 — Perceiver Resampler (latent-side)

*Removed in exp_003_4.* This section previously covered `latent_self_attn_row_jsd` and the cross-vs-self gradient split for the latent self-attention block. Because exp_003_4 removed `_SelfAttentionAmongLatents` from each Perceiver round, there is no self-attn matrix to probe and no `perc_self_r{i}` parameter bucket to track. See §7.2 for the simplified per-sub-block gradient table.

---

## Section 4 — Predictors

> **Status note:** exp_003_2 will introduce a second predictor (the **action predictor**). The current codebase only contains the next-state flow-matching predictor (`FlowMatchingPredictor` in [predictor.py](JEPA/experiments/exp_003_0_normalized_latent_jepa/models/predictor.py)). 4.1 is implementable today; 4.2–4.3 assume the exp_003_2 module exists.

### 4.1 `ode_step_cossim` (next-state flow-matching predictor)
- **What:** Cosine similarity between successive ODE denoising step outputs `x_k` and `x_{k+1}` in the rollout `predict_with_trajectory`.
- **How:** Use [`compute_ode_step_cossim`](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py) (already imported into exp_003_1). For a batch of latents, call `predict_with_trajectory(h_t, a_emb)`, get `traj = [x_0, x_{1/N}, …, x_1]` (N + 1 tensors), compute `mean_k cos(traj[k], traj[k+1])`, average across batch.
- **Why:** Near-1 ⇒ the ODE steps are not transforming their input (saturated / under-trained predictor). Healthy values are below 1; consistent monotonic decrease toward 1 is the saturation signal.
- **Soundness:** Existing logger (`health.ode_step_cossim`, [train.py:482-484](JEPA/experiments/exp_003_1_ema_target/train.py#L482-L484)) is good. Note that with `n_ode_steps = 3` we get only 3 consecutive-pair samples — average over a larger batch to denoise.

### 4.1b `ode_first_vs_final_cossim`
- **What:** Cosine similarity between the **first ODE step's output** `x_{1/N}` (after one Euler step from `x_0`) and the **final predicted output** `x_1`.
- **How:** From `predict_with_trajectory(h_t, a_emb)` get `traj = [x_0, x_{1/N}, x_{2/N}, …, x_1]`. Compute `mean_i cos(traj[1][:, i, :], traj[-1][:, i, :])` (token-wise cosine, then mean over the 4 latents), averaged over batch.
- **Why:** 4.1 measures step-to-step velocity. 4.1b measures the **total work done by the remaining steps after the first**: if 4.1b ≈ 1, the predictor essentially finishes its job in one step and the remaining `N−1` steps are no-ops (the ODE is overkill / under-utilized); if 4.1b is well below 1, later steps materially refine the prediction (multi-step ODE is paying for itself).
- **Soundness:** Cheap (the trajectory is already returned by `predict_with_trajectory`). With `N = 3`, this compares `x_{1/3}` against `x_1` — only 2 ODE updates apart, so the diagnostic resolution is modest; the metric becomes more informative if `N` is raised to e.g. 5 or 8 during eval.

### 4.2 `action_pred_input_cossim`  *(exp_003_2)*
- **What:** Mean cosine similarity between the two latents fed into the action predictor: `H_t` and `H_{t+1}`, token-wise, averaged across episode time steps.
- **How:** Identical formula to 1.4, but logged separately under the predictor section because **this is the action predictor's input signal**, and the diagnostic question differs: not "is the world model collapsing" but "is the input to the action predictor informative".
- **Why:** If `cos(H_t, H_{t+1}) ≈ 1` consistently, the action predictor literally has no signal to distinguish actions from. A useful sanity check before interpreting any action-predictor accuracy/entropy result. The user already flagged this is redundant with 1.4 — keeping it duplicated for ergonomic reasons (filter by subsystem in the dashboard).
- **Soundness:** Same caveats as 1.4. Document in the dashboard that these two series should track identically.

### 4.3 `action_pred_entropy`  *(exp_003_2)*
- **What:** Entropy of the action predictor's softmax output, in nats, averaged over the buffer batch (or eval episode).
- **How:** `entropy = -(p * p.log()).sum(-1).mean()` where `p = softmax(logits)`.
- **Display:** Always show alongside the **maximum entropy reference** `ln(n_actions)`. With `n_actions = 4`, `H_max = ln 4 ≈ 1.3863 nats`. Suggested label: `action_pred_entropy [Hmax=1.386]`.
- **Why:** If entropy ≈ `H_max` the predictor is outputting near-uniform over actions — i.e., it has not learned to invert dynamics. A drop toward 0 means it is confident (good if accuracy is also rising; bad if confidently wrong).
- **Soundness:** Requires the action predictor to exist. When implementing, ensure logits are masked by `env.available_actions` consistently with how the policy is masked (see [policy.py:25-31](JEPA/experiments/exp_003_0_normalized_latent_jepa/models/policy.py#L25-L31)), otherwise the entropy is computed over actions that are never selectable.

---

## Section 5 — Policy

### 5.1 `policy_entropy`
- **What:** Entropy of the policy's softmax distribution (after masking unavailable actions).
- **How:** Already produced by `policy.act` ([policy.py:35](JEPA/experiments/exp_003_0_normalized_latent_jepa/models/policy.py#L35)) and logged into `health.entropy` ([train.py:362](JEPA/experiments/exp_003_1_ema_target/train.py#L362)).
- **Display:** With `n_actions = 4`, **Hmax = ln 4 ≈ 1.3863 nats**. Label: `policy_entropy [Hmax=1.386]`.
- **Why:** Low entropy too early = premature commitment. Persistently near Hmax = no learning. Standard policy-gradient diagnostic.
- **Soundness:** When `available_actions` masks out some actions, the effective `H_max` is `ln(|available|)`, which varies per step. For the streaming average this is fine; if precise comparison to `H_max` matters, log `policy_entropy / ln(|available|)` as a normalized companion (in [0, 1]).

---

## Section 6 — Performance Metrics

### 6.1 Loss panel — fully decomposed display

Each scalar shown together with the **multiplier**, the **raw component value**, and the **product**. No "L_total = ..." with hidden numbers.

**Components logged:**
- `L_next_state_pred` — the flow-matching loss returned by `FlowMatchingPredictor.compute_loss` ([predictor.py:71-84](JEPA/experiments/exp_003_0_normalized_latent_jepa/models/predictor.py#L71-L84)). This is the **velocity MSE** averaged over K sampled τ values per training step, not the rollout-prediction MSE.
- `L_next_state_pred[i]` for `i = 0..3` — already exposed as `per_latent` in `FlowMatchingPredictor.compute_loss` and stored in `health.per_latent_loss[i]`. `L_next_state_pred = mean_i L_next_state_pred[i]`.
- `L_action_pred` *(exp_003_2)* — cross-entropy of the action predictor's distribution against the ground-truth `a_t`.

**Display format (one line per training-loss log):**
```
L_total  = λ_state · L_next_state_pred  + λ_action · L_action_pred
         = 1.0    · 0.0412              + 0.0     · 0.0000
         = 0.0412
  └─ L_next_state_pred per latent:
       [0] 0.0391  (23.7%)
       [1] 0.0438  (26.6%)
       [2] 0.0405  (24.6%)
       [3] 0.0414  (25.1%)
       (mean = 0.0412; weights are uniform 1/4 each in current compute_loss)
  └─ L_action_pred per ground-truth action  (exp_003_2):
       a=0 (UP)    n=18  mean_CE=0.412
       a=1 (DOWN)  n=21  mean_CE=0.083
       a=2 (LEFT)  n=17  mean_CE=0.395
       a=3 (RIGHT) n=20  mean_CE=0.117
```

- `λ_state`, `λ_action` come from `Config` and are printed in the panel header (currently `λ_state = 1.0, λ_action = 0.0` because exp_003_2's action predictor is not yet wired).
- The **per-latent share** row makes it instantly visible if one latent's MSE is dominating the gradient signal. Implementation: already exposed as `per_latent = sq_err.mean(dim=[0, 2])` in [predictor.py:83](JEPA/experiments/exp_003_0_normalized_latent_jepa/models/predictor.py#L83) and stored in `health.per_latent_loss[i]`. The percentages are `L_i / sum_j L_j`.
- The **per-action-class** row groups the batch by the ground-truth action `a_t` and reports the mean action-pred CE inside each class. Reading it: if `mean_CE` is much lower for some action than others, the model has learned to "predict that action well when it actually happens" but is missing the other action classes — a class-imbalance / mode-collapse signal in the action predictor that the aggregate loss hides. (Implementation: `for a in range(n_actions): mask = (actions == a); class_loss[a] = ce_per_sample[mask].mean()`.)
- A second line shows the rollout-MSE-based **policy reward** so the reader can spot the gap from the velocity loss (see clarification below).

**Important clarification about the predictor reward vs. the predictor loss (carry over verbatim into the dashboard tooltip):**

> During encoder/predictor training, the loss is the flow-matching **velocity** MSE over K sampled τ values: a stochastic approximation that has the same minimum as the rollout MSE but is *not* numerically equal to it on any given step. During environment rollout, the **curiosity reward** is computed from `predict_with_loss` ([train.py:336](JEPA/experiments/exp_003_1_ema_target/train.py#L336)), which runs the full N-step ODE and computes MSE between the final `x_1` and `h_{t+1}`. So: encoder is trained on velocity MSE averaged over τ samples; the policy reward is rollout-prediction MSE. They are mathematically related but distinct quantities — do not expect them to agree numerically.

### 6.2 `reachable_tile_coverage_pct`
- **What:** Fraction of reachable tiles in the LS20 map that the agent has visited at least once within an episode (or over a rolling window of episodes).
- **How:** Pre-compute the set `R` of reachable tile coordinates from the level layout (manual inspection or BFS from spawn over walkable cells). During each episode, accumulate `V = {agent_position_t}`. Metric = `|V ∩ R| / |R|`.
- **Why:** Proxy for exploration. Catches the "agent loops in a corner forever" failure mode that reward-curve plots do not.
- **Soundness:** Requires knowing the agent's `(x, y)` per frame. Check whether `LS20Env` ([env_wrapper.py](JEPA/shared/env_wrapper.py)) exposes the agent position; if not, derive it from the rendered frame (the agent sprite color is small and identifiable). Re-build `R` per game ID — `environment_files/` holds multiple LS20 variants.

### 6.3 `cross_hits_per_episode`
- **What:** Number of times the agent steps onto a cross tile in an episode.
- **How:** Identify cross pixel coordinates by manual inspection of one rendered frame per game ID (the cross is the puzzle's central mechanic). At every step, check `agent_position ∈ cross_coords`. Sum within episode; log mean and max across episodes.
- **Why:** Domain-specific success proxy. Beating the level requires interacting with the cross — until this is > 0 the agent has not discovered the core mechanic.
- **Soundness:** Cross coordinates must be re-extracted per game ID (the layout in `environment_files/g50t/`, `re86/`, `tu93/` differs). Store a JSON manifest per game ID rather than hard-coding.

---

## Section 7 — Gradient Norms (per-module, per-source)

Log `||g||_2` per module group after `loss.backward()`, before `clip_grad_norm_`. All entries should be computed via the existing `grad_norm(params)` helper ([train.py:172](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py#L172)). For each row, the **Present?** column indicates whether the scalar is already logged.

### Gradient sources (precondition for 7.1 and 7.2 splits)

For upstream encoder modules (everything **before** the two predictors), three distinct gradient signals flow back:

- **`from_state`** — gradient from `L_next_state_pred`, flowing through `H_t` (the only path to the encoder, because `H_{t+1}` is the detached target — supplied by `target_encoder` in exp_003_1 or just `.detach()`-ed earlier).
- **`from_action_via_Ht`** *(exp_003_2)* — gradient from `L_action_pred` flowing through the `H_t` input of the action predictor.
- **`from_action_via_Htp1`** *(exp_003_2)* — gradient from `L_action_pred` flowing through the `H_{t+1}` input of the action predictor.

The third signal exists **only if exp_003_2's action predictor is fed an online-encoder `H_{t+1}`** (not a target-encoder / detached `H_{t+1}`). If the action predictor uses target-`H_{t+1}` for stability, `from_action_via_Htp1 ≡ 0` for all upstream modules — flag this in the panel header so the zero is read as "by design", not "broken".

By chain-rule linearity, for any upstream parameter `θ`:
```
g_total(θ) = g_from_state(θ) + g_from_action_via_Ht(θ) + g_from_action_via_Htp1(θ)
```
so the three contributions can be measured separately and summed for the existing aggregate.

**Implementation:** three separate backward passes per training step (or two if action pred is off):
1. `g_state = torch.autograd.grad(L_state, params, retain_graph=True)` — H_t-only path; uses the current forward with `h_targets.detach()` in `predictor.compute_loss` as today.
2. `g_action_via_Ht = torch.autograd.grad(L_action_with_Htp1_detached, params, retain_graph=True)` — re-forward action predictor with `H_{t+1}.detach()` so only the H_t path remains.
3. `g_action_via_Htp1 = torch.autograd.grad(L_action_with_Ht_detached, params)` — re-forward with `H_t.detach()` so only the H_{t+1} path remains.

Costs ~2× the JEPA backward pass and one extra action-predictor forward. Gate this **per training step** under a flag (e.g., every K updates) to control cost; the aggregate `g_total` from the normal training backward stays cheap.

### 7.1 Encoder — Image-Patch Self-Attention path

Params covered: `encoder.color_embed`, `encoder.patch_proj`, `encoder.sa_blocks.*`, `encoder.sa_norm`. Four scalars (3 sources + total):

| Metric | Source | Present? |
|---|---|---|
| `gnorm_patch_sa[from_state]`           | `L_state` via `H_t`         | ❌ — add (separate backward) |
| `gnorm_patch_sa[from_action_via_Ht]`   | `L_action` via `H_t`        | ❌ — exp_003_2 |
| `gnorm_patch_sa[from_action_via_Htp1]` | `L_action` via `H_{t+1}`    | ❌ — exp_003_2 |
| `gnorm_patch_sa[total]`                | aggregate from full backward | ✅ — `health.grad_enc_s1` ([train.py:586](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py#L586)) |

### 7.2 Encoder — Perceiver Resampler (split into 3 sub-blocks × 3 sources + total)

For each sub-block below, log 4 scalars: `[from_state]`, `[from_action_via_Ht]`, `[from_action_via_Htp1]`, `[total]`. (exp_003_4 removed the two `perc_self_r{i}` buckets; only cross-attn and `perc_other` remain.)

| Sub-block | Params covered |
|---|---|
| `gnorm_perc_cross_r0[*]` | `encoder.perceiver.rounds[0].cross_attn.*` |
| `gnorm_perc_cross_r1[*]` | `encoder.perceiver.rounds[1].cross_attn.*` |
| `gnorm_perc_other[*]`    | `encoder.perceiver.placeholders`, `encoder.perceiver.output_norm` |
| `gnorm_perc_total[*]`    | union of the above — `[total]` equals current `health.grad_enc_s2` |

`[total]` for the **union** is already logged as `health.grad_enc_s2` ([train.py:587](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py#L587)); the per-sub-block `[total]` and all `[from_*]` scalars need to be added. Cache sub-block parameter lists once at startup (`cross_r0_params = list(encoder.perceiver.rounds[0].cross_attn.parameters())`, etc.), then per-source compute 5 `grad_norm()` values (one per sub-block) on each of the 3 backward passes.

**Sanity checks to display:**
- Sub-additivity over sub-blocks: `gnorm_perc_total[s] ≤ sum_{sub} gnorm_perc_sub[s]` for each source `s`.
- Source-additivity over sources: `gnorm_perc_sub[total]` should be close to (but not equal to — L2 is not linear) the L2 of the per-source vector sum. Display `||g_total|| / sqrt(sum_s ||g_s||²)` as a tag — equals 1 when sources are mutually orthogonal, < 1 when they partially cancel, > 1 when they reinforce. (See 7.6 for the cosine version of this.)

### 7.3 Predictors

| Metric | Params covered | Present? |
|---|---|---|
| `gnorm_state_pred_mlp_{0..3}` | `predictor.mlps[i]` for `i = 0..3` (per-latent flow-matching MLP) | ✅ — exactly `health.grad_pred_mlps[i]` ([train.py:588, 604](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py#L588)) |
| `gnorm_state_pred_time_emb`   | `predictor.time_embed.*` | ✅ — `health.time_emb_grad` ([train.py:589](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py#L589)) |
| `gnorm_action_embed`          | `action_embed.*` | ❌ — add (currently rolled into the predictor optimizer but not logged) |
| `gnorm_action_pred` *(exp_003_2)* | action predictor module's params (whatever architecture is chosen) | ❌ — add when module lands |

The per-latent split for the state predictor matches the user request directly; nothing to add for that piece.

### 7.4 Policy

| Metric | Params covered | Present? |
|---|---|---|
| `gnorm_policy` | `policy.*` | ✅ — `health.grad_policy` ([train.py:648](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py#L648)) |

Kept as a single scalar per user request.

### 7.5 Gradient-source cosine similarity — agreement between losses *(exp_003_2)*

For each upstream parameter group `G` in §7.1 + §7.2 (patch SA, perc_cross_r0, perc_cross_r1, perc_other):

| Metric | Definition |
|---|---|
| `gcossim_state_vs_action[G]` | `cos( g_state_G ,  g_action_G )`  where  `g_action_G = g_action_via_Ht_G + g_action_via_Htp1_G` |

Flatten each per-group gradient to a 1-D vector before the cosine.

- **Why:** A scalar in `[-1, 1]` saying whether the state predictor and the action predictor want to move the module's weights the **same direction** (positive), in **orthogonal** directions (≈ 0), or in **opposing** directions (negative). Negative values are the strongest "the two heads are fighting" signal — much more informative than just watching two grad norms grow.
- **Reading guide:** Persistent ≈ 1 means the action head is redundant with the state head (no new signal). Persistent ≈ −1 means the two heads cancel — total grad will be small even when each is large; this is the failure mode that motivates the cosine metric in the first place. Healthy bands tend to sit slightly positive (0.1–0.5).

### 7.6 Action-predictor input-path cosine similarity *(exp_003_2)*

For each upstream parameter group `G` as above:

| Metric | Definition |
|---|---|
| `gcossim_action_Ht_vs_Htp1[G]` | `cos( g_action_via_Ht_G ,  g_action_via_Htp1_G )` |

- **Why:** The action predictor's two inputs `(H_t, H_{t+1})` should encode *different* state information; if they didn't, the predictor's job is trivial and uninformative. Cosine similarity of their respective upstream-gradient pulls tells us whether the predictor is treating its two inputs as redundant (≈ 1: gradient pushes the encoder to make `H_t` and `H_{t+1}` "look similar" through the same parameters) or as complementary (≈ 0 or negative).
- **Reading guide:** Watch for a drift toward 1 as a leading indicator that the action predictor is collapsing into a "use either input, doesn't matter" regime — a precursor to action entropy (4.3) collapsing to ln(n_actions).

### 7.7 Update-to-weight ratio (per param group)

Adopted from the proposals list — see "Adopted from proposals" below for the partition. Compute as `||θ_new − θ_old||_2 / ||θ_old||_2` per group, after `optimizer.step()`. Healthy band ≈ `[1e-4, 1e-2]`. The same sub-module partition from §7.1/§7.2/§7.3/§7.4 is reused so the U/W ratios line up row-for-row with the gnorm tables.

### 7.8 Soundness notes for the gradient-norm section

- **Two backward passes for training, three for the breakdown.** The policy backward ([train.py:643](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py#L643)) runs separately from the JEPA backward. `gnorm_policy` is post the policy backward; the rest are post the JEPA backward. The three source-decomposed backward passes (§7.1, §7.2) are independent of the training backward — they are diagnostic and can be gated to every K steps. Never sum policy norms with JEPA norms; they are in different optimizer worlds.
- **Pre-clip is the right measurement point.** Current code already reads grad_norm before `clip_grad_norm_` ([train.py:586-589](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py#L586-L589)); preserve that ordering when adding the new sub-block calls. Logging post-clip would just give us back the clip threshold.
- **Sub-additivity sanity check.** L2 norm is sub-additive over a partition: `||g_total|| ≤ sum(||g_part||)`. If `gnorm_perc_total > sum(gnorm_perc_*)`, something has been double-counted (e.g., a shared parameter accidentally listed in two groups) — flag this in the dashboard.
- **`retain_graph` cost.** Three `torch.autograd.grad(..., retain_graph=True)` calls hold the forward graph in memory longer than today's pipeline. On the MPS device with `batch_size = 64` this is workable but not free — measure once on real hardware before enabling per-step.
- **Source-decomposition is only meaningful if both predictors see online-encoder inputs.** If exp_003_2 ends up using target-encoder `H_{t+1}` for the action predictor (for stability), `from_action_via_Htp1` is identically zero and 7.6 is undefined. Print "n/a (target-H_{t+1} in use)" in that case rather than logging zeros that look like a collapse.

---

## Soundness Audit — Cross-Cutting Notes

1. **Attention capture cost.** Today's code only stores attention matrices in `model.eval()` mode (`_debug_attn`). For training-time metric capture (sections 2.1, 2.2, 3.1) we need a lightweight, flag-gated capture path. Without it, every metric pass forces a re-forward in eval mode on a fresh minibatch — fine if done every `EMBED_METRIC_FREQ` updates, but flag-gating is cheaper.

2. **Pairwise *cosine* vs pairwise *L2*.** The existing `health.latent_pairwise_l2` ([train.py:474-477](JEPA/experiments/exp_003_1_ema_target/train.py#L474-L477)) measures L2 distance, not cosine. Because `output_norm` in `PerceiverResampler` is a `LayerNorm` (not L2-normalize), the latents are not strictly unit-norm, but their scale is bounded — L2 and cosine carry slightly different information here. Section 1.2 asks for cosine; add it as a new field rather than replacing the L2 one.

3. **Eval determinism.** Every metric tagged "eval episode" assumes (i) `model.eval()` is set, (ii) seeds are fixed across the eval window so the same trajectory is replayed across checkpoints, and (iii) the policy is acted upon greedily or with a fixed seed for `Categorical.sample()`. The current `set_seeds` helper ([train.py:138](JEPA/experiments/exp_003_0_normalized_latent_jepa/train.py)) handles training-time seeding but not the eval-window seed. Add a `set_eval_seeds(seed + step)` call before each eval pass to make the trajectory deterministic per checkpoint.

4. **Episode-aligned step indices (sections 1.2, 1.5).** Eval episodes run **until `is_end_of_life`**. Because LS20 is deterministic and rollouts sample from the policy distribution, `T_episode` is effectively the same across the N rollouts of a given checkpoint, but it can shift across checkpoints as the policy improves. If `T_episode < 20` at a checkpoint, the `t = 20` slot in 1.2 is `n/a` for that checkpoint. For 1.5, log `T_episode` next to the cosine value so length shifts are not misread as representation shifts. Never pad with the last latent — it silently biases the time-evolution plot.

5. **`H_max` displayed next to entropies.** Section 4.3 and 5.1 both should print `H_max = ln(n_actions)` in the legend, computed from `cfg.n_actions` so it stays correct if the action set grows.

6. **Buffer-batch vs eval-rollout statistics.** Section 1.4's training-time analogue (`health.ht_ht1_cossim`) is computed over consecutive rollout steps during data collection — so it reflects the **rollout policy**, which includes warmup random actions for the first 1k steps. The eval-time companion uses the trained policy and is the comparable signal across checkpoints.

7. **Action predictor (4.2, 4.3) does not yet exist.** Adding its metrics is conditional on the exp_003_2 commit landing. Until then, leave dashboard panels but mark them `n/a`.

---

## Adopted from proposals — placement summary

| Original proposal | Disposition |
|---|---|
| A `latent_effective_rank` | Promoted → §1.6 |
| B `across_state_latent_std` | **Dropped** (cosine sim metrics already cover this collapse mode) |
| C / H `latent_norm_per_token` | Folded into §1.7 below |
| D `ema_target_lag` | **Dropped** |
| E `predictor_velocity_norm` | Folded into §4.1c below |
| F `update_to_weight_ratio` | Folded into §7.7 (partition table reproduced below) |
| G `reward_distribution_skew` | **Dropped** |

### 1.6 `latent_effective_rank`
- **What:** Effective rank of the `4 × D` latent matrix, defined as `exp(H(p))` where `p` is the normalized singular-value distribution.
- **How:** Already computed by `health.latent_eff_rank` ([train.py:478](JEPA/experiments/exp_003_1_ema_target/train.py#L478)) via `effective_rank(h_m[0])`.
- **Why:** Falls toward 1 when the four latent tokens collapse to a single direction. Strict complement to 1.2 cosine: cosine sees pairwise alignment; effective rank sees whether the joint subspace is full-rank. Either alone can miss a collapse mode the other catches; together they triangulate it.

### 1.7 `latent_norm_per_token`
- **What:** L2 norm of each of the 4 latent tokens.
- **How:** Already computed as `health.latent_norms[i]` ([train.py:470](JEPA/experiments/exp_003_1_ema_target/train.py#L470)).
- **Why:** Useful for detecting LayerNorm-output drift — should hover near `√D ≈ 11.3` for unit-variance D=128 outputs. A drift well above that is the recurrent-norm-growth pattern the exp_003 `output_norm` was added to prevent.

### 4.1c `predictor_velocity_norm`
- **What:** Mean L2 norm of `x_1_hat - x_0` from `predict_with_trajectory`.
- **Why:** If this collapses to ~0 the predictor is outputting "no change" — a corollary of 4.1 saturating but in **absolute** terms, not similarity terms. Diagnoses the failure mode where 4.1 says "steps are similar" because the steps are all near-zero rather than because the predictor reached convergence cleanly.

### 7.7 `update_to_weight_ratio` — partition table

| Group ID | Module(s) covered |
|---|---|
| `uwr_patch_sa` | `encoder.color_embed`, `encoder.patch_proj`, `encoder.sa_blocks.*`, `encoder.sa_norm` |
| `uwr_perc_cross_r0` | `encoder.perceiver.rounds[0].cross_attn.*` |
| `uwr_perc_cross_r1` | `encoder.perceiver.rounds[1].cross_attn.*` |
| `uwr_perc_other` | `encoder.perceiver.placeholders`, `encoder.perceiver.output_norm` |
| `uwr_state_pred_mlp_{0..3}` | `predictor.mlps[i]` per latent |
| `uwr_state_pred_time_emb` | `predictor.time_embed.*` |
| `uwr_action_embed` | `action_embed.*` |
| `uwr_action_pred` *(exp_003_2)* | action predictor module's params |
| `uwr_policy` | `policy.*` |

Snapshot each group's flattened param vector pre-step, compute the ratio post-step. Reuses the same partition as §7.1/§7.2/§7.3/§7.4 so the U/W ratios line up row-for-row with the gnorm tables.

---

## Verification Sketch (how to wire this up)

1. **Hook points.** Add `register_forward_hook` to: (a) `encoder.perceiver.rounds[0].cross_attn` (captures round-0 post-CA latent for 1.3); (b) `encoder.sa_blocks[*]` for patch-attention matrices (sections 2.1, 2.2) — store as tensors on a `metrics_state` dict guarded by a `capture_attn = True` flag set only during the metric pass. Section 3 (latent self-attn) has no hook points in exp_003_4 since `_SelfAttentionAmongLatents` was removed.
2. **Eval rollout helper.** Add an `eval_episode(encoder, policy, env, T=20, seed)` function (read-only model, fixed seed) that returns the per-step list `[(H_t, attn_state_t, agent_pos_t)]`. Run it every K training steps with N seeds, fold into the existing `HealthMonitor`/`MetricsWriter` ([train.py:282](JEPA/experiments/exp_003_1_ema_target/train.py#L282)).
3. **Dashboard ingestion.** The existing dashboard ([JEPA/dashboard/server.py](JEPA/dashboard/server.py)) reads from `MetricsWriter`; extend the writer schema with the new fields, grouped by section header so the UI can render Section 1–6 as tabs.
4. **End-to-end test.** Run `uv run python -m JEPA.experiments.exp_003_1_ema_target.train --max-steps 5000` with the new metrics on, then open the dashboard and confirm Section 1.1–1.5, 3.1, 4.1, 5.1, 6.1 all populate (these are the ones implementable today without the exp_003_2 action predictor).
