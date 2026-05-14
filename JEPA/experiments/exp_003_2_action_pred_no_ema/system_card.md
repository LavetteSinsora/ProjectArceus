# System Card — exp_003_2_action_pred_no_ema

| Field | Value |
|-------|-------|
| **Experiment ID** | `exp_003_2_action_pred_no_ema` |
| **Status** | Active |
| **Parent experiment** | `exp_003_0_normalized_latent_jepa` |
| **Game** | LS20 Level 1 (`ls20-9607627b`) |
| **Reward** | Intrinsic curiosity: `prediction error = state prediction error + action prediction error` |

## 1. One-Paragraph Summary

This experiment trains a world model and policy for ARC-AGI LS20 Level 1 using a Perceiver-based JEPA with **two predictors**: (i) a flow-matching **state predictor** that learns `h_{t+1}` given `h_t` and the action, and (ii) a new **action predictor** that learns `a_t` given `(h_t, h_{t+1})`. The action predictor is introduced as an explicit anti-collapse mechanism: if the encoder maps `h_t ≈ h_{t+1}`, the transition carries no information about which action was taken and the action predictor cannot do better than chance. Gradient from the action predictor flows back through the encoder via **both** `h_t` and `h_{t+1}`, pushing the encoder to keep adjacent latents discriminative. Unlike `exp_003_1`, this experiment deliberately **does not** use an EMA target encoder — we are testing whether the action-predictor signal alone is sufficient to prevent the latent collapse observed in earlier exp_003 runs. The replay buffer now stores `frame_{t+1}` (raw uint8) instead of a precomputed `h_target`, so the target latent is re-encoded with the **current** encoder at every training step. The encoder is therefore updated through three gradient paths per step (one from the state loss, two from the action loss), and the JEPA loss is a weighted combination `0.5·L_state + 0.5·L_action`.

---

## 2. Architecture

### 2.1 Overview

```
frame (64×64 uint8)
       │
       ▼
  ┌────────────────────────────────┐
  │  Stage 1: Patch Encoder        │
  │  color_embed(16→4)             │
  │  patch_proj(1024→128)          │
  │  SA-Block 1  (4 heads, RoPE)   │
  │  SA-Block 2  (4 heads, RoPE)   │
  │  sa_norm (LayerNorm)           │
  │  → (B, 16, 128)                │
  └────────────────────────────────┘
       │
       ▼  context
  ┌────────────────────────────────┐  ← queries: h_{t-1} (B, 4, 128)
  │  Stage 2: Perceiver Resampler  │     or placeholders at episode start
  │  Round 0: Cross-Attn + SA      │
  │  Round 1: Cross-Attn + SA      │  (separate weights per round)
  │  output_norm (LayerNorm)       │
  │  → h_t (B, 4, 128)            │
  └────────────────────────────────┘
       │
       ├──────────────────────────► Policy MLP → action
       │
       ├──────────────────────────► State Predictor (flow matching)
       │                            (h_t, a_emb) → h̃_{t+1}
       │
       └──────────────────────────► Action Predictor MLP
                                    (h_t, h_{t+1}) → p(a | h_t, h_{t+1})
```

### 2.2 Stage 1 — Patch Encoder

| Attribute | Value |
|-----------|-------|
| Input | `(B, 64, 64)` uint8 frame (color indices 0–15) |
| Output | `(B, 16, 128)` SA-normed patch token sequence |
| Patch grid | 4×4 = 16 patches, 16×16 pixels each |

**Pipeline (step by step):**

1. **Color embedding:** `nn.Embedding(16, 4)` — maps each color index to a 4-dim learned vector. The 64×64 frame becomes `(B, 64, 64, 4)`.
2. **Patch reshape:** Rearrange from `(B, 4, 16, 4, 16, 4)` to `(B, 16, 1024)` — each patch is a flattened 16×16×4 region.
3. **Patch projection:** `nn.Linear(1024, 128)` — project each patch to d_model.
4. **SA-Block 1 and 2:** Each block applies pre-norm multi-head self-attention with 2D RoPE positional encoding over the 16 patch tokens, followed by a pre-norm FFN. No learned positional embeddings — position is entirely encoded in the rotary frequencies.
5. **SA norm:** `nn.LayerNorm(128)` — layer-normalizes the 16-token sequence before handing it to the Perceiver as context.

**2D Rotary Positional Encoding (RoPE):** The 4×4 patch grid has row positions {0,1,2,3} and col positions {0,1,2,3}. For each attention head, the first `d_head//2 = 16` key/query dimensions are rotated by row-based frequencies, the last 16 by column-based frequencies. This encodes 2D spatial structure without any learned parameters.

**Why no L2 normalization (unlike exp_001):** The Perceiver resampler requires the full magnitude information in its context. L2 normalization on the SA output would discard scale, which is used by the cross-attention keys/values to weight different patches.

### 2.3 Stage 2 — Perceiver Resampler

| Attribute | Value |
|-----------|-------|
| Input (context) | `(B, 16, 128)` SA-normed patch tokens |
| Input (queries) | `(B, 4, 128)` recurrent latents h_{t-1} (or placeholders at t=0) |
| Output | `(B, 4, 128)` output-normed latent vectors h_t |
| Rounds | 2, **separate weights** (not weight-tied) |
| Heads | 4 per cross/self attention |

**Per-round computation:**

```
# Cross-attention: latents attend to patch context
Q = q_proj(norm_q(h))          # (B, 4, 128)
K = k_proj(norm_kv(context))   # (B, 16, 128)
V = v_proj(context)            # (B, 16, 128)
attn_w = softmax(Q K^T / √32)  # (B, n_heads, 4, 16) — each latent attends to patches
h = h + out_proj(attn_w @ V)
h = h + FFN(norm(h))           # pre-norm FFN

# Self-attention: latents attend to each other
Q,K,V = projections(norm(h))
latent_attn = softmax(Q K^T / √32)  # (B, n_heads, 4, 4)
h = h + out_proj(latent_attn @ V)
h = h + FFN(norm(h))

# After all rounds:
h = output_norm(h)             # (B, 4, 128) — unit-scale recurrent state
```

**Separate per-round weights:** Each Perceiver round has independent parameters (not weight-tied) so gradient does not accumulate twice through shared parameters when the Perceiver is called multiple times per training step (see §4.2 for the encoder-gradient bookkeeping).

**Output LayerNorm:** The recurrent state `h_t` is passed as queries to the next step's Perceiver. The output LayerNorm clamps the recurrent state to near-unit scale at every step, preventing the linear norm growth that would otherwise accumulate over residual additions across the rollout.

**Placeholder initialization:** At episode start (t=0), the Perceiver queries are initialized from a learned `nn.Parameter` of shape `(4, 128)` (the "placeholders"), broadcast to batch size. After the first step, `h_{t-1}` provides the recurrent state.

### 2.4 State Predictor (flow matching)

| Attribute | Value |
|-----------|-------|
| Input | `(B, 4, 128)` current latents h_t + `(B, 32)` action embedding |
| Output | `(B, 4, 128)` predicted next latents h̃_{t+1} |
| Architecture | 4 independent MLPs (one per latent) + shared sinusoidal time embedding |
| ODE steps | 3 (Euler integration) |

**Flow matching training objective:** Given source distribution h_t and target h_{t+1}, define a straight-line path `x_τ = (1 − τ)·h_t + τ·h_{t+1}` for τ ∈ [0, 1]. At training time, sample τ ~ Uniform[0, 1] and train each MLP to predict the clean endpoint x₁ = h_{t+1} from the noisy intermediate x_τ:

```
τ      ~ Uniform[0, 1]
x_τ    = (1 − τ) h_t  +  τ h_{t+1}
x̂₁[i] = MLP_i(x_τ[i], action_emb, time_embed(τ))   # for latent i in {0,1,2,3}

L_state = Σ_i ‖ x̂₁[i] − h_{t+1}[i] ‖²   (mean over batch and dims)
```

**Stop gradient on h_{t+1}:** At the call site in `train.py`, the state-loss target is always detached: `state_predictor.compute_loss(h_t_fresh, h_tp1_fresh.detach(), a_emb)`. Without this, the encoder and predictor can jointly minimize the loss by collapsing all states to a single constant vector — the predictor trivially matches a constant target by outputting that constant for any input. Detaching the target eliminates that gradient path for the state loss. (The action predictor in §2.5 deliberately does **not** detach `h_{t+1}` — it relies on gradient through both endpoints to enforce non-collapse.)

**Sinusoidal time embedding:** τ is encoded as a sinusoidal vector of dimension 128, then projected to 512 via a linear layer + GELU.

**Per-latent MLPs:** `_LatentMLP = Linear(128 + 32 + 512, 512) → GELU → Linear(512, 128)`. Using 4 separate MLPs (one per latent) rather than a single shared MLP allows each latent to specialize.

**Euler ODE inference:**

```python
x_0 = h_t          # initial condition
x_k = h_t.clone()
for k in range(N):  # N = 3
    τ    = k / N
    x̂₁  = predict_clean(x_k, τ, action_emb)
    x_k  = x_k + (1/N) * (x̂₁ − x_0)
return x_k          # h̃_{t+1}
```

**No EMA target encoder.** Some JEPA variants (including `exp_003_1`) use an EMA copy of the encoder to produce `h_{t+1}` targets, on the grounds that a slowly-moving target prevents collapse. This experiment deliberately omits the EMA mechanism — we are testing whether the action predictor (§2.5) alone is sufficient to prevent representation collapse.

### 2.5 Action Predictor

| Attribute | Value |
|-----------|-------|
| Input | `(B, 4, 128)` current latents `h_t` + `(B, 4, 128)` next latents `h_{t+1}` |
| Output | `(B, 4)` logits over the four discrete actions |
| Architecture | Concatenate flattened `h_t ⊕ h_{t+1}` → MLP (512 hidden, GELU) → Linear(512, 4) |
| Module class | Standalone `ActionPredictor` — distinct from the state predictor, not a subclass of a shared `Predictor` |

**Formulation:**

```
z = concat(flatten(h_t), flatten(h_{t+1}))     # (B, 1024)
logits = Linear(512, 4)(GELU(Linear(1024, 512)(z)))
p      = softmax(logits)                       # (B, 4)
L_action = cross_entropy(logits, a_t)          # multi-class CE against one-hot a_t
```

**Gradient flow:** Gradient flows from `L_action` back through **both** `h_t` and `h_{t+1}` into the encoder — there is no detach on either side. This is the path that pushes the encoder away from the trivial `h_t ≈ h_{t+1}` collapse.

**Rationale:** A previous failure mode (observed in earlier exp_003 runs) was the encoder converging to a near-constant latent across the rollout, so that `h_t ≈ h_{t+1}` and the state-prediction loss could be driven to zero trivially. If `h_t = h_{t+1}` then the pair `(h_t, h_{t+1})` carries zero information about which action produced the transition, and the action predictor cannot do better than chance (`L_action ≈ log 4 ≈ 1.386`). Driving `L_action` down therefore requires the encoder to keep adjacent latents discriminative — exactly the property collapse violates. The action predictor thus acts as an information-theoretic regularizer on the encoder.

### 2.6 Action Embedding

`nn.Embedding(4, 32)` — maps discrete action index (0–3) to a 32-dim learned vector. Concatenated into each per-latent state-predictor MLP's input. Not used by the action predictor (the action predictor's *output* is a distribution over the same action indices; it never reads an embedded action as input).

**Why learned over one-hot:** Allows the state predictor to discover latent similarity between actions that produce similar dynamics.

### 2.7 Policy Network

| Attribute | Value |
|-----------|-------|
| Input | `(B, 4, 128)` latents, flattened to `(B, 512)` |
| Hidden | 512 (GELU activation) |
| Output | `(B, 4)` action logits |
| Recurrency | None (stateless MLP) |

```python
logits = Linear(512, 4)(GELU(Linear(512, 512)(flatten(h_t))))
# available-action masking: set logits[unavailable] = −∞
a_t ~ Categorical(softmax(logits))
```

**Why stateless:** The latent state `h_t` already carries temporal context via the recurrent Perceiver.

**Available-action masking:** LS20 Level 1 restricts which actions are legal at each step. Before softmax, logits for unavailable actions are set to −∞.

---

## 3. Key Changes from exp_003_0 / exp_003_1

| Change | exp_003_0 | exp_003_1 | exp_003_2 (this) | Motivation |
|--------|-----------|-----------|------------------|------------|
| **Anti-collapse mechanism** | Stop-grad on `h_{t+1}` target | Stop-grad + EMA target encoder | Stop-grad on state target + **Action Predictor** (no EMA) | Test whether an information-theoretic regularizer (action-from-transition) alone prevents collapse. |
| **Predictors** | State predictor only | State predictor only | State predictor **+ Action predictor** (separate module class) | Add an explicit non-trivial latent-discriminability signal. |
| **JEPA loss** | `L_state` | `L_state` | `0.5 · L_state + 0.5 · L_action` | Combine both signals; encoder receives gradient from both. |
| **Curiosity reward** | `MSE(h̃, h_{t+1})` | same | `0.5 · MSE(h̃, h_{t+1}) + 0.5 · CE(p_pred, a_t)` (cap 50) | Reward novelty in either predictor. |
| **Buffer target field** | `h_target` (float32, frozen at rollout time) | same | `next_frame` (uint8) — re-encoded fresh at training time | Eliminate stale-encoder problem; always use the current encoder for the target. |
| **Buffer sampling** | 20% recency / 80% uniform | same | **Uniform only** (no recency mix) | Re-encoding makes recency unnecessary — every target is encoded with the current weights. |
| **EMA target encoder** | None | Cosine ramp 0.996 → 0.9999 | **None** | Mechanism intentionally omitted for this experiment. |
| **Encoder gradient paths per step** | 1 (state loss via `h_t`) | 1 | **3** (state via `h_t`; action via `h_t` and via `h_{t+1}`) | See §4.2 bookkeeping. |
| **Max episode length guard** | `MAX_EP_STEPS = 300` | same | **Removed** | LS20 episodes are bounded by the in-game energy mechanic; no need for an external guard. |

---

## 4. Training

### 4.1 Data Collection (Recurrent Rollout)

At each environment step:

1. **Encode frame:** `h_current, sa_out, attn = encoder(frame_t, queries)` where `queries = h_{t-1}` (or placeholders at episode start). Run under `torch.no_grad()`.
2. **Select action:** Policy MLP on `h_current` during rollout (or uniform random during warmup). Record:
   - `action_idx` (sampled int)
   - `log_prob = Categorical(logits).log_prob(action_idx)` — **tensor with autograd graph back to policy parameters**
   - `entropy = Categorical(logits).entropy()` — tensor with graph
3. **Step environment:** Get `next_frame`, `is_terminal`.
4. **Compute curiosity reward** (dual-path, both under `no_grad`):
   - **State path:** Encode `next_frame` with `h_current` as queries → `h_next`. Run state predictor's full Euler ODE on `(h_current, action)` → `h̃_{t+1}`. `state_err = MSE(h̃_{t+1}, h_next)` (mean over 4×128 dims).
   - **Action path:** Run action predictor on `(h_current, h_next)` → distribution `p_pred` over 4 actions. `action_err = CE(p_pred, one_hot(action_idx))`.
   - `r_t = clamp(w_state · state_err + w_action · action_err, max=50.0)` with `w_state = w_action = 0.5`.
5. **Append to LatentBuffer:** `add(frame_t, h_query_np, action_idx, next_frame_np)`. Note the fourth field is now **`next_frame` (uint8, 64×64)**, not `h_target` (float32). This lets us re-encode the target at training time with the current encoder.
6. **Append to PolicyBuffer:** `(log_prob, reward, entropy)`. `log_prob` and `entropy` retain their autograd graphs so REINFORCE can backprop into the policy at update time. This works because the policy is small and the policy-buffer window is short (64 steps); if the policy is later scaled up, this storage scheme may need to switch to logits + sampled action, with `log_prob` recomputed at update time.
7. **Advance recurrent state:** `h_{t-1} ← h_current`.

**Why store `h_query` (not `h_current`):** At training time, the encoder is re-called as `encoder(batch.frames, batch.h_queries.detach())`. This exactly replicates the rollout's forward pass for `frame_t`, so the gradients computed at training correspond to the same input/query combination that was used during data collection.

### 4.2 Training Updates

**JEPA / dual-predictor update** (every `update_freq=5` environment steps, once buffer ≥ 512):

```python
batch = latent_buf.sample(batch_size=64, device)
# batch.frames:      (64, 64, 64)   uint8 — frame_t
# batch.h_queries:   (64, 4, 128)   float32 — h_{t-1} from rollout
# batch.actions:     (64,)          int64
# batch.next_frames: (64, 64, 64)   uint8 — frame_{t+1}

# 1. Re-encode h_t fresh (gradient through encoder)
h_t_fresh = encoder(batch.frames, batch.h_queries.detach())

# 2. Re-encode h_{t+1} fresh from STORED next_frame.
#    IMPORTANT: detach h_t_fresh when used as the recurrent QUERY of this
#    second encoder call, so the second forward does not back-propagate into
#    the first forward via the recurrent-query path. This keeps the encoder-
#    gradient bookkeeping clean (exactly three paths into the encoder).
h_tp1_fresh = encoder(batch.next_frames, h_t_fresh.detach())

# 3. State predictor loss — target detached at the call site
L_state, _ = state_predictor.compute_loss(
    h_t_fresh,
    h_tp1_fresh.detach(),
    action_embed(batch.actions),
)

# 4. Action predictor loss — NO detach on either side
L_action = F.cross_entropy(
    action_predictor(h_t_fresh, h_tp1_fresh),
    batch.actions,
)

# 5. Combined JEPA loss
L = lambda_state * L_state + lambda_action * L_action   # default 0.5 / 0.5
L.backward()
clip_grad_norm_(
    encoder + state_predictor + action_predictor + action_embed,
    max_norm=5.0,
)
enc_opt.step(); state_pred_opt.step(); action_pred_opt.step()
```

**Encoder gradient bookkeeping (important for this experiment):**

The encoder parameters receive gradient through **three distinct paths** per training step:

| # | Loss term | Path into encoder | Detached? |
|---|-----------|-------------------|-----------|
| 1 | `L_state`  | via `h_t_fresh` (first arg to flow-matching loss) | no |
| 2 | `L_action` | via `h_t_fresh` (first arg to action predictor)   | no |
| 3 | `L_action` | via `h_tp1_fresh` (second arg to action predictor) | no |

By contrast, exp_003_0 had only one such path (state loss via `h_t_fresh`; the stored `h_target` was a frozen numpy snapshot carrying no graph). The encoder here will therefore see roughly 1.5–2× the gradient signal per step (less than the naive 3× because the two `L_action` paths share the action-MLP's downstream Jacobian rather than duplicating it).

Detaching `h_t_fresh` when it is reused as the recurrent query for the second encoder call removes a fourth latent path — gradient flowing from `L_action` back through the second encoder forward's *query input* into the first encoder forward's parameters. Keeping that detach is what makes the 3-path count clean.

The Perceiver participates in two encode calls per step, so paths 1 and 2 flow through the Perceiver from the first call, and path 3 flows through it from the second call — the Perceiver sees gradient from all three paths.

**Note on training vs. inference for state error:** During training, `L_state` is computed from a **single random τ sample** along the flow-matching interpolation path. During reward computation at rollout time (see §5.2), the state error is computed from the **full N-step Euler integration** comparing the post-ODE prediction against the actual `h_{t+1}`. Both measure the state predictor's error; the single-τ form is used for training-efficiency, and the full-ODE form for stability of the intrinsic reward.

**Single encoder forward for `h_{t+1}`:** We deliberately do **one** encoder call for the next frame and reuse its output for both losses (detaching at the state-loss call site). Two separate forwards would compute the same activations twice for no gradient benefit.

**Policy update** (every `policy_update_freq=64` environment steps):

```python
log_probs, rewards, entropies = policy_buf.get(device)
adv = rewards − baseline.value
adv = adv / (adv.std().clamp(min=1e-8) + 1e-8)
baseline.update(rewards.mean())

pol_loss = −mean(adv * log_probs) − 0.10 * mean(entropies)
pol_loss.backward()
clip_grad_norm_(policy, max_norm=1.0)
pol_opt.step()
```

### 4.3 Per-Component Optimizer Design

Four separate optimizers partition the parameters:

| Optimizer | Parameters | LR | Weight Decay |
|-----------|-----------|-----|------|
| `enc_opt` group 1 | color_embed, patch_proj, sa_blocks, sa_norm | 1e-4 | 0.01 |
| `enc_opt` group 2 | perceiver (all rounds + output_norm + placeholders) | 5e-5 | 0.01 |
| `state_pred_opt` | state_predictor (MLPs + time_embed) + action_embed | 1e-4 | 0.01 |
| `action_pred_opt` | action_predictor | 1e-4 | 0.01 |
| `pol_opt` | policy | 1e-4 | 0 (Adam) |

**Perceiver-LR rationale.** Of the three encoder-gradient paths enumerated in §4.2, two pass through the first encoder call (paths 1 and 2 — both via `h_t_fresh`) and one passes through the second encoder call (path 3 — via `h_tp1_fresh`). The Perceiver therefore receives 2–3× the gradient accumulation per step relative to a single-loss baseline. The half-LR on the Perceiver (`5e-5`, half of the SA `1e-4`) is preserved from exp_003_0 to compensate.

**LR decision for this revision.** Encoder LRs are kept at exp_003_0 values (`sa_lr = 1e-4`, `perceiver_lr = 5e-5`) unchanged. `grad_clip_model = 5.0` will absorb the increased post-sum gradient magnitude. If monitoring shows the post-clip update consistently saturating the clip (i.e., the clip is doing nearly all the magnitude control), a follow-up revision will halve the encoder LRs.

### 4.4 LatentBuffer

The `LatentBuffer` (to be extended in `JEPA/shared/buffer.py`) stores raw-frame-based transitions:

```
fields:  frames       (uint8, capacity × 64 × 64)   — frame_t
         h_queries    (float32, capacity × 4 × 128) — h_{t-1} from rollout
         actions      (int64,  capacity)            — a_t
         next_frames  (uint8, capacity × 64 × 64)   — frame_{t+1}  (REPLACES h_targets)
```

**Sampling is uniform** over the full buffer. The `recency_fraction` / `recent_window` mechanism from exp_003_0 is **removed** in this experiment: because both `h_t` and `h_{t+1}` are re-encoded with the current encoder at every training step, there is no stale-encoder problem to mitigate by biasing toward recent transitions.

**Memory note:** replacing a `(4, 128)` float32 target (2048 B) with a `(64, 64)` uint8 next_frame (4096 B) roughly doubles the per-row target footprint, but total buffer memory at 50K capacity remains well under 500 MB.

The dying transition (s_dying → s_next_life_start) is excluded from the buffer. The episode buffer `ep_transitions` accumulates all steps; on life-end, `ep_transitions[:-1]` (all except the last step that triggered the life-end) are pushed to the LatentBuffer.

### 4.5 Training Schedule

| Phase | Steps | Actions | JEPA update | Policy update |
|-------|-------|---------|------------|---------------|
| Warmup | 0 – 999 | Uniform random | Every 5 env steps (once buffer ≥ 512) | No |
| Joint training | ≥ 1,000 | Policy | Every 5 env steps | Every 64 env steps |

**Checkpointing:** Every 5,000 steps → `checkpoints/step_XXXXXX.pt`. On training end → `step_XXXXXX_final.pt`. Each run creates a timestamped directory under `runs/` with `training.log` and `metrics.jsonl`.

**Episode termination:** Episode length is governed by the in-game energy mechanic — energy is decremented every step, and the episode terminates deterministically when energy reaches zero. No external `MAX_EP_STEPS` guard is required, and the one from exp_003_0 is removed.

---

## 5. Inference (Deployment)

### 5.1 Recurrent State Management

```python
# Episode start
h_t = encoder.perceiver.get_initial_queries(1, device)  # (1, 4, 128) placeholders

for each timestep t:
    frame_t = get_frame()
    h_current, _, _ = encoder(frame_t, h_t)   # (1, 4, 128)
    action_idx, _, _ = policy.act(h_current.squeeze(0), available_actions)
    next_frame, is_terminal = env.step(action_idx)
    h_t = h_current   # advance recurrent state
    if is_terminal: break
```

All operations run under `torch.no_grad()`.

### 5.2 Curiosity Reward at Inference (and during rollout)

Reward at each step combines the state-predictor and action-predictor errors:

```
state_err  = MSE( state_predictor.predict(h_t, a_emb) , h_{t+1} )    # full N-step Euler ODE
action_err = CE( softmax(action_predictor(h_t, h_{t+1})) , one_hot(a_t) )
r_t        = clamp( w_state · state_err + w_action · action_err , max=50.0 )

default: w_state = w_action = 0.5, N = 3 Euler steps
```

Note the asymmetry vs. training: training computes `L_state` on a single random-τ sample of the flow-matching path, whereas the reward uses the deterministic post-ODE final prediction. This makes the intrinsic reward stable shot-to-shot while keeping training cheap.

### 5.3 Action Masking

The environment provides a set of available actions at each step (`env.available_actions`). The policy applies a mask `logits[unavailable] = −∞` before softmax. Applied identically during training (rollout) and inference.

---

## 6. Reward Function

```
r_t = clamp( w_state · MSE(h̃_{t+1}, h_{t+1}) + w_action · CE(p_pred, one_hot(a_t)),
             max=50.0 )

where:
  h̃_{t+1} = state_predictor.predict(h_t, a_emb)                # full N-step Euler ODE, no_grad
  p_pred   = softmax(action_predictor(h_t, h_{t+1}))            # no_grad
  w_state  = w_action = 0.5 (default)
```

**Curiosity mechanism (state).** As the state predictor learns to correctly model frequently visited `(s, a, s')` triples, `MSE` for those triples decays toward zero, encouraging the policy to seek novel states.

**Curiosity mechanism (action).** The action-predictor reward is high whenever the action that produced a given `(h_t, h_{t+1})` transition is hard to infer. Early in training, before the encoder has learned to make adjacent latents action-discriminative, this term will dominate; it should shrink as the encoder co-trains with the action predictor. A persistently elevated action-CE component during late training would indicate either (a) the encoder has not yet escaped collapse, or (b) the environment contains genuinely action-ambiguous transitions.

**Cap at 50:** Prevents a single extremely surprising transition from dominating the REINFORCE gradient. Values above 50 indicate numerical instability rather than genuine novelty.

---

## 7. Health Monitoring

*§7 Health Monitoring — to be specified in a subsequent revision once the dual-predictor training dynamics have been observed empirically.*

---

## 8. Complete Hyperparameter Table

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model dimensions** | | |
| `d_model` | 128 | Encoder / predictor / latent dimension |
| `d_color` | 4 | Color embedding dimension (16 colors) |
| `n_actions` | 4 | Number of discrete actions |
| `d_action` | 32 | Action embedding dimension (state predictor input) |
| `patch_size` | 16 | Pixels per patch side |
| **Encoder SA blocks** | | |
| `n_sa_blocks` | 2 | Number of self-attention blocks |
| `n_sa_heads` | 4 | Attention heads per SA block (32 dims/head) |
| `sa_ffn_dim` | 512 | SA block FFN hidden dimension |
| **Perceiver Resampler** | | |
| `n_latents` | 4 | Number of latent query vectors |
| `n_placeholders` | 4 | Learned placeholder vectors (episode start) |
| `n_perceiver_rounds` | 2 | Number of Perceiver rounds (separate weights) |
| `n_perceiver_heads` | 4 | Attention heads in Perceiver |
| `perceiver_ffn_dim` | 512 | Perceiver FFN hidden dimension |
| **2D RoPE** | | |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `patch_grid_h` | 4 | Patch grid rows |
| `patch_grid_w` | 4 | Patch grid columns |
| **State Predictor (flow matching)** | | |
| `n_ode_steps` | 3 | Euler ODE integration steps |
| `predictor_hidden` | 512 | Per-latent MLP hidden dimension |
| `time_emb_dim` | 128 | Sinusoidal time embedding dimension |
| `time_proj_dim` | 512 | Projected time dimension (after linear+GELU) |
| **Action Predictor** | | |
| `action_predictor_hidden` | 512 | Action predictor MLP hidden dimension |
| `action_predictor_lr` | 1e-4 | Action predictor optimizer LR |
| `action_predictor_wd` | 0.01 | Action predictor weight decay |
| **JEPA loss weighting** | | |
| `lambda_state` | 0.5 | Weight on `L_state` in total JEPA loss |
| `lambda_action` | 0.5 | Weight on `L_action` in total JEPA loss |
| **Reward weighting** | | |
| `reward_w_state` | 0.5 | Weight on state-prediction error in curiosity reward |
| `reward_w_action` | 0.5 | Weight on action-prediction error in curiosity reward |
| `reward_clamp` | 50.0 | Per-step reward cap |
| **Policy** | | |
| `policy_hidden` | 512 | Policy MLP hidden dimension |
| **Replay buffer** | | |
| `buffer_size` | 50,000 | Total LatentBuffer capacity |
| `min_buffer_size` | 512 | Minimum transitions before JEPA training |
| `batch_size` | 64 | JEPA training batch size |
| **Training schedule** | | |
| `update_freq` | 5 | JEPA gradient step every N env steps |
| `policy_update_freq` | 64 | Policy update every N env steps |
| `warmup_steps` | 1,000 | Random exploration before policy trains |
| `max_steps` | 500,000 | Total training budget |
| **Optimisers** | | |
| `sa_lr` | 1e-4 | LR for patch embed + SA blocks |
| `perceiver_lr` | 5e-5 | LR for Perceiver (halved vs SA) |
| `state_predictor_lr` | 1e-4 | LR for state predictor + action embed |
| `policy_lr` | 1e-4 | Policy Adam LR |
| `encoder_wd` | 0.01 | Weight decay for encoder |
| `state_predictor_wd` | 0.01 | Weight decay for state predictor |
| `grad_clip_model` | 5.0 | Gradient clip for encoder + both predictors |
| `grad_clip_policy` | 1.0 | Gradient clip for policy |
| **Policy REINFORCE** | | |
| `policy_entropy_lambda` | 0.10 | Entropy regularization coefficient |
| `policy_baseline_alpha` | 0.99 | EMA decay for running reward baseline |
| **Misc** | | |
| `seed` | 42 | Random seed |
| `game_id` | `ls20-9607627b` | LS20 Level 1 environment ID |

---

## 9. File Layout

```
exp_003_2_action_pred_no_ema/
├── system_card.md              — this document
├── config.py                   — Frozen dataclass Config (inherits exp_003_0.Config)
├── train.py                    — Dual-predictor training loop
├── eval.py                     — N-episode completion rate
├── debug_runner.py             — Per-step dashboard data collection
├── reward_shaping.py           — Life/energy detection
├── panel.js                    — Dashboard visualization plugin
├── models/
│   ├── __init__.py             — load_models() factory exposing
│   │                             encoder, state_predictor, action_predictor, policy
│   ├── encoder.py              — Re-exports/extends exp_003_0 encoder
│   ├── state_predictor.py      — FlowMatchingPredictor (renamed from predictor.py)
│   ├── action_predictor.py     — NEW: ActionPredictor MLP with cross-entropy head
│   └── policy.py               — PolicyNetwork, REINFORCEBaseline
├── checkpoints/                — step_XXXXXX.pt
└── runs/                       — Per-run dirs with training.log + metrics.jsonl
```

Shared code (used by multiple experiments) lives in `JEPA/shared/`:
- `buffer.py` — `LatentBuffer` (extended to support `next_frames` field instead of `h_targets`), `PolicyBuffer`
- `env_wrapper.py` — `LS20Env`
- `action_embed.py` — `ActionEmbedding`
- `ema.py` — EMA utility (**not used in this experiment**)

---

## 10. How to Run

```bash
# ── Training (fresh) ────────────────────────────────────────────────────────
cd "Code Repo"
uv run python -m JEPA.experiments.exp_003_2_action_pred_no_ema.train

# ── Training (resume from checkpoint) ──────────────────────────────────────
uv run python -m JEPA.experiments.exp_003_2_action_pred_no_ema.train \
    --resume JEPA/experiments/exp_003_2_action_pred_no_ema/checkpoints/step_050000.pt

# ── Override hyperparameters ────────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_003_2_action_pred_no_ema.train \
    --max-steps 1000000 --batch-size 128

# ── Evaluation ──────────────────────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_003_2_action_pred_no_ema.eval \
    --checkpoint checkpoints/step_050000.pt --episodes 20

# ── Debug runner (standalone) ───────────────────────────────────────────────
uv run python JEPA/experiments/exp_003_2_action_pred_no_ema/debug_runner.py \
    JEPA/experiments/exp_003_2_action_pred_no_ema/checkpoints/step_050000.pt

# ── Dashboard ───────────────────────────────────────────────────────────────
uv run python JEPA/dashboard/server.py
# Open http://localhost:8787
# 1. Select "exp_003_2_action_pred_no_ema" in the experiment dropdown
# 2. Select a checkpoint
# 3. Click "▶ Run Episode"
```

---

## 11. Comparison with exp_003_0 and exp_003_1

| Dimension | exp_003_0 | exp_003_1 | exp_003_2 (this) |
|-----------|-----------|-----------|------------------|
| **Collapse prevention** | Stop-grad on `h_{t+1}` target | Stop-grad + EMA target encoder | Stop-grad on state target + **Action Predictor** (no EMA) |
| **Predictors** | State only | State only | State + Action |
| **JEPA loss** | `L_state` | `L_state` | `0.5·L_state + 0.5·L_action` |
| **Buffer target field** | `h_target` (frozen at rollout) | `h_target` (frozen at rollout) | `next_frame` (re-encoded fresh) |
| **Buffer sampling** | 20% recent / 80% uniform | 20% recent / 80% uniform | Uniform only |
| **Encoder gradient paths into JEPA loss** | 1 | 1 | 3 |
| **EMA target encoder** | — | Cosine 0.996 → 0.9999 | — |
| **Max-episode-length guard** | `MAX_EP_STEPS = 300` | `MAX_EP_STEPS = 300` | — (energy-bounded) |
