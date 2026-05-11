# System Card — exp_003_0_normalized_latent_jepa

| Field | Value |
|-------|-------|
| **Experiment ID** | `exp_003_0_normalized_latent_jepa` |
| **Status** | Active |
| **Parent experiment** | `exp_002_perceiver_flow_jepa` |
| **Game** | LS20 Level 1 (`ls20-9607627b`) |
| **Reward** | Intrinsic curiosity (flow-matching prediction error) |

## 1. One-Paragraph Summary

This experiment trains a world model and policy for ARC-AGI LS20 Level 1 using a Perceiver-based JEPA with flow-matching prediction. The encoder reduces a 64×64 frame to 16 SA-enriched patch tokens and then compresses them to **4 latent vectors** via a Perceiver Resampler. Those 4 latent vectors are fed recurrently: the output latents `h_t` of the previous step serve as the Perceiver's query inputs at the next step. A **flow-matching predictor** (4 independent MLPs, sinusoidal time conditioning, Euler ODE integration) learns to predict next-step latents `h_{t+1}` given `h_t` and the action. The policy is a stateless MLP that reads the flattened 4-latent vector and selects actions with REINFORCE. Three critical fixes over exp_002 prevent training pathologies: (1) **separate per-round Perceiver weights** halve gradient accumulation, (2) an **output LayerNorm** on the Perceiver prevents recurrent norm explosion, and (3) **stop-gradient on the target latent** closes the collapse attractor. A **LatentBuffer** stores latent-space transitions from the recurrent rollout so the encoder can receive gradient through re-encoded frames.

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
       ▼  + action_embed (B, 32)
  ┌─────────────────────────────────┐
  │  Flow Matching Predictor        │
  │  4 independent MLPs             │
  │  Sinusoidal time embedding      │
  │  Euler ODE (3 steps)            │
  │  → h̃_{t+1} (B, 4, 128)        │
  └─────────────────────────────────┘
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

**Separate per-round weights:** In exp_002, both rounds shared the same `_PerceiverRound` parameters (weight-tied). This caused the gradient from two encoding calls (for `h_t` and `h_{t+1}`) to flow through the same parameters twice per update, effectively giving the Perceiver 4× the gradient magnitude vs. a single-round module. The gradient norm grew from ~5 to ~17 over 500K steps. In exp_003, each round has independent parameters; gradient accumulation is halved.

**Output LayerNorm:** The recurrent state `h_t` is passed as queries to the next step's Perceiver. Without normalization, each residual connection adds ~0.15 to the L2 norm; after 8 residual additions per call × 42 steps ≈ 336 total, norms reach ~54. The output LayerNorm clamps the recurrent state to near-unit scale at every step, preventing this linear growth.

**Placeholder initialization:** At episode start (t=0), the Perceiver queries are initialized from a learned `nn.Parameter` of shape `(4, 128)` (the "placeholders"), broadcast to batch size. After the first step, `h_{t-1}` provides the recurrent state.

### 2.4 Flow Matching Predictor

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

L = Σ_i ‖ x̂₁[i] − h_{t+1}[i] ‖²   (mean over batch and dims)
```

**Stop gradient on h_{t+1}:** At the call site in `train.py`, the target is always `.detach()`-ed: `predictor.compute_loss(h_t_fresh, h_target.detach(), a_emb)`. Without this, the encoder and predictor can jointly minimize the loss by collapsing all states to a single constant vector — the predictor trivially matches a constant target by outputting that constant for any input. Detaching the target eliminates this gradient path.

**Sinusoidal time embedding:** τ is encoded as a sinusoidal vector of dimension 128, then projected to 512 via a linear layer + GELU. This gives each MLP access to where it is in the integration trajectory.

**Per-latent MLPs:** `_LatentMLP = Linear(128 + 32 + 512, 512) → GELU → Linear(512, 128)`. Using 4 separate MLPs (one per latent) rather than a single shared MLP allows each latent to specialize — latent 0 may capture spatial structure, latent 1 may capture agent state, etc. The MLPs do not communicate across latents.

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

This is the Euler discretization of the probability flow ODE for the linear interpolant `x_τ = (1 − τ)x_0 + τ x_1`, where the velocity field is `v(x, τ) = x̂₁(x, τ) − x_0`. With N=3 steps the ODE is cheap to evaluate while still capturing the curvature of the predicted trajectory.

### 2.5 Action Embedding

`nn.Embedding(4, 32)` — maps discrete action index (0–3) to a 32-dim learned vector. Concatenated into each per-latent MLP's input.

**Why learned over one-hot:** Allows the predictor to discover latent similarity between actions that produce similar dynamics (e.g., moving left and moving right both move the agent, though in opposite directions).

### 2.6 Policy Network

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

**Why stateless (no recurrent state in policy):** The latent state `h_t` already carries temporal context via the recurrent Perceiver. The policy does not need its own hidden state — it reads directly from `h_t`, which already encodes the history of observations through the recurrent Perceiver queries.

**Available-action masking:** LS20 Level 1 restricts which actions are legal at each step. Before softmax, logits for unavailable actions are set to −∞. This ensures the sampled action is always legal and the entropy signal reflects only the choice among available actions.

---

## 3. Key Changes from exp_002

| Change | exp_002 | exp_003 | Motivation |
|--------|---------|---------|------------|
| **Perceiver round weights** | Weight-tied (shared across rounds) | Separate per-round parameters | exp_002: gradient flows through the shared parameters twice per encode call (once for h_t, once for h_{t+1}), 4× accumulation total. Perceiver grad norm: ~5→17 over 500K steps. exp_003: halves accumulation. |
| **Perceiver output norm** | None | `nn.LayerNorm(128)` on output h | exp_002: recurrent h norms grew to ~54 due to residual additions. This norm clamps the recurrent state to near-unit scale. |
| **Replay buffer** | Stores raw frames; encodes fresh each batch | **LatentBuffer**: stores `(frame_t, h_query, action, h_target)` from recurrent rollout | Allows correct recurrent query to be replayed at training time, giving encoder gradient while matching the rollout's encoding path exactly. |
| **Target gradient** | Gradient flowed through h_{t+1} target (collapse attractor active) | `h_target.detach()` — stop-gradient on target | Prevents encoder from minimizing loss by mapping all states to a constant. |
| **Per-component optimizer LR** | Single optimizer for all encoder params | SA + embed: 1e-4; Perceiver: 5e-5 | Perceiver still has 2× gradient accumulation from the two encode calls per update step; lower LR further stabilizes its updates. |

---

## 4. Training

### 4.1 Data Collection (Recurrent Rollout)

At each environment step:
1. **Encode frame:** `h_current, sa_out, attn = encoder(frame_t, queries)` where `queries = h_{t-1}` (or placeholders at episode start). Run under `torch.no_grad()`.
2. **Select action:** Policy MLP on `h_current` during rollout (or uniform random during warmup).
3. **Step environment:** Get `next_frame`, `is_terminal`.
4. **Compute curiosity reward:** Encode `next_frame` with `h_current` as queries → `h_next`. Run predictor on `(h_current, action)` → `h̃`. Reward = mean MSE between `h̃` and `h_next`, capped at 50.
5. **Buffer the transition:** `LatentBuffer.add(frame_t, h_query_np, action, h_next_np)` — stores the frame, the query that was used to produce `h_current`, the action, and the target latent `h_next` computed from the next frame.
6. **Advance recurrent state:** `h_{t} ← h_current`.

**Why store h_query (not h_current):** At training time, the encoder is re-called as `encoder(batch.frames, batch.h_queries.detach())`. This exactly replicates the rollout's forward pass for `frame_t`, so the gradients computed at training correspond to the same input/query combination that was used during data collection. Storing `h_current` instead would produce a mismatch (h_current was the OUTPUT of encoding frame_t, not the query input).

### 4.2 Training Updates

**JEPA / Flow Matching update** (every `update_freq=5` environment steps):

```python
# Sample batch from LatentBuffer
batch = latent_buf.sample(batch_size=64, device)
# batch.frames:    (64, 64, 64)   uint8
# batch.h_queries: (64, 4, 128)   recurrent queries from rollout
# batch.actions:   (64,)          int64
# batch.h_targets: (64, 4, 128)   h_{t+1} computed during rollout

# Re-encode with stored queries → differentiable h_t
h_t_fresh = encoder(batch.frames, batch.h_queries.detach())  # gradients flow to encoder

# Action embedding
a_emb = action_embed(batch.actions)   # (64, 32)

# Flow matching loss — stop-gradient on target
flow_loss, per_latent_loss = predictor.compute_loss(
    h_t_fresh,
    batch.h_targets.detach(),   # ← stop gradient
    a_emb,
)

flow_loss.backward()
clip_grad_norm_(encoder + predictor + action_embed, max_norm=5.0)
enc_opt.step()     # SA params: lr=1e-4; Perceiver params: lr=5e-5
pred_opt.step()    # predictor + action_embed: lr=1e-4
```

**Policy update** (every `policy_update_freq=64` environment steps):

```python
log_probs, rewards, entropies = policy_buf.get(device)
adv = rewards − baseline.value
adv = adv / (adv.std().clamp(min=1e-8) + 1e-8)
baseline.update(rewards.mean())

pol_loss = −mean(adv * log_probs) − 0.10 * mean(entropies)
pol_loss.backward()
clip_grad_norm_(policy, max_norm=1.0)
pol_opt.step()   # Adam, lr=1e-4
```

### 4.3 Per-Component Optimizer Design

Three separate optimizers partition the parameters:

| Optimizer | Parameters | LR | Weight Decay |
|-----------|-----------|-----|------|
| `enc_opt` group 1 | color_embed, patch_proj, sa_blocks, sa_norm | 1e-4 | 0.01 |
| `enc_opt` group 2 | perceiver (all rounds + output_norm + placeholders) | 5e-5 | 0.01 |
| `pred_opt` | predictor (MLPs + time_embed) + action_embed | 1e-4 | 0.01 |
| `pol_opt` | policy | 1e-4 | 0 (Adam) |

The Perceiver runs at half the SA learning rate because it participates in TWO encode calls per training step (once for h_t and once because h_target is re-computed from the next frame during rollout collection), effectively receiving 2× the gradient signal relative to the SA blocks.

### 4.4 LatentBuffer

The `LatentBuffer` (defined in `JEPA/shared/buffer.py`) stores latent-space transitions:

```
fields:  frames      (uint8, capacity × 64 × 64)
         h_queries   (float32, capacity × 4 × 128)
         actions     (int64, capacity)
         h_targets   (float32, capacity × 4 × 128)
```

Sampling is stratified: `recency_fraction=0.20` of each batch is drawn from the most recent `recent_window=10,000` transitions; the remaining 80% from the full buffer. This prevents the encoder from drifting too far from the current policy's visitation distribution while still providing diverse off-policy experience.

The dying transition (s_dying → s_next_life_start) is excluded from the buffer. The episode buffer `ep_transitions` accumulates all steps; on life-end, `ep_transitions[:-1]` (all except the last step that triggered the life-end) are pushed to the LatentBuffer.

### 4.5 Training Schedule

| Phase | Steps | Actions | JEPA update | Policy update |
|-------|-------|---------|------------|---------------|
| Warmup | 0 – 999 | Uniform random | Every 5 env steps (once buffer ≥ 512) | No |
| Joint training | ≥ 1,000 | Policy | Every 5 env steps | Every 64 env steps |

**Checkpointing:** Every 5,000 steps → `checkpoints/step_XXXXXX.pt`. On training end → `step_XXXXXX_final.pt`. Each run creates a timestamped directory under `runs/` with `training.log` and `metrics.jsonl`.

**Max episode length guard:** Episodes longer than `MAX_EP_STEPS=300` steps are force-flushed (transitions pushed to buffer) without resetting the environment, preventing infinite loops due to deterministic wall-hitting behavior.

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

No gradient computation is needed at inference; all operations run under `torch.no_grad()`.

### 5.2 ODE Integration at Inference

The predictor's `.predict()` method runs 3 Euler steps to obtain `h̃_{t+1}` from `h_t` and the action. This is used for curiosity reward computation during rollout. The `.predict_with_trajectory()` variant returns all intermediate states `[x_0, x_{1/3}, x_{2/3}, x_1]` for dashboard visualization.

### 5.3 Action Masking

The environment provides a set of available actions at each step (`env.available_actions`). The policy applies a mask `logits[unavailable] = −∞` before softmax. This is applied identically during training (rollout) and inference.

---

## 6. Reward Function

```
r_t = clamp(MSE(h̃_{t+1}, h_{t+1}), max=50.0)

where:
  h_{t+1}  = encoder(next_frame, h_current)    — computed during rollout (no grad)
  h̃_{t+1} = predictor.predict(h_current, a_emb) — Euler ODE (no grad)
  MSE       = mean over 4 latents and 128 dims of (h̃ − h)²
```

**Curiosity mechanism:** The reward equals the predictor's surprise at the observed transition. As the predictor learns to correctly model frequently visited (s, a, s') triples, the reward for those triples decays toward zero, encouraging the policy to seek novel states. This creates an implicit exploration-exploitation trade-off without any explicit novelty detector.

**Cap at 50:** Prevents a single extremely surprising transition from dominating the REINFORCE gradient. Values above 50 indicate numerical instability (very large latent norms or NaN-propagation) rather than genuine novelty.

---

## 7. Health Monitoring

The training loop (`train.py`) tracks the following signals every `LOG_FREQ=200` steps:

| Signal | Threshold | Action | Rationale |
|--------|-----------|--------|-----------|
| NaN in flow loss | Any NaN | CRITICAL stop | NaN propagates to all subsequent steps |
| Encoder SA grad norm (mean) | > 200 | CRITICAL stop | Gradient explosion |
| Per-latent std < 0.01 for 5 consecutive checks | Any latent | CRITICAL stop | Dead latent (representation collapse on that dimension) |
| Latent L2 norm > 34 (≈3×√128) | Any latent | CRITICAL stop | Output LayerNorm bypassed or exploding |
| Loss coefficient of variation > 2.0 | Loss window | CRITICAL stop | Oscillating loss indicates divergence |
| Time-embedding grad < 1e-4 | mean | WARNING | Time embedding not receiving gradient signal |
| ODE step cosine similarity > 0.99 | mean | WARNING | Degenerate predictor — each ODE step barely moves |
| Policy entropy < 0.30 nats | mean | WARNING | Policy collapsing to near-deterministic |

**Expected healthy values** (with output_norm active):
- Latent L2 norms: ~11.3 (= √128, expected for N(0,1) vectors after LayerNorm)
- Latent pairwise L2: > 2.0 (diverse latents)
- Effective rank: > 2.0
- Flow loss: 0.05–0.30 (varies with training progress)
- ODE cosine similarity: 0.85–0.98 (steps make moderate changes)

---

## 8. Complete Hyperparameter Table

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model dimensions** | | |
| `d_model` | 128 | Encoder / predictor / latent dimension |
| `d_color` | 4 | Color embedding dimension (16 colors) |
| `n_actions` | 4 | Number of discrete actions |
| `d_action` | 32 | Action embedding dimension |
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
| **Flow Matching Predictor** | | |
| `n_ode_steps` | 3 | Euler ODE integration steps |
| `predictor_hidden` | 512 | Per-latent MLP hidden dimension |
| `time_emb_dim` | 128 | Sinusoidal time embedding dimension |
| `time_proj_dim` | 512 | Projected time dimension (after linear+GELU) |
| **Policy** | | |
| `policy_hidden` | 512 | Policy MLP hidden dimension |
| **Replay buffer** | | |
| `buffer_size` | 50,000 | Total LatentBuffer capacity |
| `min_buffer_size` | 512 | Minimum transitions before JEPA training |
| `batch_size` | 64 | JEPA training batch size |
| `recency_fraction` | 0.20 | Fraction of batch from recent window |
| `recent_buffer_size` | 10,000 | Size of recency window |
| **Training schedule** | | |
| `update_freq` | 5 | JEPA gradient step every N env steps |
| `policy_update_freq` | 64 | Policy update every N env steps |
| `warmup_steps` | 1,000 | Random exploration before policy trains |
| `max_steps` | 500,000 | Total training budget |
| **Optimisers** | | |
| `sa_lr` | 1e-4 | LR for patch embed + SA blocks |
| `perceiver_lr` | 5e-5 | LR for Perceiver (halved vs SA) |
| `predictor_lr` | 1e-4 | LR for predictor + action embed |
| `policy_lr` | 1e-4 | Policy Adam LR |
| `encoder_wd` | 0.01 | Weight decay for encoder |
| `predictor_wd` | 0.01 | Weight decay for predictor |
| `grad_clip_model` | 5.0 | Gradient clip for encoder + predictor |
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
exp_003_0_normalized_latent_jepa/
├── config.py                  — Frozen dataclass Config with all hyperparameters
├── train.py                   — Main training loop (LatentBuffer, re-encode, stop-grad)
├── eval.py                    — Runs N episodes and reports completion rate
├── debug_runner.py            — Collects per-step data for the dashboard
├── reward_shaping.py          — Life/energy detection (count_lives, is_end_of_life)
├── panel.js                   — Dashboard visualization plugin (SA, Perceiver, Predictor, Policy)
├── models/
│   ├── __init__.py            — load_models() factory, CAPABILITIES dict
│   ├── encoder.py             — SelfAttentionBlock, PerceiverResampler, Encoder
│   ├── predictor.py           — FlowMatchingPredictor, SinusoidalTimeEmbedding, _LatentMLP
│   └── policy.py              — PolicyNetwork, REINFORCEBaseline
├── checkpoints/               — Saved model checkpoints (step_XXXXXX.pt)
└── runs/                      — Per-run directories with training.log + metrics.jsonl
```

Shared code (used by multiple experiments) lives in `JEPA/shared/`:
- `buffer.py` — `LatentBuffer`, `PolicyBuffer`
- `env_wrapper.py` — `LS20Env` (wraps ARC-AGI Arcade)
- `action_embed.py` — `ActionEmbedding`
- `ema.py` — EMA utility (not used in exp_003)

---

## 10. How to Run

```bash
# ── Training (fresh) ────────────────────────────────────────────────────────
cd "Code Repo"
uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.train

# ── Training (resume from checkpoint) ──────────────────────────────────────
uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.train \
    --resume JEPA/experiments/exp_003_0_normalized_latent_jepa/checkpoints/step_050000.pt

# ── Override hyperparameters ────────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.train \
    --max-steps 1000000 --batch-size 128

# ── Evaluation ──────────────────────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.eval \
    --checkpoint checkpoints/step_050000.pt --episodes 20

# ── Debug runner (standalone) ───────────────────────────────────────────────
uv run python JEPA/experiments/exp_003_0_normalized_latent_jepa/debug_runner.py \
    JEPA/experiments/exp_003_0_normalized_latent_jepa/checkpoints/step_050000.pt

# ── Dashboard ───────────────────────────────────────────────────────────────
uv run python JEPA/dashboard/server.py
# Open http://localhost:8787
# 1. Select "exp_003_0_normalized_latent_jepa" in the experiment dropdown
# 2. Select a checkpoint
# 3. Click "▶ Run Episode"
# 4. Use the Training Monitor at http://localhost:8787/training
#    to start/stop training and view live metrics
```

---

## 11. Comparison with exp_001 (ViT JEPA Baseline)

| Dimension | exp_001 | exp_003 |
|-----------|---------|---------|
| **Encoder output** | 16 patch embeddings (B, 16, 128) | 4 latent vectors (B, 4, 128) |
| **Temporal context** | Reasoning token `h_t ∈ ℝ^128` updated by policy | Recurrent Perceiver: `h_t` fed back as queries |
| **World model** | Direct embedding regression (MSE vs EMA target) | Flow matching (ODE integration over latent trajectory) |
| **Target encoder** | EMA copy of online encoder | None — stop-gradient on h_{t+1} from rollout |
| **Prediction target** | 16 patch embeddings (pixel-weighted) | 4 latent vectors (uniform) |
| **Policy architecture** | Cross-attention reasoning token (recurrent) | Stateless MLP |
| **Collapse prevention** | Variance regularization + EMA target | Output LayerNorm + stop-gradient on target |
| **Buffer** | Raw frame replay buffer | LatentBuffer (frame + h_query + action + h_target) |
| **Compression ratio** | 16 → 16 (no compression) | 16 patches → 4 latents (4× compression) |
