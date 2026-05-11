# System Card — exp_001_vit_jepa_baseline

| Field | Value |
|-------|-------|
| **Experiment ID** | `exp_001_vit_jepa_baseline` |
| **Status** | Active (training run ongoing) |
| **Parent experiment** | None (baseline) |
| **Game** | LS20 Level 1 (`ls20-9607627b`) |
| **Best checkpoint** | `step_2205000.pt` (latest as of restructuring) |
| **Observed completion rate** | Under investigation — use `inspect_policy.py` |

## 1. One-paragraph Summary

This experiment trains a Joint Embedding Predictive Architecture (JEPA) world model together with a recurrent policy to play ARC-AGI LS20 Level 1. The encoder learns to represent 64×64 color frames as 16 patch embeddings (4×4 grid). A predictor learns to predict the next frame's embeddings given the current embeddings and the action taken — this prediction task is the signal that trains the encoder. A target encoder (updated only via EMA) provides stable regression targets. A cross-attention policy reads the current patch embeddings, updates a persistent reasoning token, and samples an action. The policy is trained with REINFORCE using intrinsic curiosity reward (prediction error) so it is incentivized to explore states the world model has not yet learned.

---

## 2. Architecture

### 2.1 Encoder (Vision Transformer)

| Attribute | Value |
|-----------|-------|
| Input | `(B, 64, 64)` uint8 frame (color indices 0–15) |
| Output | `(B, 16, 128)` L2-normalized patch embeddings |
| Patch grid | 4×4 = 16 patches, each 16×16 pixels |

**Pipeline:** Color embedding lookup (16 colors → 4-dim) → flatten patch (256 pixels × 4 dims = 1024) → linear project to 128 → add learned 2D positional embeddings → 2× pre-norm transformer blocks (4 heads, FFN hidden=512) → LayerNorm → L2 normalize.

**Why ViT over CNN:** ARC-AGI frames have discrete grid structure; patches align with the grid so each patch corresponds to a semantically coherent region. Self-attention lets patches attend to each other, capturing spatial relationships the predictor can exploit.

**Why L2 normalize output:** Prevents embedding scale from growing unboundedly under EMA. With normalized vectors, the EMA update operates in angular space — the cosine similarity metric is natural and consistent.

**Why pre-norm:** Training stability. Pre-norm (LayerNorm before attention/FFN) avoids gradient vanishing in deeper blocks without warmup tricks.

### 2.2 Target Encoder

An exact deep copy of the Encoder, updated *only* via exponential moving average (EMA). Its parameters never receive any gradient.

**Why EMA target:** If the target encoder trained jointly, it would "chase" the online encoder's predictions, destabilizing the loss. The slowly-drifting EMA target provides a stable regression target — the same principle as in DQN's frozen target network and I-JEPA.

### 2.3 Predictor

| Attribute | Value |
|-----------|-------|
| Input | `(B, 16, 128)` current patch embeddings + `(B, 32)` action embedding |
| Output | `(B, 16, 128)` predicted next-state patch embeddings |
| Architecture | Per-patch MLP: concat → Linear(160, 256) → GELU → Linear(256, 128) |

**Why the predictor exists:** Without it, the encoder has no training objective. With it, the encoder's job is to produce representations that make next-frame prediction easy. If the encoder collapses to a constant output, the predictor can't distinguish states and the JEPA loss stays high. The predictor creates a proxy objective: "learn embeddings that are useful for predicting temporal dynamics", which is a proxy for learning high-level semantics.

**Why per-patch (no cross-patch attention):** Keeps the predictor lightweight and avoids masking complexity. Patches are approximately locally independent given the action signal.

### 2.4 ActionEmbedding

`nn.Embedding(4, 32)` — maps discrete action index (0–3) to a 32-dim learned vector. Shared across all patches in the predictor.

**Why learned over one-hot:** Gives the predictor richer context — the learned vectors can encode which actions tend to cause similar dynamics, something a one-hot cannot express.

### 2.5 Policy Network (Cross-Attention Reasoning Token)

| Attribute | Value |
|-----------|-------|
| State | Reasoning token `h ∈ ℝ^128`, persists across timesteps |
| Input per step | `h_{t-1} ∈ ℝ^128`, `z_t ∈ ℝ^{16×128}` (patch embeddings) |
| Output per step | `action_idx ∈ {0,1,2,3}`, `log_prob`, `h_t`, `entropy` |

**Recurrent update (per step):**
```
Q  = Linear_Q(h_{t-1})                    # (128,)
K  = Linear_K(z_t), V = Linear_V(z_t)    # (16, 128) each
w  = softmax(Q @ K^T / sqrt(128))         # (16,) attention weights
o  = Linear_out(w @ V)                    # (128,)
h' = LayerNorm(h_{t-1} + o)              # residual + norm
h_t = LayerNorm(h' + FFN(h'))             # post-attn FFN + norm
logits = Linear_head(h_t)                 # (4,)
a_t ~ Categorical(softmax(logits))
```
FFN: `Linear(128, 256) → GELU → Linear(256, 128)`.

**Why cross-attention:** The attention weights `w` are interpretable — the dashboard shows which patches the policy focuses on at each step. Unlike concatenation, cross-attention is parameter-efficient and generalizes naturally to different numbers of patches.

**Why detach `h` between steps:** Truncated BPTT. The reasoning token is detached before each `policy.act()` call — gradients flow only through the current-step forward pass (Q/K/V projections, FFN, action head), not through the recurrent chain across 64+ steps. This prevents gradient explosion without needing LSTM gating or gradient clipping workarounds.

---

## 3. Loss Functions

### 3.1 JEPA Prediction Loss

Applied every `update_freq=5` environment steps on a batch of 64 transitions sampled from the replay buffer.

```
z_t    = online_encoder(s_t)                          # (B, 16, 128)  — gradients flow
ẑ_t+1  = predictor(z_t, action_embed(a_t))            # (B, 16, 128)  — gradients flow
z*_t+1 = stop_gradient(target_encoder(s_{t+1}))       # (B, 16, 128)  — no gradients

# Per-patch pixel-change weights
pw_i  = mean_over_pixels(|s_{t+1}[i] - s_t[i]|)      # (B, 16) in [0, 1]
w_i   = 0.1 + 0.9 * pw_i / max(pw_i.sum(-1, keepdim=True) + ε)   # floor at 0.1

# Mean weighted squared L2 distance
L_mse = mean_B(Σ_i  w_i · ‖ẑ_t+1[i] − z*_t+1[i]‖²₂)

# One-sided variance regularization (collapse safety net only)
L_var = ReLU(0.02 − std_d(z_t))    # fires only when per-dim std drops below 0.02

L_JEPA = L_mse + λ_var · L_var     where λ_var = 0.01
```

**Patch weights motivation:** Changed patches carry more information about the transition dynamics — weighting them more makes the predictor focus on learning what changes, not memorizing static background patches.

**Variance regularization motivation:** Without it, the encoder can collapse to a constant vector (all patches → same direction), making L_mse trivially zero. The floor at std=0.02 activates only when collapse begins, acting as a soft emergency brake rather than a constant regularizer.

### 3.2 Policy Loss (REINFORCE + Entropy Regularization)

Applied every `policy_update_freq=64` environment steps on the 64-step on-policy trajectory.

```
# Advantage normalization
adv_t = (r_t − mean(r)) / clamp(std(r), min=0.1)

# REINFORCE with entropy bonus
L_π = −mean_t(log π(a_t | h_t) · adv_t) − λ_H · mean_t(H(π(· | h_t)))
where λ_H = 0.10,  H(π) = −Σ_a p_a log p_a
```

**Advantage normalization:** Prevents the gradient from vanishing when reward variance is near zero (e.g., during warmup when all rewards are approximately equal).

**Entropy regularization:** Prevents premature convergence to a near-deterministic policy. λ_H=0.10 (raised from 0.02 after observing near-uniform entropy stagnation) keeps the policy exploratory throughout training.

---

## 4. Reward Function

```
r_t = mean_i(w_i · ‖ẑ_t+1[i] − z*_t+1[i]‖²₂)  +  50.0 · 𝟙[terminal ∧ level_completed]
```

**Intrinsic curiosity:** The policy reward equals the JEPA prediction error on the transition just taken. Intuitively: the policy is rewarded for visiting states the world model predicts poorly — i.e., novel states. As JEPA trains and reduces error on visited (s, a, s') triples, the reward there decays, pushing the policy to explore new states. This creates a self-supervised exploration-exploitation loop.

**Completion bonus:** A flat +50 when the agent completes Level 1. This signal is sparse but large — it gives the policy a direct incentive beyond curiosity once it discovers a completion path.

**Alternative reward** (`reward_shaping.py`): A shaped reward using player-Y progress tracking is defined but **not used in the current training loop**. It would provide denser signal (+progress per upward step, +1 if moved, -1 if wall hit) but was not activated in exp_001 to isolate the JEPA intrinsic reward effect.

---

## 5. EMA Momentum Schedule

```
m(t) = m_start + (m_end − m_start) · (1 − cos(π · t/T)) / 2

where:
  m_start = 0.996   (initial — higher momentum at start for stable early training)
  m_end   = 0.9999  (final — nearly frozen target in late training)
  T       = max_steps (total training budget)

θ_target ← m(t) · θ_target + (1−m(t)) · θ_online   (applied every JEPA gradient step)
```

**Intuition:** High momentum early → target encoder nearly frozen → stable loss landscape. Low momentum late would let the target drift too fast, undermining the stable-target property. This cosine schedule follows the I-JEPA paper.

---

## 6. Complete Hyperparameter Table

| Parameter | Value | Description | Notes |
|-----------|-------|-------------|-------|
| `d_model` | 128 | Transformer / reasoning token dim | Balanced capacity vs. speed |
| `d_color` | 4 | Color embedding dim (16 colors → 4d) | Sufficient for color identity |
| `n_heads` | 4 | Attention heads in encoder (32d/head) | |
| `n_blocks` | 2 | Transformer blocks in encoder | Shallow encoder; more blocks = diminishing returns |
| `ffn_dim` | 512 | FFN hidden dim in encoder (4× d_model) | Standard transformer ratio |
| `d_action` | 32 | Action embedding dim | |
| `patch_size` | 16 | Pixels per patch side → 4×4=16 patches | Aligned with ARC grid structure |
| `n_actions` | 4 | LS20 discrete actions (ACTION1–4) | |
| `ema_start` | 0.996 | Initial EMA momentum | I-JEPA default |
| `ema_end` | 0.9999 | Final EMA momentum | |
| `change_weight_max` | 3.0 | Legacy field (unused in current loss) | Was used in older per-patch weighting scheme |
| `variance_reg_lambda` | 0.01 | Collapse prevention weight | One-sided floor at std=0.02 |
| `buffer_size` | 50,000 | Replay buffer total capacity | ~200 MB (uint8 frames) |
| `min_buffer_size` | 512 | Minimum transitions before JEPA training starts | |
| `batch_size` | 64 | JEPA update batch size | |
| `recency_fraction` | 0.2 | Fraction of batch from recent transitions | Keeps encoder aligned with current policy |
| `recent_buffer_size` | 10,000 | Size of "recent" window for oversampling | |
| `update_freq` | 5 | JEPA gradient step every N env steps | |
| `policy_update_freq` | 64 | Policy update every N env steps (on-policy window) | |
| `warmup_steps` | 1,000 | Random actions + JEPA-only warmup | Policy not updated during warmup |
| `max_steps` | 50,000 | Training budget (nominal) | Can be extended with --resume |
| `jepa_lr` | 1e-4 | AdamW learning rate for encoder+predictor | Reduced from 3e-4 to prevent embedding scale drift |
| `jepa_weight_decay` | 0.01 | L2 regularization for JEPA optimizer | Prevents parameter growth under long training |
| `policy_lr` | 1e-4 | Adam learning rate for policy | |
| `policy_entropy_lambda` | 0.10 | Entropy regularization coefficient | Raised from 0.02 after observing stagnation |
| `grad_clip_jepa` | 5.0 | JEPA gradient clipping threshold | |
| `grad_clip_policy` | 1.0 | Policy gradient clipping threshold | |
| `game_id` | `ls20-9607627b` | LS20 Level 1 environment ID | |

---

## 7. Training Schedule

### Phase 0 — Warmup (steps 0–999)
- Actions: random (uniform over available actions)
- JEPA: trains every 5 env steps (builds world model from random transitions)
- Policy: NOT updated (no on-policy trajectory stored)
- Buffer: fills up to min_buffer_size before JEPA begins

### Phase 1 — Joint Training (steps ≥ 1000)
- JEPA updates every 5 env steps (off-policy, from replay buffer)
- Policy updates every 64 env steps (on-policy, from PolicyBuffer)
- EMA momentum cosine-anneals from 0.996 → 0.9999

### Checkpointing
- Every 5,000 steps: save `checkpoints/step_XXXXXX.pt`
- On training end: save `checkpoints/step_XXXXXX_final.pt`

### Stopping Conditions
| Type | Condition |
|------|-----------|
| **SUCCESS** | Completion rate ≥ 30% over rolling 20-episode window |
| **CRITICAL — NaN** | NaN detected in loss or model parameters |
| **CRITICAL — Collapse** | Pairwise L2 between patch embeddings < 0.05 for 10 consecutive checks |
| **CRITICAL — Explosion** | JEPA grad norm > 200 |

---

## 8. Health Monitoring Thresholds

| Constant | Value | Meaning |
|----------|-------|---------|
| `COLLAPSE_PAIRWISE_CRITICAL` | 0.05 | Pairwise L2 below this for 10 checks → CRITICAL stop |
| `COLLAPSE_PAIRWISE_WARN` | 0.15 | Pairwise L2 below this → WARNING log |
| `EFFECTIVE_RANK_WARN` | 2.0 | Effective rank of embedding matrix below this → WARNING |
| `ACROSS_STD_WARN` | 0.05 | Across-state per-dim std below this → WARNING |
| `DEAD_ACT_WARN` | 0.60 | GELU dead-activation rate above 60% → WARNING |
| `ENTROPY_WARN` | 0.30 | Policy entropy below 0.3 nats → WARNING |
| `GRAD_NORM_CRITICAL` | 200.0 | JEPA gradient norm above this → CRITICAL stop |

---

## 9. Evaluation Metrics

| Metric | Target (healthy run) | How to measure |
|--------|---------------------|----------------|
| 20-episode completion rate | ≥ 30% = SUCCESS | Tracked during training; `eval.py` |
| JEPA MSE loss | ~0.10–0.15 (well-trained) | Dashboard Episode Overview tab |
| Pairwise cosine similarity | < 0.5 (diverse embeddings) | Dashboard right panel |
| Effective rank | ≥ 4 (good representation) | Dashboard right panel |
| Policy entropy | ≥ 0.6 (exploratory) | Dashboard Episode Overview tab |
| Dead GELU activations | < 30% | Logged during training |

---

## 10. How to Run

```bash
# ── Training ────────────────────────────────────────────────────────
# From scratch
cd "Code Repo"
uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.train

# With custom hyperparameters
uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.train \
    --batch-size 128 --jepa-lr 3e-4

# Resume from checkpoint
uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.train \
    --resume JEPA/experiments/exp_001_vit_jepa_baseline/checkpoints/step_2205000.pt

# ── Evaluation ──────────────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.eval --episodes 10

# ── Policy diagnostics ──────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.inspect_policy --episodes 5

# ── Dashboard ───────────────────────────────────────────────────────
uv run python JEPA/dashboard/server.py
# Open http://localhost:8787
# 1. Select "exp_001_vit_jepa_baseline" in the experiment dropdown
# 2. Select a checkpoint in the checkpoint dropdown
# 3. Click "▶ Run Episode"

# ── Official scoring (requires ARC_API_KEY) ─────────────────────────
uv run python -m JEPA.experiments.exp_001_vit_jepa_baseline.run_online --episodes 3
```

---

## 11. Creating exp_002 from This Experiment

To create a new experiment (e.g., replacing the ViT encoder with a CNN):

```bash
cp -r JEPA/experiments/exp_001_vit_jepa_baseline JEPA/experiments/exp_002_cnn_jepa
```

Then edit:
1. **`models/encoder.py`** — Replace `Encoder` class with CNN implementation
2. **`models/__init__.py`** — Update `load_models()` to use `CNNEncoder`; set `CAPABILITIES["has_encoder_attention"] = False`
3. **`config.py`** — Adjust hyperparameters (e.g., remove `n_heads`, `n_blocks`; add CNN-specific params)
4. **`system_card.md`** — Fill in "Parent experiment: exp_001", add "Changes from parent" section

All other files (`train.py`, `reward_shaping.py`, `eval.py`, `run_online.py`, `inspect_policy.py`) can be reused unchanged unless the loss function or evaluation logic also changes.
