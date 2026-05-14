# System Card — exp_004_0_ls20_tu93

| Field | Value |
|-------|-------|
| **Experiment ID** | `exp_004_0_ls20_tu93` |
| **Status** | TU93 probe complete (§7); implementation in progress |
| **Parent experiments** | `exp_003_2_action_pred_no_ema` (action predictor, no EMA), `exp_003_3_state_only_reward` (state-only reward), `exp_003_4_no_resampler_self_attn` (cross-attn-only resampler) |
| **Games** | LS20 Level 1 (`ls20-9607627b`) **+** TU93 Level 1 (`tu93-0768757b`) — both 4-action environments |
| **Reward** | Intrinsic curiosity, **state-prediction error only**, computed independently per environment |

---

## 1. One-Paragraph Summary

This experiment merges the three isolated changes explored in `exp_003_2`, `exp_003_3`, and `exp_003_4`, **and introduces a new axis**: joint training of a single shared encoder on transitions from two different 4-action games. From `exp_003_2` it keeps the dual-predictor architecture with **no EMA target encoder** — the action predictor is the anti-collapse mechanism. From `exp_003_3` it keeps the **state-only intrinsic reward** (action-CE term removed from the reward but retained in the JEPA loss, to avoid the corner-attractor pathology). From `exp_003_4` it keeps the **cross-attention-only Perceiver Resampler** (no self-attention among latents). The new axis is multi-environment training: a single patch encoder and a single Perceiver Resampler are co-trained on LS20 **and** TU93 transitions, with **per-environment action embeddings** (the same action index means different things in different games) and **per-environment policies** (each policy only sees on-policy data from its own game). Replay is split into two per-env buffers and each JEPA gradient step sees a balanced batch (half from each env), so the longer-episode game cannot dominate the encoder's training signal. The headline question is whether a shared encoder trained on two games learns transition-relevant features that generalise across games, or whether the two games' gradients fight each other.

A small training-loop bug from earlier experiments is also fixed here: when a life ends, the rollout's curiosity-reward computation, all rollout health-metric appends, and the policy-buffer append are now gated behind `not life_end` — they were previously logged for the degenerate dying transition before the existing `ep_transitions[:-1]` slice excluded it from the replay buffer.

---

## 2. Architecture

### 2.1 Overview

```
                    ┌─────────────── LS20 frame ─────────────────┐
                    │  (64×64 uint8)                              │
                    └─────────────────────┬───────────────────────┘
                                          │
                    ┌─────────────── TU93 frame ─────────────────┐
                    │  (64×64 uint8)                              │
                    └─────────────────────┬───────────────────────┘
                                          │
                                          ▼   (one env per rollout episode;
                                              JEPA batch mixes both halves)
                          ┌────────────────────────────────┐
                          │  Stage 1: Patch Encoder        │   SHARED
                          │  color_embed(16→4)             │
                          │  patch_proj(1024→128)          │
                          │  SA-Block 1  (4 heads, RoPE)   │
                          │  SA-Block 2  (4 heads, RoPE)   │
                          │  sa_norm (LayerNorm)           │
                          │  → (B, 16, 128)                │
                          └────────────────────────────────┘
                                          │
                                          ▼  context
                          ┌────────────────────────────────┐   SHARED
                          │  Stage 2: Perceiver Resampler  │
                          │  Round 0: Cross-Attn only      │  (no self-attn,
                          │  Round 1: Cross-Attn only      │   inherited from
                          │  output_norm (LayerNorm)       │   exp_003_4)
                          │  → h_t (B, 4, 128)             │
                          └────────────────────────────────┘
                                          │
                          ┌───────────────┼─────────────────────────┐
                          ▼               ▼                         ▼
              ┌─────────────────┐ ┌─────────────────┐   ┌──────────────────────┐
              │ State Predictor │ │ Action          │   │ Policy (PER-ENV)     │
              │   SHARED        │ │   Predictor     │   │  policy_ls20         │
              │ (h_t, a_emb_env)│ │   SHARED        │   │  policy_tu93         │
              │    → h̃_{t+1}   │ │ (h_t, h_{t+1})  │   │  (independent weights│
              │                 │ │  → p(a | …)     │   │   and baselines)     │
              └────────┬────────┘ └─────────────────┘   └──────────────────────┘
                       │
                       │ uses
                       ▼
               ┌─────────────────────────────┐
               │  Action Embedding (PER-ENV) │   action_embed_ls20: Embedding(4, 32)
               │                             │   action_embed_tu93: Embedding(4, 32)
               └─────────────────────────────┘
```

What is shared and what is per-environment:

| Module | Shared / Per-env | Notes |
|--------|------------------|-------|
| Patch encoder (color embed, patch proj, SA blocks, SA norm) | **Shared** | Single set of weights trained on both envs |
| Perceiver Resampler (cross-attn rounds, output norm, placeholders) | **Shared** | Cross-attention only — no self-attn among latents (inherited from exp_003_4) |
| State predictor (4 per-latent MLPs + time embed) | **Shared** | Takes env-specific action embedding as input |
| Action predictor (Linear → GELU → Linear) | **Shared** | Both envs have exactly 4 actions, so a single 4-way head is sufficient. CE is computed against the env-local action index. |
| Action embedding | **Per-env** | Two `Embedding(4, 32)` tables: `action_embed_ls20`, `action_embed_tu93` |
| Policy network | **Per-env** | Two independent `PolicyNetwork` instances + two independent REINFORCE baselines |
| Replay buffer (`NextFrameLatentBuffer`) | **Per-env** | Capacity 25K each |
| Policy buffer | **Per-env** | Capacity 64 each |

### 2.2 Stage 1 — Patch Encoder

Identical to exp_003_2 / exp_003_3 / exp_003_4. All weights shared across the two games — there is no env conditioning at this stage.

| Attribute | Value |
|-----------|-------|
| Input | `(B, 64, 64)` uint8 frame (color indices 0–15) |
| Output | `(B, 16, 128)` SA-normed patch token sequence |
| Patch grid | 4×4 = 16 patches, 16×16 pixels each |

Pipeline: color embedding `nn.Embedding(16, 4)` → patch reshape to `(B, 16, 1024)` → linear projection to `(B, 16, 128)` → two pre-norm SA blocks with 2D RoPE (4 heads, 32 dims/head) → final `LayerNorm`. No learned positional embeddings; position is encoded entirely in the rotary frequencies over the 4×4 patch grid. See `exp_003_4/system_card.md §2.2` for the full step-by-step.

**Why a shared patch encoder makes sense here.** LS20 and TU93 are both rendered in the ARC-AGI-3 16-color palette at 64×64. Both use grid-cell-aligned sprites. The colour-index→4-dim embedding, the 16×16 patching, and the 2-block SA stack are completely game-agnostic operations on those frames; nothing in their parameterisation references game-specific semantics. The shared encoder is therefore a fair test of whether low-level visual features (which patches contain agent, walls, goal-like sprites) are learned the same way regardless of which game produced the transition.

### 2.3 Stage 2 — Perceiver Resampler (cross-attention only)

Identical to exp_003_4. Shared across the two games.

| Attribute | Value |
|-----------|-------|
| Input (context) | `(B, 16, 128)` SA-normed patch tokens |
| Input (queries) | `(B, 4, 128)` recurrent latents `h_{t-1}` (or placeholders at episode start) |
| Output | `(B, 4, 128)` output-normed latent vectors `h_t` |
| Rounds | 2, **separate weights** per round (not weight-tied) |
| Heads per cross-attn | 4 |
| Self-attn among latents | **absent** (inherited from exp_003_4) |

Per round:

```
Q = q_proj(norm_q(h))          # (B, 4, 128)
K = k_proj(norm_kv(context))   # (B, 16, 128)
V = v_proj(context)            # (B, 16, 128)
attn_w = softmax(Q K^T / √32)
h = h + out_proj(attn_w @ V)
h = h + FFN(norm(h))           # round output (no SA step)
# After all rounds:
h = output_norm(h)             # (B, 4, 128) — unit-scale recurrent state
```

The **placeholder vectors** (a learned `nn.Parameter` of shape `(4, 128)`) are also shared — both games' episode starts begin from the same placeholders.

**Risks specific to this experiment.** Two separate risks now compound the single-experiment collapse risk inherited from exp_003_4:

1. *Inter-latent collapse* (exp_003_4 risk, unchanged): with no self-attn channel, the 4 latents only differ via 4 distinct placeholders and 4 distinct cross-attn query projections. Monitor `eff_rank`, `latent_pairwise_cossim`.
2. *Cross-env representation collision* (new risk): two unrelated games are forced through the same Perceiver. If the two games' patch distributions are too dissimilar, the resampler may carve out two near-disjoint sub-regions of the 4-latent space, behaving like two independent encoders that happen to share weights. If they are too similar, useful game-discriminative features may be averaged away. Either failure mode is detectable in the cross-env gradient cosine metrics (§9, Section 9).

### 2.4 State Predictor (flow matching) — shared, with per-env action embedding input

Same flow-matching predictor as exp_003_2/3/4 — four per-latent MLPs plus a shared sinusoidal time embedding, integrated by Euler with `N = 3` steps. **The predictor weights are shared between games.** Game-specific information enters only via the action-embedding input:

```
τ      ~ Uniform[0, 1]
x_τ    = (1 − τ) h_t  +  τ h_{t+1}
x̂₁[i] = MLP_i(x_τ[i], a_emb_env, time_embed(τ))    # i ∈ {0,1,2,3}, env ∈ {ls20, tu93}

L_state = Σ_i ‖ x̂₁[i] − h_{t+1}[i] ‖²   (mean over batch and dims; target detached)
```

Stop-grad on `h_{t+1}` is unchanged from exp_003_2. The single-τ training form is the same; the full N-step Euler ODE is used at rollout time for the curiosity-reward computation. See `exp_003_4/system_card.md §2.4` for the full description.

**Why action embeddings are per-env but the MLP is shared.** Both games use action indices 0–3, but `ACTION1` ("up") in LS20 represents motion in pixel space, whereas in TU93 it represents traversal along a maze edge. A shared embedding row would force the predictor to model two different state-transition operators with the same conditioning vector. Per-env action embeddings let the predictor learn `(latent shift due to "LS20 up")` vs. `(latent shift due to "TU93 up")` separately while sharing all the heavy compute.

### 2.5 Action Predictor — shared (both games are 4-way)

Identical to exp_003_2/3/4. A single MLP with a single 4-way output head, used for both games:

```
z      = concat(flatten(h_t), flatten(h_{t+1}))     # (B, 1024)
logits = Linear(512, 4)(GELU(Linear(1024, 512)(z)))
L_action = cross_entropy(logits, a_t)               # multi-class CE; a_t ∈ {0,1,2,3}
```

No detach on either `h_t` or `h_{t+1}`. Gradient flows through both endpoints into the shared encoder, providing the anti-collapse signal that replaces the EMA target encoder.

**Why a shared head works.** Both LS20 and TU93 use action indices 0–3 (ACTION1–ACTION4). The output logits index over the *same set of integer labels*. The semantics differ between games — index 0 represents different physical motions — but that game-specific semantics is supplied to the predictor via the per-env action-embedding table in `L_state`. The action *predictor* only needs to recover which index was used, and CE against `a_t ∈ {0,1,2,3}` does that regardless of which game produced the transition.

### 2.6 Action Embedding — per environment

Two separate `nn.Embedding(4, 32)` tables:

```python
self.action_embed = nn.ModuleDict({
    "ls20": nn.Embedding(4, 32),
    "tu93": nn.Embedding(4, 32),
})
```

Used by the state predictor only (the action predictor does not consume action embeddings — it predicts actions as output). When a JEPA batch is assembled from balanced per-env halves (§4.2), each half looks up its actions in its own table, and the two embedded halves are concatenated before being passed to the state predictor.

### 2.7 Policy Networks — per environment

Two independent `PolicyNetwork` instances with identical architecture but independent weights and independent REINFORCE baselines:

```
logits = Linear(512, 4)(GELU(Linear(512, 512)(flatten(h_t))))
# available-action masking: set logits[unavailable] = −∞
a_t ~ Categorical(softmax(logits))
```

`baseline_ls20`, `baseline_tu93` — each maintains an independent EMA of mean reward.

**Why separate policies.** The two games have different transition dynamics, different reward distributions, and different available-action masks at any given state. A single shared policy would have to learn an "env-id detection" channel internally and switch behaviour on it — possible, but adds an unrelated representation-learning load on the policy. Separate policies are the simpler, cleaner control: any cross-game generalisation we observe must be coming from the shared encoder / predictor, not from a policy that has secretly learned a joint representation.

---

## 3. Key Changes

### 3.1 Changes from exp_003_2 / exp_003_3 / exp_003_4 (the three merged isolates)

| Dimension | exp_003_2 | exp_003_3 | exp_003_4 | exp_004 (this) | Inherited from |
|-----------|-----------|-----------|-----------|----------------|----------------|
| **Resampler self-attn among latents** | Yes | Yes | No | **No** | exp_003_4 |
| **Reward = state + action error** | Yes (`0.5/0.5`) | No (state only, `1.0/0.0`) | Yes (`0.5/0.5`) | **No (state only, `1.0/0.0`)** | exp_003_3 |
| **JEPA loss** | `0.5·L_state + 0.5·L_action` | `0.5·L_state + 0.5·L_action` | `0.5·L_state + 0.5·L_action` | **`0.5·L_state + 0.5·L_action`** | exp_003_2 / 3 / 4 (unchanged) |
| **EMA target encoder** | None | None | None | **None** | exp_003_2 |
| **Buffer target field** | `next_frame` (uint8) | `next_frame` (uint8) | `next_frame` (uint8) | `next_frame` (uint8) | exp_003_2 |
| **Buffer sampling** | Uniform | Uniform | Uniform | **Balanced per env** (§4.2) | new in this experiment |

### 3.2 New in this experiment (multi-environment axis)

| Change | exp_003_4 (single-env baseline) | exp_004 (this) | Motivation |
|--------|---------------------------------|----------------|------------|
| **Games trained on** | LS20 only | LS20 + TU93 (both 4-action) | Test whether a shared encoder learns cross-game transition features, or whether the two gradients fight each other. |
| **Patch encoder weights** | Single | Single (shared) | Headline shared module — what the experiment is actually testing. |
| **Perceiver Resampler weights** | Single | Single (shared) | Same — under joint test. |
| **State predictor weights** | Single | Single (shared) | Shared dynamics MLP; per-env action conditioning. |
| **Action predictor weights** | Single | Single (shared) | Both games are 4-way; one CE head suffices. |
| **Action embedding** | 1 × `Embedding(4, 32)` | 2 × `Embedding(4, 32)` | Same index has different physical meaning per game. |
| **Policy network** | 1 × `PolicyNetwork` | 2 × `PolicyNetwork` (`policy_ls20`, `policy_tu93`) | Each game gets its own on-policy controller; no policy-level multi-task interference. |
| **Replay buffer** | 1 × `NextFrameLatentBuffer(50K)` | 2 × `NextFrameLatentBuffer(25K)` | Balanced JEPA sampling guarantees both games contribute equally per gradient step. |
| **Policy buffer** | 1 × `PolicyBuffer(64)` | 2 × `PolicyBuffer(64)` | Per-env on-policy REINFORCE; no cross-env contamination. |
| **Rollout structure** | One env loop | Round-robin: one full episode in LS20, then one full episode in TU93, repeat | Long-episode games (TU93 may be substantially longer than LS20's ~130 steps) cannot dominate the replay; the JEPA balanced sampler is the second layer of protection. |

### 3.3 Bug fix in the rollout loop (gating the dying step)

In exp_003_4/train.py, the dying step (the env step that triggered `life_end = True`) is excluded from the replay buffer via `ep_transitions[:-1]` ([exp_003_4/train.py:419-421](../exp_003_4_no_resampler_self_attn/train.py#L419-L421)). But the same step *is* pushed to the policy buffer (line 403) and *is* logged into rollout health metrics (lines 385-391). Because the rollout's reward components are computed from `(h_current, h_next)` where `h_next` is the encoder's output on the post-death (life-decrement or game-over) frame, those values are degenerate.

This experiment defers all per-step appends behind `not life_end`:

```python
next_np, is_terminal = env.step(action_idx)
life_end = is_end_of_life_env(frame_np, next_np, is_terminal, env=env_name)

if not life_end:
    # Compute reward components and ht_htp1_cossim under no_grad
    ...
    # Append to rollout health logs
    health.sec1["ht_htp1_cossim_rollout"].append(ht_htp1_cs)
    health.sec6["reward_state_component"].append(state_err)
    # (reward_action_component is also appended for logging, even though
    #  reward_w_action == 0 and it does not affect curiosity_reward)
    # Append to policy buffer
    if step >= cfg.warmup_steps and log_prob is not None:
        policy_buf.add(log_prob, curiosity_reward, entropy)

# Append to ep_transitions unconditionally — the dying step is the slice tail.
ep_transitions.append((frame_np.copy(), h_query_np.copy(), action_idx, next_np.copy()))

if life_end:
    for frame_i, hq_i, action_i, next_i in ep_transitions[:-1]:
        latent_buf.add(frame_i, hq_i, action_i, next_i)
    ...
```

The change is local to this experiment's `train.py` only. exp_003_2/3/4's `train.py` are left intact so previous runs remain reproducible against the exact code that produced them.

### 3.4 Multi-environment data-collection structure

A single rollout iteration runs **one full episode in one env**, then switches to the other env. Pseudocode:

```python
env_cycle = itertools.cycle(["ls20", "tu93"])

while step < cfg.max_steps:
    env_name = next(env_cycle)
    env       = envs[env_name]
    latent_buf = latent_bufs[env_name]
    policy    = policies[env_name]
    aemb      = action_embeds[env_name]
    policy_buf = policy_bufs[env_name]
    baseline   = baselines[env_name]

    # One episode end-to-end in this env (with bug fix from §3.3)
    run_one_episode(
        env=env,
        env_name=env_name,
        encoder=encoder,
        state_predictor=state_predictor,
        action_predictor=action_predictor,
        action_embed=aemb,
        policy=policy,
        latent_buf=latent_buf,
        policy_buf=policy_buf,
        baseline=baseline,
        health=health,
        cfg=cfg,
    )

    # After the episode, advance global step counter and run JEPA / policy updates.
```

JEPA updates and policy updates are still gated by their respective step-frequency counters (`update_freq = 5`, `policy_update_freq = 64`) measured in **global** env steps (LS20 steps + TU93 steps).

---

## 4. Training

### 4.1 Data Collection (Recurrent Rollout, Per Episode)

Inside `run_one_episode(env, env_name, ...)`:

1. **Reset env** if at episode start. Initialize recurrent state `h_t = encoder.perceiver.get_initial_queries(1, device)` (the shared learned placeholders).
2. **Encode frame**: `h_current, _, _ = encoder(frame_t, h_query_t)` under `torch.no_grad()`.
3. **Select action**: `action_idx, log_prob, entropy = policy.act(h_current.squeeze(0), env.available_actions)` (uniform random during warmup; `log_prob`, `entropy` retain autograd graphs).
4. **Step env**: `next_np, is_terminal = env.step(action_idx)`. Compute `life_end = is_end_of_life_env(env_name, frame_np, next_np, is_terminal)` — see §7 for the per-env detection logic.
5. **If `not life_end`**: compute curiosity reward under `no_grad()`:
   - Encode the next frame with `h_current` as recurrent query → `h_next`.
   - Run state predictor's full N=3 Euler ODE on `(h_current, action_embed_env(a_t))` → `h̃_{t+1}`.
   - `state_err = MSE(h̃_{t+1}, h_next)` (mean over 4 × 128 dims).
   - Compute `action_err = CE(action_predictor(h_current, h_next), one_hot(a_t))` **for logging only**; it does not enter `curiosity_reward` because `reward_w_action = 0.0`.
   - `curiosity_reward = clamp(1.0 * state_err + 0.0 * action_err, max=50.0)`.
   - Append to health logs (`ht_htp1_cossim_rollout`, `reward_state_component`, `reward_action_component`, `reward_total`).
   - Append `(log_prob, curiosity_reward, entropy)` to `policy_buf[env_name]`.
6. **Append to `ep_transitions`** unconditionally: `(frame_np, h_query_np, action_idx, next_np)`.
7. **Advance recurrent state**: `h_query_np = h_current.squeeze(0).cpu().numpy()`.
8. **If `life_end`**: flush `ep_transitions[:-1]` into `latent_buf[env_name]`; reset `ep_transitions = []`, `h_query_np = None`; reset env if `is_terminal`, otherwise continue from `next_np` (life-decrement case for LS20).

The dying step's curiosity-reward is **never computed** under this gating — encoder forward passes for the post-death frame are skipped, which is also a small efficiency win.

### 4.2 JEPA Training Update (balanced cross-env batch)

Every `update_freq = 5` global env steps, once **both** per-env buffers have ≥ `min_buffer_size`:

```python
batch_ls20 = latent_bufs["ls20"].sample(batch_size // 2, device)
batch_tu93 = latent_bufs["tu93"].sample(batch_size // 2, device)

# Re-encode h_t fresh for each half. The encoder is shared.
h_t_ls20 = encoder(batch_ls20.frames, batch_ls20.h_queries.detach())
h_t_tu93 = encoder(batch_tu93.frames, batch_tu93.h_queries.detach())

# Re-encode h_{t+1} fresh for each half. Detach the recurrent-query path.
h_tp1_ls20 = encoder(batch_ls20.next_frames, h_t_ls20.detach())
h_tp1_tu93 = encoder(batch_tu93.next_frames, h_t_tu93.detach())

# Per-env action embeddings into the shared state predictor.
a_emb_ls20 = action_embeds["ls20"](batch_ls20.actions)
a_emb_tu93 = action_embeds["tu93"](batch_tu93.actions)

# State loss — target detached.
L_state_ls20, _ = state_predictor.compute_loss(h_t_ls20, h_tp1_ls20.detach(), a_emb_ls20)
L_state_tu93, _ = state_predictor.compute_loss(h_t_tu93, h_tp1_tu93.detach(), a_emb_tu93)
L_state = 0.5 * (L_state_ls20 + L_state_tu93)

# Action loss — no detach on either endpoint. Shared 4-way head.
L_action_ls20 = F.cross_entropy(
    action_predictor(h_t_ls20, h_tp1_ls20), batch_ls20.actions)
L_action_tu93 = F.cross_entropy(
    action_predictor(h_t_tu93, h_tp1_tu93), batch_tu93.actions)
L_action = 0.5 * (L_action_ls20 + L_action_tu93)

# Combined JEPA loss
L = cfg.lambda_state * L_state + cfg.lambda_action * L_action     # 0.5 / 0.5
L.backward()
clip_grad_norm_(
    list(encoder.parameters())
    + list(state_predictor.parameters())
    + list(action_predictor.parameters())
    + list(action_embeds["ls20"].parameters())
    + list(action_embeds["tu93"].parameters()),
    max_norm=cfg.grad_clip_model,        # 5.0
)
enc_opt.step()
state_pred_opt.step()
action_pred_opt.step()
```

Implementation note: the four forward passes (`encoder` × 4) can be packed into a single batched call by concatenating `(batch_ls20.frames, batch_tu93.frames)` before the encoder and splitting after, which reduces overhead — but the per-env split is logically what is happening. Keep the unbatched form in code for clarity unless wall-clock becomes a bottleneck.

**Encoder gradient paths per JEPA step.** Now there are **six** distinct paths into the encoder per JEPA step (three per env, identical structure to exp_003_4's three-path bookkeeping, doubled across envs):

| # | Source | Env | Path into encoder | Detached? |
|---|--------|-----|-------------------|-----------|
| 1 | `L_state_ls20`  | LS20 | via `h_t_ls20` (first encoder call) | no |
| 2 | `L_action_ls20` | LS20 | via `h_t_ls20` (first encoder call)   | no |
| 3 | `L_action_ls20` | LS20 | via `h_tp1_ls20` (second encoder call) | no |
| 4 | `L_state_tu93`  | TU93 | via `h_t_tu93` (first encoder call) | no |
| 5 | `L_action_tu93` | TU93 | via `h_t_tu93` (first encoder call) | no |
| 6 | `L_action_tu93` | TU93 | via `h_tp1_tu93` (second encoder call) | no |

The Perceiver participates in four encoder calls per step (two per env, one for `h_t` and one for `h_{t+1}`), so paths 1/2/4/5 flow through it from the `h_t` calls and paths 3/6 from the `h_{t+1}` calls — the Perceiver sees gradient from all six paths.

We keep `perceiver_lr = 5e-5` (half of `sa_lr = 1e-4`) inherited from exp_003_4. The 6-path accumulation per JEPA step (vs. 3 in single-env experiments) does roughly double the post-sum gradient magnitude into the shared Perceiver, but the batch contributions are half-size (`batch_size // 2` per env), which partially offsets it. `grad_clip_model = 5.0` continues to act as the safety net. If the LR turns out to be too high after smoke-running (clip saturating consistently), halve it.

### 4.3 Policy Update (per env, independent)

Each `policy_buf[env_name]` empties when full (capacity 64), independently of the other env's buffer. The corresponding policy is updated using only its own on-policy data:

```python
log_probs, rewards, entropies = policy_buf[env_name].get(device)
adv = rewards - baseline[env_name].value
adv = adv / (adv.std().clamp(min=1e-8) + 1e-8)
baseline[env_name].update(rewards.mean())
pol_loss = -mean(adv * log_probs) - cfg.policy_entropy_lambda * mean(entropies)
pol_loss.backward()
clip_grad_norm_(policies[env_name].parameters(), max_norm=cfg.grad_clip_policy)   # 1.0
pol_opts[env_name].step()
```

Each policy's baseline is independent, so the policy update for one game cannot bias the advantage signal of the other.

### 4.4 Per-Component Optimizer Design

| Optimizer | Parameters | LR | Weight Decay |
|-----------|-----------|-----|------|
| `enc_opt` group 1 | color_embed, patch_proj, sa_blocks, sa_norm | 1e-4 | 0.01 |
| `enc_opt` group 2 | perceiver (all rounds + output_norm + placeholders) | 5e-5 | 0.01 |
| `state_pred_opt` | state_predictor (MLPs + time_embed) + **both** `action_embed_ls20` and `action_embed_tu93` | 1e-4 | 0.01 |
| `action_pred_opt` | action_predictor (shared 4-way head) | 1e-4 | 0.01 |
| `pol_opt_ls20` | `policy_ls20` | 1e-4 | 0 |
| `pol_opt_tu93` | `policy_tu93` | 1e-4 | 0 |

Both per-env action embeddings go into `state_pred_opt` because their gradients flow through `L_state` only (the action predictor doesn't read them).

### 4.5 Replay Buffers (per env, `NextFrameLatentBuffer`)

Two independent `NextFrameLatentBuffer` instances:

| Buffer | Capacity | Stored fields | Sampling |
|--------|----------|---------------|----------|
| `latent_bufs["ls20"]` | 25,000 | `(frame_t, h_{t-1}, a_t, next_frame_t)` | Uniform |
| `latent_bufs["tu93"]` | 25,000 | same | Uniform |

Total memory ≈ 510 MB (matches a single 50K buffer in earlier experiments).

A small helper assembles the balanced JEPA batch:

```python
def sample_balanced(latent_bufs, batch_size, device):
    """Return (batch_ls20, batch_tu93) each of size batch_size // 2."""
    assert batch_size % 2 == 0, "batch_size must be even for balanced sampling"
    half = batch_size // 2
    return (
        latent_bufs["ls20"].sample(half, device),
        latent_bufs["tu93"].sample(half, device),
    )
```

The dying transition is excluded from each buffer by the same `ep_transitions[:-1]` mechanism as exp_003_4. The bug-fix gating (§3.3) ensures the dying step is also excluded from the policy buffer and from rollout health logs.

### 4.6 Training Schedule

| Phase | Steps (global) | Actions | JEPA update | Policy update |
|-------|----------------|---------|------------|---------------|
| Warmup | 0 – 999 | Uniform random in the currently-rolling env | Every 5 env steps (once both buffers ≥ 512) | No |
| Joint training | ≥ 1,000 | Env-specific policy | Every 5 env steps | Every 64 env steps **per env** (the buffer that just filled triggers its own policy update) |

**Round-robin granularity:** one full episode in one env at a time (not interleaved by step). This guarantees recurrent latent continuity within an episode — `h_{t-1}` always belongs to the same game as `frame_t`.

**Checkpointing:** every 5,000 global env steps. Each checkpoint contains the shared encoder, the shared predictors, both action-embedding tables, and both policies + optimizers + baselines, so a resume restores the entire two-game system in one file.

---

## 5. Inference (Deployment)

### 5.1 Per-Env Inference Loop

```python
def run_episode(env_name):
    env       = envs[env_name]
    policy    = policies[env_name]
    aemb      = action_embeds[env_name]

    frame_t = env.reset()
    h_t = encoder.perceiver.get_initial_queries(1, device)
    while True:
        frame_t_torch = torch.from_numpy(frame_t).unsqueeze(0).to(device)
        h_current, _, _ = encoder(frame_t_torch, h_t)
        action_idx, _, _ = policy.act(h_current.squeeze(0), env.available_actions)
        next_frame, is_terminal = env.step(action_idx)
        h_t = h_current
        if is_terminal: break
        frame_t = next_frame
```

Both envs share the encoder forward; only the policy and the action-embedding lookup are env-specific. The action-embedding lookup is not used at inference because the policy reads `h_current` directly — `aemb` is only consumed during training inside the state predictor.

### 5.2 Curiosity Reward at Rollout

Per env, identical formula (inherited from exp_003_3):

```
state_err  = MSE( state_predictor.predict(h_t, action_embeds[env_name](a_t)),  h_{t+1} )    # full Euler ODE
action_err = CE( softmax(action_predictor(h_t, h_{t+1})) , one_hot(a_t) )                   # for logging only
r_t        = clamp( 1.0 · state_err  +  0.0 · action_err , max=50.0 )
           = clamp( state_err , max=50.0 )
```

Training-time `L_state` is still computed on a single random-τ sample of the flow-matching path; the reward uses the deterministic post-ODE prediction. This asymmetry is unchanged from exp_003_2.

### 5.3 Action Masking

Each env exposes `env.available_actions` per step. The corresponding policy applies `logits[unavailable] = −∞` before softmax. Applied identically at rollout and at evaluation.

---

## 6. Reward Function

```
r_t = clamp( 1.0 · MSE(h̃_{t+1}, h_{t+1})  +  0.0 · CE(p_pred, one_hot(a_t)) ,
             max=50.0 )
    = clamp( state_err , max=50.0 )
```

The action-CE term is computed and logged at every step (so we can still inspect `reward_action_component` as a diagnostic) but is multiplied by zero and does not enter `curiosity_reward`. The rationale is exactly the same as exp_003_3's: in maze-like environments with many blocked actions per state (LS20 corners, TU93 dead-end maze nodes), the action predictor's CE saturates at `ln(4) ≈ 1.386` on those transitions and turns walls into maximum-reward states. State error decays to zero on the identity transition `h_t → h_t` that a competent state predictor learns for blocked actions, giving the agent a gradient out.

**Why we keep the action predictor in the JEPA loss anyway.** The anti-collapse mechanism inherited from exp_003_2 depends on gradient flowing through `(h_t, h_{t+1})` from `L_action`. Removing the action predictor from the JEPA loss would re-introduce the latent-collapse risk that motivated the dual-predictor design. The reward and the loss are decoupled: the action predictor stays in the loss, but its error is no longer a reward signal.

---

## 7. TU93 Life-End Detection (resolved by probe)

### 7.1 Probe protocol

The probe `JEPA/experiments/exp_004_0_ls20_tu93/probe_tu93_lives.py` rolled a random-action agent through 8 attempts on `tu93-0768757b` (offline mode), capturing every frame plus `raw.state`, `raw.available_actions`, `raw.levels_completed`, `raw.win_levels`, and a one-shot `vars(raw)` + `dir(raw)` dump on the very first reset. Outputs persisted under `probe_runs/<timestamp>/`.

### 7.2 What the raw env exposes

From the first-reset `vars(raw)` dump (`raw_fields.txt`):

```
game_id: str
state: GameState  (NOT_FINISHED / WIN / GAME_OVER)
levels_completed: int
win_levels: int           (9 for TU93 — total winnable levels)
action_input: ActionInput
guid: str
full_reset: bool
available_actions: list[int]   (1-indexed GameAction values)
frame: list[ndarray]
```

The probe explicitly scanned for `lives`, `remaining_lives`, `energy`, `score`, `step_count`, `time_left`, `level` attributes on every step of every attempt. **None of these attributes are ever present.** There is no intra-game lives counter exposed by the env.

### 7.3 Empirical episode behaviour

All 8 random-action attempts terminated identically:

```
attempt 0..7: steps=50  final_state=GameState.GAME_OVER  levels_completed=0
```

- **Episode length under random policy: exactly 50 steps.** This is deterministic across all 8 attempts (no variance — implies a hard step-count cap rather than a stochastic per-step death probability).
- **Terminal state: always GAME_OVER**, never WIN. Random play never clears Level 1.
- **levels_completed**: stays at 0 throughout every attempt.

A pixel-level scan for regions that change at-and-only-at terminal (`changes_near_end & ~changes_throughout`, intersected across all 8 attempts) identified a stable region of **3 pixels at row 63, cols 0–2**. This is the natural tail of the step-count bar (rows 63, colour 6) shrinking to zero — not a separate life indicator. The bar starts at 64 columns and shrinks one column per step; cols 0–2 are simply the last to disappear. Row 63 is already in `_MASKED_ROWS` of `Tu93Env` and is excluded from frame diffs used for reward computation.

### 7.4 Verdict

**TU93 has no intra-game life concept.** One game = one life = one episode. Termination is determined entirely by the framework-level `_is_terminal()` flag (`GameState.GAME_OVER` after 50 random-action steps; `levels_completed >= 1` if the agent ever solves a level).

`is_end_of_life_tu93` therefore reduces to:

```python
def is_end_of_life_tu93(frame, next_frame, is_terminal):
    return is_terminal
```

### 7.5 Implication for buffer dynamics

Per-episode transition counts under random policy:

| Env | Steps / episode | Transitions stored after `[:-1]` slice |
|-----|-----------------|----------------------------------------|
| LS20 | ~130 (energy-bounded; 3 lives × ~42 energy + extras) | ~129 (one life ended) or ~387 (all three lives, full game) |
| TU93 | 50 (deterministic step cap) | 49 |

TU93 generates transitions roughly 2.5×–7× more slowly per episode than LS20. The round-robin (one episode in each env per cycle) plus balanced JEPA sampling (`batch_size//2` from each per-env buffer) guarantees this asymmetry does not bias the encoder's training signal. The per-env buffer capacities (25K each) accommodate ~500 TU93 episodes vs ~64 LS20 full games before circular overwrite — both substantially exceed the recency horizon needed for uniform sampling to be statistically well-mixed.

### 7.6 Dispatcher

`reward_shaping.py` exposes a single dispatcher used by `train.py`:

```python
def is_end_of_life(env_name, frame, next_frame, is_terminal):
    if env_name == "ls20":
        return is_end_of_life_ls20(frame, next_frame, is_terminal)   # existing logic
    if env_name == "tu93":
        return is_end_of_life_tu93(frame, next_frame, is_terminal)   # is_terminal
    raise ValueError(f"unknown env_name: {env_name}")
```

The LS20 implementation (`count_lives` + life-counter decrement check) is unchanged from prior experiments.

---

## 8. Health Monitoring

The single-env metric set is inherited from `exp_003_4/metrics.md` (the closest sibling — same architecture choices except for the multi-env axis). All Section 1–7 metrics from that file are duplicated per env with an env-tag suffix; new Section 8–10 metrics are added for the multi-env axis.

### 8.1 Per-environment basics (Section 8 of metrics.md — NEW)

Mirror every single-env metric with `_ls20` / `_tu93` suffixes. At minimum:

- `L_next_state_pred_ls20`, `L_next_state_pred_tu93`
- `L_action_pred_ls20`, `L_action_pred_tu93`
- `reward_state_component_ls20`, `reward_state_component_tu93`
- `reward_total_ls20`, `reward_total_tu93`
- `episode_length_ls20`, `episode_length_tu93`
- `episodes_completed_ls20`, `episodes_completed_tu93`
- `policy_entropy_ls20`, `policy_entropy_tu93`
- `policy_entropy_normalized_ls20`, `policy_entropy_normalized_tu93`
- `tile_coverage_pct_ls20`, `tile_coverage_pct_tu93` — separate manifests under `monitors/exploration_manifests/`

The shared-encoder metrics (`eff_rank`, `latent_pairwise_cossim_*`, `ht_htp1_cossim_*`, `latent_norm_per_token`, etc.) remain single-valued: they describe the shared encoder regardless of which env produced the batch.

### 8.2 Cross-env gradient interference (Section 9 of metrics.md — NEW)

For each parameter subset `k ∈ {patch_sa, perc_cross_r0, perc_cross_r1, state_pred_mlp_0..3, action_pred}`, run **two separate backward passes** per probe (one per env, on the same balanced JEPA batch) and compute:

- `gnorm_jepa_ls20_<k>` — L2 norm of grads from the LS20 half of the JEPA batch
- `gnorm_jepa_tu93_<k>` — same for TU93 half
- `gcossim_jepa_ls20_vs_tu93_<k>` — flatten-cosine between the two
- Broken down by loss source (state-only, action-only, and the cross terms):
  - `gcossim_state_ls20_vs_state_tu93_<k>`
  - `gcossim_action_ls20_vs_action_tu93_<k>`
  - `gcossim_state_ls20_vs_action_tu93_<k>`
  - `gcossim_action_ls20_vs_state_tu93_<k>`

Reuse the `grad_cosine()` and `compute_source_decomposition()` patterns from [exp_003_4/monitors/gradients.py:49-187](../exp_003_4_no_resampler_self_attn/monitors/gradients.py#L49-L187). Gating: the same `grad_decomp_freq = 25` schedule — these probes do extra backward passes and are not free.

**Why this is the headline new metric.** Negative or near-zero `gcossim_jepa_ls20_vs_tu93_<k>` on the shared encoder modules would be direct evidence that the two games are pulling the encoder in conflicting directions — which would mean a shared encoder is wrong for this pair. Strongly positive cosines would mean the two games' gradients reinforce each other, validating the shared-encoder hypothesis.

### 8.3 Beyond cosine — per-element disagreement (Section 10 of metrics.md — NEW)

Module-level cosine compresses thousands of parameter dimensions into a single scalar and loses fine-grained sign-disagreement information (see §12 for the full discussion). To complement it, for each subset `k` log:

- **`gsign_disagree_frac_<k>`** — fraction of parameter entries where `sign(g_ls20) ≠ sign(g_tu93)`, treating zeros as agreement. Range `[0, 1]`; `0.5` = random sign relationship.
- **`gsign_disagree_frac_magweighted_<k>`** — same fraction weighted by `|g_ls20| · |g_tu93|`, so disagreement at heavily-updated parameters counts more.
- **`gcossim_perlayer_dist_<k>`** — histogram of per-layer cosines inside subset `k` (e.g. each `nn.Linear` weight, each attention projection). Reveals localised conflict that the module-level cosine averages away.

These are computed on the same two gradient vectors already gathered for Section 9; the marginal compute cost is small.

### 8.4 Shared-encoder collapse monitoring (unchanged structure)

The collapse-monitoring metrics from exp_003_4 — `eff_rank`, `latent_pairwise_cossim_buf`, `latent_pairwise_cossim_t{1,10,20}`, `ht_htp1_cossim_*`, `H1_HT_cossim`, `latent_norm_per_token`, `round0_postCA_pairwise_cossim_t1`, `action_pred_entropy_eval` — all remain in place and remain single-valued. The risk profile is unchanged from exp_003_4 (no SA among latents) and is in fact slightly higher under multi-env training because the shared encoder is asked to handle two distributions at once.

### 8.5 Eval-pass instrumentation

`monitors/eval_pass.py` runs an evaluation rollout under hooks every `eval_freq` env steps. In this experiment it runs **once per env per eval cycle**, producing env-suffixed metrics where applicable. Module-level metrics (e.g. round-0 post-CA cosine) report the value from each env's eval rollout under env-suffixed keys; metrics that are inherently env-agnostic (e.g. `eff_rank`) report once per cycle on a combined eval pass.

---

## 9. Complete Hyperparameter Table

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model dimensions** | | |
| `d_model` | 128 | Encoder / predictor / latent dimension |
| `d_color` | 4 | Color embedding dimension (16 colors) |
| `n_actions` | 4 | Discrete actions per env (both LS20 and TU93) |
| `d_action` | 32 | Action embedding dimension (state predictor input) |
| `patch_size` | 16 | Pixels per patch side |
| **Encoder SA blocks (shared)** | | |
| `n_sa_blocks` | 2 | Number of self-attention blocks |
| `n_sa_heads` | 4 | Attention heads per SA block (32 dims/head) |
| `sa_ffn_dim` | 512 | SA block FFN hidden dimension |
| **Perceiver Resampler (shared, cross-attn only)** | | |
| `n_latents` | 4 | Number of latent query vectors |
| `n_placeholders` | 4 | Learned placeholder vectors (episode start) |
| `n_perceiver_rounds` | 2 | Perceiver rounds (separate weights) |
| `n_perceiver_heads` | 4 | Cross-attn heads per round |
| `perceiver_ffn_dim` | 512 | Perceiver FFN hidden dimension |
| `perceiver_self_attn_among_latents` | **False** | Inherited from exp_003_4 |
| **2D RoPE** | | |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `patch_grid_h` / `patch_grid_w` | 4 / 4 | Patch grid dimensions |
| **State Predictor (shared)** | | |
| `n_ode_steps` | 3 | Euler ODE integration steps at rollout |
| `predictor_hidden` | 512 | Per-latent MLP hidden dimension |
| `time_emb_dim` | 128 | Sinusoidal time embedding dimension |
| `time_proj_dim` | 512 | Projected time dimension |
| **Action Predictor (shared)** | | |
| `action_predictor_hidden` | 512 | Action predictor MLP hidden dimension |
| **JEPA loss weighting** | | |
| `lambda_state` | 0.5 | Weight on `L_state` in total JEPA loss |
| `lambda_action` | 0.5 | Weight on `L_action` in total JEPA loss |
| **Reward weighting (state-only, inherited from exp_003_3)** | | |
| `reward_w_state` | 1.0 | Weight on state-prediction error in curiosity reward |
| `reward_w_action` | 0.0 | Weight on action-prediction error (REMOVED from reward) |
| `reward_clamp` | 50.0 | Per-step reward cap |
| **Policy networks (per env, identical architecture)** | | |
| `policy_hidden` | 512 | Policy MLP hidden dimension |
| **Replay buffers (per env)** | | |
| `buffer_size_per_env` | 25,000 | Per-env `NextFrameLatentBuffer` capacity |
| `min_buffer_size` | 512 | Minimum transitions per env before JEPA training |
| `batch_size` | 64 | JEPA training batch size (half from each env) |
| **Training schedule** | | |
| `update_freq` | 5 | JEPA gradient step every N global env steps |
| `policy_update_freq` | 64 | Per-env policy update every N steps in that env |
| `warmup_steps` | 1,000 | Random exploration before policy trains |
| `max_steps` | 500,000 | Total training budget (global env steps) |
| **Optimisers** | | |
| `sa_lr` | 1e-4 | LR for patch embed + SA blocks |
| `perceiver_lr` | 5e-5 | LR for Perceiver (half of SA) |
| `state_predictor_lr` | 1e-4 | LR for state predictor + both action embeddings |
| `action_predictor_lr` | 1e-4 | Action predictor LR |
| `policy_lr` | 1e-4 | Policy Adam LR (per env) |
| `encoder_wd` | 0.01 | Encoder weight decay |
| `state_predictor_wd` | 0.01 | State predictor weight decay |
| `action_predictor_wd` | 0.01 | Action predictor weight decay |
| `grad_clip_model` | 5.0 | Gradient clip for encoder + both predictors + both action embeddings |
| `grad_clip_policy` | 1.0 | Gradient clip per policy |
| **Policy REINFORCE (per env)** | | |
| `policy_entropy_lambda` | 0.10 | Entropy regularisation coefficient |
| `policy_baseline_alpha` | 0.99 | EMA decay for running reward baseline |
| **Cross-env gradient probe** | | |
| `grad_decomp_freq` | 25 | Cross-env gradient cosine / sign-disagreement probe every N JEPA updates |
| **Misc** | | |
| `seed` | 42 | Random seed |
| `game_ids` | `["ls20-9607627b", "tu93-0768757b"]` | Game IDs trained on |
| `env_cycle_unit` | `episode` | Round-robin granularity is one full episode per env |

---

## 10. File Layout (to create)

```
JEPA/experiments/exp_004_0_ls20_tu93/
├── system_card.md             — this document
├── metrics.md                 — copied from exp_003_4 and extended with §8.1–8.3
├── __init__.py
├── config.py                  — Frozen dataclass Config; reward_w_action = 0.0,
│                                 game_ids = [ls20, tu93], per_env_buffer_size = 25_000
├── train.py                   — Two-env round-robin training loop with bug fix
├── eval.py                    — Per-env eval (`--env ls20` / `--env tu93`)
├── probe_tu93_lives.py        — Pre-implementation life-end probe (§7)
├── debug_runner.py            — Per-step dashboard data collection, env-aware
├── reward_shaping.py          — LS20 life detection (re-export) + TU93 (TBD by probe)
├── analyze_cross_attn.py      — Post-hoc cross-attn analysis, env-tagged outputs
├── analyze_cross_attn_t0_full.py — t=0 full-bandwidth cross-attn analysis
├── panel.js                   — Dashboard plugin: env-tagged plot panes + cross-env grad cossim
├── models/
│   ├── __init__.py            — load_models() factory exposing
│   │                            encoder, state_predictor, action_predictor,
│   │                            action_embeds (dict), policies (dict)
│   ├── encoder.py             — Re-export from exp_003_4 (no architectural change)
│   ├── state_predictor.py     — FlowMatchingPredictor (shared)
│   ├── action_predictor.py    — ActionPredictor MLP, 4-way head (shared)
│   └── policy.py              — PolicyNetwork, REINFORCEBaseline (instantiated twice)
├── monitors/
│   ├── __init__.py
│   ├── health.py              — env-tagged metric keys
│   ├── attention.py
│   ├── gradients.py           — extended: cross-env grad cosine + sign-disagreement (§8.2-8.3)
│   ├── representation.py
│   ├── predictors.py
│   ├── exploration.py         — separate manifests per game
│   ├── eval_pass.py           — runs once per env per cycle
│   ├── writer.py              — handles env-tagged keys + multi-env-aware run dir
│   └── exploration_manifests/ — per-game-ID JSON (reachable tiles, etc.)
├── checkpoints/               — empty at init; contains both envs' state in one file
└── runs/                      — Per-run dirs with training.log + metrics.jsonl + probe artefacts
```

Shared infrastructure (no changes needed):

- `JEPA/shared/buffer.py` — `NextFrameLatentBuffer` is instantiated twice, no code change.
- `JEPA/shared/env_wrapper.py` — `LS20Env` and `Tu93Env` are already present at [lines 144-168](../../../JEPA/shared/env_wrapper.py#L144-L168).
- `JEPA/shared/action_embed.py` — instantiated twice as `nn.ModuleDict`.
- `JEPA/shared/ema.py` — unused.

---

## 11. How to Run

```bash
# ── Step 0: TU93 life-end probe (must run FIRST) ───────────────────────────
cd "Code Repo"
uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.probe_tu93_lives
# Inspect runs/probe_tu93/ and update §7 of this card with the conclusion.

# ── Training (fresh) ───────────────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.train

# ── Training (resume from checkpoint) ──────────────────────────────────────
uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.train \
    --resume JEPA/experiments/exp_004_0_ls20_tu93/checkpoints/step_050000.pt

# ── Override hyperparameters ───────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.train \
    --max-steps 1000000 --batch-size 128

# ── Evaluation (per env) ───────────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.eval \
    --checkpoint checkpoints/step_050000.pt --env ls20 --episodes 50
uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.eval \
    --checkpoint checkpoints/step_050000.pt --env tu93 --episodes 50

# ── Debug runner (standalone, choose env) ─────────────────────────────────
uv run python JEPA/experiments/exp_004_0_ls20_tu93/debug_runner.py \
    --checkpoint .../checkpoints/step_050000.pt --env ls20

# ── Dashboard ──────────────────────────────────────────────────────────────
uv run python JEPA/dashboard/server.py
# Open http://localhost:8787
# Select "exp_004_0_ls20_tu93" in the experiment dropdown.
# The panel.js plugin exposes per-env tabs plus a cross-env gradient-cosine pane.
```

---

## 12. On the meaning of gradient cosine similarity (discussion)

This section documents what the cross-env gradient cosine metric in Section 9 of `metrics.md` does and does not measure, because the metric is one of the headline signals in this experiment and its interpretation is more subtle than it first appears.

### 12.1 What we compute today (verified in code)

The function used today is [exp_003_4/monitors/gradients.py:49-63](../exp_003_4_no_resampler_self_attn/monitors/gradients.py#L49-L63):

```python
def grad_cosine(g_a, g_b):
    a, b = [], []
    for ga, gb in zip(g_a, g_b):
        if ga is None or gb is None: continue
        a.append(ga.detach().reshape(-1))
        b.append(gb.detach().reshape(-1))
    if not a: return float("nan")
    va, vb = torch.cat(a), torch.cat(b)
    if va.norm() < 1e-12 or vb.norm() < 1e-12: return float("nan")
    return float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())
```

That is: for each parameter `p` in a module group, take `g_a[p]` (gradient from loss A — in our case the LS20 half of the JEPA batch) and `g_b[p]` (gradient from loss B — the TU93 half), flatten each tensor to 1-D, concatenate everything into two long vectors `v_a, v_b ∈ R^N` (where `N` = total parameter count of the group), and return `cos(v_a, v_b) ∈ [-1, 1]`. Every scalar partial derivative `∂L/∂θ_i` is one entry of the flattened vector.

### 12.2 What this captures

- The aggregate direction-alignment of the two gradient vectors in the module's full parameter space.
- **`cos = +1`** ⇒ `v_b = α · v_a` for some `α > 0` — the two halves of the batch recommend proportional updates *everywhere*.
- **`cos = −1`** ⇒ `v_b = α · v_a` for some `α < 0` — exact opposition everywhere.
- **`cos = 0`** ⇒ the two vectors are orthogonal in `R^N`.

### 12.3 What this loses (the compression from N dimensions to one scalar)

The reader's intuition asked: "if cosine is 0, does that mean each entry in one vector cancels its counterpart in the other?" The answer is **no, not in general**.

- *Orthogonal does not mean per-entry cancellation.* The two vectors `(1, 1, 0, …, 0)` and `(1, −1, 0, …, 0)` have cosine 0. Their dot product is zero not because each entry has opposing signs, but because the positive and negative entry-wise products sum to zero across coordinates. Per-entry sign agreement is 50%, not 0%.
- *Per-entry sign disagreement is not the same as low cosine.* Two vectors that agree in sign on 99% of entries can still have cosine near 0 if the disagreeing 1% have enormous magnitudes that dominate the dot product. Conversely, two vectors with 50/50 sign agreement on small entries can have cosine near 1 if a few high-magnitude entries are well-aligned.
- *A module-global cosine averages many sub-modules.* The patch SA stack and the Perceiver each contain multiple `nn.Linear` weight matrices and biases. Within `gcossim_perceiver_<…>` a single scalar mixes the cross-attention `q_proj`, `k_proj`, `v_proj`, `out_proj` plus the FFN layers across two rounds. If `q_proj` is in conflict between the two envs but the FFN is well-aligned, the module cosine can hide the local conflict.

So when we read `gcossim_jepa_ls20_vs_tu93_perc_cross_r0 = 0`, the right interpretation is "the two halves' gradient sums are perpendicular in this module's parameter space", **not** "the two halves want to push each individual parameter in opposite directions." The latter statement is closer to **sign-disagreement fraction**, which is why we also log it (Section 10 / §8.3).

### 12.4 Practical reading

To decide whether the two games are constructively or destructively coupled on a given module `k`, look at all three signals together:

- **`gcossim_jepa_ls20_vs_tu93_<k>`** — coarse summary of overall direction-alignment in the module's full parameter space.
- **`gsign_disagree_frac_<k>`** and **`gsign_disagree_frac_magweighted_<k>`** — per-element sign-disagreement, capturing the kind of "pull in opposite directions on this individual parameter" intuition.
- **`gcossim_perlayer_dist_<k>`** — distribution of cosines across the sub-layers inside `k`, exposing localised conflict that the module-level cosine averages away.

A scenario diagnosis:

| `gcossim` | `gsign_disagree_frac` | Most likely interpretation |
|-----------|------------------------|----------------------------|
| ≈ +1 | ≈ 0 (close to 0) | Two games strongly reinforce each other — shared module is doing useful joint work. |
| ≈ 0 | ≈ 0.5 | Two games' gradients are uncorrelated; the module is effectively averaging over two unrelated tasks. May not be using its capacity efficiently. |
| ≈ 0 | ≈ 0 | Two games agree on signs but the *magnitudes* are misaligned across coordinates. The module is being driven in compatible-but-uneven directions. |
| ≈ 0 | ≈ 1 | Strong per-element conflict but with magnitudes that happen to cancel in the global dot product. Cross-env interference; this is the worst case for a shared module. |
| ≈ −1 | ≈ 1 | Exact opposition everywhere — the two games want to push the module the opposite way. Strongest possible cross-env interference signal. |

None of these states is inherently catastrophic — a `gcossim ≈ 0` does not mean training will fail — but they map to qualitatively different stories about what the shared encoder is learning, which is exactly what this experiment is trying to find out.

---

## 13. Expected Outcomes and Open Risks

### 13.1 Expected outcomes

- **High confidence:** With state-only reward, the corner-attractor pathology from exp_003_2 does not return in either game. Per-env tile-coverage curves should rise. (Same expectation as exp_003_3, applied per env.)
- **Medium confidence:** The shared encoder remains non-collapsed by exp_003_4's standards — `eff_rank` stays around 3–4, latent pairwise cosine stays comfortably below 1, `ht_htp1_cossim_eval` stays in roughly the 0.4–0.7 band. Cross-env training is asking more of the encoder, but the action predictor's anti-collapse signal is doubled across envs.
- **Low confidence:** Either policy completes Level 1 of its game. With pure state-novelty reward, no terminal bonus, and no temporal credit assignment, completion is expected at or near 0% in both games (same baseline as exp_003_3 in single-env). The value of this experiment is the cross-env representation-learning signal, not policy performance.

### 13.2 Risks

- **R1 — TU93 life-end heuristic wrong.** If the §7 probe converges on a wrong heuristic, every TU93 episode's last K transitions could be either degenerate (`is_end_of_life_tu93` triggers too early) or missing (triggers too late), contaminating the TU93 buffer. *Mitigation:* the probe is gated as a prerequisite; the diagnostic frames are saved to `runs/probe_tu93/` for hand-verification before training starts.
- **R2 — Cross-env reward-magnitude imbalance.** LS20's `state_err` distribution may differ substantially from TU93's, biasing the JEPA loss toward whichever env has the larger error magnitude. *Mitigation:* per-env loss breakdowns are logged (`L_next_state_pred_ls20`, `_tu93`); if the imbalance is severe, switch the JEPA combine from `0.5 * (L_state_ls20 + L_state_tu93)` to a normalised form (e.g. divide each env's contribution by its EMA).
- **R3 — Shared-encoder representation collision.** The Perceiver may carve the latent space into two near-disjoint regions per game, behaving like two encoders that happen to share weights but produce uncoupled representations. *Mitigation:* monitor `eff_rank` across mixed-env eval batches and the cross-env grad cosine in Section 9 — a persistently negative cosine across the encoder is the early-warning signal.
- **R4 — Round-robin granularity too coarse.** Running one full episode per env may produce long stretches of in-distribution drift between JEPA updates. *Mitigation:* JEPA updates already trigger every 5 global env steps regardless of which env is rolling, so the encoder is updated mid-episode. If wall-clock allows, an alternative is per-step env switching, but it complicates recurrent-state tracking.
- **R5 — Buffer asymmetry by episode length.** If TU93 episodes are substantially longer than LS20's ~130 steps, the TU93 buffer fills faster, which would be fine under balanced sampling but means TU93's per-transition recency horizon is shorter. *Mitigation:* monitor `episodes_completed_ls20` vs `_tu93` and `buffer_fill_<env>`; if the ratio is extreme, consider per-env capacities tuned to the observed episode-length ratio.

### 13.3 What a "successful" run looks like

- Per-env policies reach ~uniform-random-baseline performance in both games (i.e. the encoder doesn't break the policy) and ideally exceed it on tile coverage.
- `gcossim_jepa_ls20_vs_tu93_*` on the encoder modules is **not** persistently negative.
- `eff_rank` of the shared latent space stays in roughly `[2.5, 4.0]` across training.
- `gsign_disagree_frac_*` on encoder modules is in roughly `[0.3, 0.5]` (mostly agreeing signs).

A run that fails any of these is still informative — it answers the central question (with negative evidence) and informs the next experiment.

---

## 14. Comparison with prior experiments

| Dimension | exp_003_2 | exp_003_3 | exp_003_4 | **exp_004 (this)** |
|-----------|-----------|-----------|-----------|--------------------|
| **Games** | LS20 | LS20 | LS20 | **LS20 + TU93** |
| **Encoder weights** | Single | Single | Single | **Single, shared across games** |
| **Resampler self-attn among latents** | Yes | Yes | No | **No** (from exp_003_4) |
| **Action predictor in JEPA loss** | Yes | Yes | Yes | **Yes** (from exp_003_2) |
| **Action predictor in reward** | Yes (`0.5`) | **No** (`0.0`) | Yes (`0.5`) | **No** (`0.0`, from exp_003_3) |
| **Reward weighting** | `0.5/0.5` | `1.0/0.0` | `0.5/0.5` | **`1.0/0.0`** |
| **JEPA loss weighting** | `0.5/0.5` | `0.5/0.5` | `0.5/0.5` | **`0.5/0.5`** |
| **EMA target encoder** | None | None | None | **None** |
| **Buffer target field** | `next_frame` | `next_frame` | `next_frame` | `next_frame` |
| **Buffer count / capacity** | 1 × 50K | 1 × 50K | 1 × 50K | **2 × 25K (per env)** |
| **Buffer sampling** | Uniform | Uniform | Uniform | **Balanced per env** |
| **Action embedding count** | 1 | 1 | 1 | **2 (per env)** |
| **Policy count** | 1 | 1 | 1 | **2 (per env)** |
| **Encoder gradient paths per JEPA step** | 3 | 3 | 3 | **6 (3 per env)** |
| **Rollout dying-step gated from policy buf + health logs** | No | No | No | **Yes** (§3.3) |
| **Approx. encoder params** | ~966k | ~966k | ~924k | **~924k** (encoder weights unchanged from exp_003_4) |

---

## 15. Acceptance Tests (run after implementation)

1. **TU93 probe is decisive.** `probe_tu93_lives.py` produces a written conclusion and supporting frames; §7 of this card is updated accordingly.
2. **Static checks.**
   - `grep -n 'reward_w_action' .../config.py` returns `0.0`.
   - `grep -n 'n_actions' .../config.py` returns `4`.
   - `grep -n 'game_ids' .../config.py` returns the two-element list.
   - `grep -n 'self_attn_among_latents' .../models/encoder.py` returns no matches (or returns `False` for an explicit flag).
3. **Smoke run.** From repo root:
   ```
   uv run python -m JEPA.experiments.exp_004_0_ls20_tu93.train --max-steps 3000
   ```
   Completes without errors; both buffers exceed `min_buffer_size` within warmup; per-env episode counts grow comparably.
4. **Balanced JEPA batches.** Add a runtime assertion in the JEPA update path that each batch contains exactly `batch_size // 2` transitions from each env. The assertion should never fire during training; if it does, balanced sampling is broken.
5. **Dying-step gating.** Replay the buffer dump after a smoke run. For each transition, the `next_frame` is never the reset state of the *next* episode. Add a one-time invariant check in `sample_balanced` if cheap.
6. **Metric surface.** Dashboard shows: per-env loss curves, per-env policy entropy, per-env tile coverage, and the cross-env grad-cosine panel updating every `grad_decomp_freq = 25` JEPA updates.
7. **Per-env eval.** `eval.py --env ls20 --episodes 50` and `eval.py --env tu93 --episodes 50` both complete with the same checkpoint and produce a level-completion-rate + episode-length distribution per env.
8. **No-regression check (sanity).** Run `exp_003_4` for the same number of env steps. The shared-encoder collapse metrics in this experiment (`eff_rank`, `latent_pairwise_cossim_buf`, `ht_htp1_cossim_eval`) should be qualitatively comparable to exp_003_4 at the same step count — if they are drastically worse, multi-env training is breaking representational health and should be reported back rather than patched ad-hoc.

---

## 16. Out of Scope (deliberately deferred)

These are **not** in this experiment. They are listed so the implementing agent knows not to add them on its own initiative.

1. More than two games. Once we know whether two-game shared training works, scaling to three (RE86 / G50T) requires solving the heterogeneous-action-space problem (5-action games) and is a separate experiment.
2. Per-env predictors. The state and action predictors are shared on purpose — we are testing a shared world model, not two co-trained world models.
3. Per-env policy update schedule. Both policies share the same `policy_update_freq`. If they end up wildly imbalanced in episodes-completed, revisit.
4. Cross-env policy transfer / fine-tuning. Each policy is trained from scratch alongside the shared encoder; no transfer step.
5. PPO / value heads / GAE. REINFORCE with EMA baseline retained per env.
6. Goal-conditioned policy, HER, k-NN-novelty bonus, RND-style fixed-target curiosity, large terminal bonus on `levels_completed >= 1`. Same deferral list as exp_003_3.
7. Back-porting the §3.3 bug fix to exp_003_2 / exp_003_3 / exp_003_4. Their existing checkpoints are still meaningful comparison baselines and should not be re-run with subtly different rollout statistics.

Strategy for which of these to pursue next is decided in the session that produced this card; the implementing agent should not pre-empt those choices.
