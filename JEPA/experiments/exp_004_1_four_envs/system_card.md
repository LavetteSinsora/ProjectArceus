# System Card — exp_004_1_four_envs

| Field | Value |
|-------|-------|
| **Experiment ID** | `exp_004_1_four_envs` |
| **Status** | Implementation complete; RE86 + G50T life-end probes required before training |
| **Parent experiment** | `exp_004_0_ls20_tu93` (two-env shared-encoder JEPA) |
| **Games** | LS20 (`ls20-9607627b`), TU93 (`tu93-0768757b`), RE86 (`re86-8af5384d`), G50T (`g50t-5849a774`) |
| **Action spaces** | LS20, TU93 → 4 actions; RE86, G50T → 5 actions (max-size single 5-way head + per-env masking) |
| **Reward** | Intrinsic curiosity, **state-prediction error only**, computed independently per environment |

---

## 1. One-Paragraph Summary

This experiment lifts `exp_004_0_ls20_tu93`'s deferral of "more than two games / heterogeneous action spaces" (§16 of that card) and jointly trains a single shared encoder + Perceiver Resampler + state-predictor + action-predictor on **all four ARC-AGI-3 envs** registered in `JEPA/shared/env_wrapper.py`. The architectural design is identical to exp_004_0 — no EMA target encoder (action predictor is the anti-collapse mechanism), state-only intrinsic reward, cross-attention-only Perceiver Resampler — and the new axis is two-fold: (a) the four-way generalisation of the cross-env joint training set-up (per-env action embeddings, per-env policies, per-env replay buffers, **`batch_size // 4 = 16`** balanced JEPA samples from each), and (b) a single max-size action head: the action predictor outputs **5** logits and per-env `Embedding(5, 32)` action embeddings and per-env `PolicyNetwork(out=5)` are used uniformly, with the wrappers' `available_actions` masking preventing the 4-action envs (LS20, TU93) from ever sampling index 4. The headline question is whether a shared encoder trained on **four** heterogeneous games learns transition-relevant features that cohere across all games, or whether the gradients fight each other at the 4-way scale.

---

## 2. Architecture

### 2.1 Overview

```
                LS20 frame ──┐
                TU93 frame ──┤   (one env per rollout episode;
                RE86 frame ──┤    JEPA batch = batch_size // 4 from
                G50T frame ──┘    each per-env buffer, equal share)
                              │
                              ▼
                  ┌────────────────────────────────┐   SHARED
                  │  Stage 1: Patch Encoder        │
                  │  color_embed(16→4)             │
                  │  patch_proj(1024→128)          │
                  │  SA-Block 1  (4 heads, RoPE)   │
                  │  SA-Block 2  (4 heads, RoPE)   │
                  │  sa_norm (LayerNorm)           │
                  └────────────────────────────────┘
                              │
                              ▼
                  ┌────────────────────────────────┐   SHARED
                  │  Stage 2: Perceiver Resampler  │
                  │  Round 0: Cross-Attn only      │
                  │  Round 1: Cross-Attn only      │
                  │  → h_t (B, 4, 128)             │
                  └────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────────┐
        ▼                     ▼                         ▼
┌──────────────┐    ┌──────────────────┐     ┌──────────────────────┐
│ State        │    │ Action Predictor │     │ Policy (PER-ENV × 4) │
│ Predictor    │    │ SHARED (5-way)   │     │   policy_ls20        │
│ SHARED       │    │ (h_t, h_{t+1})   │     │   policy_tu93        │
│ (h_t, a_emb) │    │   → logits[5]    │     │   policy_re86        │
│ → h̃_{t+1}    │    │                  │     │   policy_g50t        │
└──────┬───────┘    └──────────────────┘     └──────────────────────┘
       │
       │ uses
       ▼
┌────────────────────────────────────┐
│  Action Embedding (PER-ENV × 4)    │
│  action_embed_<env>: Embedding(5,32)│
└────────────────────────────────────┘
```

What is shared and what is per-environment:

| Module | Shared / Per-env | Notes |
|--------|------------------|-------|
| Patch encoder (color embed, patch proj, SA blocks, SA norm) | **Shared** | One set of weights trained on all four envs |
| Perceiver Resampler (cross-attn rounds, output norm, placeholders) | **Shared** | Cross-attention only — no self-attn among latents |
| State predictor (4 per-latent MLPs + time embed) | **Shared** | Takes env-specific action embedding as input |
| Action predictor (Linear → GELU → Linear) | **Shared, 5-way** | Single 5-way head. 4-action envs' CE targets are in {0..3}; 5-action envs in {0..4}. |
| Action embedding | **Per-env (×4)** | Four `Embedding(5, 32)` tables: `action_embed_<ls20|tu93|re86|g50t>` |
| Policy network | **Per-env (×4)** | Four independent `PolicyNetwork(out=5)` + four REINFORCE baselines |
| Replay buffer (`NextFrameLatentBuffer`) | **Per-env (×4)** | Capacity 15K each |
| Policy buffer | **Per-env (×4)** | Capacity 64 each |

### 2.2 Stage 1 — Patch Encoder

Identical to exp_004_0 / exp_003_4 — game-agnostic operations on (64, 64) colour-indexed frames. See `exp_004_0/system_card.md §2.2` for the step-by-step. All four games render to the same 16-colour palette at 64×64 with grid-aligned sprites, so the shared encoder is a fair test of cross-game low-level visual feature learning.

### 2.3 Stage 2 — Perceiver Resampler (cross-attention only)

Identical to exp_004_0 / exp_003_4. Shared across all four games. The **placeholder vectors** are a single learned `nn.Parameter` of shape `(4, 128)` — all four games' episode starts begin from the same placeholders.

**Risks specific to this experiment.** Two risks from exp_004_0 compound at the 4-env scale:

1. *Inter-latent collapse* (unchanged risk from exp_003_4): 4 latents differ only via 4 distinct placeholders and 4 distinct cross-attn query projections — no SA among latents. Monitor `latent_eff_rank`, `latent_pairwise_cossim_buf`.
2. *Cross-env representation collision* (amplified): four unrelated games now share the Perceiver. The resampler may carve four near-disjoint sub-regions of the 4-latent space, behaving like four encoders that share weights. Detected via cross-env grad cosine — see §8.

### 2.4 State Predictor (flow matching) — shared, per-env action embedding input

Same flow-matching predictor as exp_004_0 — four per-latent MLPs plus a shared sinusoidal time embedding, integrated by Euler with `N = 3` steps. Predictor weights are shared across all four games. Game-specific information enters only via the per-env action embedding.

`a_emb_env` has shape `(B, 32)` regardless of env. Per-env tables let the predictor learn the latent-space transition operator for each game separately while sharing all the heavy compute.

### 2.5 Action Predictor — shared, 5-way head

Single MLP with a single 5-way output head, used for all four games:

```
z      = concat(flatten(h_t), flatten(h_{t+1}))     # (B, 1024)
logits = Linear(512, 5)(GELU(Linear(1024, 512)(z)))
L_action = cross_entropy(logits, a_t)               # a_t ∈ {0,1,2,3}     for ls20/tu93
                                                    # a_t ∈ {0,1,2,3,4}   for re86/g50t
```

No detach on either `h_t` or `h_{t+1}`. Gradient flows through both endpoints into the shared encoder — the anti-collapse signal that replaces the EMA target encoder.

**Why a single 5-way head works.** Both 5-action games (RE86, G50T) need the extra class; both 4-action games (LS20, TU93) never use it because `available_actions` returns only the four legal `GameAction.value` ints and the wrappers' `_ACTIONS` list has length 4. The policy's masking ensures `action_idx ∈ {0..3}` always for LS20/TU93, so the CE target is always in-range for the 5-way softmax. The class-4 logit on LS20/TU93 batches is pushed *down* by the softmax (it's never the target), which is the desired behaviour — see R6 in §13.

### 2.6 Action Embedding — per environment (×4)

Four separate `nn.Embedding(5, 32)` tables:

```python
self.action_embed = nn.ModuleDict({
    "ls20": nn.Embedding(5, 32),
    "tu93": nn.Embedding(5, 32),
    "re86": nn.Embedding(5, 32),
    "g50t": nn.Embedding(5, 32),
})
```

For LS20 / TU93, row 4 is unused at training time — `action_idx` is never 4 because `available_actions` excludes it. The unused row is therefore only touched by weight decay (state_predictor_wd = 0.01) and stays near initialisation.

### 2.7 Policy Networks — per environment (×4)

Four independent `PolicyNetwork(d_model=128, n_actions=5, hidden=512)` instances with independent REINFORCE baselines:

```
logits = Linear(512, 5)(GELU(Linear(512, 512)(flatten(h_t))))
# available-action masking: set logits[unavailable] = −∞
a_t ~ Categorical(softmax(logits))
```

For 4-action envs, the `available_actions` list contains only the four legal `GameAction.value` ints; `policy.act()` masks the 5th logit (and any blocked legal action) to −∞ before softmax. **No 4-action env can ever sample `action_idx = 4`** — a runtime assertion in `train.py:run_one_episode` enforces this.

---

## 3. Key Changes

### 3.1 Changes from exp_004_0

| Dimension | exp_004_0 | exp_004_1 (this) | Motivation |
|-----------|-----------|------------------|------------|
| **Games trained on** | LS20 + TU93 | LS20 + TU93 + RE86 + G50T | Scale cross-env shared-encoder test to all 4 ARC-AGI-3 envs |
| **Action predictor head width** | 4 | **5** | RE86 / G50T have 5 actions; max-size head + masking is simplest extension |
| **Action embeddings** | 2 × `Embedding(4, 32)` | **4 × `Embedding(5, 32)`** | Per-env tables sized to max action space; 4-action envs leave row 4 unused |
| **Policy networks** | 2 × `PolicyNetwork(out=4)` | **4 × `PolicyNetwork(out=5)`** | Per-env policies sized to max action space |
| **Replay buffers** | 2 × 25K (= 50K) | **4 × 15K (= 60K)** | Per-env recency horizon stays generous; total memory stays similar |
| **Balanced JEPA batch** | `batch_size // 2 = 32` per env | **`batch_size // 4 = 16` per env** | Equal-share sampling across all 4 envs |
| **Encoder gradient paths per JEPA step** | 6 (2 envs × 3 paths) | **12 (4 envs × 3 paths)** | More accumulated gradient into the shared Perceiver per step |
| **Rollout cycle** | Round-robin over 2 envs | **Round-robin over 4 envs** | One full episode per env per cycle of 4 |
| **Warm-up random action** | `randint(0, cfg.n_actions=4)` | `randint(0, env.n_actions)` — per env | Prevents 4-action envs from sampling index 4 during warmup |
| **Life-end detection** | `is_end_of_life` dispatches on 2 envs | Dispatches on 4 envs | New `is_end_of_life_re86`, `is_end_of_life_g50t` (default to `is_terminal`; see §7) |

### 3.2 Heterogeneous action space — the masking design

The cleanest extension under the exp_004_0 design is to size every action-indexed module to **`max(n_actions) = 5`** and rely on the environment wrappers' `available_actions` to constrain sampling:

- **Policy time** — `PolicyNetwork.act(h, avail)` sets `logits[i] = −∞` for any `i` not in `avail`. For LS20/TU93 the wrapper's `_ACTIONS` list has length 4, so action index 4 is *never* in `avail` (it doesn't even map to a `GameAction`); the policy mask sets `logits[4] = −∞` unconditionally for these envs.
- **Warm-up random sampling** — `np.random.randint(0, env.n_actions)` per env (was `cfg.n_actions` global). LS20/TU93 sample uniformly from `{0, 1, 2, 3}`; RE86/G50T from `{0, 1, 2, 3, 4}`.
- **Replay buffer** — stores `action_idx`, an integer. For LS20/TU93 entries it's always `< 4`; for RE86/G50T it can be 4. The 5-way action predictor handles both without code paths.
- **L_action CE** — `cross_entropy(logits, action_idx)` accepts targets in `{0, ..., 4}`. The 5-way softmax normalises over all 5 logits regardless of which env produced the batch. On LS20/TU93 batches the class-4 logit is implicitly pushed down by softmax — see R6 in §13.

A safety assertion in `run_one_episode` validates `action_idx < env.n_actions` per env per step; if it ever fires, the available-actions masking is broken.

### 3.3 Bug fix in the rollout loop (inherited from exp_004_0 §3.3)

Unchanged from exp_004_0. The dying step is excluded from the replay buffer (`ep_transitions[:-1]`), the policy buffer (`if not life_end`), and rollout health metrics (`if not life_end`). See exp_004_0/system_card.md §3.3 for the rationale.

### 3.4 Multi-environment data-collection structure

`itertools.cycle(cfg.env_names)` over `("ls20", "tu93", "re86", "g50t")`. One full episode in one env at a time, then advance to the next env. JEPA updates and policy updates are gated by global env-step counts (`update_freq = 5`, `policy_update_freq = 64`).

---

## 4. Training

### 4.1 Data Collection (per episode in a single env)

Inside `run_one_episode(env, env_name, ...)`:

1. **Reset env** if at episode start. Initialise recurrent state from the shared learned placeholders.
2. **Encode frame** under `torch.no_grad()`.
3. **Select action**: during warm-up, `action_idx = np.random.randint(0, env.n_actions)`; afterwards `policy[env_name].act(h_current, env.available_actions)`. Assert `action_idx < env.n_actions`.
4. **Step env**. Compute `life_end = is_end_of_life(env_name, frame_np, next_np, is_terminal)`.
5. **If `not life_end`**: compute curiosity reward = `clamp(state_err, max=50.0)` under `no_grad`; append to per-env health logs (`reward_state_component_<env>`, `reward_action_component_<env>`, `reward_total_<env>`, etc.) and to `policy_buf[env_name]`.
6. **Append to `ep_transitions`** unconditionally (the dying step is the tail of the slice).
7. **Advance recurrent state**.
8. **If `life_end`**: flush `ep_transitions[:-1]` into `latent_bufs[env_name]`; mark episode end.

### 4.2 JEPA Training Update (balanced cross-env batch)

Every `update_freq = 5` global env steps, once **all four** per-env buffers have ≥ `min_buffer_size`:

```python
n_envs = len(env_names)                                  # 4
half   = cfg.batch_size // n_envs                        # 64 // 4 = 16
assert cfg.batch_size % n_envs == 0
batches = {e: latent_bufs[e].sample(half, device) for e in env_names}

for e in env_names:
    b = batches[e]
    h_t_e,   _, _ = encoder(b.frames,      b.h_queries.detach())
    h_tp1_e, _, _ = encoder(b.next_frames, h_t_e.detach())
    a_emb_e       = action_embeds[e](b.actions)

    L_state_per_env[e],  _    = state_predictor.compute_loss(h_t_e, h_tp1_e.detach(), a_emb_e)
    L_action_per_env[e]       = F.cross_entropy(action_predictor(h_t_e, h_tp1_e), b.actions)

L_state  = sum(L_state_per_env.values())  / n_envs       # mean over 4 envs
L_action = sum(L_action_per_env.values()) / n_envs
L        = cfg.lambda_state * L_state + cfg.lambda_action * L_action     # 0.5 / 0.5

L.backward()
clip_grad_norm_( <all shared params + all per-env action embeds> , cfg.grad_clip_model )
enc_opt.step(); state_pred_opt.step(); action_pred_opt.step()
```

**Encoder gradient paths per JEPA step.** Four envs × three paths = **12** distinct paths into the encoder per JEPA step. The Perceiver sees gradient from all 12 (eight encoder calls per step — two per env, for `h_t` and `h_{t+1}`).

| # | Source | Env | Path into encoder | Detached? |
|---|--------|-----|-------------------|-----------|
| 1, 2, 3   | `L_state` / `L_action` (via h_t) / `L_action` (via h_{t+1}) | LS20 | three paths | no |
| 4, 5, 6   | same three | TU93 | three paths | no |
| 7, 8, 9   | same three | RE86 | three paths | no |
| 10, 11, 12| same three | G50T | three paths | no |

Per-env batch contributions are 4× smaller than exp_004_0's, partially offsetting the 4× path-count increase into the shared Perceiver. We keep `perceiver_lr = 5e-5` inherited from exp_003_4 — gate adjustment on observed grad-clip saturation during the smoke run.

### 4.3 Policy Update (per env, independent)

Each `policy_buf[env_name]` empties when full (capacity 64) and the corresponding policy is updated using only its own on-policy data — independently of the other three. See `exp_004_0/system_card.md §4.3` for the REINFORCE update; nothing changed.

### 4.4 Per-Component Optimizer Design

| Optimizer | Parameters | LR | Weight Decay |
|-----------|-----------|-----|------|
| `enc_opt` group 1 | color_embed, patch_proj, sa_blocks, sa_norm | 1e-4 | 0.01 |
| `enc_opt` group 2 | perceiver (all rounds + output_norm + placeholders) | 5e-5 | 0.01 |
| `state_pred_opt` | state_predictor (MLPs + time_embed) + **all four** `action_embed_<env>` | 1e-4 | 0.01 |
| `action_pred_opt` | action_predictor (shared 5-way head) | 1e-4 | 0.01 |
| `pol_opt_<env>` | `policy_<env>` (×4 instances) | 1e-4 | 0 |

All four per-env action embeddings go into `state_pred_opt` because their gradients flow through `L_state` only.

### 4.5 Replay Buffers (per env)

Four independent `NextFrameLatentBuffer` instances:

| Buffer | Capacity | Stored fields | Sampling |
|--------|----------|---------------|----------|
| `latent_bufs[<env>]` (×4) | 15,000 each | `(frame_t, h_{t-1}, a_t, next_frame_t)` | Uniform |

Total memory ≈ 600 MB (vs exp_004_0's 510 MB at 2 × 25K).

Per-env episode lengths (probe-confirmed; all four envs are step-bounded, not skill-bounded):

| Env | Steps / episode | What bounds the episode | Horizon (15K transitions ÷ steps/ep) |
|-----|-----------------|-------------------------|--------------------------------------|
| LS20 | ~130 | Energy = 42 per life, **decrements 1 per action**; 3 lives → ~126–130 actions | ~115 episodes |
| TU93 | 50 | Deterministic step cap | 300 episodes |
| RE86 | 100 | Step bar, decrements per action (probe 2026-05-14) | 150 episodes |
| G50T | 130 | Timer strip, decrements per action (probe 2026-05-14) | ~115 episodes |

**All four envs are step-counter-bounded** — episode length is fixed by an action-count budget, *not* by policy skill. LS20's "energy" decrements once per action (every action, not per wall hit), so a better LS20 policy does not lengthen the episode. The spread across envs is ~2.6× (TU93 = 50 vs LS20/G50T = 130) — modest, and balanced JEPA sampling neutralises it for the encoder. All four horizons comfortably exceed what uniform sampling needs to be statistically well-mixed.

### 4.6 Training Schedule

| Phase | Steps (global) | Actions | JEPA update | Policy update |
|-------|----------------|---------|------------|---------------|
| Warmup | 0 – 3,999 | Uniform random per env (sampled from `env.n_actions`) | Every 5 env steps (once **all four** buffers ≥ 512) | No |
| Joint training | ≥ 4,000 | Env-specific policy | Every 5 env steps | Every 64 env steps **per env** |

**Warm-up budget.** `warmup_steps = 4000` global env steps. Under equal round-robin sampling across the four envs this is **≈ 1000 random-exploration steps per env** before that env's policy turns on — enough to seed each per-env buffer well above `min_buffer_size = 512`. We use a single global step counter (not per-env episode counts) for simplicity; the per-env spread is small because all four episode lengths are within ~2.6× of each other.

**Round-robin granularity:** one full episode per env at a time. JEPA updates fire on every 5 global env steps regardless of which env is currently rolling.

**Checkpointing:** every 5,000 global env steps. Each checkpoint contains the shared encoder, the shared predictors, all four action-embedding tables, and all four policies + optimisers + baselines.

---

## 5. Inference (Deployment)

Per-env inference loop (identical structure to exp_004_0 §5.1, with `env_name ∈ {ls20, tu93, re86, g50t}`):

```python
def run_episode(env_name):
    env       = envs[env_name]
    policy    = policies[env_name]

    frame_t = env.reset()
    h_t = encoder.perceiver.get_initial_queries(1, device)
    while True:
        h_current, _, _ = encoder(frame_t.unsqueeze(0).to(device), h_t)
        action_idx, _, _ = policy.act(h_current.squeeze(0), env.available_actions)
        assert action_idx < env.n_actions
        next_frame, is_terminal = env.step(action_idx)
        h_t = h_current
        if is_terminal: break
        frame_t = next_frame
```

All four envs share the encoder forward; only the policy is env-specific at inference. (The action-embedding lookup is consumed only during training inside the state predictor.)

---

## 6. Reward Function

```
r_t = clamp( 1.0 · MSE(h̃_{t+1}, h_{t+1})  +  0.0 · CE(p_pred, one_hot(a_t)) ,
             max = 50.0 )
    = clamp( state_err , max = 50.0 )
```

Inherited from exp_004_0 / exp_003_3. Per-env breakdown: `reward_state_component_<env>` and `reward_total_<env>` logged separately for the four envs. The action-CE term is logged at every step for diagnostics but is multiplied by zero and does not enter `curiosity_reward`.

---

## 7. Life-End Detection (per env)

### 7.1 LS20 (3 lives, energy-bar bounded)

Unchanged from exp_004_0 / earlier experiments. `is_end_of_life_ls20` = `is_terminal OR (count_lives_ls20(next_frame) < count_lives_ls20(frame))`. Life counter at row 61, cols 55–63, color-8 pixel pattern at offsets `[(1,2), (4,5), (7,8)]`.

### 7.2 TU93 (no intra-game lives; 50-step cap)

Unchanged from exp_004_0 §7 (probe-confirmed). `is_end_of_life_tu93` = `is_terminal`. Random play terminates at exactly 50 steps with `GameState.GAME_OVER`.

### 7.3 RE86 (verdict from `probe_re86_lives.py` — run 2026-05-14)

**No intra-game lives. `is_end_of_life_re86` = `is_terminal` is correct.** Probe: 8 random-policy attempts, all terminated at exactly 100 steps with `GameState.GAME_OVER` (σ = 0). `raw` exposes no `lives`/`hp`/`energy`/`score`/`step_count` attribute — only the standard `state` / `levels_completed` / `available_actions`. The terminal-only pixel scan found exactly **one** stable pixel at (row 63, col 0) across all attempts — that is the tail of the row-63 step bar reaching zero, not a life indicator (row 63 is already masked by the `Re86Env` wrapper). RE86 behaves single-life under random play; every game-over is the only life-end signal.

- Episode length under random policy: **100 steps, deterministic** (step-bar-bounded).
- Buffer implication: each episode contributes 99 transitions to `latent_bufs["re86"]` after the `[:-1]` dying-step slice.

### 7.4 G50T (verdict from `probe_g50t_lives.py` — run 2026-05-14)

**No intra-game lives. `is_end_of_life_g50t` = `is_terminal` is correct.** Probe: 8 random-policy attempts, all terminated at exactly 130 steps with `GameState.GAME_OVER` (σ = 0). `raw` exposes no `lives`/`hp`/`energy`/`score`/`step_count` attribute. The terminal-only pixel scan found **zero** stable pixels — because the row-63 timer strip scrolls continuously, no pixel region is static during the episode and then changes only near terminal. G50T behaves single-life under random play; every game-over is the only life-end signal.

- Episode length under random policy: **130 steps, deterministic** (timer-bounded).
- Buffer implication: each episode contributes 129 transitions to `latent_bufs["g50t"]` after the `[:-1]` dying-step slice.

### 7.5 Dispatcher

`reward_shaping.py` exposes a single dispatcher used by `train.py` and `eval.py`:

```python
def is_end_of_life(env_name, frame, next_frame, is_terminal):
    if env_name == "ls20": return is_end_of_life_ls20(...)
    if env_name == "tu93": return is_end_of_life_tu93(...)
    if env_name == "re86": return is_end_of_life_re86(...)
    if env_name == "g50t": return is_end_of_life_g50t(...)
    raise ValueError(...)
```

---

## 8. Health Monitoring

The single-env metric set is inherited via exp_004_0 from exp_003_4. All sections 1–7 metrics about the shared modules transfer directly. Per-env breakdowns and cross-env diagnostics extend mechanically from 2 envs to 4. See `metrics.md` for the full enumeration.

### 8.1 Per-environment basics (sec5 / sec6 — extends exp_004_0 §8.1)

For each `env ∈ {ls20, tu93, re86, g50t}`, suffix:

- `L_state_<env>`, `L_action_<env>`
- `reward_state_component_<env>`, `reward_action_component_<env>`, `reward_total_<env>`
- `episode_length_<env>`, `completion_rate_<env>`
- `policy_entropy_<env>`
- `pol_loss_<env>`

The shared-encoder metrics (`latent_eff_rank`, `latent_pairwise_cossim_*`, `ht_htp1_cossim_*`, `latent_norm_per_token`, etc.) remain single-valued — they describe the shared encoder regardless of which env produced the batch.

### 8.2 Cross-env gradient interference (extends exp_004_0 §8.2 from 2-way to 4-way)

With 4 envs the pairwise count is C(4, 2) = 6. To keep the metric surface compact:

- **Summary scalars (default, on every probe cycle)** per shared-module subset `k`:
  - `gcossim_avg_pairs_<k>` — mean of the 6 pairwise cosines
  - `gcossim_min_pairs_<k>` — minimum (worst-case interference)
- **Per-pair breakdowns** (verbose-flag gated): six `gcossim_<envA>_vs_<envB>_<k>` for `(envA, envB)` in lexicographic order of `env_names`.

Reuses `grad_cosine()` and `compute_source_decomposition()` from `exp_003_4/monitors/gradients.py:49-187`. Same cadence as exp_004_0 (`grad_decomp_freq = 25`).

**Why summary + worst-case rather than just the mean.** A single negative pair can drag the mean toward 0 even when the other five pairs are healthy. `gcossim_min_pairs_<k>` surfaces it. The full 6-pair breakdown is available behind the verbose flag for triage.

### 8.3 Beyond cosine — per-element disagreement (4-way generalisation of exp_004_0 §8.3)

Per shared-module subset `k`, log:

- `gsign_disagree_frac_avg_pairs_<k>` — mean of the 6 pairwise sign-disagreement fractions
- `gsign_disagree_frac_max_pairs_<k>` — maximum (worst-case)
- `gsign_disagree_frac_magweighted_avg_pairs_<k>` — magnitude-weighted version, average
- `gcossim_perlayer_dist_<k>` — histogram of per-layer cosines averaged over the 6 pairs

Interpretation table is identical to exp_004_0 §8.3 / §12.4 — apply per pair.

### 8.4 Shared-encoder collapse monitoring (unchanged)

The collapse-monitoring metrics from exp_003_4 — `latent_eff_rank`, `latent_pairwise_cossim_buf`, `latent_pairwise_cossim_t{1,10,20}`, `ht_htp1_cossim_*`, `H1_HT_cossim`, `latent_norm_per_token`, `round0_postCA_pairwise_cossim_t1`, `action_pred_entropy_eval` — all remain in place and remain single-valued. Cross-env training at 4-way scale is asking more of the encoder, so this monitoring matters more.

### 8.5 Eval-pass instrumentation

`monitors/eval_pass.py` runs an evaluation rollout under hooks every `eval_freq` env steps. In this experiment it runs **once per env per eval cycle** (4 rollouts), producing env-suffixed metrics. Module-level metrics that are inherently env-agnostic (e.g. `latent_eff_rank`) report once per cycle on a combined eval pass over a balanced 4-env mini-batch.

---

## 9. Complete Hyperparameter Table

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model dimensions** | | |
| `d_model` | 128 | Encoder / predictor / latent dimension |
| `d_color` | 4 | Colour embedding dimension (16 colours) |
| `n_actions` | **5** | Discrete-action head width (max over envs) |
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
| **Reward weighting (state-only, inherited)** | | |
| `reward_w_state` | 1.0 | Weight on state-prediction error in curiosity reward |
| `reward_w_action` | 0.0 | Weight on action-prediction error (REMOVED from reward) |
| `reward_clamp` | 50.0 | Per-step reward cap |
| **Policy networks (per env, identical architecture)** | | |
| `policy_hidden` | 512 | Policy MLP hidden dimension |
| **Replay buffers (per env, ×4)** | | |
| `buffer_size_per_env` | **15,000** | Per-env `NextFrameLatentBuffer` capacity |
| `min_buffer_size` | 512 | Minimum transitions per env before JEPA training |
| `batch_size` | 64 | JEPA batch size — `batch_size // 4 = 16` per env |
| **Training schedule** | | |
| `update_freq` | 5 | JEPA gradient step every N global env steps |
| `policy_update_freq` | 64 | Per-env policy update every N steps in that env |
| `warmup_steps` | 4,000 | Random exploration before policy trains (≈ 1000 per env under round-robin) |
| `max_steps` | 500,000 | Total training budget (global env steps) |
| **Optimisers** | | |
| `sa_lr` | 1e-4 | LR for patch embed + SA blocks |
| `perceiver_lr` | 5e-5 | LR for Perceiver (half of SA) |
| `state_predictor_lr` | 1e-4 | LR for state predictor + all four action embeddings |
| `action_predictor_lr` | 1e-4 | Action predictor LR |
| `policy_lr` | 1e-4 | Policy Adam LR (per env) |
| `encoder_wd` | 0.01 | Encoder weight decay |
| `state_predictor_wd` | 0.01 | State predictor weight decay |
| `action_predictor_wd` | 0.01 | Action predictor weight decay |
| `grad_clip_model` | 5.0 | Gradient clip for encoder + predictors + all four action embeddings |
| `grad_clip_policy` | 1.0 | Gradient clip per policy |
| **Policy REINFORCE (per env)** | | |
| `policy_entropy_lambda` | 0.10 | Entropy regularisation coefficient |
| `policy_baseline_alpha` | 0.99 | EMA decay for running reward baseline |
| **Cross-env gradient probe** | | |
| `grad_decomp_freq` | 25 | Cross-env gradient cosine / sign-disagreement probe every N JEPA updates |
| **Misc** | | |
| `seed` | 42 | Random seed |
| `game_ids` | `["ls20-9607627b", "tu93-0768757b", "re86-8af5384d", "g50t-5849a774"]` | Game IDs trained on |
| `env_names` | `["ls20", "tu93", "re86", "g50t"]` | Short env names |
| `env_cycle_unit` | `episode` | Round-robin granularity is one full episode per env |

---

## 10. File Layout

```
JEPA/experiments/exp_004_1_four_envs/
├── system_card.md             — this document
├── metrics.md                 — copied from exp_004_0 and extended for 4 envs
├── __init__.py
├── config.py                  — Config (4-tuple game_ids; n_actions=5; buffer_size_per_env=15_000)
├── train.py                   — 4-env round-robin training loop
├── eval.py                    — Per-env eval (--env ls20|tu93|re86|g50t|all)
├── debug_runner.py            — Per-step dashboard data, env-aware (delegates to exp_003_4 runner)
├── reward_shaping.py          — LS20 / TU93 (verified) + RE86 / G50T (default = is_terminal, pending probes)
├── probe_lives_common.py      — Shared life-end probe protocol
├── probe_re86_lives.py        — RE86 life-end probe (gating prerequisite)
├── probe_g50t_lives.py        — G50T life-end probe (gating prerequisite)
├── models/
│   ├── __init__.py            — load_models factory: encoder, state_predictor,
│   │                            action_predictor (5-way), action_embeds (4), policies (4),
│   │                            baselines (4)
├── checkpoints/               — empty at init
└── runs/                      — Per-run dirs with training.log + metrics.jsonl + probe artefacts
```

Shared infrastructure (no changes needed):

- `JEPA/shared/buffer.py` — `NextFrameLatentBuffer` instantiated 4×, no code change.
- `JEPA/shared/env_wrapper.py` — all four wrappers (`LS20Env`, `Tu93Env`, `Re86Env`, `G50tEnv`) and the registry already exist.
- `JEPA/shared/action_embed.py` — instantiated 4×.
- `JEPA/experiments/exp_003_4_no_resampler_self_attn/models/*` — re-exported as-is.
- `JEPA/experiments/exp_003_4_no_resampler_self_attn/monitors/*` — re-exported as-is; per-env keys flow through `health.sec*.setdefault(..., deque)` automatically.

---

## 11. How to Run

```bash
# ── Step 0a: RE86 life-end probe (must run BEFORE training) ────────────────
cd "Code Repo"
uv run python -m JEPA.experiments.exp_004_1_four_envs.probe_re86_lives
# Inspect probe_runs/re86/<timestamp>/ and update §7.3 of this card.

# ── Step 0b: G50T life-end probe (must run BEFORE training) ────────────────
uv run python -m JEPA.experiments.exp_004_1_four_envs.probe_g50t_lives
# Inspect probe_runs/g50t/<timestamp>/ and update §7.4 of this card.

# ── Training (fresh) ───────────────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_004_1_four_envs.train

# ── Training (resume from checkpoint) ──────────────────────────────────────
uv run python -m JEPA.experiments.exp_004_1_four_envs.train \
    --resume JEPA/experiments/exp_004_1_four_envs/checkpoints/step_050000.pt

# ── Override hyperparameters ───────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_004_1_four_envs.train \
    --max-steps 1000000 --batch-size 128

# ── Evaluation (per env or all) ────────────────────────────────────────────
uv run python -m JEPA.experiments.exp_004_1_four_envs.eval \
    --checkpoint checkpoints/step_050000.pt --env ls20 --episodes 50
uv run python -m JEPA.experiments.exp_004_1_four_envs.eval \
    --checkpoint checkpoints/step_050000.pt --env all  --episodes 50

# ── Debug runner (standalone, choose env) ─────────────────────────────────
uv run python JEPA/experiments/exp_004_1_four_envs/debug_runner.py \
    .../checkpoints/step_050000.pt ls20

# ── Dashboard ──────────────────────────────────────────────────────────────
uv run python JEPA/dashboard/server.py
# Open http://localhost:8787
# Select "exp_004_1_four_envs" in the experiment dropdown.
```

---

## 12. On the meaning of gradient cosine similarity (pointer)

Same interpretation framework as exp_004_0 §12 — the cross-env cosine for a module compresses the full parameter space into one scalar; per-element sign disagreement complements it. At the 4-env scale we summarise the C(4,2) = 6 pairs by their mean and minimum to keep the metric surface manageable; the full 6-pair breakdown is available behind a verbose flag.

`gcossim_min_pairs_<k>` is the worst-case signal: a single env-pair in conflict can be hidden by the mean even when five other pairs are well-aligned. Surface it.

---

## 13. Expected Outcomes and Open Risks

### 13.1 Expected outcomes

- **High confidence:** State-only reward continues to avoid the corner-attractor pathology in all four envs. Per-env tile-coverage curves rise.
- **Medium confidence:** The shared encoder remains non-collapsed by exp_003_4's standards — `latent_eff_rank` stays around 3–4, latent pairwise cosine well below 1, `ht_htp1_cossim_eval` in roughly the 0.4–0.7 band. Cross-env training is asking more of the encoder at the 4-env scale, but the action predictor's anti-collapse signal is quadrupled.
- **Low confidence:** Either policy completes Level 1 of its game under pure intrinsic-only reward. Same expectation as exp_004_0 — the experiment's value is the cross-env representation-learning signal, not policy performance.

### 13.2 Risks

- **R1 — RE86 / G50T life-end heuristic wrong.** ✅ **Closed (probes run 2026-05-14).** Both probes confirmed no intra-game lives — every episode terminates once via GAME_OVER at a deterministic step count (RE86 = 100, G50T = 130). `is_end_of_life_<env>` = `is_terminal` is provably correct under random play. See §7.3 / §7.4.
- **R2 — Cross-env reward-magnitude imbalance (4-way).** The `state_err` distribution may differ substantially across the four envs, biasing the JEPA loss toward whichever env has the larger error magnitude. *Mitigation:* per-env loss breakdowns are logged; if the imbalance is severe, switch the JEPA combine from the unweighted mean of four to a per-env-EMA-normalised form.
- **R3 — Shared-encoder representation collision.** The Perceiver may carve four near-disjoint sub-regions of the 4-latent space (4 encoders that share weights but produce uncoupled representations). *Mitigation:* monitor `latent_eff_rank` and `gcossim_min_pairs_<k>` on the encoder modules.
- **R4 — Round-robin granularity too coarse.** One full episode per env may produce long stretches of in-distribution drift between JEPA updates, especially if any env's episodes are very long. *Mitigation:* JEPA updates fire every 5 global env steps; if wall-clock allows, switch to per-step env rotation (complicates recurrent-state tracking).
- **R5 — Per-env buffer-fill / recency imbalance.** Probe-confirmed episode lengths are 50 (TU93) / 100 (RE86) / 130 (LS20, G50T) — a **~2.6× spread**, not order-of-magnitude. At equal episode counts the fast-episode env (TU93) cycles its buffer ~2.6× slower in *transition* terms, so its recency horizon is ~300 episodes vs ~115 for LS20/G50T. Balanced JEPA sampling neutralises this for the encoder; the residual effect is only that TU93's buffer holds older transitions. *Mitigation:* per-env `buffer_fill_<env>` and `episodes_completed_<env>` monitored. No action needed unless `buffer_fill` divergence becomes extreme.
- **R6 — 5-way action head on 4-action envs.** LS20 / TU93 batches' CE targets never include class 4. The class-4 logit is pushed down by the softmax over many steps. This is the desired behaviour for policy time (the unused logit is also masked to −∞ by the policy on those envs), but the action predictor's class-4 prior could be compressed in a way that affects RE86 / G50T learning. *Mitigation:* `action_pred_entropy_eval` per env catches this; if the class-4 prior collapses on 5-action envs, switch to a per-env-head design.
- **R7 — Smaller per-env buffer recency horizon (15K vs 25K).** Probe-confirmed: LS20/G50T ≈ 130-step episodes → ~115-episode horizon; TU93 ≈ 50-step → ~300-episode horizon; RE86 = 100-step → ~150-episode horizon. All four comfortably well-mixed for uniform sampling.
- **R8 — 4-way cross-env interference is qualitatively different from 2-way.** Even with positive pairwise cosines, the *mean direction* may be inconsistent with each pair (no single direction satisfies all four). `gcossim_avg_pairs_<k>` can hide this; `gcossim_min_pairs_<k>` surfaces the worst pair.
- **R9 — Perceiver LR may be too high.** 12 grad paths per JEPA step (vs 6 in exp_004_0) doubles the accumulated grad into the shared Perceiver. Per-env batches are 4× smaller (vs 2× in exp_004_0), partially offsetting. `grad_clip_model = 5.0` is the safety net. *Mitigation:* if `gnorm_perceiver_total` clips saturate consistently in the first 1K JEPA steps, halve `perceiver_lr` to 2.5e-5.

### 13.3 What a "successful" run looks like

- Per-env policies reach ~uniform-random-baseline performance in all four games and ideally exceed it on tile coverage.
- `gcossim_avg_pairs_<k>` on the encoder modules is **not** persistently negative; `gcossim_min_pairs_<k>` does not stay below -0.3 for extended windows.
- `latent_eff_rank` of the shared latent space stays in roughly `[2.5, 4.0]` across training.
- `gsign_disagree_frac_avg_pairs_<k>` on encoder modules in roughly `[0.3, 0.5]` (signs mostly agreeing).
- `action_pred_entropy_eval` per env stays clearly above 0 — particularly on 5-action envs, where the class-4 prior must remain usable.

---

## 14. Comparison with prior experiments

| Dimension | exp_003_2 | exp_003_3 | exp_003_4 | exp_004_0 | **exp_004_1 (this)** |
|-----------|-----------|-----------|-----------|-----------|----------------------|
| **Games** | LS20 | LS20 | LS20 | LS20 + TU93 | **LS20 + TU93 + RE86 + G50T** |
| **Encoder weights** | Single | Single | Single | Shared (×2) | **Shared (×4)** |
| **Resampler self-attn among latents** | Yes | Yes | No | No | **No** |
| **Action predictor in JEPA loss** | Yes | Yes | Yes | Yes (4-way) | **Yes (5-way)** |
| **Action predictor in reward** | Yes (`0.5`) | **No** (`0.0`) | Yes (`0.5`) | **No** (`0.0`) | **No** (`0.0`) |
| **Reward weighting** | `0.5/0.5` | `1.0/0.0` | `0.5/0.5` | `1.0/0.0` | **`1.0/0.0`** |
| **JEPA loss weighting** | `0.5/0.5` | `0.5/0.5` | `0.5/0.5` | `0.5/0.5` | **`0.5/0.5`** |
| **EMA target encoder** | None | None | None | None | **None** |
| **Buffer count / capacity** | 1 × 50K | 1 × 50K | 1 × 50K | 2 × 25K | **4 × 15K** |
| **Buffer sampling** | Uniform | Uniform | Uniform | Balanced (2-way) | **Balanced (4-way)** |
| **Action embedding count** | 1 | 1 | 1 | 2 | **4** |
| **Policy count** | 1 | 1 | 1 | 2 | **4** |
| **n_actions (head width)** | 4 | 4 | 4 | 4 | **5** |
| **Encoder gradient paths / JEPA step** | 3 | 3 | 3 | 6 | **12** |

---

## 15. Acceptance Tests (run after implementation)

1. **Probe verdicts written.** `probe_re86_lives.py` and `probe_g50t_lives.py` each produce `verdict.md` under `probe_runs/<env>/<ts>/`; §7.3 and §7.4 of this card are updated accordingly *before* training is started.
2. **Static checks.**
   - `grep -n 'n_actions' .../config.py` returns `5`.
   - `grep -n 'game_ids' .../config.py` lists the 4 game IDs.
   - `grep -n 'buffer_size_per_env' .../config.py` returns `15_000`.
   - `grep -n 'env_names' .../config.py` returns the 4-tuple.
3. **Smoke run.** From repo root:
   ```
   uv run python -m JEPA.experiments.exp_004_1_four_envs.train --max-steps 8000
   ```
   `--max-steps 8000` = the 4000-step warm-up plus ~4000 steps of policy-on joint training. Completes without errors; all four per-env buffers exceed `min_buffer_size = 512` within warm-up; per-env episode counts grow within an order of magnitude of each other.
4. **Action-mask runtime assertion.** The assertion in `run_one_episode` (`action_idx < env.n_actions`) never fires. If it does, available-actions masking is broken on a 4-action env.
5. **Balanced JEPA batches.** Each JEPA call sees exactly `batch_size // 4` transitions from each env (asserted by `jepa_update`).
6. **Dying-step gating.** Replay the buffer dump after a smoke run. For each transition, `next_frame` is never the reset state of the next episode.
7. **Metric surface.** Dashboard shows: per-env loss curves, per-env policy entropy, per-env tile coverage, and the cross-env grad-cosine summary (`avg_pairs`, `min_pairs`).
8. **Per-env eval.** `eval.py --env <e>` for each of the 4 envs completes with the same checkpoint and reports level-completion + episode-length distribution.
9. **No-regression check (vs exp_004_0).** At the same global step count, the shared-encoder collapse metrics (`latent_eff_rank`, `latent_pairwise_cossim_buf`, `ht_htp1_cossim_eval`) should be qualitatively comparable to exp_004_0. Drastic worsening = stop and report; do not patch ad-hoc.

---

## 16. Out of Scope (deliberately deferred)

These are **not** in this experiment.

1. **Per-env predictor weights.** The state and action predictors are shared on purpose — we are testing a shared world model, not four co-trained world models.
2. **Per-env action predictor heads.** Single 5-way head is the design under test. If R6 fires, switch to per-env heads in a follow-up.
3. **Per-env policy update schedule.** All four policies share `policy_update_freq`. If they end up wildly imbalanced in episodes-completed, revisit.
4. **Cross-env policy transfer / fine-tuning.** Each policy is trained from scratch alongside the shared encoder; no transfer step.
5. **PPO / value heads / GAE.** REINFORCE with EMA baseline retained per env.
6. **Goal-conditioned policy, HER, k-NN-novelty bonus, RND-style fixed-target curiosity, large terminal bonus on `levels_completed >= 1`.** Same deferral list as exp_004_0.
7. **Per-env loss normalisation.** Listed as the R2 mitigation; only applied if the cross-env reward-magnitude imbalance is severe in the smoke run.
8. **Halving `perceiver_lr`.** Only applied if R9 fires in the smoke run.
9. **Verbose per-pair cross-env cosine breakdown by default.** Only `avg_pairs` and `min_pairs` summary scalars are logged on every probe; the full 6-pair breakdown is behind a flag.
10. **Compensating for per-env episode-length differences.** Probe-confirmed episode lengths differ across envs (TU93 = 50, RE86 = 100, LS20/G50T = 130 — a ~2.6× spread). Under episode-level round-robin this means per cycle the four envs contribute unequal *transition* counts, and policy updates (gated by `policy_update_freq` per env) fire at different wall-clock cadences. **For this experiment we deliberately do not compensate** — the spread is modest and balanced JEPA sampling already equalises the encoder's exposure. **This is a known item to revisit later** if it causes observable per-env policy-learning imbalance: candidate fixes are (a) transition-count round-robin instead of episode round-robin, (b) per-env `policy_update_freq` scaled to episode length, or (c) sub-episode env rotation. Monitor `episode_length_<env>`, `episodes_completed_<env>`, and `pol_loss_<env>` cadence to decide if/when this needs addressing.
