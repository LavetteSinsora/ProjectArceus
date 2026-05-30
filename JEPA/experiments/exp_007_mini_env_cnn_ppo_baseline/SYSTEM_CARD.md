# exp_007 — CNN + PPO Baseline on MiniLS20 Level 1

> A *deliberately naive* baseline. We are not trying to set a new state of the
> art here — we are trying to characterise what the simplest plausible
> pixel-input policy-gradient agent can and cannot do on a reduced version of
> ARC-AGI-3 Level 1, so that every later experiment has a meaningful zero
> against which to measure progress.

---

## 1. Motivation

### 1.1 The two difficulties from the project notes

ARC-AGI-3 presents (at least) two structural difficulties for any learning
agent:

1. **Generalisation** — across games (can one set of weights work everywhere?)
   and across levels within a single game (does mechanism X learnt in level 1
   transfer when X reappears rotated or relocated in level 4?).
2. **Sparse reward** — the engine returns a binary terminal signal at most
   once per 42-step episode. The probability of stumbling onto a successful
   trajectory by uniform random action is roughly $\exp(-N)$ in the path
   length $N$. For Montezuma's-Revenge-class problems this is the bottleneck.

exp_007 addresses **only the sparse-reward axis**, and only on a single
level. Generalisation will be revisited once the simpler problem is
understood.

### 1.2 Experimental philosophy

> Take a step back. Start from the simplest. Figure out what the naive
> approach can and cannot do. Gradually add complexity. See when does naive
> break, why, how to fix it.

This is the load-bearing principle. exp_007 deliberately strips away every
machine that prior experiments (JEPA latent world models in exp_001–004,
Dreamer V3 in exp_005, interleaved freezing in exp_006) bring to the
problem, and asks: *with just a CNN, just PPO, on a 32×32 frame, what
happens?* The expected story going in:

- **exp_007_0** (terminal reward only): probably fails. We expect to see the
  policy collapse to a near-deterministic loop somewhere safe and never
  discover the goal. If it doesn't fail, that is a *much more interesting*
  result than success would have been.
- **exp_007_1** (+ wall-hit penalty): probably also fails. A wall penalty
  nudges the agent *away from* dead ends but provides no *gradient toward*
  the goal. The expectation is that variant 1 simply learns to wander in
  open space.
- **exp_007_2** (+ rotation-match shaping): the first variant we expect to
  actually solve the level. The shaping makes the cross (the only rotation-
  changing entity) intrinsically rewarding when stepping on it brings the
  player into rotational alignment with the goal, and intrinsically
  punishing when it breaks an alignment. This converts the puzzle from
  "find a 13-step needle in a 4^42 haystack" into a two-phase reward
  landscape: get aligned → walk to the goal.

We expect to be wrong about at least one of these predictions. That is the
point. The system card and metrics are designed so that *whichever one*
turns out wrong, the failure mode is legible.

### 1.3 Why complexity is being reduced (and where)

| axis | full ARC-AGI-3 | mini-LS20 (this exp) |
|---|---|---|
| frame | 64 × 64 | 32 × 32 |
| episode length | 64–128 | 42 |
| action space | up to ~12 | 4 (cardinal) |
| reward | binary terminal | binary terminal + optional shaping |
| level count | 12 | 1 (level_01) |
| engine | `arcengine` (TS port) | pure-numpy `mini_env` |

The 32 × 32 pixel-space reduction is the critical lever. Random-walk hitting
probability of an arbitrary cell scales with the square of the grid, so
moving from 64 × 64 to 32 × 32 quadruples the chance that random exploration
stumbles onto the goal at least once. Combined with the 42-step horizon,
this puts the level in the regime where naive policy gradient *could*
plausibly work — which is exactly what we want to test.

---

## 2. What is and is not being tested

**Tested:** can a vanilla CNN + PPO with the standard set of implementation
details (orthogonal init, clipped surrogate, GAE, entropy bonus, value-loss
clipping, advantage normalisation, gradient clipping) solve a single ARC-like
puzzle level given (a) terminal reward only, (b) wall-hit penalty, or (c)
rotation-match shaping?

**Not tested:** generalisation, transfer, world models, latent dynamics,
exploration bonuses (count-based, RND, P2E), recurrence, behaviour cloning,
imitation learning. These are the *next* experiments — exp_007 is the floor
against which they will be measured.

---

## 3. The three sub-experiments

All three share the same model, optimiser, hyperparameters, vectorised
environment setup, and metric logging. The **only** difference is the
per-step reward function. This is deliberate: any difference in outcome can
be attributed unambiguously to the shaping.

### 3.1 exp_007_0_naive — terminal reward only

```
r_t = +1 if env.won this step, else 0
```

This is the honest sparse-reward baseline. If naive PPO ever solves this
level, this variant is where we will see it. The metric we will watch
hardest here is `success_rate` over the last 100 evaluation episodes — does
it ever depart from random?

### 3.2 exp_007_1_wallpen — wall-hit penalty

```
r_t  = +1 if env.won this step, else 0
r_t += −0.05 if this step was wall-blocked or out-of-bounds, else 0
```

A wall hit is detected by comparing `(player_c, player_r)` before and after
`env.step()` — if the position is unchanged and the episode did not just
terminate by reaching the goal, the step was blocked. The penalty is small
(−0.05) so a single successful 13-step trajectory containing some wall hits
is still net positive (`+1 − k·0.05 > 0` for `k < 20`). This variant tests
the hypothesis that ridding the policy of obviously-pointless actions is
enough to let it stumble onto the goal — we suspect it is not.

### 3.3 exp_007_2_match — rotation-match shaping

```
r_t  = +1 if env.won this step, else 0
r_t += −0.05 if wall-blocked, else 0
r_t += +0.1 if this step transitioned player_rotation: mismatch → match
r_t += −0.1 if this step transitioned player_rotation: match → mismatch
```

The "pattern" the player can change is their rotation (only changed by
stepping on the cross). "Matching the goal pattern" means
`player_rotation == goal_rotation`. The +0.1 fires on the single step the
agent crosses into alignment; the −0.1 fires if it then steps on the cross
again and breaks that alignment. No shaping on any other step.

Scale rationale: terminal reward dominates at +1; the per-step shaping is
±0.1 so that a single rotation-flip is roughly 10× cheaper than completing
the level. This keeps shaping informative without overwhelming the true
objective.

---

## 4. Architecture (shared)

### 4.1 Input encoding
Raw observation: `(32, 32) uint8` with palette values in `[0, 15]`.
Encoded for the network: `(16, 32, 32) float32` via per-pixel one-hot over
the 16 palette indices.

Why one-hot: colour indices are *categorical*, not ordinal. There is no
sense in which "palette 4 (energy)" is "between palette 3 and palette 5".
Linear/conv layers on raw integers would force the network to undo the
arithmetic encoding before it could use the labels; one-hot avoids that
entirely. Standard practice for ARC-style symbolic grids.

The UI strip rows (`frame[28:32, :]`) carry the rotation preview tile and
the energy bar, both of which are necessary state. **They are NOT masked
out of the policy input.** The `_MASKED_ROWS` slice in `mini_env.env` is
used only for `frame_diff` / `patch_weights` introspection utilities and
does not touch the network.

### 4.2 CNN encoder

```
Conv2d(16 → 32, k=3, s=1, pad=1)   ReLU     # (B, 32, 32, 32)
Conv2d(32 → 64, k=3, s=2, pad=1)   ReLU     # (B, 64, 16, 16)
Conv2d(64 → 64, k=3, s=2, pad=1)   ReLU     # (B, 64,  8,  8)
Flatten                                       # (B, 4096)
Linear(4096 → 256)                  ReLU     # (B, 256)   "trunk feature h"
```

Why these choices:
- **3×3 kernels, strided downsampling, no pooling.** Pooling discards
  positional information that the policy needs to localise actions; strided
  convolutions let the network learn its own downsampling. Standard modern
  practice and the universal default in PPO codebases.
- **First conv preserves resolution (stride 1).** Each grid cell is only
  4×4 pixels and the L-mark inside the goal tile is just 3 pixels wide.
  Aggressive early downsampling (à la Nature DQN's stride-4 first conv)
  would smear those features. We can afford a stride-1 first conv because
  the input is only 32×32 — there is no compute reason to be stingy.
- **256-dim trunk feature.** Wide enough to encode `(player_c, player_r,
  player_rotation, goal_rotation_match, remaining_energy)` plus several
  more bits of layout context, narrow enough that the value head is well-
  regularised by the bottleneck.

### 4.3 Heads

```
policy_head: Linear(256 → 4)   logits, no activation
value_head:  Linear(256 → 1)   scalar, no activation
```

### 4.4 Initialisation
Orthogonal init with per-layer gain:
- Conv + trunk Linear (ReLU after): `gain = √2`. Compensates for ReLU
  killing half the variance, so signal magnitude stays roughly constant
  through the network.
- `value_head`: `gain = 1.0`. No nonlinearity to correct for.
- `policy_head`: `gain = 0.01`. **The single most important hyperparameter
  in this whole stack.** Tiny initial logits make the initial policy almost
  exactly uniform across the 4 actions, so the very first rollouts are real
  exploration rather than a self-fulfilling lottery on whichever logit
  random init happened to favour. Combined with the entropy bonus, this is
  what gives PPO any chance against sparse reward at all.

### 4.5 No BatchNorm
BatchNorm interacts pathologically with PPO's policy/old-policy ratio:
running statistics shift between rollout collection and the update epochs,
so `π_old(a|s)` computed at rollout time uses different normalisation than
`π_new(a|s)` computed at update time, silently breaking the importance-
sampling ratio. Standard PPO codebases use no normalisation, or LayerNorm /
GroupNorm. We use none.

### 4.6 Parameter count
≈ 1.1 M, dominated by the 4096 → 256 trunk Linear. Easily fits and trains
on Apple M3 Pro MPS.

---

## 5. PPO

### 5.1 Loss
```
ratio    = exp(log_prob_new − log_prob_old)
L_policy = − E[ min( ratio · A_norm,
                     clip(ratio, 1 − ε, 1 + ε) · A_norm ) ]
L_value  = 0.5 · E[ (V_new − V_target)² ]
L_ent    = − E[ H(π) ]
L_total  = L_policy + c_v · L_value + c_ent · L_ent
```

with `ε = 0.2`, `c_v = 0.5`, `c_ent = 0.01`. Advantages are normalised
per-minibatch. We do **not** clip the value loss in this baseline (one
fewer knob; PPO is robust enough without it on this scale).

### 5.2 GAE
```
δ_t   = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
A_t   = δ_t + γ · λ · (1 − done_t) · A_{t+1}     # backward sweep
V_tgt = A_t + V(s_t)
```
`γ = 0.99`, `λ = 0.95`. Step-limit truncation is treated as `done=True`
(since `r_terminal = 0` on truncation, the bootstrap is irrelevant).

### 5.3 Rollout
- 8 synchronous vectorised mini envs (chosen for MPS — keeps batch size
  within MPS bandwidth and avoids subprocess overhead, which dominates for
  an env this cheap).
- 128 steps per env per rollout → **1024 transitions per update**.
- 4 epochs × 4 minibatches of 256 = 16 gradient steps per rollout.

### 5.4 Other implementation details
- Adam, `lr = 3e-4`, no schedule (baseline).
- `clip_grad_norm_(model.parameters(), 0.5)` after every backward.
- Approximate KL is *logged* but does not early-stop epochs (keeps the
  baseline minimal).

---

## 6. Metrics

Metrics are split into **performance** (does the agent solve the problem?)
and **diagnostics** (is training healthy under the hood?). Both kinds are
logged to `runs/<run_name>/metrics.jsonl`. Heavier metrics that require
extra forward passes (eval rollouts, gradient-source decomposition) are
computed every `eval_every` updates rather than every update.

### 6.1 Performance metrics (computed during periodic eval rollouts)

Evaluated by running `eval_episodes` (default 32) freshly-reset episodes
with **stochastic** action sampling from the current policy.

- `success_rate` — fraction of evaluation episodes that ended with
  `env.won == True`. The headline metric.
- `min_steps_to_solve` — minimum step count over successful episodes
  (`None` if no episode succeeded). Tells us the best trajectory the
  current policy ever produces.
- `avg_steps_to_solve` — mean step count over successful episodes only.
  Should approach 13 (the optimal solve length found by
  `claude_automate`) as the policy converges.
- `pattern_matched_at_end_rate` — fraction of episodes in which the player
  ended with `player_rotation == goal_rotation`, regardless of whether the
  agent actually reached the goal. Decouples "the agent learnt rotation
  matters" from "the agent learnt to navigate to the goal". For exp_007_0
  this is expected to stay near random; for exp_007_2 we expect it to rise
  much earlier than `success_rate` does.
- `coverage_rate` — average over evaluation episodes of (unique non-wall
  cells visited) / (total non-wall cells). Measures exploration breadth.
  We expect this to be high early in training and decay as the policy
  becomes more directed.
- `wall_hit_rate` — mean wall-blocked steps per episode. We expect this to
  start near `4/4 · (wall_fraction)` for a uniform-random policy and to
  drop sharply once exp_007_1 / exp_007_2 punish it.
- `mean_episode_return` — unshaped return (i.e. just whether the agent
  won). For exp_007_2 we also log `mean_episode_return_shaped` (including
  the shaping signal) to confirm shaping is non-degenerate.

### 6.2 Diagnostic metrics (per training update)

- `policy_entropy` — mean entropy of `π(·|s)` over the rollout. Falling
  smoothly is healthy; collapsing in the first ~100 updates means
  exploration is failing.
- `approx_kl` — sample estimate of `KL(π_old ‖ π_new)` after the update.
  Should stay below ~0.02. Spikes mean the clip is failing.
- `clipfrac` — fraction of samples for which the PPO ratio was clipped.
  Healthy range is 0.1–0.3.
- `value_loss`, `policy_loss`, `entropy_loss` — three components of the
  combined loss.
- `mean_feature_cosine` — average cosine similarity between trunk feature
  `h_t` and `h_{t+1}` over consecutive **same-episode** time steps in the
  rollout. High values (close to 1) mean the encoder is producing nearly
  identical representations for adjacent frames — i.e. the representation
  is *temporally stable*. Very high values may indicate the encoder is
  collapsing (representational rigidity); very low values may indicate
  noisy or chaotic features. This is the encoder-health analogue of
  representational-collapse metrics in JEPA/SSL.
- `grad_norm_total` — `‖∇L_total‖₂` post-clip.
- `grad_norm_encoder_from_actor` — `‖∇(L_policy + c_ent·L_ent)‖` measured
  on encoder parameters only. Computed every `grad_decomp_every` updates
  (default 10) via a separate backward pass with `retain_graph=True`.
- `grad_norm_encoder_from_critic` — `‖∇(c_v·L_value)‖` on encoder
  parameters only. Same backward-pass machinery.
- `grad_norm_ratio_critic_over_actor` — the ratio between the above two.
  This is the diagnostic for "is the value loss bullying the shared
  encoder?". If this ratio sits above ~5, the encoder is being driven
  predominantly by the critic and the policy gradient is effectively
  starved. The standard fix is to lower `c_v` or split the encoder.

### 6.3 Logging cadence

- Per-update (cheap metrics): every `log_every` updates (default 1).
- Periodic eval (performance metrics): every `eval_every` updates
  (default 50, ≈ every ~50K env steps).
- Gradient decomposition: every `grad_decomp_every` updates (default 10).
- Checkpoint: every `save_every` updates (default 100).

---

## 7. Expected outcomes

| variant | hypothesis | what falsifies it |
|---|---|---|
| 007_0 | `success_rate` stays below 5% throughout training | success_rate climbs above 20% — would mean sparse-reward PPO is more capable than expected on this level, and we re-evaluate the whole motivation |
| 007_1 | `success_rate` similar to 007_0; `wall_hit_rate` drops; `coverage_rate` rises | wall_hit_rate does *not* drop — means the wall penalty isn't even being learned, broken implementation |
| 007_2 | `success_rate > 50%`, `avg_steps_to_solve < 30` within 2M env steps | success_rate stays below 007_0's level — means the shaping signal is either too small, miscomputed, or is creating an exploitable local optimum that prevents goal-seeking |

For 007_2 specifically, we also expect `pattern_matched_at_end_rate` to
rise *before* `success_rate` does, because rotation alignment is the easier
subproblem.

---

## 8. How to run

From the repo root (`Code Repo/`):

```bash
# Train
uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_0_naive.train
uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_1_wallpen.train
uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_2_match.train

# Evaluate a checkpoint
uv run python -m JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_0_naive.eval \
    --checkpoint JEPA/experiments/exp_007_mini_env_cnn_ppo_baseline/runs/<run_name>/checkpoints/final.pt
```

Each `train.py` reads its variant-specific `config.py`, builds the reward
function, and calls into the shared trainer.

---

## 9. Caveats / known limitations

- **Single seed.** All three variants are reported as single-seed runs.
  Across seeds, PPO on sparse-reward tasks is notoriously high-variance,
  so the *qualitative* story (does it work or not) is more informative
  than precise numbers. A multi-seed sweep is left for a follow-up exp
  once the qualitative picture is locked in.
- **Single level.** No claim about generalisation. The fact that 007_2
  might solve level_01 says nothing about whether it would solve
  level_02 or level_03 (also present in `mini_env/configs/`). That
  question is the explicit subject of a follow-up.
- **No frame stacking.** Standard Atari recipe stacks 4 frames to give
  the agent velocity information. Mini-LS20 is fully observable from a
  single frame (player position, rotation preview, energy bar are all
  rendered in the current frame), so frame stacking is omitted.
- **Synchronous vec env.** With 8 envs and a pure-numpy step function we
  use a synchronous loop rather than subprocesses. If wall-clock becomes
  the bottleneck we can move to subproc, but on M3 Pro MPS the gradient
  step dominates env stepping, so it isn't worth the complexity.
