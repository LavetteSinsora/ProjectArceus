# exp_012_1 — Random Network Distillation (RND) exploration baseline on the *real* LS20 game

> **Family.** exp_012 is the **intrinsic-reward exploration** family on the real
> 64×64 ARC-AGI-3 LS20 Level 1 game. It asks the question exp_010 left open:
> exp_010_0 showed that vanilla terminal-only CNN+PPO has to *stumble* onto the
> first reward by chance, and a uniform-random policy needs **≈ 50,000 env steps
> to its first level completion** (Markov-chain estimate; see
> `memory/finding_random_policy_ls20_l1.md`). exp_012 replaces blind luck with a
> directed exploration bonus and measures how much that bonus shrinks the
> **environment steps to first extrinsic reward**.
>
> | id | dir | method |
> |---|---|---|
> | 12_0 | `exp_012_0_icm_baseline/` | Intrinsic Curiosity Module (Pathak et al. 2017) |
> | **12_1** | **`exp_012_1_rnd_baseline/`** | **Random Network Distillation (Burda et al. 2018)** — *this card* |
>
> Both reuse the exp_010 real-LS20 CNN+PPO infrastructure so the *only* thing
> that changes versus exp_010_0 is the exploration mechanism.

---

## 1. Motivation

exp_010_0 is the honest sparse-reward zero: a 64×64 CNN + PPO with **terminal-only
reward** (`r=+1` iff the step cleared the level, else `0`). On real LS20 Level 1
that reward is reached by a uniform-random agent only about once per ~50k env
steps, so PPO has nothing to climb until it gets lucky. exp_012_1 keeps that
exact extrinsic reward and adds an **RND intrinsic bonus** that rewards visiting
states the agent has not yet learned to predict, pulling exploration toward the
novel parts of the maze instead of leaving it to chance.

The headline question is a sample-efficiency one, not an asymptotic one:

> **How many environment steps until the first extrinsic reward (first level
> completion), with RND, versus the ~50k-step random-exploration baseline and
> versus exp_010_0?**

RND is also the natural counterpart to 12_0 (ICM): ICM derives novelty from a
*learned forward dynamics* prediction error in an inverse-dynamics feature
space; RND derives it from a *fixed random* target's prediction error and needs
no dynamics model. exp_012 runs both faithfully on identical PPO scaffolding so
the comparison is apples-to-apples.

---

## 2. What RND is (faithful module breakdown)

Random Network Distillation (Burda, Edwards, Storkey, Klimov, 2018,
[arXiv:1810.12894](https://arxiv.org/abs/1810.12894)). Three pieces:

1. **Target network** `f` — a CNN, **randomly initialised and frozen forever**.
   It maps an observation to a `k`-d feature vector. Its only job is to be a
   fixed, arbitrary but deterministic function of the state.
2. **Predictor network** `f̂` — a CNN with the *same input* and output dim,
   **trained by gradient descent** to regress the target's output on the states
   the agent actually visits.
3. **Intrinsic reward** — the predictor's error on the *next* state:
   ```
   i_t = ½ · ‖ f̂(s_{t+1}) − f(s_{t+1}) ‖²
   ```
   On states seen many times the predictor has fit `f` well → error ≈ 0 → low
   bonus. On novel states the predictor has not caught up → large error → large
   bonus. As exploration proceeds and `f̂` learns, the bonus on familiar regions
   decays automatically. There is no dynamics model and no action conditioning —
   novelty is purely "have I seen states like this."

RND's two famous failure-mode mitigations, both **kept**:

- **Intrinsic-reward normalisation.** Raw `i_t` has an arbitrary, drifting scale
  (it depends on the random target). RND divides `i_t` by a **running estimate
  of the standard deviation of the intrinsic *returns*** (the discounted sum,
  not the per-step reward), tracked with a `RewardForwardFilter` + a
  `RunningMeanStd`. This is the single most important detail for RND to work and
  is replicated exactly.
- **Two reward streams / two value heads.** The intrinsic reward is treated as
  **non-episodic** (novelty does not "reset" at a death/level boundary), while
  the extrinsic reward is episodic. They get **separate value heads** and
  **separate discount factors**, and their advantages are summed (§4).

---

## 3. Architecture

Everything imports the exp_010 shared library; only the listed deltas are new.

### 3.1 Actor-critic with **two value heads** (modified from exp_010 `model.py`)

```
encoder = exp_010 CNNEncoder           # one-hot (16,64,64) → 4 strided convs → Linear(4096→256) ReLU
policy_head : 256 → 4                   # orthogonal gain 0.01
value_head_ext (V_E) : 256 → 1          # orthogonal gain 1.0   — extrinsic returns
value_head_int (V_I) : 256 → 1          # orthogonal gain 1.0   — intrinsic returns   (NEW)
```

The only change to exp_010's `ActorCritic` is the **second value head** `V_I`.
The encoder is byte-for-byte the exp_010 / "7_0" CNN (§4 of exp_010's card):
`Conv2d(16→32,k3,s1) → Conv2d(32→64,k3,s2) → Conv2d(64→64,k3,s2) →
Conv2d(64→64,k3,s2) → Flatten → Linear(4096→256)`, all ReLU, orthogonal init,
gain √2 on the ReLU layers. No BatchNorm, no pooling.

### 3.2 RND target & predictor (NEW — `rnd.py`)

Per the Q3 decision, both reuse the **exp_010 CNNEncoder backbone** (so RND lives
in the same encoder architecture as the rest of the project) with **256-d
output** (the project `trunk_dim`, not the paper's 512). They are **independent
of the policy encoder** — see §6 for why we do *not* read the actor-critic's `h`.

```
RNDTarget   : CNNEncoder(→256)                          # random, FROZEN (requires_grad=False, eval())
RNDPredictor: CNNEncoder(→256) → Linear(256→256) ReLU
                                → Linear(256→256) ReLU
                                → Linear(256→256)        # extra FC capacity vs target (per RND)
```

Both take the **same one-hot `(16,64,64)` frame** the policy encoder consumes
(Q2 decision). The predictor is deliberately higher-capacity than the target
(extra FC stack) so it *can* fit the random target on visited states — the RND
paper's reason for the asymmetry. The target's weights are drawn once at init
and never updated.

**Intrinsic reward** on a rollout step:
`i_t = ½ · mean_j ( f̂(s_{t+1})_j − f(s_{t+1})_j )²` (mean over the 256 feature
dims), computed under `torch.no_grad()` for `f` (target) and detaching `f` in
the predictor's training loss.

**Predictor loss** (added to the optimiser, trained jointly with PPO each
update): `L_rnd = ½ · ‖ f̂(s_{t+1}) − sg(f(s_{t+1})) ‖²`, mean over the minibatch.

---

## 4. Dual-stream PPO (faithful RND)

This is the substantive change to exp_010's single-head PPO. We extend
`rollout.py` / `ppo.py` to carry **two reward streams**.

| quantity | extrinsic (E) | intrinsic (I) |
|---|---|---|
| reward | `r^E_t` = terminal-only `{0,1}` (unchanged from exp_010) | `r^I_t = i_t / σ̂(returns)` (normalised, §2) |
| value head | `V_E` | `V_I` |
| discount γ | **0.999** | **0.99** |
| episodic? | **yes** — `V_E` bootstrap masked by `(1−done_t)` | **no** — `V_I` is **non-episodic**: GAE for the intrinsic stream does **not** mask at episode boundaries |
| GAE λ | 0.95 | 0.95 |

**Advantage combination** (exactly the paper):
```
A_t = ext_coef · A^E_t + int_coef · A^I_t        with  ext_coef = 2.0,  int_coef = 1.0
```
`A_t` drives the clipped policy surrogate. The two value heads are trained
against their *own* returns: `L_value = ½·(V_E − R^E)² + ½·(V_I − R^I)²`
(each with the exp_010 value-clip of 0.2). Total loss:
`L = L_policy + c_value·L_value + c_entropy·L_entropy + L_rnd`.

**Non-episodic intrinsic, concretely.** The intrinsic GAE recursion uses
`δ^I_t = r^I_t + γ_I·V_I(s_{t+1}) − V_I(s_t)` with **no `(1−done)` mask** on the
bootstrap or the accumulator, whereas the extrinsic GAE keeps exp_010's
off-by-one-fixed mask. This is the single most common thing reimplementations
get wrong, so it is called out here explicitly.

The clipped-surrogate maths, value-clip, advantage-normalisation, grad-clip,
epoch/minibatch loop, and Adam optimiser are otherwise **identical to exp_010
`ppo.py`**.

---

## 5. Hyperparameters

### 5.1 Inherited PPO (from exp_010 `config_base.py`, unchanged unless noted)

| param | value | note |
|---|---|---|
| `n_envs` | 8 | real LS20 is ~1.6k steps/s/env; 8 synchronous envs (RND paper used 128) |
| `rollout_steps` | 128 | 128 × 8 = 1024 transitions / update |
| `max_episode_steps` | 200 | our truncation; treated as `done` for the **extrinsic** stream only |
| `epochs` / `minibatches` | 4 / 4 | |
| `clip_eps` / `vf_clip_eps` | 0.2 / 0.2 | |
| `c_value` / `c_entropy` / `grad_clip` | 0.5 / 0.01 / 0.5 | |
| `learning_rate` | 3e-4 | exp_010 value (RND paper used 1e-4 — see §6) |
| `total_env_steps` | 3,000,000 | matches exp_010_0's sparse-reward budget |

### 5.2 RND-specific (NEW)

| param | value | source |
|---|---|---|
| `gamma_ext` (γ_E) | **0.999** | RND paper |
| `gamma_int` (γ_I) | **0.99** | RND paper |
| `gae_lambda` | 0.95 | RND paper / exp_010 |
| `ext_coef` | **2.0** | RND paper hyperparameter table |
| `int_coef` | **1.0** | RND paper hyperparameter table |
| `rnd_feature_dim` (k) | 256 | Q3 decision (paper: 512) |
| `predictor_update_proportion` | **1.0** | RND uses 1.0 at ≤32 envs; we have 8 → keep all experience |
| `obs_norm_init_steps` | 0 (omitted) | see §6 — bounded one-hot input, pixel-norm dropped |
| intrinsic-return-std normalisation | **on** | RND paper (essential) |

---

## 6. Deviations from the paper (and why) — read this for "faithful"

The user asked for an exact replication; these are the *only* places exp_012_1
departs from Burda et al. 2018, each forced by the LS20 setting or a project
consistency rule, and each documented so the deviation is a known quantity:

1. **Encoder backbone & feature dim.** Paper uses the Mnih-2015 *Nature CNN* with
   a **512**-d RND output. We reuse the project's **exp_010 CNNEncoder** with a
   **256**-d output (Q3). *Why:* architectural consistency with exp_007/010/012_0
   so any 12_1-vs-12_0-vs-10_0 difference is attributable to the exploration
   mechanism, not the encoder. Smaller `k` only changes the bonus scale, which
   the running-std normalisation absorbs.
2. **RND networks are dedicated, not the policy encoder.** RND's target must be a
   **fixed** function of the state. The actor-critic encoder is retrained by PPO
   every update, so feeding its `h_t` to RND would make "novelty" drift with the
   encoder — a previously-visited state re-embeds differently and earns a phantom
   bonus, and the encoder could even collapse `h` to suppress the bonus. So `f`
   and `f̂` each own a **separate** copy of the CNN backbone on the raw frame.
   This is the answer to the Q2 question "can we use the encoder's state
   representation": yes to a *CNN-encoded* state, no to *the shared policy
   encoder's* output.
3. **Input is one-hot, pixel obs-norm dropped.** Paper normalises a single
   grayscale frame as `clip((x−μ)/σ, −5, 5)` with stats seeded by a random agent.
   LS20 frames are **categorical 16-colour palette indices** (one-hot, not
   ordinal), already bounded to `[0,1]` after one-hot. We therefore feed one-hot
   and **omit the `(x−μ)/σ` pixel normalisation** (and its random-agent
   warm-up). The *intrinsic-reward* normalisation (÷ running-std of returns) is
   the load-bearing one and is **kept**.
4. **8 envs, not 128.** Real LS20 throughput caps us at 8 synchronous envs.
   Consequently `predictor_update_proportion = 1.0` (the paper's value at ≤32
   envs), i.e. every collected transition trains the predictor.
5. **Learning rate 3e-4, not 1e-4.** We keep exp_010's PPO lr for cross-experiment
   comparability; the RND paper used 1e-4. (Flag for a later sweep if PPO is
   unstable with the dual-stream advantage.)
6. **γ_E = 0.999 with 200-step episodes.** Our episodes truncate at 200 steps, so
   γ_E = 0.999 has effective horizon comfortably inside the episode; kept faithful.

Everything else — frozen random target, higher-capacity predictor, `i_t` as
next-state prediction error, non-episodic intrinsic stream, dual value heads,
ext/int coefficients 2/1, intrinsic-return-std normalisation — is the paper's
recipe unchanged.

---

## 7. Shared code to add (`exp_012.../shared/`)

The family clones the exp_010 shared library and adds RND. Proposed layout:

| module | change vs exp_010 |
|---|---|
| `model.py` | `ActorCritic` gains a **second value head** `V_I`; `forward` returns `(logits, v_ext, v_int, feat)` |
| `rnd.py` | **NEW** — `RNDTarget`, `RNDPredictor`, `intrinsic_reward(next_obs)`, `predictor_loss`, `RewardForwardFilter` + `RunningMeanStd` for return-std normalisation |
| `rollout.py` | store `rewards_ext` and per-step intrinsic reward; **two** advantage/return tensors (`adv_ext/ret_ext`, `adv_int/ret_int`); `compute_gae` runs once per stream, intrinsic stream **without** the done-mask |
| `ppo.py` | combined advantage `2·A_E + 1·A_I`; two value-loss terms; add `L_rnd` to the total loss; predictor params in the optimiser & grad-clip set |
| `trainer.py` | build RND nets; per rollout compute `i_t` on `next_obs`, update the return-std normaliser, normalise, then dual-stream GAE; log RND metrics |
| `config_base.py` | add `gamma_ext`, `gamma_int`, `ext_coef`, `int_coef`, `rnd_feature_dim`, `predictor_update_proportion` |
| `ls20_vec_env.py`, `evaluator.py`, `metrics.py`, `device.py` | unchanged from exp_010 |

`exp_012_1_rnd_baseline/` itself is just a thin `config.py` + `train.py` +
`debug_runner.py` (the exp_010 per-variant pattern).

---

## 8. Metrics (logged to `runs/<run>/metrics.jsonl`)

exp_010's full record **plus** the RND/dual-stream additions:

- **`env_steps_to_first_extrinsic_reward`** — the headline metric: cumulative env
  steps when the first `r^E=1` is observed in training. Compared against the
  ~50k random baseline and exp_010_0. `null` until the first reward.
- `intrinsic_reward_mean` / `intrinsic_reward_std` — raw `i_t` before
  normalisation (watch it **decay** as the predictor fits).
- `intrinsic_return_std` — the running normaliser's current value.
- `rnd_predictor_loss` — the distillation loss (mirrors `intrinsic_reward_mean`).
- `value_loss_ext` / `value_loss_int`, `v_ext_mean` / `v_int_mean`.
- `adv_ext_mean` / `adv_int_mean` (post-combination diagnostics).
- inherited: `policy_loss`, `policy_entropy`, `approx_kl`, `clipfrac`,
  `grad_norm_total`, `mean_feature_cosine`, `train_success_rate`, `sps`, and on
  eval cadence `success_rate`, `avg_steps_to_solve`, `min_steps_to_solve`,
  `truncation_rate`, `feat_std/effective_rank/pairwise_l2`.

`success` is `env.level_completed` at a terminal step (same as exp_010).

---

## 9. How to run

From the repo root (`Code Repo/`). Append `--smoke` for a few-update plumbing
run (< 1 min, CPU/MPS).

```bash
# 12_1 — RND exploration baseline on real LS20
uv run python -m JEPA.experiments.exp_012_ls20_intrinsic_exploration.exp_012_1_rnd_baseline.train
uv run python -m JEPA.experiments.exp_012_ls20_intrinsic_exploration.exp_012_1_rnd_baseline.train --smoke
```

Dashboard wiring is identical to exp_010 §7 (port 8787; flat
`checkpoints/step_*.pt` + `runs/<run>/metrics.jsonl`; `debug_runner.py` sets the
ViT/JEPA capability flags `False`). Because exploration is high-variance,
**run multiple seeds** and report the distribution of
`env_steps_to_first_extrinsic_reward`, not a single number.

---

## 10. Expected outcomes

| hypothesis | what would falsify it |
|---|---|
| RND reaches its **first extrinsic reward in markedly fewer env steps than the ~50k random baseline** (and fewer than exp_010_0), because the bonus drives the agent through the maze instead of waiting for a lucky terminal hit | first-reward step is no better than ~50k / exp_010_0 → the bonus is not steering exploration usefully on LS20 (e.g. novelty saturates before reaching the goal corridor) |
| `intrinsic_reward_mean` **rises early then decays** as `f̂` distils `f` over visited states | flat/rising-forever intrinsic reward → predictor not learning (lr/capacity bug) or the "noisy-TV" pathology (constant novelty from stochastic frame content) |
| the **non-episodic** intrinsic stream + dual heads matter: an ablation that makes intrinsic episodic / single-head should explore worse | no difference → on this short-horizon env the dual-stream machinery is inert and a single head would do |

Single level (LS20 L1). Multi-seed strongly recommended given exploration
variance (cf. the 12_0/12_1 "Caution: run multiple seeds" note).

---

## 11. Caveats / limitations

- **Exploration is high-variance.** A single seed says little; the metric is a
  *distribution* over seeds of steps-to-first-reward.
- **Noisy-TV risk.** RND can fixate on irreducibly stochastic frame content. LS20
  frames include a step-counter UI region (rows 61–62) that changes every step —
  a potential cheap novelty sink. If `intrinsic_reward_mean` never decays,
  masking those rows from the RND input is the first thing to try.
- **Reused encoder, not the paper's Nature CNN, and 256-d not 512-d** (§6) — this
  is a baseline-consistency choice, not an architecture search.
- **Single level, terminal-only extrinsic reward** — same scope caveats as
  exp_007/010. No generalisation or transfer claims.
- **Directory/name provisional.** Parent `exp_012_ls20_intrinsic_exploration`
  and sibling `exp_012_0_icm_baseline` names are assumed to match the 12_0/12_1
  plan; rename if the repo settles on different slugs.
```
