# exp_011_0 — ICM baseline (Pathak 2017 curiosity) on the *real* LS20 game

> exp_010 established that on the real 64×64 LS20 game with **terminal-only**
> reward, plain CNN+PPO and the JEPA-encoder variants are all bottlenecked on
> the same thing: **exploration**. PPO only learns *after* it has stumbled onto
> the first reward, and on a sparse 64×64 puzzle that first reward can take a
> very long time to appear by chance. exp_011 is the "exploration methods"
> series, where we bolt a dedicated exploration mechanism onto the exp_010
> recipe and measure whether it shortens the time to that first reward.
>
> exp_011_0 is the first of those methods: a **faithful replication of the
> Intrinsic Curiosity Module** (ICM; Pathak, Agrawal, Efros, Darrell, ICML
> 2017, *Curiosity-driven Exploration by Self-supervised Prediction*,
> arXiv:1705.05363). A sibling experiment (exp_011_1, Go-Explore) is planned
> under the same parent and is **out of scope here**.

---

## 1. Motivation

### 1.1 Why exploration is the binding constraint

In a reward-sparse environment, RL is not the hard part — **finding the first
reward is**. Once any reward signal exists, PPO has a gradient to climb. The
whole difficulty of LS20 Level 1 under terminal-only reward is that a uniform-
random agent rarely completes the level at all, so for a long time every
rollout return is exactly zero and the policy gradient is identically zero
(only the entropy bonus moves the weights). This is the Montezuma's-Revenge
regime, scaled down.

ICM attacks exactly this: it manufactures a **dense intrinsic reward** from the
agent's *prediction error about the consequences of its own actions*, so the
policy has something to climb even before it has ever seen the extrinsic +1.

### 1.2 Why ICM specifically (and what makes it clever)

The naive way to reward "surprise" is prediction error in **pixel space** —
but that rewards the agent for staring at anything it cannot predict, including
intrinsically unpredictable noise (the paper's "tree leaves in a breeze"
trap). ICM's contribution is to compute prediction error in a **learned
feature space** that is trained, via an **inverse dynamics model**, to encode
*only the parts of the observation the agent can control or that affect it*. A
feature φ(s) that is forced to be sufficient for predicting the action that
took s→s' has no incentive to represent uncontrollable distractors. Forward-
prediction error in that φ-space is therefore a curiosity signal robust to
noise.

For LS20 the "uncontrollable distractor" analogue is the **step-counter UI rows
(61–62)** that change every frame regardless of action — exactly the kind of
nuisance variation ICM's inverse-model feature space is designed to ignore.
That makes LS20 a genuinely appropriate test of the ICM mechanism, not just a
sparse-reward stress test.

### 1.3 The zero we measure against

| reference | meaning |
|---|---|
| **pure-random exploration** | per the experiment brief, ~**50,000 env-steps** expected before the first extrinsic reward (geometric over the per-episode random success probability). This is the "no exploration mechanism at all" floor. |
| **exp_010_0 (plain CNN+PPO)** | terminal-only PPO on the same env. PPO's entropy bonus is its *only* exploration pressure; whether it beats pure-random-to-first-reward is itself an open exp_010 result. |

ICM is interesting only if it drives the first reward **earlier** than both.

---

## 2. What is and is not being tested

**Tested.** Does adding the ICM intrinsic reward to the exp_010 CNN+PPO recipe
reduce the number of environment steps needed to reach the **first** extrinsic
reward on real LS20 Level 1, and then reach a high success rate?

**Faithful to the paper.** The ICM module structure — separate self-supervised
φ encoder, inverse dynamics head, forward dynamics head, forward-error-as-
reward, the (1−β)·L_I + β·L_F loss with β = 0.2 — is reproduced exactly.

**Deliberately adapted to our series (documented deviations in §7).** The RL
algorithm (PPO, not A3C), the φ-encoder *architecture* (the exp_007/exp_010
CNN, not the paper's 42×42 ELU stack), single-frame input (no 4-frame stack),
and 4 cardinal actions (not VizDoom's/Mario's action set). These keep exp_011_0
directly comparable to exp_010 so any difference is attributable to ICM, not to
a divergent backbone.

**Not tested.** Generalisation/transfer, Go-Explore (exp_011_1), pixel-space
curiosity (the paper's ICM-pixels ablation), or β/η sweeps. Single level.

---

## 3. The two networks (they do NOT share weights)

The single most important fidelity point: **the policy and the curiosity
feature space are separate networks.** This is exactly the paper's design and
the answer chosen for this card.

```
                 ┌─────────────────────────┐
   s_t  ───────▶ │  PolicyEncoder (PPO)     │ ─▶ trunk h_t ─▶ π(a|s), V(s)
                 │  = exp_010 ActorCritic   │
                 └─────────────────────────┘
                 ┌─────────────────────────┐
   s_t, s_{t+1}▶ │  φ encoder (ICM-only)    │ ─▶ φ(s_t), φ(s_{t+1})
                 │  trained ONLY by L_I,L_F │       │        │
                 └─────────────────────────┘       ▼        ▼
                                       inverse g(φ_t,φ_{t+1})→â_t   (L_I)
                                       forward f(φ_t,a_t)→φ̂_{t+1}   (L_F)
                                                              │
                                  r^i_t = (η/2)‖φ̂_{t+1}−φ(s_{t+1})‖²
```

The PPO encoder receives **no gradient** from any ICM loss; the φ encoder
receives **no gradient** from PPO. They couple in exactly one place: the scalar
intrinsic reward r^i is added to the extrinsic reward before GAE. This is the
faithful ICM information flow and matches Figure 2 of the paper.

---

## 4. Architecture

### 4.1 Policy / value network (unchanged from exp_010)

The exp_010 `ActorCritic` verbatim (see
[exp_010 model.py](../../exp_010_ls20_cnn_ppo_jepa/shared/model.py)):

```
input  : (64,64) uint8 palette frame, indices [0,15]  →  one-hot (16,64,64)
encoder: Conv2d(16→32,k3,s1,p1) ReLU                  # 64×64
         Conv2d(32→64,k3,s2,p1) ReLU                  # 32×32
         Conv2d(64→64,k3,s2,p1) ReLU                  # 16×16
         Conv2d(64→64,k3,s2,p1) ReLU                  #  8×8
         Flatten → Linear(4096→256) ReLU              # trunk h
heads  : policy_head Linear(256→4)  (orthogonal gain 0.01)
         value_head  Linear(256→1)  (orthogonal gain 1.0)
init   : orthogonal, gain √2 on ReLU layers; no BatchNorm, no pooling
```

The step-counter UI rows (61–62) are **not** masked from the policy input
(same as exp_010 §4).

### 4.2 ICM feature encoder φ (separate instance, same CNN family)

φ is a **second, independent** copy of the same `CNNEncoder` architecture, with
its own weights, trained **only** by the inverse + forward losses. We
deliberately reuse the exp_007/exp_010 CNN here (the project's standing encoder)
rather than the paper's 42×42 / ELU / 288-d stack, so φ and the policy encoder
have identical capacity and the comparison to exp_010 stays clean.

```
φ : one-hot(16,64,64) → 4 strided convs (16→32→64→64→64) → Flatten
                       → Linear(4096→256) ReLU
φ-dim = 256          # the paper's φ is 288-d; ours is the 256-d trunk (see §7)
```

### 4.3 Inverse dynamics model g  (faithful structure)

Predicts the action that produced the transition, from the two φ encodings:

```
g : concat[φ(s_t); φ(s_{t+1})]  (512)
      → Linear(512→256) ReLU      # paper: FC 256
      → Linear(256→4)             # logits over the 4 actions
L_I = cross_entropy(g(φ_t,φ_{t+1}), a_t)        # eq. (3), discrete ⇒ MLE/softmax
```

### 4.4 Forward dynamics model f  (faithful structure)

Predicts the *next* φ from the current φ and the action:

```
f : concat[φ(s_t); one_hot(a_t)]  (256+4 = 260)
      → Linear(260→256) ReLU       # paper: FC 256
      → Linear(256→256)            # predicts φ̂_{t+1}, dim = φ-dim
L_F = ½ · ‖ φ̂_{t+1} − φ(s_{t+1}) ‖²₂            # eq. (5)
```

The action is fed as a **one-hot vector** (faithful to the paper), not a learned
embedding (which the exp_010 JEPA predictor used). The forward target
`φ(s_{t+1})` is produced by the same φ encoder; per eq. (5)/(6) the forward
loss flows into both f **and** φ — i.e. φ is shaped by the inverse loss *and*
the forward loss jointly, exactly as in the paper (the inverse loss is what
keeps the forward objective from collapsing φ to a constant).

### 4.5 Intrinsic reward

```
r^i_t = (η/2) · ‖ φ̂_{t+1} − φ(s_{t+1}) ‖²₂          # eq. (6), η > 0
total reward fed to GAE :  r_t = r^e_t + r^i_t      # single reward stream
```

`r^e_t` is the LS20 terminal reward (+1 on level completion, else 0). There is a
**single value head** estimating the value of the **summed** reward stream
(faithful ICM; no separate intrinsic value head).

**On η.** The paper states only η > 0 and gives no numeric value (it is the one
unspecified scalar in ICM). η is the single free knob here. Default
`eta = 0.01`, but it must be **calibrated against the +1 terminal reward**: at
initialisation L_F over a 256-d φ can be large, so we log `intrinsic_reward_mean`
and set η so the per-step intrinsic reward is ≈ 10⁻² early — large enough to
give PPO a gradient in the all-zero-extrinsic regime, small enough that a real
level completion (+1) still dominates. We do **not** apply RND-style running
normalisation (it is not part of faithful ICM); if intrinsic reward proves
wildly scale-unstable across seeds, normalisation is noted as a fallback, not
the default.

---

## 5. Optimisation

### 5.1 PPO (untouched exp_010 recipe)

Identical to [exp_010 §5](../../exp_010_ls20_cnn_ppo_jepa/SYSTEM_CARD.md):
8 synchronous real-LS20 envs × `rollout_steps=128` → 1024 transitions/update;
γ=0.99, λ_GAE=0.95, clip ε=0.2, value-clip 0.2, c_v=0.5, c_ent=0.01,
grad-clip 0.5, Adam **lr 3e-4**, 4 epochs × 4 minibatches. Truncation
(`max_episode_steps=200`) is treated as `done=True`. The **only** change is that
the reward GAE consumes is `r^e + r^i` instead of `r^e`.

### 5.2 ICM (auxiliary, its own optimiser)

A **second Adam optimiser** owns {φ, g, f} and minimises, on the same rollout:

```
L_ICM = (1 − β)·L_I + β·L_F ,   β = 0.2          # eq. (7) inner terms
Adam lr = 1e-3                                   # the paper's ICM-side lr
```

ICM is updated on the rollout's transitions **after** the intrinsic reward for
that rollout has already been computed (otherwise we would shrink the very
prediction error we just rewarded). Episode-ending steps are **excluded** (their
`s_{t+1}` is a reset frame, not a real transition — same rule as exp_010's
`jepa.py`). Default `icm_epochs = 1` pass over the rollout (a knob); this keeps
the forward model from over-fitting and prematurely flattening the curiosity
signal.

### 5.3 On β and λ

- **β = 0.2** — reproduced exactly; weights inverse vs forward loss as in eq. (7).
- **λ = 0.1** (paper) weights the policy-gradient term against the ICM losses
  *inside the single joint A3C objective*. Under our chosen **separate-optimiser
  auxiliary** structure there is no single objective, so λ is not applied as a
  literal coefficient; its role — keep the intrinsic/representation learning
  from overwhelming the task objective — is instead carried by (a) the intrinsic
  scale η and (b) the two independent learning rates (PPO 3e-4, ICM 1e-3). This
  is the one structural place where "faithful to ICM" and "keep the exp_010 PPO
  recipe" genuinely diverge, and it is called out as a deviation in §7.

---

## 6. Metrics (logged to `runs/<run>/metrics.jsonl`, dashboard format)

**Headline (the brief's requested metric):**
- `env_steps_to_first_reward` — environment-step index at which **any** env
  first returns a nonzero extrinsic reward (first LS20 completion). Reported per
  seed and as mean ± s.e.m. across seeds; compared to the ~50k pure-random
  reference and to exp_010_0.

**Performance (periodic stochastic eval, `eval_every`):**
`success_rate`, `avg_steps_to_solve`, `min_steps_to_solve`,
`mean_episode_steps`, `truncation_rate` — same as exp_010 §6.

**ICM health (per update):**
- `intrinsic_reward_mean` / `intrinsic_reward_std` — magnitude of r^i; expected
  to start high and **decay** as the forward model learns the controllable
  dynamics (the canonical ICM curve).
- `forward_loss` (L_F), `inverse_loss` (L_I).
- `inverse_acc` — accuracy of g at predicting a_t. This is the **direct health
  check on φ**: if inverse accuracy never rises above chance (0.25 for 4
  actions), φ is not learning a controllable-feature space and the curiosity
  signal is meaningless.

**PPO diagnostics (per update):** `policy_loss`, `value_loss`, `policy_entropy`,
`approx_kl`, `clipfrac`, `grad_norm_total`, plus the exp_010 feature-collapse
diagnostics (`feat_std`, `feat_effective_rank`, `feat_pairwise_l2`) computed on
**both** encoders so we can see whether φ and the policy encoder diverge in
geometry.

`success` is `env.level_completed` at a terminal step.

---

## 7. Documented deviations from Pathak et al. 2017

| # | paper | exp_011_0 | why |
|---|---|---|---|
| 1 | A3C, 20 async workers | **PPO**, 8 sync envs | the standing exp_007–010 RL backbone; keeps comparison to exp_010 exact |
| 2 | φ: 4×conv(32,3,s2,p1)+ELU → 288-d, on 42×42 input | the exp_010 CNN (ReLU, orthogonal) → 256-d, on 64×64 | "encoder same as 7_0" — identical capacity to the policy encoder for a clean ablation |
| 3 | inverse/forward heads use **ELU** | **ReLU** | match the codebase's activation; cosmetic |
| 4 | 4-frame grayscale stack, action-repeat 4/6 | **single** one-hot frame, no action repeat | LS20 is fully observable from one frame (exp_007 §9 rationale); keeps input identical to exp_010 |
| 5 | single joint objective, λ=0.1, lr 1e-3 | two optimisers: PPO lr 3e-4 + ICM lr 1e-3; λ not applied literally (§5.3) | preserve the validated exp_010 PPO recipe; λ's role absorbed by η + separate lrs |
| 6 | η unspecified | η = 0.01 default, calibrated against the +1 terminal (§4.5) | the paper leaves η free; we pin and log it |

Items 1–4 are surface adaptations that keep exp_011_0 comparable to its own
series. Item 5 is the only deviation that touches the *training dynamics*, and
it is the deliberate consequence of the "keep exp_010 PPO + ICM auxiliary"
design choice. The **ICM mechanism itself** — separate inverse-model feature
space, forward-error-as-reward, (1−β)L_I + β L_F with β=0.2, r=r^e+r^i — is
reproduced faithfully.

---

## 8. How to run

> **Status: design / system card only.** The shared `icm.py` module, per-variant
> config, and `train.py` are not yet implemented; this card specifies them. The
> commands below are the intended entrypoints, following the exp_010 layout
> (run from the repo root `Code Repo/`; append `--smoke` for a plumbing run).

```bash
# exp_011_0 — ICM + PPO on real LS20, one seed
uv run python -m JEPA.experiments.exp_011_ls20_icm.exp_011_0_icm_baseline.train

# multi-seed sweep (exploration is high-variance — run ≥3–5 seeds, see §9)
for s in 0 1 2 3 4; do
  uv run python -m JEPA.experiments.exp_011_ls20_icm.exp_011_0_icm_baseline.train --seed $s
done
```

Surfaces on the **main JEPA dashboard** (port 8787) via the same flat
`checkpoints/step_<env_step>.pt` + `runs/<run>/metrics.jsonl` layout exp_010
uses; ship a `debug_runner.py` re-export so checkpoints play back on real LS20.

---

## 9. Expected outcomes

| signal | hypothesis | what would falsify it |
|---|---|---|
| `env_steps_to_first_reward` | ICM reaches the first reward **well under** the ~50k random floor and earlier than exp_010_0 — the intrinsic reward steers exploration toward novel transitions instead of waiting on chance | ICM ≈ random/exp_010_0 → the curiosity signal is not informative on LS20 (e.g. φ collapsed, η mis-scaled) |
| `inverse_acc` | rises clearly above 0.25 within early training — φ is learning controllable features and ignoring the UI-row distractor | stays at chance → inverse model isn't learning; r^i is noise; everything downstream is suspect |
| `intrinsic_reward_mean` | starts high, **decays** as f learns the dynamics | flat-high forever → forward model not learning (lr/scale bug); flat-zero → η too small or φ collapsed |
| `success_rate` | climbs sooner than exp_010_0 once the first reward is found | no improvement over exp_010_0 despite earlier first-reward → exploration helped find reward but PPO still can't exploit it |

---

## 10. Caveats / limitations

- **Exploration is high-variance — multiple seeds are mandatory.** Per the
  brief, curiosity/exploration results swing hard across seeds; a single run is
  not interpretable. Report ≥3–5 seeds with mean ± s.e.m. (the paper itself
  reports 3 runs and notes the very-sparse case succeeds in only 66% of runs).
- **The ~50k random baseline is an estimate.** Prior limited evals on real
  64×64 LS20 saw ~0% random completion, so "first reward in ~50k steps" is a
  target/expectation from the brief, not a measured constant; the binding
  difficulty may be reaching the first reward *at all*. We log
  `env_steps_to_first_reward` per seed precisely so this is observed, not assumed.
- **η is the one un-pinned knob.** The headline result is sensitive to it; we
  fix a default and a calibration recipe (§4.5) but a bad η can mask a working
  ICM. η is logged and, if necessary, re-calibrated once per environment, never
  per seed.
- **Single seed family ≠ tuned SOTA.** This is a faithful baseline, not an ICM
  hyperparameter search. Single level, LS20 Level 1 only; no transfer claim.
- **λ deviation (§5.3, §7 item 5).** The separate-optimiser structure means we
  do not reproduce the paper's exact joint-objective weighting. If exp_011_0
  underperforms, reproducing the literal single-objective Eq. (7) (one optimiser,
  λ=0.1, lr 1e-3) is the first variant to try (a candidate exp_011_0_b).
```
