# exp_008 — Mini-env JEPA Tests

> A family of *diagnostic* experiments aimed at one observation from
> exp_007_4: the JEPA encoder keeps reshaping its representation long after
> the policy has saturated at ~100% success on MiniLS20 Level 1. Before
> building any new architecture on top of JEPA we want to know whether the
> JEPA we already have is actually learning a transition model, or whether
> it is overfitting to the narrow on-policy trajectory the trained agent
> keeps replaying.

---

## 1. Motivation

### 1.1 The exp_007_4 observation

exp_007_4 (CNN + JEPA + IDM) reaches 100% success rate on Level 1 around
~5000 episodes. After that point:

- The policy is, for practical purposes, deterministic — the same ~13-step
  trajectory is rolled out on essentially every reset.
- Despite training reward having saturated, the JEPA representation keeps
  drifting: feature-pair similarity (a collapse-diagnostic) **continues to
  decrease** for thousands of episodes after success rate plateaus.

Two readings of this are consistent with what we observe:

1. **Healthy reading.** The encoder is still receiving gradient from the
   JEPA loss and is genuinely refining the geometry of its latent space —
   producing finer distinctions between states it has already seen. Under
   this reading, the late-stage encoder is *better*, not just *different*.
2. **Overfit reading.** Because the agent is deterministic, the encoder
   and predictor only ever see one narrow tube of `(s, a, s')` tuples.
   The JEPA loss has no incentive to model the rest of the state space.
   The "continued progress" we see in the diagnostic is the encoder
   carving sharper boundaries between a small finite set of cached states,
   not learning a better world model. Off-trajectory transitions — the
   ones a *better* policy or a downstream planner would actually need —
   would be predicted no better than at initialisation.

These two readings have very different implications for what the JEPA is
worth as a foundation for later work (Dreamer-style imagination rollouts,
model-based search, exploration bonuses derived from prediction error).
Distinguishing between them is a prerequisite for everything in that
direction.

### 1.2 What this experiment family is for

exp_008 is the diagnostic family for the overfit-vs-refinement question.
Each sub-experiment isolates one piece of evidence:

- **008_1** asks the question directly using the *existing* 7_4 run, with
  no retraining: compare predictor and IDM loss on transitions the policy
  produces vs. transitions a uniform-random agent produces.
- Higher-numbered slots are reserved for the followups outlined in §3
  (training a new JEPA on random-agent data; replicating the test on
  exp_007_3 checkpoints; etc.) once we know what 008_1 says.

The whole family is intentionally *offline / cheap*. None of it touches
the env's reward or changes the architecture. The point is to learn
something about what we already have before deciding what to build next.

---

## 2. What is and is not being tested

**Tested:** does the trained 7_4 JEPA predictor (and its IDM head)
generalise to transitions outside the trajectory distribution the policy
actually visits? Concretely: how does the per-transition JEPA MSE and IDM
cross-entropy on uniform-random rollouts compare to the same quantities
on the policy's own rollouts, and how does that gap evolve across the
21 saved checkpoints of the 7_4 run?

**Not tested:** whether a less-overfit JEPA would yield a better policy
(that requires retraining and downstream eval — left for 008_2). Whether
the same pattern holds on harder levels, other env variants, or different
JEPA architectures. Whether IDM-style auxiliary tasks help or hurt the
overfit gap (would require an exp_007_3-only comparison — left for 008_3).

---

## 3. The sub-experiments

| id | status | one-line description |
|---|---|---|
| 008_1 | implemented now | offline: compare JEPA / IDM loss on trained-policy vs uniform-random transitions across the full 7_4 checkpoint sweep |
| 008_2 | reserved | train a new JEPA-only run on data collected by a permanently-random agent; compare its loss on the same two sources to see whether on-policy data is what *causes* the gap |
| 008_3 | reserved | repeat 008_1 against the exp_007_3 (JEPA, no IDM) checkpoint sweep, to isolate whether the IDM head changes the overfit picture |

Only 008_1 is built as part of this writeup. 008_2 and 008_3 are
mentioned so it is clear what 008_1's result will *not* answer on its own.

### 3.1 exp_008_1_JEPA_overfit_test

#### Hypothesis

If the overfit reading from §1.1 is correct, then for any sufficiently
late checkpoint of the 7_4 run we should see:

- `JEPA_MSE(random transitions) >> JEPA_MSE(trained-policy transitions)`
- `IDM_accuracy(random transitions) << IDM_accuracy(trained-policy
  transitions)`
- The gap should *widen* as the checkpoint index grows (because the
  encoder has had more updates to specialise on the trajectory tube).
- Within the random source, per-action breakdown should show the largest
  losses for actions the trained policy rarely takes — those are the
  transitions JEPA was most starved of during training.

If instead both sources show comparable loss across the whole sweep, the
refinement reading wins: the encoder is generalising and the late-stage
drift in feature-pair similarity is genuine geometric refinement, not
overfit.

#### Method

The experiment is a pure offline analysis. No training, no environment
reward changes, no new model code. Per checkpoint in
`exp_007_4_jepa_sg_idm_novfclip_20260525_161738/checkpoints/` (21 files:
`update_000050.pt` through `update_000950.pt`, plus `update_000976.pt`
and `final.pt`):

1. **Load.** `torch.load(...)` and reconstruct the three modules saved by
   exp_007_4 from their state dicts: the `ActorCritic` (CNN encoder +
   policy/value heads), the `ActionConditionedPredictor`, and the
   `InverseDynamicsModel`. All set to `.eval()`.

2. **Collect trained-source transitions.** Run the loaded `ActorCritic`
   in 8 synchronous mini-envs, sampling actions stochastically from
   `Categorical(logits=policy_head(h))`. Drain into a flat buffer until
   50,000 valid `(s, a, s')` transitions are collected. Transitions
   crossing an episode boundary (`done=True`) are discarded — they
   correspond to a reset, not a real env transition, and would not be
   fed to the predictor during training either.

3. **Collect random-source transitions.** Same env setup, but actions
   are drawn uniformly from `{0, 1, 2, 3}`. 50,000 valid transitions.
   This collection is policy-independent and is therefore done **once
   at startup** and reused across all 21 checkpoints.

4. **Score, under `torch.no_grad()`, batched at 1024:**
   - `h_t = encoder(one_hot(obs))`, `h_next = encoder(one_hot(next_obs))`
   - JEPA prediction: `h_pred = predictor(h_t, a)`
   - JEPA per-transition MSE: `((h_pred - h_next.detach()) ** 2).mean(-1)`
   - IDM per-transition CE: `cross_entropy(idm(h_t, h_next), a,
     reduction='none')`
   - IDM per-transition correctness: `idm_logits.argmax(-1) == a`

5. **Aggregate per (checkpoint, source):**
   - Overall mean of JEPA MSE, IDM CE, IDM accuracy.
   - The same three sliced per action `a ∈ {0, 1, 2, 3}`.
   - Action histogram (sanity check — the trained source should be
     visibly skewed; the random source should be uniform).

6. **Dump.** Append rows to `results/per_checkpoint.csv` with columns
   `update, source, action, n_samples, jepa_mse, idm_ce, idm_acc`. Write
   `results/summary.json` with the headline gap numbers from the final
   checkpoint, and `results/run_meta.json` with the config snapshot and
   timestamps.

#### Why the design is what it is

- **The full checkpoint sweep, not just `final.pt`.** A single final
  number could only confirm or deny the gap. The sweep tells us *when*
  the gap emerges — does it appear immediately, does it track the policy
  becoming deterministic, or does it grow steadily across all of
  training? Each of those answers points at a different mechanism.
- **Stochastic sampling for the trained source.** This matches exactly
  what JEPA saw during training (PPO collects with stochastic actions).
  An argmax-deterministic source would understate the trained-side loss
  by visiting *even less* of the state space than training did. We want
  the most charitable possible trained-side number, so the gap is a
  conservative estimate.
- **50k transitions per source.** Roughly 1200 episodes of 42 steps.
  Large enough that per-action slices have a few thousand samples even
  for the rarest action; small enough to run the whole sweep in a few
  minutes on M3 Pro MPS.
- **Random transitions collected once.** Policy-independent. Avoids
  paying for ~95% of the random-rollout wall time. The same fixed
  evaluation set is scored against every checkpoint, which also makes
  the across-checkpoint trend cleaner (no source-sampling noise on the
  x-axis).
- **Skipping `done` transitions.** A `(s_terminal, a, s_reset)` pair is
  not a real env transition; the predictor never sees it during
  training. Including it would inflate both sources' loss roughly
  equally but would obscure the actual physics being modelled.

---

## 4. Architecture (inherited)

No new models. The three modules are taken verbatim from exp_007_4 — see
that experiment's `models.py` and `SYSTEM_CARD.md` §4 for the CNN
encoder, action-conditioned predictor, and inverse-dynamics model. The
only thing this experiment adds is a different *evaluation distribution*
to score them against.

---

## 5. Metrics

Per `(checkpoint, source)`:

- `jepa_mse` — mean of per-transition `‖predictor(h_t, a) − sg(h_{t+1})‖²
  / d`. Directly comparable to the `jepa_loss` field logged in the 7_4
  run's `metrics.jsonl`. Lower is better; the gap of interest is
  `jepa_mse(random) − jepa_mse(trained)`.
- `idm_ce` — mean of per-transition `cross_entropy(idm(h_t, h_{t+1}),
  a)`. Comparable to the `idm_loss` field in 7_4's metrics.
- `idm_acc` — fraction of transitions where `argmax(idm_logits) == a`.
  A bounded sanity-check companion to `idm_ce` that is easier to read.
- `action_histogram` — count of each action in the source. Not a
  diagnostic of the model; a diagnostic of the source itself. Confirms
  the trained-policy distribution is collapsed and the random
  distribution is uniform.

Per `(checkpoint, source, action)`: the same `jepa_mse`, `idm_ce`,
`idm_acc`, restricted to the slice where `a == action`. This is the
per-action breakdown — the place we expect to see the cleanest signature
of overfit if it is real (large random-side loss on the actions the
trained policy under-uses).

---

## 6. Expected outcomes

| reading | predicts |
|---|---|
| overfit | `jepa_mse(random) / jepa_mse(trained)` is large (≥3×) by mid-training and grows toward `final.pt`; `idm_acc(random)` is well below `idm_acc(trained)`; the per-action breakdown shows the largest random-side losses on actions with the smallest entries in the trained-source action histogram |
| refinement | both sources track each other within ~20% across the whole sweep; per-action breakdown is flat |
| mixed | absolute losses on both sources fall in lockstep early in training, then the random source plateaus while the trained source keeps falling — i.e. the encoder *did* generalise initially but specialisation kicks in once the policy locks in |

The "mixed" outcome is the most interesting one for designing 008_2: it
would suggest that the overfit is specifically caused by the late-stage
on-policy phase, not by JEPA training in general, and would directly
motivate a curriculum that injects random data later in training.

---

## 7. How to run

From the repo root (`Code Repo/`):

```bash
# Smoke test — final.pt only, 1024 transitions per source. < 30s.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_1_JEPA_overfit_test.eval_overfit --smoke

# Full sweep — all 21 checkpoints, 50k transitions per source.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_1_JEPA_overfit_test.eval_overfit
```

Outputs land in
`JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_1_JEPA_overfit_test/results/`.

---

## 8. Caveats / known limitations

- **Single training run as the substrate.** The 21 checkpoints come from
  one 7_4 seed. If the overfit gap is real but its size depends on the
  particular trajectory the policy locked into, a single seed will not
  reveal that. A multi-seed 7_4 sweep is a prerequisite for any
  quantitative claim — for now we only claim qualitative direction.
- **Uniform random is one specific alternative distribution.** It is the
  natural "maximum coverage" baseline but it is not the only off-policy
  source worth scoring against. A semi-random policy that biases toward
  reachable but unvisited cells would be a more informative comparison;
  building one is left for a future sub-experiment.
- **No causal claim about downstream policy performance.** Even a large
  overfit gap does not by itself prove the resulting policy is *worse*
  than it would be with a more generalising JEPA — only that the JEPA is
  not the world model we hoped it was. The causal question lives in
  008_2.
- **Mini-env, Level 1 only.** Same scope as exp_007.
