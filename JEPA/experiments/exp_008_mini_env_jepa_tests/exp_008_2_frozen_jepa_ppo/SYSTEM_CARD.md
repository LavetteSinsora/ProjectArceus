# exp_008_2 — Frozen-JEPA PPO vs joint-training baseline

> Does the representation a JEPA learns on *off-policy random-agent
> data* help downstream policy learning, compared to training the
> encoder and policy jointly from scratch on the same env?
>
> Parent: [exp_008_mini_env_jepa_tests/SYSTEM_CARD.md](../SYSTEM_CARD.md).
> Sibling: [exp_008_1_JEPA_overfit_test](../exp_008_1_JEPA_overfit_test/).

---

## 1. Motivation

### 1.1 The question

The original reason for putting JEPA in front of a policy at all
([exp_007_3](../../exp_007_mini_env_cnn_ppo_baseline/exp_007_3_jepa_sg/),
[exp_007_4](../../exp_007_mini_env_cnn_ppo_baseline/exp_007_4_jepa_sg_idm/))
was to learn a representation that is **transferable** — a set of
object/edge detectors that are not specialised to one task's reward
signal. exp_008_1 is currently testing whether the *joint*-trained
JEPA from exp_007_4 has, in fact, ended up specialised in the wrong
sense — overfit to the narrow trajectory tube the trained policy
actually visits.

`exp_008_2` asks the complementary, *constructive* question:

> If we train JEPA on a distribution where the overfit-to-policy
> failure mode is impossible by construction — i.e. on
> uniform-random-agent data — and then freeze it, is the resulting
> encoder a **useful** scaffolding for PPO? Better than, comparable
> to, or worse than letting the encoder and policy learn jointly from
> scratch?

Both directions of the answer matter:

- **Helpful reading.** The JEPA-learned encoder has good edge / object
  detectors for the task before PPO ever runs. PPO only has to learn a
  policy head on top, which should converge in fewer environment
  steps. Even if the absolute final reward is the same, the
  sample-efficiency gap is the headline number.
- **Not-helpful reading.** A jointly-trained encoder is shaped by the
  reward gradient and so ends up more task-aligned than any
  reward-agnostic encoder can be. Random-data JEPA learns generic
  geometry that PPO has to *re-shape* through its detached path —
  possibly slower than just learning a CNN from scratch.

The point of the experiment is to make this empirically decidable
inside the cheap mini-env, before any larger-scale claim is made.

### 1.2 Why two env configs

A single env tells us very little about generalisation. The notebook
proposal specifies two task variants:

- **simple (1 rotation needed)** — `simple_1_rotation.json`. The
  player must perform one rotation to align with the goal. This is
  the env every `exp_007_*` run was trained in, so an established
  joint-training baseline already exists.
- **hard (2 rotations needed)** — `simple_2_rotation.json`. Same
  grid, same walls, same goal cell, same cross cell — only the
  initial player rotation differs (180° instead of 270°), so the
  optimal trajectory now requires *two* rotation actions instead of
  one. No `exp_007_*` baseline exists for this env; a joint-training
  control is trained here for the first time as part of this
  experiment.

Running the comparison in both configs lets us separate two effects
that look identical on a single env: (a) does the frozen JEPA help on
the env it was *collected on*, and (b) does the help, if any, hold up
as the task gets harder.

---

## 2. What is and is not being tested

**Tested:** the sample efficiency and final success rate of PPO when
its encoder is a frozen JEPA trained on uniform-random-agent data,
compared to PPO with the same architecture but encoder learned
jointly from scratch. In each of two env configs (1-rot and 2-rot).

**Not tested:**

- Whether *intrinsic curiosity*-collected data gives a better JEPA
  than uniform-random data. (Future work, §8.)
- Whether the same conclusion holds when the JEPA encoder is allowed
  to *unfreeze* late in PPO training, or partially trained. The
  binary frozen/joint comparison is the whole point here.
- Whether a different downstream algorithm (DQN, SAC, model-based
  search) would extract more value from the frozen JEPA than PPO
  does.
- Whether the result transfers to larger envs (full LS20, Atari).
  Mini-env, Level 1 family only.

---

## 3. Experiment matrix

Four PPO runs in total — two envs × two encoder treatments — plus one
JEPA pretraining run per env (data is env-specific, see §4.2). The
`exp_007_0_naive` run in the 1-rot env is *reused* as the
joint-training control for that env, so only three new PPO runs are
trained here.

| run id                              | env config                  | encoder                                | role                              |
|-------------------------------------|-----------------------------|----------------------------------------|-----------------------------------|
| `exp_007_0_naive` (reused)          | `simple_1_rotation.json`    | CNN, joint                             | baseline, 1-rot                   |
| `008_2_frozen_jepa_ppo__1rot`       | `simple_1_rotation.json`    | JEPA on random@1rot, **frozen**        | treatment, 1-rot                  |
| `008_2_joint_cnn_ppo__2rot`         | `simple_2_rotation.json`    | CNN, joint                             | baseline, 2-rot (new)             |
| `008_2_frozen_jepa_ppo__2rot`       | `simple_2_rotation.json`    | JEPA on random@2rot, **frozen**        | treatment, 2-rot                  |

Every run's checkpoint directory name includes its **env tag**
(`__1rot` / `__2rot`) so the `(run, env)` mapping is unambiguous from
the filename alone. The `config` dict embedded in each checkpoint
also carries the absolute `level_path` it was trained against.

### 3.1 Naming and labelling rules

To keep this experiment family auditable, every saved artefact
follows the same labelling convention:

- Run directory: `<run_id>_<YYYYMMDD>_<HHMMSS>/`
- Checkpoint config field: `cfg.level_path` (already in
  `shared.config_base.Config`).
- JEPA buffers and checkpoints: file basename includes `__1rot` or
  `__2rot` and the random-seed used to collect data.
- Plots and tables in `results/` are produced from these tags — never
  from manual filename joins.

---

## 4. Method

### 4.1 What we call the "JEPA"

For this experiment, **JEPA = encoder + predictor**, where
*predictor* is the union of:

- the **forward state predictor**
  `ActionConditionedPredictor(h_t, a) → ĥ_{t+1}` from exp_007_3, and
- the **inverse dynamics model**
  `InverseDynamicsModel(h_t, h_{t+1}) → â` from exp_007_4.

Both heads are trained jointly with the encoder during pretraining;
both are discarded at PPO time. Only the encoder weights move forward
into the frozen-encoder PPO run. (The predictor heads are kept on
disk for diagnostics — they are what would be used if we later wanted
to score predictive loss on the eval distribution the way exp_008_1
does.)

The encoder architecture, action-conditioned predictor, and IDM are
re-used verbatim from
[shared/model.py](../../exp_007_mini_env_cnn_ppo_baseline/shared/model.py)
and the exp_007_3 / exp_007_4 module files. No new model code is
introduced.

### 4.2 JEPA pretraining (per env)

Per env config `c ∈ {1rot, 2rot}`:

1. **Collect.** Run `n_envs=8` parallel mini-envs with uniform-random
   actions from `{0, 1, 2, 3}`. Drain into a flat buffer until
   `N_pretrain` valid `(obs_t, a_t, obs_{t+1})` tuples are stored.
   `done=True` transitions are skipped (they correspond to a reset,
   not a real transition, same rule as exp_008_1).
2. **Train.** Optimise the JEPA loss
   `L = L_jepa + λ_idm · L_idm`
   where
   - `L_jepa = MSE(predictor(encoder(obs_t), a_t),
     sg(encoder(obs_{t+1})))`
   - `L_idm = CE(idm(encoder(obs_t), encoder(obs_{t+1})), a_t)`
   `sg(·)` is stop-gradient on the target branch (matches exp_007_3
   and exp_007_4). All three modules — encoder, forward predictor,
   IDM — receive gradients. There is no PPO loss and no reward signal
   at all in this phase.
3. **Checkpoint.** Save `(encoder_state_dict, predictor_state_dict,
   idm_state_dict, cfg)` periodically. The final encoder is what gets
   handed to the frozen-encoder PPO run.

Default sizes (subject to change after smoke run):

- `N_pretrain = 200_000` transitions (~4760 random episodes at 42
  steps).
- Batch size 256, Adam, `lr = 3e-4`, `λ_idm = 1.0`, train until
  `L_jepa` plateaus on a held-out slice of the same random buffer
  (target ~10 epochs).
- Determinism: fixed seed per env tag, recorded in
  `results/run_meta.json`.

### 4.3 Frozen-encoder PPO (treatment)

Take the pretrained encoder for env `c`, set `requires_grad=False` on
every parameter, set it to `.eval()`, and plug it into the
`ActorCritic` module in place of the freshly-initialised CNN encoder.
The policy head and value head are reinitialised from scratch.

PPO training otherwise follows exactly the
[exp_007_0_naive](../../exp_007_mini_env_cnn_ppo_baseline/exp_007_0_naive/)
recipe via `shared/trainer.train(cfg)`:

- `reward_mode = "terminal_only"`
- `n_envs = 8`, `rollout_steps = 128` (1024 transitions/update)
- `total_env_steps = 1_000_000`
- `learning_rate = 3e-4` on the policy/value heads only
- `minibatches = 4`, `epochs = 2`, GAE, clipped surrogate

`level_path` is the only `Config` field that differs between the
1-rot run and the 2-rot run.

### 4.4 Joint-training PPO (baseline)

- **1-rot env.** Already trained; we reuse the existing
  `exp_007_0_naive` final checkpoints and metrics. No re-run.
- **2-rot env.** New. Train exp_007_0_naive **with one change**:
  `cfg.level_path = "mini_env/configs/level_01/simple_2_rotation.json"`,
  and the run is tagged `..._2rot`. Everything else — architecture,
  hyperparameters, seed strategy — matches the 1-rot baseline so the
  only varying factor between the two baselines is the env. This run
  has no JEPA loss and no frozen modules; the CNN encoder is updated
  by PPO from random init.

---

## 5. Metrics

For each of the four PPO runs:

- **Sample efficiency:** mean episode return / success rate as a
  function of `env_step`. The headline plot is two panels (1-rot,
  2-rot), each overlaying the frozen-JEPA-PPO curve with the joint
  baseline curve.
- **Steps-to-solve:** number of environment steps until rolling 100-
  episode success rate crosses 90% / 99% (whichever is reached). NaN
  if never reached within the budget.
- **Final 100-episode success rate** at end of training.
- **Average solve length** in steps at end of training. The optimal
  solve length differs between envs (1-rot ≈ 13, 2-rot ≈ 14+), so
  this is reported per env.

For the two JEPA-pretrain runs (one per env), independently:

- `L_jepa` and `L_idm` over pretraining steps (sanity: the loss
  should fall and plateau cleanly; if it doesn't, the pretrained
  encoder is not "good" in any meaningful sense and the downstream
  PPO comparison is uninformative).
- IDM accuracy on a held-out slice of the random buffer (bounded
  companion to `L_idm`).

---

## 6. Expected outcomes

| outcome                                    | what we'd see                                                                                                                                                                                                                                        | implication                                                                                                                                                            |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **JEPA helps**                             | Frozen-JEPA PPO reaches 90% success in noticeably fewer env steps than the joint baseline, in at least one env (probably the harder 2-rot env, where representation learning is the bottleneck).                                                     | The JEPA representation transfers — frozen random-data JEPA is a useful warm start. Motivates investing in better data-collection policies (curiosity, §8) for JEPA.   |
| **JEPA doesn't help**                      | Both curves are within noise of each other across both envs.                                                                                                                                                                                         | A random-data JEPA encodes the env's *geometry* but not anything the policy couldn't have learned with the reward gradient just as cheaply. Re-examine the motivation. |
| **JEPA hurts**                             | Frozen-JEPA PPO is slower, or plateaus below the joint baseline. Worse on 2-rot than 1-rot.                                                                                                                                                          | The frozen representation is missing information the policy actually needs. May be that random data under-samples the goal cell or the rotation-aligned states.       |
| **Asymmetric (helps on 2-rot, not 1-rot)** | The interesting case. Frozen JEPA neutral on 1-rot, positive on 2-rot. Consistent with the story that representation pretraining helps most when the task is hard enough that PPO would otherwise spend many steps on representation by itself.      | Headline result. Sets up the curiosity-based follow-up as the natural next step.                                                                                       |

Any outcome with the JEPA pretraining losses *not* plateauing is
reported with a caveat that the encoder is not "good" in the sense
the experiment intends, and the PPO comparison should be re-run after
fixing pretraining.

---

## 7. How to run

From the repo root (`Code Repo/`), with `uv` per the project
convention:

```bash
# 1. Collect random-agent data, per env. Re-runs are idempotent on the
#    seed; output buffer is tagged __1rot / __2rot.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.collect --env 1rot
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.collect --env 2rot

# 2. Train JEPA (encoder + forward predictor + IDM) on the collected
#    buffer for each env. Saves encoder state dict to
#    results/<env_tag>/encoder_final.pt.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_jepa --env 1rot
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_jepa --env 2rot

# 3. Frozen-encoder PPO, per env.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_ppo --env 1rot
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_ppo --env 2rot

# 4. Joint-training baseline in the 2-rot env (the 1-rot baseline
#    reuses exp_007_0_naive; do not re-run).
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_baseline --env 2rot

# 5. Plot. Builds the headline 2-panel success-rate-vs-env-step figure
#    and writes summary.json into results/.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.plot
```

Smoke variants (`--smoke` flag) cap each phase at a handful of
updates so the whole pipeline runs end-to-end in under a minute on
M3 Pro MPS, for plumbing verification.

Outputs land in
`JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_2_frozen_jepa_ppo/results/`,
split into `<env_tag>/` subdirectories.

---

## 8. Caveats and future work

- **Single seed per cell of the matrix.** Four PPO runs and two JEPA
  pretrains, one seed each — sufficient for a qualitative read but
  not for any quantitative significance claim. A multi-seed extension
  is a prerequisite for publishing any of these numbers.
- **One specific off-policy distribution.** Uniform-random is the
  obvious baseline but it under-samples states deep into the env that
  a smarter exploration policy would reach. If frozen-JEPA-PPO
  underperforms, this is a plausible confound, not necessarily a
  refutation.
- **Future work — curiosity-driven JEPA data.** A natural follow-up
  is to replace the uniform-random data-collection step with a policy
  driven by an intrinsic-curiosity signal (e.g. prediction error of
  the *current* JEPA), and ask whether the resulting encoder is more
  helpful for downstream PPO than the random-data one. This would be
  exp_008_4 (or similar); not designed here, only flagged so that the
  reader knows the random-agent choice was deliberate-and-limited,
  not the only option considered.
- **No causal claim about features.** Even if frozen-JEPA-PPO is
  faster, this experiment does not by itself identify *which*
  features the JEPA contributed. A linear-probe analysis on top of
  the frozen encoder (predicting player cell, rotation, cross
  alignment) would be the right tool; left for a separate
  sub-experiment.
- **Mini-env, Level 1 only.** Same scope as the rest of exp_007 and
  exp_008.
