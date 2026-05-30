# exp_008_3 — Encoder transfer: JEPA vs PPO encoder on a *new* env

> Two encoders are learned on the **same** env (`simple_1_rotation`): one by
> JEPA on uniform-random data (the 008_2 encoder), one by CNN+PPO joint
> training (the 007_0 encoder). Move both to a **different** env and train a
> fresh policy on top. Which representation transfers better — the
> reward-agnostic JEPA one, or the reward-shaped PPO one? And does freezing
> vs. fine-tuning the encoder change the answer?
>
> Parent: [exp_008_mini_env_jepa_tests/SYSTEM_CARD.md](../SYSTEM_CARD.md).
> Siblings: [exp_008_2_frozen_jepa_ppo](../exp_008_2_frozen_jepa_ppo/),
> [exp_008_4_pretrained_jepa_unfrozen_ppo](../exp_008_4_pretrained_jepa_unfrozen_ppo/).

---

## 1. Motivation

### 1.1 The question

The whole reason for putting a JEPA in front of a policy
([exp_007_3](../../exp_007_mini_env_cnn_ppo_baseline/exp_007_3_jepa_sg/),
[exp_007_4](../../exp_007_mini_env_cnn_ppo_baseline/exp_007_4_jepa_sg_idm/),
008_2) was the hope that it learns a **transferable** representation — generic
object/edge/geometry detectors that are *not* specialised to one task's reward
signal. So far we have only ever measured the JEPA encoder on the env it was
collected on. That is the wrong test for transfer, and it is also unfair to
JEPA: 008_2 found that on the **same** env a frozen random-data JEPA is a
*worse* warm-start than just joint-training a CNN with PPO (≈2–3× slower to
90%). But "worse on the env you trained the competitor on" is exactly what you
would expect if the competitor is allowed to overfit.

`exp_008_3` runs the comparison the way it should be run:

> Train two encoders on `simple_1_rotation` — one JEPA-on-random, one CNN+PPO.
> Then drop each one into a **new** env it has never seen
> (`hard_1_rotation`, `hard_2_rotation`) and ask how many environment steps a
> fresh PPO policy needs to learn a good policy on top of it.

This neutralises the home-field advantage 008_2 gave the PPO encoder. The
spec's intuition is explicit and worth stating up front:

- **JEPA should transfer better.** It was trained on uniform-random
  transitions, so it has seen a broad slice of the state space and the env's
  shared mechanics (touch-cross, pattern-rotate, walls). It was never shaped
  by one task's reward.
- **The PPO encoder is suspected of overfitting** to the single winning
  trajectory its policy converged on in `simple_1_rotation`. Whatever it
  encodes may be useless — or actively misleading — off that trajectory.

### 1.2 Why two freeze treatments

The same frozen-vs-fine-tuned axis that separated 008_2 from 008_4 is the
second factor here, and it interacts with transfer:

- **Frozen** isolates the *quality of the transferred representation itself*:
  the policy/value heads are the only thing that can adapt, so a bad encoder
  cannot be rescued. This is the cleanest read on "is this representation
  good for the new env".
- **Unfrozen** lets PPO reshape the transferred encoder. This measures the
  encoder as a *warm start* — even a partly-wrong representation can help if
  it is a better-than-random initialisation, and can hurt if it is an
  adversarial one.

### 1.3 Why the PPO encoder is swept at two checkpoints

The overfit suspicion in §1.1 has a direct, testable consequence: an **early**
PPO encoder — taken the moment the policy first solves `simple_1_rotation`,
before thousands of further updates carve the representation down onto the
winning tube — should transfer *better* than the **fully-converged** one. So
the PPO encoder is not a single artefact; it is two: `ppo_early` and
`ppo_final`. If `ppo_early` beats `ppo_final` on transfer, the overfit reading
is supported and the spec's "perhaps select an earlier checkpoint" caveat is
vindicated.

---

## 2. What is and is not being tested

**Tested:** the sample efficiency (and final success rate) of PPO on a *new*
env when its encoder is **transferred from a different env** (`simple_1_rotation`),
across three encoder sources (JEPA-on-random, early-PPO, final-PPO), two freeze
treatments (frozen / unfrozen), and two transfer targets (`hard_1_rotation`,
`hard_2_rotation`). A from-scratch CNN+PPO run on each target env is the
zero-transfer floor.

**Not tested:**

- **Multi-seed significance.** One seed per cell of the matrix, in line with
  the rest of exp_008. Qualitative direction only.
- **Partial unfreezing / layer-wise LR.** The freeze axis is binary here.
- **Transfer beyond the Level-1 family.** Mini-env, 8×8, only.
- **JEPA loss *during* the PPO phase.** That is the exp_007_3/4 joint setup; in
  008_3 the transferred encoder receives only PPO gradients (when unfrozen) or
  none (when frozen). We isolate the effect of the *initial representation*.
- **A feature-level explanation of *why* one transfers better.** We measure
  downstream sample efficiency, not which features carried over. A linear-probe
  follow-up is flagged in §8.

---

## 3. Experiment matrix

All encoders are trained on `simple_1_rotation` and reused as-is — no encoder
is re-trained here. The only training in 008_3 is the downstream PPO runs.

| run id                                       | source      | encoder origin (all @ simple_1_rotation)         | freeze   | target env        | role                  |
|----------------------------------------------|-------------|--------------------------------------------------|----------|-------------------|-----------------------|
| `008_3_transfer__jepa_frozen__hard1`         | `jepa`      | 008_2 `jepa_runs/1rot_*/encoder_final.pt`        | frozen   | hard_1_rotation   | treatment             |
| `008_3_transfer__jepa_unfrozen__hard1`       | `jepa`      | "                                                | unfrozen | hard_1_rotation   | treatment             |
| `008_3_transfer__ppo_early_frozen__hard1`    | `ppo_early` | 007_0_naive first-solve ckpt (`encoder.*`)       | frozen   | hard_1_rotation   | treatment             |
| `008_3_transfer__ppo_early_unfrozen__hard1`  | `ppo_early` | "                                                | unfrozen | hard_1_rotation   | treatment             |
| `008_3_transfer__ppo_final_frozen__hard1`    | `ppo_final` | 007_0_naive `final.pt` (`encoder.*`)             | frozen   | hard_1_rotation   | treatment             |
| `008_3_transfer__ppo_final_unfrozen__hard1`  | `ppo_final` | "                                                | unfrozen | hard_1_rotation   | treatment             |
| `008_3_transfer__scratch_unfrozen__hard1`    | `scratch`   | random init                                      | unfrozen | hard_1_rotation   | **zero-transfer floor** |
| …and the same seven rows for `hard2`         |             |                                                  |          | hard_2_rotation   |                       |

= 3 sources × 2 freeze × 2 envs (12) + `scratch` × 1 freeze × 2 envs (2) =
**14 PPO runs**. `scratch + frozen` is intentionally omitted — a frozen
randomly-initialised encoder is not a meaningful condition.

### 3.1 Naming and labelling rules

Same auditability discipline as 008_2/008_4. Every saved artefact carries its
`<source>_<freeze>__<envtag>` tag in its run-directory name, so the
`(source, freeze, env)` triple is unambiguous from the path alone. Each
checkpoint's embedded `config` dict additionally records `level_path` (the
*target* env) and `encoder_ckpt` (the absolute path of the transferred
encoder, including which `simple_1_rotation` run it came from). Plots and tables
in `results/` are produced from these tags — never from manual filename joins.

---

## 4. Method

### 4.1 Encoder sources and how each is loaded

All three sources resolve to a state-dict that fits `CNNEncoder` verbatim
(one-hot 32×32 → 256-d trunk;
[shared/model.py](../../exp_007_mini_env_cnn_ppo_baseline/shared/model.py)).
Because the architecture is byte-for-byte identical across JEPA and PPO, the
comparison is clean — **only the initial weights differ.**

- **`jepa`.** Reuse 008_2's offline encoder, glob
  `../exp_008_2_frozen_jepa_ppo/jepa_runs/1rot_*/encoder_final.pt` and take the
  latest. The checkpoint stores a bare `encoder_state_dict`, loaded directly:
  `model.encoder.load_state_dict(ckpt["encoder_state_dict"])` — the exact
  pattern already used in
  [008_2's train_ppo.py](../exp_008_2_frozen_jepa_ppo/train_ppo.py).
- **`ppo_final`.** Use `final.pt` from an `exp_007_0_naive` run. exp_007 saves
  the **full ActorCritic** `model_state_dict`, so the encoder is extracted by
  keeping the `encoder.`-prefixed keys and stripping the prefix:
  `{k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}`,
  then `model.encoder.load_state_dict(...)`. This is the one new bit of loader
  code 008_3 introduces; it lives in a small `encoders.py`.
- **`ppo_early`.** Same extraction, but from the **first checkpoint that solves
  `simple_1_rotation`**. A helper `select_first_solve_ckpt(run_dir, thr=0.99)`
  reads `metrics.jsonl`, finds the earliest `update` whose `eval_success_rate`
  crosses `thr`, and snaps to the nearest saved checkpoint at or after that
  update. We prefer the `exp_007_0_naive` run with the finer 50-update
  checkpoint cadence (`runs/exp_007_0_naive_20260524_121307/`) so the
  "first-solve" snap is tight; the resolved checkpoint path is recorded in
  `results/run_meta.json`.
- **`scratch`.** No load. A fresh `ActorCritic` with default orthogonal init,
  trained unfrozen — i.e. ordinary joint CNN+PPO on the target env. This is the
  control that defines "no transfer".

### 4.2 Frozen and unfrozen treatments

A single parametrised driver `train_ppo.py` handles all four sources and both
freeze modes; it is a merge of the 008_2 and 008_4 drivers with a `--source`
selector.

- **Frozen** (`--freeze`): after loading the encoder, set `requires_grad=False`
  on every encoder param and `model.encoder.eval()`; the Adam optimiser is built
  over `policy_head.parameters() + value_head.parameters()` only. A **freeze-leak
  assert** at the end checks the encoder parameter signature is unchanged —
  reused verbatim from
  [008_2's `_freeze_encoder` / `_param_signature`](../exp_008_2_frozen_jepa_ppo/train_ppo.py).
- **Unfrozen** (`--no-freeze`): leave `requires_grad=True`, build a single Adam
  over `model.parameters()`. An **encoder-moved assert** at the end checks the
  signature *did* change — reused from
  [008_4's train_ppo.py](../exp_008_4_pretrained_jepa_unfrozen_ppo/train_ppo.py).
  `scratch` is always unfrozen and trivially passes this assert.

Everything else — rollout collection, GAE, the clipped PPO update, evaluation —
is the shared exp_007 code used unchanged
([shared/rollout.py](../../exp_007_mini_env_cnn_ppo_baseline/shared/rollout.py),
[shared/ppo.py](../../exp_007_mini_env_cnn_ppo_baseline/shared/ppo.py),
[shared/metrics.py](../../exp_007_mini_env_cnn_ppo_baseline/shared/metrics.py)).

### 4.3 From-scratch baseline

`--source scratch --no-freeze --env {hard1,hard2}` constructs a fresh
`ActorCritic` and trains it on the target env with the exact exp_007_0_naive
recipe. This is the reference curve every transfer run is judged against: a
transfer condition only "helps" if it reaches a given success threshold in
fewer env-steps than scratch on the same env.

### 4.4 Hyperparameters (inherited)

PPO hyperparameters are inherited verbatim from
[shared/config_base.py](../../exp_007_mini_env_cnn_ppo_baseline/shared/config_base.py)
/ exp_007_0_naive, the same as 008_2/008_4:

- `reward_mode = "terminal_only"`, `n_envs = 8`, `rollout_steps = 128`
  (1024 transitions/update).
- `learning_rate = 3e-4`, `gamma = 0.99`, `gae_lambda = 0.95`,
  `clip_eps = 0.2`, `epochs = 2`, `minibatches = 4`, `grad_clip = 0.5`.
- Budget matched to 008_2/008_4 (`--updates 488` ≈ 500K env-steps) so all
  curves share an x-axis. `level_path` is the only env-dependent field; the new
  env tags are `hard1 → hard_1_rotation.json`, `hard2 → hard_2_rotation.json`.

---

## 5. Metrics

Per PPO run:

- **Sample efficiency:** `eval_success_rate` vs `env_step`. **Headline figure:
  two panels (hard1, hard2)**, each overlaying all seven curves for that env
  (3 sources × frozen, 3 × unfrozen — minus that `scratch` has no frozen row —
  so 7 curves per panel: jepa/ppo_early/ppo_final × {frozen,unfrozen} plus
  scratch-unfrozen).
- **Steps-to-90% / steps-to-99%:** env-steps until rolling 100-episode eval
  success crosses 90% / 99%. NaN if never reached within budget. This is the
  primary "how fast does each learn a good policy" number the spec asks for.
- **Final 100-episode eval success rate** at end of training.
- **Average solve length** (steps) at end of training; reported per env since
  the optimal length differs between hard1 and hard2.

For the **unfrozen** runs, additionally log the 008_4-style collapse
diagnostics per update so we can see how fast (and how far) PPO reshapes each
transferred encoder away from its inherited geometry:

- **`mean_feature_cosine`** (from
  [shared/metrics.py](../../exp_007_mini_env_cnn_ppo_baseline/shared/metrics.py)).
- **`feat_std`**, **`feat_pairwise_l2`**, **`feat_effective_rank`** (from
  [exp_007_3_jepa_sg/diagnostics.py](../../exp_007_mini_env_cnn_ppo_baseline/exp_007_3_jepa_sg/diagnostics.py)).

A useful secondary read: if a transferred encoder's diagnostics drift fast and
far, PPO is overwriting the inherited representation (so the *init* mattered
little); if they stay near their inherited values while success climbs, the
transferred features were directly usable.

---

## 6. Expected outcomes

| outcome                                          | what we'd see                                                                                                                                  | implication                                                                                                                            |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **JEPA transfers best (frozen)**                 | Frozen-`jepa` reaches 90% in fewer steps than frozen-`ppo_*` on at least one env; ideally also beats `scratch`.                                | The reward-agnostic, broad-coverage representation is the more transferable one — the original JEPA motivation holds once tested fairly.|
| **PPO encoder overfit (early > final)**          | `ppo_early` transfers noticeably better than `ppo_final` (frozen, and/or unfrozen), the gap widening with the harder env.                       | Late PPO training carves the encoder onto the `simple_1_rotation` winning tube. Confirms the spec caveat — checkpoint selection matters.|
| **Freezing is the bottleneck (unfrozen recovers)** | Frozen transfer is poor for all sources, but unfrozen closes most of the gap to `scratch`/each other.                                          | The *init* is a weak signal; PPO re-learns the encoder on the new env regardless of where it started. Transfer ≈ warm-start only.       |
| **Negative transfer**                            | A transfer run is *slower* than `scratch_unfrozen` on the same env.                                                                             | That inherited representation is an adversarial init for the new env — worse than starting fresh. Most likely candidate: `ppo_final`.   |
| **Everything ties**                              | All seven curves overlap within noise on both envs.                                                                                            | At mini-env scale the encoder init is washed out; transfer signal (if any) needs harder envs or multi-seed to surface.                  |

The cleanest "JEPA wins the transfer argument" signature is: **frozen**
`jepa` ≥ frozen `ppo_early` > frozen `ppo_final`, and `ppo_early` > `ppo_final`
holding under unfreezing too. The cleanest null is the "freezing is the
bottleneck" row. Any run that fails to beat `scratch` is reported as negative
transfer, not as a wash.

---

## 7. How to run

From the repo root (`Code Repo/`), with `uv`. The PPO budget matches 008_2/008_4
(`--updates 488` ≈ 500K env-steps). All 14 runs:

```bash
# JEPA encoder (008_2 1rot), frozen + unfrozen, on each target env.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source jepa      --freeze    --env hard1 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source jepa      --no-freeze --env hard1 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source jepa      --freeze    --env hard2 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source jepa      --no-freeze --env hard2 --updates 488

# PPO encoder, first-solve checkpoint (overfit-control), frozen + unfrozen.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source ppo_early --freeze    --env hard1 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source ppo_early --no-freeze --env hard1 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source ppo_early --freeze    --env hard2 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source ppo_early --no-freeze --env hard2 --updates 488

# PPO encoder, fully-converged checkpoint, frozen + unfrozen.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source ppo_final --freeze    --env hard1 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source ppo_final --no-freeze --env hard1 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source ppo_final --freeze    --env hard2 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source ppo_final --no-freeze --env hard2 --updates 488

# From-scratch zero-transfer floor (always unfrozen), one per target env.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source scratch   --no-freeze --env hard1 --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo --source scratch   --no-freeze --env hard2 --updates 488

# Plot: per-env headline figure (7 curves each) + the unfrozen collapse traces,
# and write summary.json (steps-to-90/99, final success, solve length per run).
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.plot
```

`--smoke` caps each run at 5 updates so the whole pipeline runs end-to-end in
under a minute on M3 Pro MPS, for plumbing verification.
`--encoder_ckpt <path>` overrides the auto-resolved encoder for the JEPA/PPO
sources. Outputs land in
`JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_3_encoder_transfer/results/<envtag>/`.

---

## 8. Caveats and future work

- **Single seed per cell.** Fourteen runs, one seed each — qualitative read
  only. Whether any transfer gap survives multi-seed averaging is unknown and is
  a prerequisite before any number here is reported as a result.
- **"First-solve" checkpoint is a heuristic.** `ppo_early` is defined by the
  first checkpoint crossing 99% eval success, snapped to the saved-checkpoint
  grid. A different threshold or a finer cadence would pick a slightly different
  encoder; the resolved path is logged so the choice is auditable, but the
  early-vs-final contrast is only as sharp as the checkpoint spacing allows.
- **One specific JEPA data distribution.** The JEPA encoder was trained on
  uniform-random data (008_2). A curiosity-collected encoder might transfer
  differently; that comparison is exp_008-future work, not tested here.
- **No partial unfreezing / layer-wise LR.** If unfrozen transfer ties scratch,
  the natural follow-up is a small encoder LR that preserves the inherited prior
  while allowing fine-tuning. Left for later.
- **Sample efficiency, not feature attribution.** Even a clear transfer win does
  not say *which* features carried over. A linear-probe analysis on the frozen
  transferred encoders (predicting player cell, rotation, cross alignment in the
  new env) is the right tool and is left for a separate sub-experiment.
- **Mini-env, Level 1 only.** Same scope as the rest of exp_007 / exp_008. The
  two transfer targets differ from the source in walls, goal cell, and cross
  cell, but remain 8×8 / 32×32 / step-limit-42 — a within-family transfer test,
  not a cross-domain one.
