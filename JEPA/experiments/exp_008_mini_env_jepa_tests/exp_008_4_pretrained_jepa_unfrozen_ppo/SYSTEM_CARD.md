# exp_008_4 — Pretrained JEPA + *Unfrozen* PPO

> Same setup as [exp_008_2](../exp_008_2_frozen_jepa_ppo/) — start from the
> exact same offline JEPA encoders pretrained on uniform-random data — but
> this time **do not freeze** the encoder during PPO. PPO gradients flow
> into the encoder as well as the policy/value heads. Does that recover
> the loss exp_008_2 saw against the joint baseline?
>
> Parent: [exp_008_mini_env_jepa_tests/SYSTEM_CARD.md](../SYSTEM_CARD.md).
> Predecessor: [exp_008_2_frozen_jepa_ppo](../exp_008_2_frozen_jepa_ppo/).

---

## 1. Motivation

### 1.1 What exp_008_2 found

exp_008_2 took a JEPA pretrained on uniform-random-agent data and used its
encoder as a **frozen** feature extractor for downstream PPO. Result
(single seed, 500K env-steps):

| env  | joint CNN+PPO (baseline) | frozen JEPA + PPO |
|------|--------------------------|--------------------|
| 1rot | 51,200 steps to 90%       | **102,400** (≈2× slower) |
| 2rot | 51,200 steps to 90%       | **153,600** (≈3× slower); plateaus around 80–90% |

i.e. under that protocol, the random-data JEPA encoder is a *worse* warm
start than just training the CNN from scratch with PPO. The frozen
representation is missing whatever task-specific shape the joint baseline
manages to put into its encoder through the reward gradient.

### 1.2 The question this experiment asks

There are two distinct hypotheses consistent with the 008_2 result:

1. **JEPA's representation is genuinely worse.** Whatever the random-data
   JEPA encoded, it's the wrong subspace for this task — even letting PPO
   reshape it from there should be no better, possibly worse (the
   pretrained weights are an *adversarial* init for the gradient PPO would
   otherwise have followed).
2. **The freezing is what hurt.** The JEPA-shaped initial subspace is
   actually fine, possibly helpful — but the frozen-encoder protocol
   prevented the policy/value heads from re-shaping the upstream features
   to fit the reward gradient. Unfreeze the encoder and the run should
   match or beat the joint baseline.

`exp_008_4` is the cleanest possible test of hypothesis (2): it changes
*one* thing relative to 008_2 — `requires_grad` on the encoder — and
keeps everything else identical (same pretrained checkpoints, same PPO
hyperparameters, same env configs, same budget).

---

## 2. What is and is not being tested

**Tested:** sample efficiency and final success rate of PPO when the
encoder is **initialised from the 008_2 JEPA checkpoint** but is allowed
to receive PPO gradients (i.e. is *not* frozen). Compared, on each of
the 1-rot and 2-rot env configs, against (a) the joint-from-scratch
baseline and (b) the frozen-encoder treatment from 008_2.

**Not tested:**

- Whether the encoder *should* additionally receive a JEPA-style loss
  during the PPO phase (that is the joint-JEPA setup of exp_007_3 /
  exp_007_4 and is out of scope here). In 008_4, the encoder only sees
  PPO gradients during the policy-learning phase — no JEPA loss is
  applied after pretraining.
- Whether the conclusion generalises across seeds; we still run a
  single seed, in line with the rest of exp_008.
- Whether *partial* unfreezing (e.g. only the top conv layer, or with a
  lower learning rate on the encoder) would change the picture.

---

## 3. Experiment matrix

Reuses every artefact 008_2 already produced. The only new training is
the two unfrozen-encoder PPO runs.

| run id                                        | env config                  | encoder                            | role                                 |
|-----------------------------------------------|-----------------------------|------------------------------------|--------------------------------------|
| `exp_007_0_naive` (reused)                    | `simple_1_rotation.json`    | CNN, joint                         | baseline, 1-rot                      |
| `exp_008_2_joint_cnn_ppo__2rot` (reused)      | `simple_2_rotation.json`    | CNN, joint                         | baseline, 2-rot                      |
| `exp_008_2_frozen_jepa_ppo__1rot` (reused)    | `simple_1_rotation.json`    | JEPA@1rot, **frozen**              | predecessor treatment, 1-rot         |
| `exp_008_2_frozen_jepa_ppo__2rot` (reused)    | `simple_2_rotation.json`    | JEPA@2rot, **frozen**              | predecessor treatment, 2-rot         |
| `008_4_pretrained_jepa_unfrozen_ppo__1rot`    | `simple_1_rotation.json`    | JEPA@1rot, **unfrozen** (this exp) | new treatment, 1-rot                 |
| `008_4_pretrained_jepa_unfrozen_ppo__2rot`    | `simple_2_rotation.json`    | JEPA@2rot, **unfrozen** (this exp) | new treatment, 2-rot                 |

JEPA encoder source: the `encoder_final.pt` produced by
[exp_008_2's train_jepa.py](../exp_008_2_frozen_jepa_ppo/train_jepa.py)
runs in `exp_008_2_frozen_jepa_ppo/jepa_runs/<env_tag>_*/`. No
re-collection of random data and no re-pretraining of JEPA happens here.

---

## 4. Method

### 4.1 What "unfrozen" means here

After loading the pretrained encoder state-dict into `ActorCritic.encoder`,
we leave `requires_grad=True` on every encoder parameter and add the
encoder parameters to the PPO optimiser alongside the policy/value heads.
PPO's combined loss

```
L = L_policy + c_v · L_value + c_ent · L_entropy
```

is the only signal acting on the encoder during this phase. No JEPA loss,
no IDM loss, no auxiliary objective — the only thing that changes vs.
joint-from-scratch PPO is the **initial weights** of the encoder.

This is, deliberately, the most "vanilla" possible way to use the JEPA
encoder as a warm start: same model, same loss, same hyperparameters as
the exp_007_0_naive baseline, only the encoder init differs.

### 4.2 Training procedure

Per env tag `c ∈ {1rot, 2rot}`:

1. **Load.** Construct a fresh `ActorCritic`. Locate the latest
   `encoder_final.pt` under `../exp_008_2_frozen_jepa_ppo/jepa_runs/<c>_*/`
   and copy `encoder_state_dict` into `model.encoder`. Policy and value
   heads remain at their default orthogonal init.
2. **Optimise.** Single Adam optimiser over **all** of
   `encoder.parameters() + policy_head.parameters() + value_head.parameters()`,
   `lr = 3e-4` (same as 007_0_naive). No separate `opt_enc` group.
3. **Train.** Reuse `shared.rollout.collect_rollout`,
   `shared.rollout.compute_gae`, `shared.ppo.ppo_update`, and the
   evaluation utilities from
   [shared/](../../exp_007_mini_env_cnn_ppo_baseline/shared/) verbatim.
4. **Budget.** 488 PPO updates × 1024 env-steps/update ≈ 500,000 env
   steps — matches the 008_2 budget so the curves line up on the same
   x-axis.
5. **Checkpoint** every 100 updates and at the end of training.

### 4.3 Sanity check

At the start of training we record a parameter signature of the loaded
encoder. At the end we assert the signature has *changed* — if it
hadn't, PPO would not actually have been training the encoder and the
"unfrozen" claim would be a lie. (This is the mirror image of the
freeze-leak check in 008_2.)

---

## 5. Metrics

For every PPO run (both new and reused), we plot and tabulate the same
quantities as 008_2:

- **`eval_success_rate`** vs `env_step` — overlapped on one figure with
  *all six* runs at once (2 baselines + 2 frozen + 2 unfrozen).
- **Steps-to-90% / steps-to-99%** rolling eval success rate.
- **Final eval success rate** at end of training.
- **Average solve length** at end of training.

For the new unfrozen runs we additionally log, per PPO update, the same
collapse diagnostics that exp_008_2 logged for its JEPA pretraining (so
we can see whether and how fast PPO undoes the JEPA-shaped geometry):

- **`mean_feature_cosine`** — cosine between consecutive same-episode
  trunk features (already in `shared.metrics`).
- **`feat_std`**, **`feat_pairwise_l2`**, **`feat_effective_rank`** —
  the three diagnostics from
  [exp_007_3_jepa_sg/diagnostics.py](../../exp_007_mini_env_cnn_ppo_baseline/exp_007_3_jepa_sg/diagnostics.py).

The headline figure for the experiment is the overlapped six-curve
success-rate plot; the collapse figure is the secondary diagnostic.

---

## 6. Expected outcomes

| outcome                                          | what we'd see                                                                                                                                          | implication                                                                                                                                                       |
|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Unfrozen recovers, beats baseline**            | Unfrozen-JEPA-PPO reaches 90% in noticeably *fewer* env-steps than the joint baseline on at least one env.                                             | The JEPA-shaped init is genuinely helpful; the 008_2 result was the *freezing*, not the representation. Worth investing in better JEPA pretraining (curiosity §7).|
| **Unfrozen ties baseline**                       | Curves overlap the joint baseline within noise across both envs.                                                                                       | The JEPA init is neutral as a warm start once the encoder is allowed to move; PPO erases it. Frozen-vs-joint gap from 008_2 was due to freezing alone.            |
| **Unfrozen still worse than baseline**           | Unfrozen-JEPA-PPO is slower or plateaus below the joint baseline, but better than frozen-JEPA-PPO.                                                     | JEPA init partially poisons later learning. PPO has to fight the prior, not just learn from scratch. Suggests random-data JEPA is *actively misaligned*.          |
| **Unfrozen matches frozen-JEPA**                 | Unfrozen tracks frozen-JEPA-PPO instead of the joint baseline.                                                                                         | The encoder barely moves under PPO's gradient (e.g. trapped in a flat region), so the unfreezing nominally happened but had no effect. Investigate gradient norms.|

The most informative collapse-metric signature would be: if PPO is
actively reshaping the encoder, `feat_effective_rank` and
`feat_pairwise_l2` should drift away from their JEPA-end values within
a few hundred updates. If they stay pinned, the encoder is effectively
still frozen even though we removed the `requires_grad` flag.

---

## 7. How to run

From the repo root. The PPO budget matches what 008_2 used.

```bash
# 1. Unfrozen-encoder PPO. Locates the latest 008_2 JEPA encoder for the
#    selected env_tag automatically; pass --encoder_ckpt to override.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_4_pretrained_jepa_unfrozen_ppo.train_ppo --env 1rot --updates 488
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_4_pretrained_jepa_unfrozen_ppo.train_ppo --env 2rot --updates 488

# 2. Plot. Assembles the overlapped six-curve eval plot AND the
#    collapse-metric trace for the two new unfrozen runs.
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_4_pretrained_jepa_unfrozen_ppo.plot
```

Outputs land in
`JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_4_pretrained_jepa_unfrozen_ppo/results/`.

The two reused-baseline runs (exp_007_0 1rot, exp_008_2 joint 2rot) and
the two reused-frozen runs (exp_008_2 frozen 1rot/2rot) are located by
glob from the existing experiment dirs — no copying or re-running.

---

## 8. Caveats and future work

- **Single seed across all six cells.** Whether the gap (in either
  direction) survives multi-seed averaging is unknown.
- **Same JEPA pretraining as 008_2.** If that pretrain was under-trained
  or over-trained, this experiment inherits that limitation; differences
  between 008_4 and 008_2 are interpretable, but absolute claims about
  "is JEPA helpful" depend on the pretraining quality.
- **No partial unfreezing or layer-wise LR.** The natural follow-up if
  unfrozen tracks the joint baseline is: does a *small* encoder LR
  preserve the JEPA prior while still allowing fine-tuning? Left for
  later.
- **No JEPA loss during PPO.** Adding `L_JEPA` alongside `L_PPO` is
  exp_007_3 / exp_007_4 territory and is intentionally not what 008_4
  tests; 008_4 isolates the effect of the *init* alone.
