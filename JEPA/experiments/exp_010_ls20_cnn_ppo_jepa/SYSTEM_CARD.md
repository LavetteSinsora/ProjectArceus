# exp_010 — CNN+PPO and JEPA-encoder baselines on the *real* LS20 game

> exp_007 and exp_008 established the CNN+PPO and JEPA-encoder story on the
> cheap pure-numpy **mini-env** (a 32×32 recreation of LS20 Level 1). exp_010
> repeats the core of that story on the **real 64×64 ARC-AGI-3 LS20 game**
> (`environment_files/ls20`, via `JEPA.shared.env_wrapper.LS20Env`) — the same
> environment the main JEPA dashboard (port 8787) and exp_001–004 use. The
> question is whether the conclusions from the toy env survive the jump to the
> actual game's pixels, action semantics, and step budget.

---

## 1. Motivation

mini-env let us iterate in seconds, but it is our own reconstruction: 32×32,
fully observable through accessors we wrote (player cell, rotation), and with
reward shaping that reads those accessors. The real LS20 game gives us none of
that — only **64×64 colour-index frames** and a terminal `level_completed`
flag. Everything we concluded on mini-env (PPO can/can't crack sparse reward;
a JEPA encoder helps/doesn't as a warm start) is only interesting if it holds
on the real thing. exp_010 is the bridge.

Two structural differences from exp_007/008 force real design decisions:

1. **Observation.** 64×64 instead of 32×32 → one extra strided conv so the
   flattened map stays 8×8 and the trunk Linear stays ~exp_007 size.
2. **Reward.** No introspection → **terminal-only** reward is the *only*
   option (the rotation-match shaping of exp_007_2 is impossible without the
   player-rotation accessor). So every exp_010 run is the honest sparse-reward
   regime — exactly exp_007_0_naive ("7_0"), ported.

---

## 2. The three sub-experiments

| id | dir | one-line |
|---|---|---|
| 10_0 | [exp_010_0_cnn_ppo_baseline](exp_010_0_cnn_ppo_baseline/) | vanilla CNN+PPO, terminal-only reward (the 7_0 recipe on real LS20) |
| 10_1 | [exp_010_1_jepa_joint_online](exp_010_1_jepa_joint_online/) | **joint online** JEPA+PPO: random-init encoder+policy, PPO from scratch, encoder *also* trained by a JEPA(+IDM) loss on the agent's **own on-policy** rollout transitions every update |
| 10_2 | [exp_010_2_jepa_random_pretrain](exp_010_2_jepa_random_pretrain/) | JEPA encoder **pretrained on random-policy data** until plateau, then PPO from that init (encoder **unfrozen**) |

10_1 and 10_2 are the two "JEPA-trained encoder" variants from the project
spec, distinguished by *where the encoder's training data comes from*:

- **10_1 / on-policy:** the data is whatever the current PPO policy visits.
  Encoder and policy co-adapt online; there is no separate pretraining phase.
- **10_2 / random-policy:** the data is a fixed buffer collected by a uniform-
  random agent *before* any policy exists. The encoder is trained to plateau
  offline (we report the number of environment steps that data cost), then
  handed to PPO as an initial weight and fine-tuned (unfrozen).

The plain baseline (10_0) is the zero against which both JEPA variants are
measured: does a world-model objective on the encoder — from either data
source — buy any sample efficiency over just letting PPO learn the encoder?

---

## 3. What is shared (`shared/`)

All three sub-experiments import one library so any difference in outcome is
attributable to the encoder-training treatment, never to a divergent PPO impl.

| module | role |
|---|---|
| `ls20_vec_env.py` | N synchronous **real** LS20 envs on one offline `Arcade`; terminal-only reward; `max_episode_steps` truncation; per-episode success tracking |
| `model.py` | `ActorCritic` (64×64 one-hot → 4 strided convs → 256-d trunk → policy/value heads) + `ActionConditionedPredictor` (forward JEPA head) + `InverseDynamicsModel` (IDM head) |
| `rollout.py` | (T,N) rollout buffer (stores `next_obs` too, for JEPA transition pairs) + GAE (exp_007's off-by-one-fixed convention) |
| `ppo.py` | clipped-surrogate PPO update (value clip, advantage norm, grad clip) |
| `jepa.py` | JEPA + IDM loss on `(s,a,s')`; online update from a rollout (skips episode-ending steps) |
| `trainer.py` | one `train(cfg)` for all three: collect → GAE → PPO → (optional online JEPA) → eval → log → checkpoint |
| `pretrain.py` | random-data collection + offline JEPA pretraining-to-plateau (10_2) |
| `evaluator.py` | periodic stochastic-eval rollouts → `success_rate`, steps-to-solve, truncation rate |
| `metrics.py` | feature-collapse diagnostics + `metrics.jsonl` writer (dashboard format) |
| `debug_runner.py` | dashboard episode playback (re-exported by each child) |

Architecture, optimiser, and PPO hyperparameters are identical across the
three; the **only** differences are `jepa_mode` and `init_encoder_ckpt`.

---

## 4. Architecture

**Input.** Raw `(64,64) uint8` palette frame, indices `[0,15]`. One-hot to
`(16,64,64)` — palette indices are categorical, not ordinal (same rationale as
exp_007 §4.1). The step-counter UI rows (61–62) are **not** masked from the
policy input.

**Encoder (CNN).**
```
Conv2d(16→32, k3,s1,p1)  ReLU     # 64×64
Conv2d(32→64, k3,s2,p1)  ReLU     # 32×32
Conv2d(64→64, k3,s2,p1)  ReLU     # 16×16
Conv2d(64→64, k3,s2,p1)  ReLU     #  8×8
Flatten → Linear(4096→256) ReLU   # trunk feature h
```
One more stride-2 conv than the 32×32 model, so the flattened map is 8×8
(=4096) — keeping the trunk Linear and total params (~1.15M) close to exp_007.

**Heads.** `policy_head: 256→4` (orthogonal gain 0.01, near-uniform init),
`value_head: 256→1` (gain 1.0). Orthogonal init, gain √2 on ReLU layers. No
BatchNorm, no pooling (same reasoning as exp_007 §4).

**JEPA modules (10_1, 10_2 only).**
- `ActionConditionedPredictor(h_t, a) → ĥ_{t+1}`: MLP over `[h_t ; emb(a)]`.
- `InverseDynamicsModel(h_t, h_{t+1}) → action logits`: MLP over `[h_t;h_{t+1}]`.
- JEPA loss `MSE(predictor(h_t,a), sg(h_{t+1})) + idm_coef·CE(idm(h_t,h_{t+1}),a)`
  with a stop-gradient target branch (cf. exp_007_3 / exp_007_4).

---

## 5. PPO

Identical recipe to exp_007 §5, ported to 64×64: 8 synchronous envs ×
`rollout_steps=128` → 1024 transitions/update; γ=0.99, λ=0.95, clip ε=0.2,
value clip 0.2, c_v=0.5, c_ent=0.01, grad clip 0.5, Adam lr 3e-4,
4 epochs × 4 minibatches. Truncation (`max_episode_steps`) is treated as
`done=True`; since terminal reward is 0 on truncation, the bootstrap is
irrelevant there.

---

## 6. Metrics (logged to `runs/<run>/metrics.jsonl`)

Per update: `policy_loss`, `value_loss`, `policy_entropy`, `approx_kl`,
`clipfrac`, `grad_norm_total`, `mean_feature_cosine`, `train_success_rate`,
`sps`. For 10_1 also `jepa_loss`, `idm_loss`, `idm_acc` (the online JEPA).
Every `eval_every` updates: `success_rate`, `avg_steps_to_solve`,
`min_steps_to_solve`, `mean_episode_steps`, `truncation_rate`, plus the
collapse diagnostics `feat_std` / `feat_effective_rank` / `feat_pairwise_l2`.

`success` is `env.level_completed` (cleared ≥1 LS20 level) at a terminal step.

---

## 7. Dashboard integration

exp_010 surfaces on the **main JEPA dashboard** (port 8787), the one built
around the real LS20 game, exactly like exp_001–004 — *not* a standalone
server like exp_007's. The wiring:

- Checkpoints are written flat to `<child>/checkpoints/step_<env_step>.pt`
  and metrics to `<child>/runs/<run>/metrics.jsonl` — the layout the dashboard's
  `/api/checkpoints` and `/api/training/metrics` read.
- Each child ships a `debug_runner.py` (re-exporting `shared/debug_runner.py`)
  so the dashboard plays a CNN+PPO/JEPA checkpoint on real LS20. Its payload
  sets the ViT/JEPA capability flags to `False`, so the dashboard cleanly
  hides the exp_001-specific patch/attention cards and shows frame playback,
  action probabilities, and value.
- The dashboard lists nested sub-experiments as `parent/child`. Two one-line
  patches (`JEPA/dashboard/server.py`, `JEPA/dashboard/debug_runner.py`)
  normalise that `/` to `.` when importing the child's modules. Backward-
  compatible: top-level experiments (exp_001–004) contain no `/`, so their
  behaviour is unchanged.

Launch: `uv run python JEPA/dashboard/server.py` → http://localhost:8787.

---

## 8. How to run

From the repo root (`Code Repo/`). Append `--smoke` to any command for a
few-update plumbing run (< 1 min, CPU/MPS).

```bash
# 10_0 — CNN+PPO baseline
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_0_cnn_ppo_baseline.train

# 10_1 — joint online JEPA + PPO
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_1_jepa_joint_online.train

# 10_2 — random-data JEPA pretrain, then unfrozen PPO  (three stages)
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_2_jepa_random_pretrain.collect
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_2_jepa_random_pretrain.train_jepa
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_2_jepa_random_pretrain.train_ppo
```

A self-contained Google Colab notebook (`colab_exp_010.ipynb`) runs all three
on a GPU and zips the checkpoints for download — see §10.

---

## 9. Expected outcomes

| variant | hypothesis | what would falsify it |
|---|---|---|
| 10_0 | terminal-only PPO on real 64×64 LS20 is *harder* than on mini-env; `success_rate` may stay low for a long time | early, high success would mean real LS20 Level 1 is as easy as the mini-env under sparse reward |
| 10_1 | the on-policy JEPA loss regularises the encoder (watch `feat_effective_rank`, `mean_feature_cosine`); whether it speeds up PPO is the open question | no measurable effect on success-vs-steps vs 10_0 → the auxiliary loss is inert on this env |
| 10_2 | random-data JEPA gives a generic-geometry encoder; per exp_008_2/008_4 the *frozen* version hurt but *unfrozen* may tie the baseline | unfrozen 10_2 beats 10_0 by a clear margin → random-data pretraining is a useful warm start on real LS20 |

These mirror the open questions exp_008 raised on mini-env; exp_010 is where we
find out if they hold on the actual game. Single-seed, single-level — same
caveats as exp_007/008.

---

## 10. Caveats / limitations

- **Terminal-only reward only.** No shaping is available on the real env, so
  10_0 is strictly the hardest (sparse) regime. If nothing learns, that is a
  statement about exploration, not about the encoder treatments.
- **Single seed, LS20 Level 1 only.** No generalisation or transfer claims.
- **Synchronous vec env.** The real LS20 step is ~1.6k steps/s/env, so env
  stepping is not the bottleneck; the gradient step dominates (more so at
  64×64 than at 32×32).
- **Reused architecture.** Encoder/predictor/IDM are deliberately the exp_007
  designs scaled to 64×64 — this is a baseline family, not an architecture
  search.
