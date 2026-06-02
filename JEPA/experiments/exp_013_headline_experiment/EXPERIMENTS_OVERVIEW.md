# exp_013 — methods, issues found, fixes, and promising directions

*Reference for compiling the Colab sweep. Goal metric: **env-steps to the first positive
extrinsic reward** (= first success). Intrinsic-only; extrinsic +1 used only as the stop
signal. Stop-on-first-reward + per-cell cap. ≥ multiple seeds, mean ± spread.*

## 0. Baselines (the benchmark to beat)
- **Random policy** — computed per game×level (`baseline_random_policy/SUMMARY.md`). Only **4
  cells are reachable by random** (finite E): tu93 L2 ≈2k, ls20 L1 ≈50k, tu93 L1 ≈500k,
  re86 L1 ≈2M. All others E = ∞ ("beat-the-impossible" cells: a solve is the whole result).
- **ICM** (exp_011) and **RND** (exp_012) — the two standard intrinsic baselines.

## 1. Methods tried (under exp_013)
| id | method | combine type | status |
|---|---|---|---|
| **13_1** `exp_013_1b_leaky_rnd_on_icm_phi` | RND novelty computed *inside* ICM's inverse-dynamics φ-space; leaky predictor; φ frozen after warm-up | **composition** (ICM = the representation; one reward) | built, runs; high seed-variance on ls20 L1 |
| **13_2** `exp_013_2_additive_rnd_icm` | `w·norm(ICM-fwd-error) + (1−w)·norm(RND-on-φ)`, w=0.5 | **additive** (two rewards summed) | **DEAD** — see issue (4) |
| **13_4** `exp_013_4_plan2explore` | Plan2Explore-style **ensemble disagreement** (variance of K forward models predicting a FROZEN-RANDOM φ(s')) | non-redundant signal; no ICM/φ-freeze | built, smoke-passes; not yet run |
| **13_5 (D)** `exp_013_3_mcts_lookahead` | **MCTS-organized 1-step lookahead-softmax**, actor-free: act by `softmax(standardize(nov(φ̂'_a)+γV(φ̂'_a))/τ)`; V_int trained on REAL returns | **structural fix for the entropy collapse** (no policy gradient → no phantom advantage); model used for decision only | built, smoke-passes |
| **13_1 phi_mode=frozen (A)** | RND+leak on a frozen-random φ (no ICM) | `--phi-mode frozen` | built |
| **13_1 init_phi_ckpt (B-xfer)** | B on L2 with φ initialised from a trained L1 φ (cross-level transfer) | `--init-phi-ckpt` | built |

## 2. Issues we discovered (each with the probe that found it)
1. **Entropy collapse → worse-than-random** (`probes/occ_power_limits.md`): on ls20 L1, 13_1 solved 1/3 seeds; the policy collapses to a deterministic loop. *Coverage ≠ solving* (a specific hidden-state rotation→goal sequence).
2. **Root cause = value-lag / "phantom advantage"** (`probes/entropy_collapse_diagnosis.md`, `entropy_collapse_explained.html`): controls prove it's the novelty *return structure*, not generic PPO (reward≡0 or noise don't collapse). The non-episodic return inflates (V 0.3→8) and bursts; the single critic under-shoots (mean Ret−V = +0.76); per-batch advantage-norm leaves a *structured* positive advantage on the visited region → PPO commits → entropy→0. Same pattern as exp_010 `finding_phantom_advantages`.
3. **φ is only partially controllable + a fooled freeze** (`probes/inv_acc_causality.md`): on-policy inv_acc inflates to 0.98 (narrow-data artifact) while **held-out inv_acc ≈ 0.76** (below the 0.90 freeze gate, but well above chance) — so the freeze trigger was reading a fooled metric (fixed: trigger on held-out).
4. **Signals are redundant AND anti-informative** (`probes/signal_redundancy.md`): both ICM-fwd and RND-on-φ correlate **negatively** with true novelty (−0.46, −0.56) → reward higher on more-visited states. No convex w helps ⇒ 13_2 dead. Root cause = (3) + (5).
5. **🔴 Observation confound — a marching step-timer** (`probes/signal_redundancy.md`): the env wrapper masks UI rows for its *diff utilities* but **not in the frame fed to the models**, so a step-timer (ls20 rows 61–62) marches every step → every frame is unique (1073 vs **43** distinct board states after masking) → swamps the novelty signal. Present since exp_010; poisons every count/RND result.

## 3. Fixes applied (in the harness/configs now)
- **Normalized-ICM** (running-std, no frozen-η) — the 2018 large-scale-study fix for ICM's frozen-η collapse.
- **Held-out-inv_acc freeze trigger** (not on-policy) + a WARNING when it falls back on a poor φ.
- **Reward clip** (adaptive, k× running-mean raw) before the normalizer.
- **γ_int 0.99→0.95** + **c_value 0.5→1.0** (value-lag mitigation: smaller return inflation, faster value tracking). Non-episodic kept; **no entropy floor** (by choice).
- **Checkpoint saving** + richer logging (clipfrac, V vs Ret, holdout_inv_acc, novelty_raw_max).

## 4. Promising directions (ranked, not all built)
1. **Masked-frame count** *(not built; strongest lever on these deterministic games)* — intrinsic reward = count novelty on **UI-masked** frames (exact/hash → 1/√N). Stationary, informative (43 states), no φ, no timer confound. Directly fixes issues (4)+(5).
2. **Ensemble disagreement** *(13_4, built)* — the non-redundant signal (epistemic; handles aleatoric noise; no stationary-ruler needed). Should also use **masked** frames for its frozen φ.
3. **PopArt value normalization** *(not built)* — the principled cure for "normalize the return without introducing error": normalises value targets + preserves outputs → lets the critic track the non-stationary return (attacks issue 2 at the critic).
4. **Go-Explore frontier-return** *(not built)* — directed return to the frontier; the real cure for *coverage ≠ sequence*; solved LS20 L2–L4 historically (`claude_automate`).
5. **NGU episodic memory** *(not built)* — within-episode kNN count; complementary timescale to lifelong novelty.
6. **L1→L2 encoder transfer** *(probe-gated)* — init L2's φ from an L1-trained ICM; see `probes/` once run.

## 5. For the sweep (match to the random benchmark)
Mirror `colab_calibration.ipynb`: methods × game × level × seed, `--max-env-steps` per-cell cap
scaled to the random E (SUMMARY.md), stop-on-first-reward, collect `result.json`
(`env_steps_to_first_reward`, solved/censored). Compare each method's mean steps-to-first-reward
against the random E per cell; on E=∞ cells report solve-rate. Candidate methods for the sweep:
**ICM, RND, 13_1 (RND-on-φ), 13_4 (disagreement), masked-frame-count** (once built). 13_2 excluded (dead).
Apply the **timer-mask to all novelty paths** before running, or results stay confounded.
