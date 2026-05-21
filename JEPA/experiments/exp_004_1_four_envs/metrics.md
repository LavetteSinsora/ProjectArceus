# Metrics — exp_004_1_four_envs

This experiment inherits the single-env metric set from `exp_003_4_no_resampler_self_attn` (via `exp_004_0_ls20_tu93`). Sections 1–7 describe **shared-encoder / shared-predictor / shared-action-predictor** properties that are env-agnostic and remain single-valued. Sections 8–10 cover the multi-env axis at the 4-env scale.

For sections 1–7, see [`exp_003_4_no_resampler_self_attn/metrics.md`](../exp_003_4_no_resampler_self_attn/metrics.md). For the 2-env precedent of sections 8–10, see [`exp_004_0_ls20_tu93/metrics.md`](../exp_004_0_ls20_tu93/metrics.md). This document covers only the deltas and additions for the 4-env scale.

---

## Section 8 — Per-environment basics (extends exp_004_0 §8 from 2 envs to 4)

Every per-env metric appears in `metrics.jsonl` under its section path with an `_<env>` suffix for `env ∈ {ls20, tu93, re86, g50t}`. All four envs use the same 5-way action head, but the per-env action-space size differs (LS20 / TU93 → 4; RE86 / G50T → 5), so policy entropy ranges differ:

- `policy_entropy_ls20`, `_tu93` max value `ln(4) ≈ 1.386`.
- `policy_entropy_re86`, `_g50t` max value `ln(5) ≈ 1.609`.

Use `policy_entropy_normalized_<env>` (divided by `ln(|avail_t|)` at each step) for cross-env comparison.

### sec5 (policy)

- **`policy_entropy_<env>`** — REINFORCE policy entropy per env (raw, in nats), for `<env> ∈ {ls20, tu93, re86, g50t}`.
- **`policy_entropy_normalized_<env>`** — divided by `ln(|avail_t|)` at each step.
- The aggregate `sec5/policy_entropy` is the mean over all envs' rollouts within the log window.

### sec6 (performance / losses)

For each `<env> ∈ {ls20, tu93, re86, g50t}`:

- **`L_state_<env>`** — JEPA state-prediction loss on the quarter-batch from that env.
- **`L_action_<env>`** — JEPA action-prediction CE loss on the quarter-batch from that env. Targets are in `{0..3}` for ls20/tu93; `{0..4}` for re86/g50t. Same 5-way softmax.
- **`reward_state_component_<env>`** — per-step `state_err` (= reward, because `reward_w_action = 0`) used as the policy's curiosity signal. Computed during rollout, NOT during JEPA update.
- **`reward_action_component_<env>`** — per-step action-CE error (LOGGED ONLY; multiplied by 0 in the reward).
- **`reward_total_<env>`** — per-env clamped curiosity reward distribution.
- **`pol_loss_<env>`** — REINFORCE policy loss per env.
- **`episode_length_<env>`** — distribution of episode lengths.
- **`completion_rate_<env>`** — fraction of recent episodes that cleared a level. Expected near 0 under intrinsic-only reward.
- **`buffer_fill_<env>`** — fill ratio of the per-env replay buffer in `[0, 1]`. Faster-episode envs reach 1 sooner.

Aggregate `sec6/L_state`, `sec6/L_action`, `sec6/L_total`, `sec6/reward_total` are kept as the average across the four envs (one number per JEPA update / rollout window).

### Reading the per-env rows

- `L_state_<env>` divergence across the four envs > 2× is the early warning that one env is unmodelled or numerically unstable. The shared `state_predictor` cannot drive all four to zero simultaneously if the envs' dynamics differ in difficulty.
- `reward_state_component_<env>` close to `reward_clamp = 50.0` indicates that env's state predictor is unable to fit transitions — diagnose via `L_state_<env>`.
- Persistent `policy_entropy_<env>` collapse on a single env is the per-env analogue of exp_003_2's policy-collapse risk; check `policy_entropy_normalized_<env>`.

---

## Section 9 — Cross-environment gradient interference (extends exp_004_0 §9 from 2 envs to 4)

With 4 envs the pairwise count is C(4, 2) = 6. To keep the metric surface compact we log two summary scalars per shared-module subset `k ∈ {patch_sa, perc_cross_r0, perc_cross_r1, state_pred_mlp_0..3, action_pred}`; the full 6-pair breakdown is gated behind a verbose flag.

### gnorm (per env, per source, per sub-block)

For each `<env> ∈ {ls20, tu93, re86, g50t}` and each subset `k`:

- **`gnorm_jepa_<env>_<k>`** — L2 norm of `<env>`-half gradient vector in subset `k`.
- Source-decomposed forms (mirror exp_003_4):
  - `gnorm_state_<env>_<k>`
  - `gnorm_action_via_Ht_<env>_<k>`
  - `gnorm_action_via_Htp1_<env>_<k>`

### Cross-env gradient cosine (summary scalars)

For each subset `k`:

- **`gcossim_avg_pairs_<k>`** — mean of the 6 pairwise cosines over `(<envA>, <envB>) ∈ choose(env_names, 2)`. The aggregate cross-env signal.
- **`gcossim_min_pairs_<k>`** — minimum of the 6 pairwise cosines. The worst-case interference signal — a single bad pair stays visible even when the mean is healthy.

### Cross-env gradient cosine (verbose per-pair breakdown — flag-gated)

For each ordered pair `(<envA>, <envB>)` in lexicographic order of `env_names` and each subset `k`:

- **`gcossim_<envA>_vs_<envB>_<k>`** — flatten-cosine between `<envA>` and `<envB>` total gradient vectors in subset `k`. Range `[-1, 1]`.
- Source-decomposed:
  - `gcossim_state_<envA>_vs_state_<envB>_<k>`
  - `gcossim_action_<envA>_vs_action_<envB>_<k>`
  - `gcossim_state_<envA>_vs_action_<envB>_<k>`
  - `gcossim_action_<envA>_vs_state_<envB>_<k>`

### Interpretation reference

Same as exp_004_0 §9, applied per pair:

| `gcossim_<A>_vs_<B>_<k>` | Reading |
|--------------------------|---------|
| ≈ +1 | Two envs reinforce — shared module is doing useful joint work. |
| ≈  0 | Two envs uncorrelated — shared module is averaging unrelated tasks. |
| ≈ −1 | Two envs in direct opposition — strongest cross-env interference. |

When summarised to `min_pairs` and `avg_pairs`:

- `avg_pairs ≈ +0.5, min_pairs ≈ +0.3` — broadly healthy; some asymmetry but no destructive pair.
- `avg_pairs ≈ 0,   min_pairs ≈ -0.5` — one destructive pair hidden by five neutral pairs. Surface the verbose breakdown.
- `avg_pairs ≈ -0.3, min_pairs ≈ -0.8` — encoder is being pulled in conflicting directions. Strong signal that a shared module is the wrong design for this 4-env mix.

### Cadence

Same as exp_003_4's `grad_decomp_freq = 25` — these probes do extra backward passes and are not free. Implemented by extending `compute_source_decomposition` to operate on per-env batches and aggregate the 6 pairs.

---

## Section 10 — Beyond cosine: per-element disagreement (extends exp_004_0 §10 from 2 envs to 4)

Cosine compresses an entire module's parameter space into one scalar. Per-element sign disagreement complements it. Same gating as Section 9.

### Per-element sign disagreement (summary scalars)

For each shared-module subset `k`:

- **`gsign_disagree_frac_avg_pairs_<k>`** — `mean over pairs of mean(sign(g_<A>) != sign(g_<B>))`. Range `[0, 1]`; 0.5 = random sign relationship per pair averaged across the 6 pairs.
- **`gsign_disagree_frac_max_pairs_<k>`** — maximum across the 6 pairs (worst-case sign conflict).
- **`gsign_disagree_frac_magweighted_avg_pairs_<k>`** — magnitude-weighted version, averaged. Captures whether disagreement is concentrated at heavily-updated parameters.

### Per-layer cosine distribution

For each shared-module subset `k`:

- **`gcossim_perlayer_<k>_<layer>_avg_pairs`** — cosine computed over each `nn.Linear` weight, attention projection, FFN layer inside subset `k`, averaged over the 6 env pairs. Reveals layer-localised conflict that the module-global cosine averages away.

### Quick diagnosis table (per pair)

| `gcossim` | `gsign_disagree_frac` | Interpretation |
|-----------|------------------------|----------------|
| ≈ +1 | ≈ 0 | Strong constructive coupling. |
| ≈ 0 | ≈ 0.5 | Uncorrelated — capacity underused. |
| ≈ 0 | ≈ 0 | Signs agree but magnitudes are uneven. |
| ≈ 0 | ≈ 1 | Per-element conflict, summed magnitudes happen to cancel. |
| ≈ −1 | ≈ 1 | Direct opposition — worst case. |

---

## What stays unchanged from exp_003_4

The following inherited metric families do **not** get env-suffix variants — they describe shared modules and are computed once per JEPA / probe cycle on a *combined* sample from all four envs' buffers:

- sec1 (representation health) — `placeholder_pairwise_cossim`, `latent_pairwise_cossim_buf`, `latent_eff_rank`, `latent_norm_*`, `placeholder_drift_from_init_*`, `ht_htp1_cossim_rollout`, `H1_HT_cossim`
- sec2 (image-patch SA) — `patch_sa_row_jsd`, `patch_sa_temporal_jsd`
- sec4 (predictors) — `ode_step_cossim`, `ode_first_vs_final_cossim`, `predictor_velocity_norm`, `action_pred_entropy*`, `action_pred_input_cossim`
- sec7 (gradient norms / UWR per sub-block) — `gnorm_<sub>_total`, `uwr_<sub>`. Note: the cross-env breakdowns added in Section 9 ABOVE coexist with these aggregate keys.

The eval-pass time-series (`*_eval`, `*_t1`, `*_t10`, `*_t20`, `H1_HT_cossim`) are computed once per env per eval cycle (four rollouts) and reported under env-suffixed keys.

### Per-env action-predictor entropy — note on the 5-way head

`action_pred_entropy_eval_<env>` should be inspected per env. Specifically:

- On `<env> ∈ {re86, g50t}` it should be measurably above zero — class 4 is a legal target and must remain usable.
- On `<env> ∈ {ls20, tu93}` the class-4 logit is never the target, so its posterior probability will trend toward zero. A measurable difference between `action_pred_entropy_eval_re86` (≈ should be near `ln(5)` early; declines with training) and `action_pred_entropy_eval_ls20` (≈ should be capped near `ln(4)`) is expected and not a bug — it is the signature of the 5-way head correctly handling heterogeneous action sets.

---

## Implementation pointers

- `HealthMonitor` (re-exported from `exp_003_4_no_resampler_self_attn/monitors/health.py`) accepts arbitrary keys via `push_sec7(key, value)`. Per-env keys are added on-the-fly in `train.py` via `health.sec*.setdefault(f"{base}_{env}", deque(maxlen=200)).append(v)` for `<env> ∈ env_names`.
- `MetricsWriter` (also re-exported) iterates `health.sec1..sec7` dicts, so any key added there flows into `metrics.jsonl` automatically — no writer changes required.
- Cross-env gradient probe (Section 9): the 2-env precedent in `exp_004_0` left this as a planned extension of `monitors/gradients.py`. The 4-env version requires the same `compute_source_decomposition_cross_env(...)` extension, with the additional aggregation step that collapses the 6 pairs to `avg_pairs` and `min_pairs` summaries. Until that lands, only the aggregate sec7 sub-block gnorms (single-valued) are tracked.
- Action-mask runtime assertion in `train.py:run_one_episode` (`assert action_idx < env.n_actions`) catches a broken `available_actions` mask immediately on the 4-action envs. No metric — it raises.
