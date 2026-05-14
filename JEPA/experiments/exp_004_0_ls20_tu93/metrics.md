# Metrics — exp_004_0_ls20_tu93

This experiment inherits the single-env metric set from `exp_003_4_no_resampler_self_attn`. Sections 1–7 below are unchanged — they describe **shared-encoder / shared-predictor / shared-action-predictor** properties that are env-agnostic. Sections 8–10 are new and describe the multi-env axis.

For sections 1–7, see [`exp_003_4_no_resampler_self_attn/metrics.md`](../exp_003_4_no_resampler_self_attn/metrics.md). This document covers only the deltas and additions.

---

## Section 8 — Per-environment basics (NEW)

Every per-env metric appears in `metrics.jsonl` under its section path with an `_<env>` suffix. Both envs use 4 actions, so per-class action stats have the same shape across envs.

### sec5 (policy)

- **`policy_entropy_ls20`**, **`policy_entropy_tu93`** — REINFORCE policy entropy per env (raw, in nats). Max value `ln(4) ≈ 1.386`.
- The aggregate `sec5/policy_entropy` is the mean of both envs' rollouts within the log window.

### sec6 (performance / losses)

- **`L_state_ls20`**, **`L_state_tu93`** — JEPA state-prediction loss on the half-batch from that env.
- **`L_action_ls20`**, **`L_action_tu93`** — JEPA action-prediction CE loss on the half-batch from that env.
- **`reward_state_component_ls20`**, **`reward_state_component_tu93`** — per-step `state_err` (= reward, because `reward_w_action = 0`) used as the policy's curiosity signal. Computed during rollout, NOT during JEPA update.
- **`reward_total_ls20`**, **`reward_total_tu93`** — per-env clamped curiosity reward distribution.
- **`pol_loss_ls20`**, **`pol_loss_tu93`** — REINFORCE policy loss per env.
- **`episode_length_<env>`** — distribution of episode lengths.
- **`completion_rate_<env>`** — fraction of recent episodes that cleared a level. Expected near 0 under intrinsic-only reward (same baseline as exp_003_3).

Aggregate `sec6/L_state`, `sec6/L_action`, `sec6/L_total`, `sec6/reward_total` are kept as the average across the two envs (one number per JEPA update / rollout window).

### Reading the per-env rows

- `L_state_ls20 ≠ L_state_tu93` is expected. Persistent divergence > 2× is the early warning that one env is unmodelled or numerically unstable. The shared `state_predictor` cannot drive both to zero simultaneously if the two envs' dynamics differ in difficulty.
- `reward_state_component_<env>` close to `reward_clamp = 50.0` indicates that env's state predictor is unable to fit transitions — diagnose via `L_state_<env>`.

---

## Section 9 — Cross-environment gradient interference (NEW)

Headline metric for this experiment. For each parameter subset
`k ∈ {patch_sa, perc_cross_r0, perc_cross_r1, state_pred_mlp_0..3, action_pred}`,
run **two separate backward passes** on the same JEPA balanced batch (LS20 half, then TU93 half), collect per-parameter gradients via `torch.autograd.grad(..., retain_graph=True, allow_unused=True)`, and compute:

### gnorm (per env, per source, per sub-block)
- **`gnorm_jepa_ls20_<k>`** — L2 norm of LS20-half gradient vector in subset `k`.
- **`gnorm_jepa_tu93_<k>`** — same for TU93 half.
- Source-decomposed forms (mirror exp_003_4):
  - `gnorm_state_ls20_<k>`, `gnorm_state_tu93_<k>`
  - `gnorm_action_via_Ht_ls20_<k>`, `gnorm_action_via_Ht_tu93_<k>`
  - `gnorm_action_via_Htp1_ls20_<k>`, `gnorm_action_via_Htp1_tu93_<k>`

### Cross-env gradient cosine
- **`gcossim_jepa_ls20_vs_tu93_<k>`** — flatten-cosine between LS20 and TU93 total gradient vectors in subset `k`. Range `[-1, 1]`. The headline cross-env signal.
- Broken down by loss source:
  - `gcossim_state_ls20_vs_state_tu93_<k>`
  - `gcossim_action_ls20_vs_action_tu93_<k>`
  - `gcossim_state_ls20_vs_action_tu93_<k>`
  - `gcossim_action_ls20_vs_state_tu93_<k>`

### Interpretation reference (also in system_card.md §12)

| `gcossim_jepa_ls20_vs_tu93_<k>` | Reading |
|--------------------------------|---------|
| ≈ +1 | Two envs reinforce — shared module is doing useful joint work. |
| ≈  0 | Two envs uncorrelated — shared module is averaging unrelated tasks. |
| ≈ −1 | Two envs in direct opposition — strongest cross-env interference. |

### Cadence
Same as exp_003_4's `grad_decomp_freq = 25` — these probes do extra backward passes and are not free. Implemented in `monitors/gradients.py` by extending `compute_source_decomposition` to operate on per-env batches.

---

## Section 10 — Beyond cosine: per-element disagreement (NEW)

Cosine compresses an entire module's parameter space into one scalar. Per-element sign disagreement complements it. Same gating as Section 9.

### Per-element sign disagreement
- **`gsign_disagree_frac_<k>`** — `mean(sign(g_ls20) != sign(g_tu93))`, treating zeros as agreement. Range `[0, 1]`; 0.5 = random sign relationship; 0 = perfect sign agreement.
- **`gsign_disagree_frac_magweighted_<k>`** — sign disagreement weighted by `|g_ls20| * |g_tu93|`. Captures whether disagreement is concentrated at heavily-updated parameters.

### Per-layer cosine distribution
- **`gcossim_perlayer_<k>_<layer>`** — cosine computed over each `nn.Linear` weight, attention projection, FFN layer inside subset `k`. Reveals layer-localised conflict that the module-global cosine averages away.

### Quick diagnosis table

| `gcossim` | `gsign_disagree_frac` | Interpretation |
|-----------|------------------------|----------------|
| ≈ +1 | ≈ 0 | Strong constructive coupling. |
| ≈ 0 | ≈ 0.5 | Uncorrelated — capacity underused. |
| ≈ 0 | ≈ 0 | Signs agree but magnitudes are uneven. |
| ≈ 0 | ≈ 1 | Per-element conflict, summed magnitudes happen to cancel. |
| ≈ −1 | ≈ 1 | Direct opposition — worst case. |

---

## What stays unchanged from exp_003_4

The following inherited metric families do **not** get env-suffix variants. They describe shared modules and are computed once per JEPA / probe cycle on a *combined* sample from both envs' buffers:

- sec1 (representation health) — `placeholder_pairwise_cossim`, `latent_pairwise_cossim_buf`, `latent_eff_rank`, `latent_norm_*`, `placeholder_drift_from_init_*`, `ht_htp1_cossim_rollout`, `H1_HT_cossim`
- sec2 (image-patch SA) — `patch_sa_row_jsd`, `patch_sa_temporal_jsd`
- sec4 (predictors) — `ode_step_cossim`, `ode_first_vs_final_cossim`, `predictor_velocity_norm`, `action_pred_entropy*`, `action_pred_input_cossim`
- sec7 (gradient norms / UWR per sub-block) — `gnorm_<sub>_total`, `uwr_<sub>`. Note: the cross-env breakdowns added in Section 9 ABOVE coexist with these aggregate keys.

The eval-pass time-series (`*_eval`, `*_t1`, `*_t10`, `*_t20`, `H1_HT_cossim`) are computed twice per eval cycle — once per env — and reported under env-suffixed keys.

---

## Implementation pointers

- HealthMonitor (re-exported from `exp_003_4_no_resampler_self_attn/monitors/health.py`) already accepts arbitrary keys via `push_sec7(key, value)`. Per-env keys are added on-the-fly via `health.sec6.setdefault(f"{base}_{env}", deque(maxlen=200)).append(v)` in `train.py`.
- MetricsWriter (also re-exported) iterates `health.sec1..sec7` dicts, so any key added there flows into `metrics.jsonl` automatically — no writer changes required.
- Cross-env gradient probe (Section 9) lives in a planned extension of `monitors/gradients.py` — TODO: implement `compute_source_decomposition_cross_env(...)`. Until then, only the aggregate sec7 sub-block gnorms are tracked.
