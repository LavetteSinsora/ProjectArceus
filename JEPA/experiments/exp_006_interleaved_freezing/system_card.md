# System Card — exp_006_interleaved_freezing

| Field | Value |
|-------|-------|
| **Experiment ID** | `exp_006_interleaved_freezing` |
| **Status** | To be implemented |
| **Parent experiment** | `exp_003_3_state_only_reward` |
| **Game** | LS20 Level 1 (`ls20-9607627b`) |
| **Reward** | Inherited from parent — intrinsic curiosity, **state-prediction error only** |

---

## 1. One-Paragraph Summary

This experiment is a minimal-diff fork of `exp_003_3_state_only_reward`. **Architecture, reward, JEPA loss, optimisers, buffers, and the policy are unchanged.** The only change is the *training schedule*: a `FreezeScheduler` toggles between two top-level modes — `INTERLEAVED` (alternates a predictor-frozen / encoder-trains phase with an encoder-frozen / predictors-train phase) and `JOINT` (the parent's behaviour). Transitions between modes are event-driven: `INTERLEAVED → JOINT` when the predictor is healthy *and* representations are spread (conjunction on running `L_action` and running `ht_htp1_cossim`), and `JOINT → INTERLEAVED` re-entry when `ht_htp1_cossim` rises above a higher hysteresis threshold (collapse re-detected). The motivation is empirical: in `exp_003_3` we observe representation collapse (`ht_htp1_cossim_eval ≈ 0.6–0.8`, intra-`H_t` `latent_pairwise_cossim_t1 ≈ 0.96`) even as `L_action` drops to 0.12 — the action predictor succeeds via residual differences rather than by forcing the encoder to spread states. The hypothesis is that the joint optimisation cannot escape this attractor because both sides are moving targets; alternating freezes give each side a stable target to fit against (see §3).

---

## 2. Change vs. parent (`exp_003_3_state_only_reward`)

Two localised changes. **No model-class, reward, loss, or buffer code changes.**

### 2.1 Config additions

**File:** `JEPA/experiments/exp_006_interleaved_freezing/config.py`

```python
from dataclasses import dataclass
from JEPA.experiments.exp_003_3_state_only_reward.config import Config as _Base033

@dataclass(frozen=True)
class Config(_Base033):
    # ── Interleaved freeze schedule (NEW) ──────────────────────────────────────
    freeze_enabled: bool = True
    freeze_phase_len: int = 500              # JEPA updates per freeze phase
    freeze_initial_mode: str = "interleaved" # "interleaved" | "joint" — what the run starts in
    freeze_initial_phase: str = "encoder_frozen"  # which freeze phase fires first when interleaved
    freeze_threshold_window: int = 200       # JEPA updates of running mean used for thresholds
    freeze_threshold_min_phases: int = 4     # ≥ N complete phases before INTERLEAVED→JOINT can fire

    # Exit criterion (INTERLEAVED → JOINT): both must hold simultaneously
    freeze_l_action_exit: float   = 0.05     # PLACEHOLDER — tune via calibration run (§6.3)
    freeze_cossim_exit:   float   = 0.30     # PLACEHOLDER — tune via calibration run (§6.3)

    # Re-entry criterion (JOINT → INTERLEAVED): collapse detected
    freeze_cossim_reentry: float  = 0.70     # PLACEHOLDER — tune via calibration run (§6.3)
    freeze_reentry_cooldown: int  = 1000     # JEPA updates after exit before re-entry can fire
```

All five threshold defaults are intentionally placeholders, to be replaced after the calibration run described in §6.3. The exit/re-entry thresholds form a hysteresis band on `ht_htp1_cossim` (exit at ≤ 0.30, re-enter at ≥ 0.70) so the schedule does not chatter at the boundary.

### 2.2 Training-loop changes

The only structural change to `train.py` is inside the JEPA update block (replacing lines ~440-493 of [`exp_003_3_state_only_reward/train.py`](../exp_003_3_state_only_reward/train.py#L440-L493)):

- A `FreezeScheduler` (defined in `train.py`) is instantiated once before the main loop.
- Before each JEPA update, query `enc_frozen, pred_frozen = scheduler.freeze_state()`.
- `requires_grad` is toggled accordingly on `encoder`, `state_predictor`, `action_predictor`, `action_embed` params.
- `zero_grad()` is skipped for frozen optimisers; `.step()` is skipped for frozen optimisers (this matters — AdamW decoupled weight decay would otherwise drift frozen params).
- When the encoder is frozen, its two forward passes are wrapped in `with torch.no_grad():` for compute savings.
- After the loss step, call `scheduler.step(L_action.item(), <running cossim>)` with the running `ht_htp1_cossim_rollout` mean. This advances the schedule and applies mode-transition logic.

Everything else — the encoder forward chain, the dual-loss construction (`L_state + L_action`), gradient clipping, gradient-source decomposition, UWR snapshots, sub-block gnorm totals, the policy update path, the reward computation, the buffer, and all eval probes — is **inherited unchanged** from `exp_003_3`.

**Action embed grouping.** `action_embed` shares the state-predictor optimiser group in the parent ([`exp_003_3/train.py:294-297`](../exp_003_3_state_only_reward/train.py#L294-L297)). For consistency with that grouping, `action_embed` is frozen and unfrozen *with the predictor side*: in the encoder-trains phase its params are frozen alongside `state_predictor` and `action_predictor`, in the predictors-train phase it updates with them. This treats `action_embed` as part of the "what the predictor sees" surface, which is the right semantics for the targeted intervention.

---

## 3. Rationale — Why interleaved freezing?

### 3.1 Observed collapse signature in `exp_003_3`

From [`runs/run_20260521_113423_fresh/metrics.jsonl`](../exp_003_3_state_only_reward/runs/run_20260521_113423_fresh/metrics.jsonl):

| metric                                | step 25k | step 50k | step 100k | interpretation                                       |
|---------------------------------------|----------|----------|-----------|------------------------------------------------------|
| `sec1/ht_htp1_cossim_eval`            | 0.815    | 0.691    | 0.606     | h_t ≈ h_{t+1} in latent space                        |
| `sec1/latent_pairwise_cossim_t1`      | 0.986    | 0.961    | 0.966     | The 4 latent slots inside one H_t are near-identical |
| `sec4/ode_first_vs_final_cossim`      | 0.841    | 0.743    | 0.756     | State predictor barely moves — near-identity mapping |
| `sec4/action_pred_input_cossim`       | 0.625    | 0.601    | 0.617     | Action predictor sees barely-distinguishable inputs  |
| `sec6/L_action`                       | 0.528    | 0.136    | 0.116     | Action predictor still drives CE down anyway         |
| `sec6/completion_rate`                | 0.0      | 0.0      | 0.0       | Agent never solves the level                         |

The action predictor *does* drive `L_action` down (from `ln 4 ≈ 1.386` to ~0.12) even while `h_t ≈ h_{t+1}` — it succeeds via tiny residual differences rather than by forcing the encoder to produce informative reps. The state predictor settles near the identity mapping (`ode_first_vs_final_cossim ≈ 0.75`). The joint optimum is *"encoder maps everything close to one point + predictor extracts a sliver"* rather than *"encoder spreads states out + predictor decodes cleanly"*. The action-predictor anti-collapse signal added in exp_003_2 / exp_003_3 was intended to prevent this; it does not.

### 3.2 Why joint training cannot escape this attractor

Both the encoder and the predictors are moving targets for each other. When the encoder moves to make `(h_t, h_{t+1})` more informative, the predictor that *would* benefit from the new representation has already moved on. When the predictor improves on the current representation, the encoder has already drifted. The cheapest joint solution under SGD is the collapse attractor: a (near-)constant encoder paired with a (near-)identity state predictor and a (near-)trivial action predictor — all three can sit there indefinitely because their local gradients all point inward.

The action-predictor anti-collapse signal exists, but `L_action ≈ 0.12` shows that it does not dominate: the predictor is happy decoding actions from a near-collapsed representation via small residual differences.

### 3.3 Why interleaved freezing should help

Each freeze phase converts the moving-target optimisation into a fixed-target one:

- **Predictors-train phase (encoder frozen).** Predictors fit a stationary representation. They learn the best possible action / next-state map for whatever spread is currently in `H`. Because the encoder cannot drift to make their job easier, they cannot rely on the encoder's collapse as a crutch.
- **Encoder-trains phase (predictors frozen).** The encoder cannot reshape its output to make the predictor's lazy attractor easier — the predictor's input/output map is locked. The encoder's only path to lower loss is to produce `(h_t, h_{t+1})` pairs that the *current* predictor can actually decode well, which requires real separation in latent space (when the predictor is not yet at the trivial solution).

This is the same trick used by BYOL (frozen target encoder), two-time-scale GAN training, and EM-style alternating optimisation. The phase length (`freeze_phase_len`) controls the time-scale separation.

### 3.4 Why a hysteresis band on `ht_htp1_cossim` drives mode transitions

A pure "run interleaving as a fixed warmup, then resume joint training" schedule has a failure mode: the encoder may re-collapse after the schedule hands control back to joint SGD — the attractor is still there. Empirically, parent runs show `ht_htp1_cossim_rollout` remaining above 0.6 throughout, with no sign of structural escape. So the schedule defends against re-collapse actively:

- **`INTERLEAVED → JOINT`** fires only when *both* `L_action` is low (predictor is healthy on the current encoder) *and* `ht_htp1_cossim` is low (representations are actually spread). `L_action` alone is necessary but not sufficient — see §3.1, where the parent reaches `L_action ≈ 0.12` while still collapsed. The conjunction prevents premature exit.
- **`JOINT → INTERLEAVED`** fires when `ht_htp1_cossim` rises above a higher threshold (`freeze_cossim_reentry`, default 0.70 vs. exit at 0.30). The gap between exit and re-entry thresholds is the hysteresis band — it prevents chatter at the boundary.
- A cooldown (`freeze_reentry_cooldown`) further debounces re-entry, and a `freeze_threshold_window` running mean smooths the signal.

The `n_exits` counter is the diagnostic of structural success: if the schedule fires once and never re-enters, the alternating-freeze phase produced a genuine escape from the collapse basin. If `n_exits` keeps incrementing, the schedule is masking a still-attractive collapse basin and a follow-up intervention is needed.

---

## 4. What stays the same (inherited from exp_003_3)

- **Encoder** (Stage 1 SA + Stage 2 Perceiver Resampler), `d_model = 128`, 4 latent vectors.
- **State predictor** (per-latent flow-matching MLPs, 3 Euler steps).
- **Action predictor** (MLP 1024 → 512 → 4, no detach on either endpoint).
- **JEPA loss:** `0.5 · L_state + 0.5 · L_action` — unchanged.
- **Reward:** `reward_w_state = 1.0`, `reward_w_action = 0.0`, `reward_clamp = 50.0` — unchanged. `reward_action_component` is still computed and logged.
- **Buffer:** stores raw `next_frame` (uint8); both `h_t` and `h_{t+1}` re-encoded fresh every training step; uniform sampling.
- **Policy:** stateless 512-hidden MLP, REINFORCE with scalar EMA baseline, entropy `λ = 0.10`. Trains in *both* modes — its update path is independent of the freeze schedule.
- **Schedule:** 1k warmup, JEPA every 5 env steps, policy every 64 env steps, max 500k steps. (The freeze schedule overlays this without changing it.)
- **No EMA target encoder.**
- **Anti-collapse stays via action predictor** in the JEPA loss; the freeze schedule is layered on top.

For any detail not explicitly listed in §2 of this document, see the parent system card at [`exp_003_3_state_only_reward/system_card.md`](../exp_003_3_state_only_reward/system_card.md).

---

## 5. File Layout (to create)

```
JEPA/experiments/exp_006_interleaved_freezing/
├── system_card.md             — this document
├── __init__.py
├── config.py                  — see §2.1; inherits exp_003_3.Config, adds freeze_* fields
├── train.py                   — copy of exp_003_3/train.py with FreezeScheduler + JEPA-block change
├── eval.py                    — copy of exp_003_3/eval.py with config import swapped
├── debug_runner.py            — copy of exp_003_3/debug_runner.py with config import swapped
├── reward_shaping.py          — re-export of exp_003_3/reward_shaping.py
├── panel.js                   — copy of exp_003_3/panel.js (dashboard plugin)
├── models/                    — copy of exp_003_3/models/ unchanged
├── monitors/                  — copy of exp_003_3/monitors/ unchanged
├── checkpoints/               — empty, created at first save
└── runs/                      — empty, created at first run
```

**Implementation note:** Copy the whole experiment dir over importing parent modules wholesale, because the parent's `train.py` imports its own config and models via package path. The fork only needs those import paths swapped to point at `exp_006_interleaved_freezing.*`. Do not refactor the parent — keep it intact as the comparison baseline.

### 5.1 New metrics (added under `sec8/`)

The metrics writer adds the following keys (no removals):

| key                                   | meaning                                                                  |
|---------------------------------------|--------------------------------------------------------------------------|
| `sec8/freeze_mode`                    | 0 = JOINT, 1 = INTERLEAVED                                               |
| `sec8/freeze_phase`                   | 0 = encoder_frozen (predictors train), 1 = predictors_frozen (encoder), −1 in JOINT |
| `sec8/phase_step`                     | progress through the current freeze phase                                |
| `sec8/phases_completed_this_block`    | count within current INTERLEAVED block (resets on each re-entry)         |
| `sec8/n_exits`                        | total INTERLEAVED→JOINT transitions so far in this run                   |
| `sec8/l_action_running_mean`          | running mean over `freeze_threshold_window` updates                      |
| `sec8/cossim_running_mean`            | running mean over `freeze_threshold_window` updates                      |

### 5.2 Checkpoint additions

`save_checkpoint` / `load_checkpoint` persist:
`freeze_mode`, `freeze_phase`, `freeze_phase_step`, `freeze_phases_completed_this_block`, `freeze_n_exits`, `freeze_updates_since_last_exit`.

The two running windows are *not* persisted — they re-warm within `freeze_threshold_window` JEPA updates after resume, which is short relative to typical resume gaps.

---

## 6. Acceptance Tests (run after implementation)

1. **Static.**
   ```
   grep -n 'freeze_enabled' JEPA/experiments/exp_006_interleaved_freezing/config.py
   grep -rn 'exp_003_3' JEPA/experiments/exp_006_interleaved_freezing/   # should be 0 matches
   ```

2. **Smoke run, freeze enabled.**
   ```
   uv run python -m JEPA.experiments.exp_006_interleaved_freezing.train --max-steps 3000
   ```
   Confirm in `metrics.jsonl`:
   - `sec8/freeze_mode == 1` for early rows (INTERLEAVED).
   - `sec8/freeze_phase == 0` at the very start (encoder_frozen / predictors train, per `freeze_initial_phase`).
   - `freeze_phase` flips to 1 after `freeze_phase_len` JEPA updates, then back to 0.

3. **Threshold calibration run.** Short run (e.g. 30k env steps) with `freeze_enabled=False` so the run is permanently in JOINT — this reproduces exp_003_3's curves inside this directory. From those curves pick:
   - `freeze_l_action_exit` ≈ the `L_action` value reached *only after* `ht_htp1_cossim_eval` has clearly dropped (avoids the Risk #3 trap of exiting while collapsed).
   - `freeze_cossim_exit` ≈ a low-water mark on `ht_htp1_cossim_rollout` that the parent reaches transiently.
   - `freeze_cossim_reentry` ≈ a value comfortably above `freeze_cossim_exit` (hysteresis ≥ 0.2 absolute) at which the parent run shows the collapse pathology.

4. **Mode-transition smoke.** Set thresholds tight enough that a transition fires within 5k env steps. Confirm `sec8/n_exits` increments and that one full JOINT→INTERLEAVED re-entry fires within an instrumented run.

5. **Checkpoint resume.** Train 2000 steps, save mid-INTERLEAVED, resume. Verify `freeze_mode`, `freeze_phase`, `phase_step`, `phases_completed_this_block`, `n_exits` match across the boundary.

6. **No-regression on JEPA in JOINT mode.** With `freeze_enabled=False`, training should be functionally identical to exp_003_3 — `L_state`, `L_action`, `ht_htp1_cossim` curves should overlap an exp_003_3 run from the same seed.

7. **Behaviour check vs. parent.** At matched env-step count (freeze enabled), compare `ht_htp1_cossim_eval` and `latent_pairwise_cossim_t1` against an exp_003_3 run. Expectation: consistently lower in exp_006, with a "saw-tooth" if any JOINT→INTERLEAVED re-entry fires.

8. **Grad-decomp robustness.** Confirm `compute_source_decomposition` does not raise when called during a freeze phase: it already uses `allow_unused=True` ([`exp_003_3/monitors/gradients.py:167-175`](../exp_003_3_state_only_reward/monitors/gradients.py#L167-L175)) so `requires_grad=False` params surface as `None` grads and the helper reports 0. Dashboard lines dip to 0 during the corresponding freeze phase rather than vanish.

---

## 7. Risks & Out-of-Scope

### 7.1 Risks (flagged for analysis, not for pre-emptive patching)

1. **Initial encoder-frozen phase may not actually escape collapse.** Mitigated by starting with `freeze_initial_phase = "encoder_frozen"` (predictors train *first*) so the predictor first learns from a random-init encoder's reps — which are not yet collapsed — before the encoder phase starts.
2. **`L_action` alone is necessary but not sufficient as exit criterion.** The parent reaches `L_action ≈ 0.12` while still collapsed (§3.1). Handled by the conjunction `L_action < freeze_l_action_exit` AND `ht_htp1_cossim < freeze_cossim_exit`.
3. **Chatter / starvation.** If thresholds are mis-tuned the run can oscillate (frequent mode flips) or one side can starve. Mitigated by hysteresis band, `freeze_reentry_cooldown`, running-mean windowing, and `freeze_threshold_min_phases`. The `sec8/n_exits` metric makes pathological schedules visible — `n_exits > 5` per run is a signal to widen the band.
4. **Post-resume re-collapse.** Actively handled by the re-entry trigger; if `n_exits` keeps rising over a long run, the conclusion is "interleaving is a transient fix, not a structural one", and follow-up work (e.g. EMA target encoder, BYOL-style stop-grad) is warranted.

### 7.2 Out of scope (deliberately deferred — discuss in session)

These are **not** in this experiment. Listed so the implementing agent knows not to add them on its own initiative.

1. EMA target encoder (BYOL-style stop-grad on a moving-average copy).
2. Learning-rate schedule that differs between INTERLEAVED phases (e.g. higher predictor LR during predictors-train phase).
3. Phase length that scales with training step or with running L_action.
4. Three-way phase rotation (encoder / state-predictor / action-predictor each separately frozen).
5. Reward-side changes (this experiment keeps state-only reward from exp_003_3 verbatim).
6. Episodic novelty, value head + GAE, goal-conditioned policy, RND — all inherited as out-of-scope from exp_003_3 §7.

---

## 8. Expected Outcome

- **High confidence:** `sec1/ht_htp1_cossim_eval` and `sec1/latent_pairwise_cossim_t1` stay lower than exp_003_3 at matched env-step count, because the re-entry mechanism actively pulls them back below threshold.
- **Medium confidence:** `sec4/ode_first_vs_final_cossim` decreases relative to parent — the state predictor moves away from near-identity once it has informative `(h_t, h_{t+1})` to fit.
- **Medium confidence:** the policy meaningfully visits new states (reachable-tile coverage rises) once representations are spread; whether this translates into Level-1 completions remains an open question because there is no temporal credit assignment.
- **Diagnostic:** `sec8/n_exits` is the load-bearing read-out. `n_exits = 1` (one exit, never re-enters) → structural fix; `n_exits` keeps rising → schedule masks a still-attractive collapse basin; either way the run produces a clean signal about whether interleaved freezing solves the underlying optimisation difficulty or merely papers over it.

If the agent still gets stuck after this change — but with `sec8/n_exits` saturating, or with collapse metrics improving while completion rate stays at 0% — that is a useful and distinct signal and should be reported back rather than patched ad-hoc.
