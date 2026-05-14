# System Card — exp_003_3_state_only_reward

| Field | Value |
|-------|-------|
| **Experiment ID** | `exp_003_3_state_only_reward` |
| **Status** | To be implemented |
| **Parent experiment** | `exp_003_2_action_pred_no_ema` |
| **Game** | LS20 Level 1 (`ls20-9607627b`) |
| **Reward** | Intrinsic curiosity, **state-prediction error only** (action-prediction error removed from reward, retained in JEPA loss) |

---

## 1. One-Paragraph Summary

This experiment is a minimal-diff fork of `exp_003_2_action_pred_no_ema`. **Architecture, training procedure, optimisers, buffers, and the JEPA loss are unchanged.** The only change is the per-step intrinsic curiosity reward: the action-predictor cross-entropy term is **removed from the reward**, while the action predictor itself remains in place and continues to contribute to the JEPA loss for anti-collapse. The motivation is empirical: in `exp_003_2` the agent settles into a corner and refuses to explore, and inspection of the reward formula shows that the corner is in fact a maximum-reward state for this reward function (see §3). Dropping the action-CE term removes that pathology while preserving the action predictor's representation-learning role.

---

## 2. Change vs. parent (`exp_003_2_action_pred_no_ema`)

Exactly one config-level change. No model-class, training-loop, buffer, or optimiser code changes.

**File:** `JEPA/experiments/exp_003_3_state_only_reward/config.py`

```python
from dataclasses import dataclass
from JEPA.experiments.exp_003_2_action_pred_no_ema.config import Config as _Base032

@dataclass(frozen=True)
class Config(_Base032):
    # ── Reward weighting (only fields that differ from exp_003_2) ─────────────
    reward_w_state:  float = 1.0      # was 0.5 — restore magnitude after dropping action term
    reward_w_action: float = 0.0      # was 0.5 — REMOVED; see system card §3
    # reward_clamp stays at 50.0 (inherited)
```

Everything else — encoder, state predictor, action predictor, action embedding, policy, optimisers, schedule, buffer, RoPE, JEPA loss weights (`lambda_state = lambda_action = 0.5`) — is **inherited unchanged** from `exp_003_2`.

The action-CE component is still computed at rollout time and still logged independently as `reward_action_component` (see `train.py` around line 391). Because `reward_w_action = 0.0`, it simply does not contribute to `curiosity_reward` and therefore does not drive the policy.

---

## 3. Rationale — Why drop the action-CE reward term?

The reward in the parent experiment ([`exp_003_2/train.py:381-384`](../exp_003_2_action_pred_no_ema/train.py#L381-L384)) is:

```
raw = w_state * state_err  +  w_action * action_err
    = w_state * MSE(state_predictor(h_t, a_t), h_{t+1})
    + w_action * CE(action_predictor(h_t, h_{t+1}), one_hot(a_t))
```

The intent of the `action_err` term is to reward the agent for visiting transitions that the action predictor cannot yet decode. In LS20 Level 1, however, the action predictor is structurally adversarial to navigation:

- LS20 walls block 3 of the 4 actions in any corner / wall-adjacent cell. Those blocked-action transitions produce a `next_frame` essentially equal to `frame_t` (after masking the step-counter rows).
- After re-encoding, `h_t ≈ h_{t+1}` for those blocked actions, so the pair `(h_t, h_{t+1})` carries near-zero information about which action was taken.
- The action predictor therefore cannot do better than chance on those transitions: cross-entropy stays near its maximum `ln 4 ≈ 1.386` and **does not decay with training**.
- Consequently, the corner is a *maximum-reward state* for the parent's reward function. This is the noisy-TV problem in reverse — the agent farms ambiguity rather than novelty.

`state_err` does not have this pathology: a competent state predictor learns the identity transition `h_t → h_t` for blocked actions, drives `state_err → 0` in those regions, and gives the agent a gradient out of corners.

The action predictor is still needed as the encoder's anti-collapse regulariser (parent §2.5, §4.2 path 3). Removing it from the **reward** while keeping it in the **JEPA loss** preserves the anti-collapse property without the noisy-TV trap.

**Why `reward_w_state = 1.0` (not 0.5):** keep per-step reward magnitude comparable to the parent's average so the existing entropy coefficient (`policy_entropy_lambda = 0.10`) and advantage normalisation do not need re-tuning. The policy update std-normalises the advantage anyway, so the absolute scale is not load-bearing, but matching it minimises unrelated regressions.

---

## 4. What stays the same (inherited from exp_003_2)

- **Encoder** (Stage 1 SA + Stage 2 Perceiver Resampler), `d_model = 128`, 4 latent vectors.
- **State predictor** (per-latent flow-matching MLPs, 3 Euler steps).
- **Action predictor** (MLP 1024 → 512 → 4, no detach on either endpoint).
- **JEPA loss:** `0.5 · L_state + 0.5 · L_action`.
- **Buffer:** stores raw `next_frame` (uint8); both `h_t` and `h_{t+1}` re-encoded fresh every training step; uniform sampling.
- **Policy:** stateless 512-hidden MLP, REINFORCE with scalar EMA baseline, entropy `λ = 0.10`.
- **Schedule:** 1k warmup, JEPA every 5 env steps, policy every 64 env steps, max 500k steps.
- **Reward clamp:** 50.0.
- **No EMA target encoder.**
- **Anti-collapse remains via action predictor** (gradient through both `h_t` and `h_{t+1}`).

For any detail not explicitly listed in §2 of this document, see the parent system card at [`exp_003_2_action_pred_no_ema/system_card.md`](../exp_003_2_action_pred_no_ema/system_card.md).

---

## 5. File Layout (to create)

```
JEPA/experiments/exp_003_3_state_only_reward/
├── system_card.md             — this document
├── __init__.py
├── config.py                  — see §2; inherits exp_003_2.Config, two field overrides
├── train.py                   — copy of exp_003_2/train.py with config import swapped
├── eval.py                    — copy of exp_003_2/eval.py with config import swapped
├── debug_runner.py            — copy of exp_003_2/debug_runner.py with config import swapped
├── reward_shaping.py          — identical re-export of exp_003_2/reward_shaping.py (or import-and-re-export)
├── panel.js                   — copy of exp_003_2/panel.js (dashboard plugin)
├── models/                    — copy of exp_003_2/models/ unchanged
├── monitors/                  — copy of exp_003_2/monitors/ unchanged
├── checkpoints/               — empty, created at first save
└── runs/                      — empty, created at first run
```

**Implementation note:** Prefer copying the whole experiment dir over importing parent modules wholesale, because the parent's `train.py` imports its own config via package path (`from JEPA.experiments.exp_003_2_action_pred_no_ema.config import Config`). The fork only needs that one import line changed to point at `exp_003_3_state_only_reward.config`. Do not refactor the parent — keep it intact as the comparison baseline.

---

## 6. Acceptance Tests (run after implementation)

1. **Static.** `grep -n 'reward_w_action' JEPA/experiments/exp_003_3_state_only_reward/config.py` returns the `0.0` line. `grep -rn 'exp_003_2' JEPA/experiments/exp_003_3_state_only_reward/` returns **zero** matches (no leaked parent imports).
2. **Smoke run.** From repo root:
   ```
   uv run python -m JEPA.experiments.exp_003_3_state_only_reward.train --max-steps 3000
   ```
   Should complete without errors and produce a checkpoint dir under `runs/<timestamp>/`.
3. **Reward log invariant.** In the smoke-run `metrics.jsonl`, confirm:
   - `reward_action_component` is still populated (the action_err is still being computed for logging).
   - `reward_total` ≈ `state_err` (within float noise — `reward_w_state = 1.0`).
4. **Behaviour check.** Visualise 5 eval rollouts (`debug_runner.py` on the latest checkpoint). Compare against a parent-experiment checkpoint of similar age:
   - Does the agent leave the starting region within the first ~20 steps of an episode? (Parent stays in the corner.)
   - Episode-length distribution — parent piles up at the ~129-step energy bound; expect a broader distribution if exploration unblocks.
5. **No-regression on JEPA.** `L_state`, `L_action`, latent-variance, and `ht_htp1_cossim` curves should be qualitatively unchanged from a parent run at matched env-step count — the JEPA training path is untouched.

---

## 7. Out of Scope (deliberately deferred — discuss in session)

These are **not** in this experiment. They are listed so the implementing agent knows not to add them on its own initiative.

1. Episodic novelty (k-NN bonus in latent space, per-episode reset).
2. Value head + GAE / PPO (REINFORCE retained).
3. Large terminal bonus on `levels_completed >= 1` (no extrinsic reward of any kind).
4. Goal-conditioned policy + HER.
5. Learning-progress reward (decrease in `state_err` over a window).
6. RND-style fixed-target curiosity.

Strategy for which of these (or other ideas) to try next is being worked out in the session that produced this card; the implementing agent should not pre-empt that decision.

---

## 8. Expected Outcome

- **High confidence:** The corner-attractor pathology disappears. The agent moves around the map.
- **Low confidence:** Level 1 completion rate improves meaningfully. With pure state-novelty curiosity, no temporal credit assignment, and no goal signal, completion is expected to remain at or near 0%. The value of this experiment is to **unblock exploration** so that follow-up reward layers (§7) have a base policy that actually visits new states.

If the agent still gets stuck after this change — but in a different state, or with `state_err` saturating the clamp — that is a useful and distinct signal and should be reported back rather than patched ad-hoc.
