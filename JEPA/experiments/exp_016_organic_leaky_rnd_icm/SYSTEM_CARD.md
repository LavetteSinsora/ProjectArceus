# exp_016 — System Card: Leaky RND on ICM-φ

*What this method is, every technique baked into it and **why**, and which of those
techniques are actually load-bearing — so exp_016 can strip the rest and ship a
smaller, organic version.*

Each technique carries a **KEEP / SIMPLIFY / STRIP** verdict. Sources: the code in
`exp_011…exp_015` and the consolidated diagnoses in
`exp_014_figures_and_results/leaky_rnd_lineage_recap.html`. Exact hyperparameters
live in the **Appendix** so the body stays readable.

---

## 1. The method in one picture

The task: drop a PPO agent into a level with **terminal-only reward**, measure
**env-steps to the first success**, stop there. During learning there is *no reward
to optimize* — the whole job is shaping an **intrinsic** signal that reaches a first
success fast.

```
every update (= 2048 env steps, on-policy, no replay):
    collect rollout
    novelty(s') = ½‖P(φ(s')) − T(φ(s'))‖²        # RND in a feature space φ, not pixels
    reward      = normalize(novelty)              # one signal; the env +1 is only the STOP flag
    PPO update  (single intrinsic value head, non-episodic GAE)
    train φ     (ICM inverse+forward) until frozen
    train P, then LEAK it toward init            # the core idea
```

Three ideas stacked on PPO: **(1)** measure novelty with **RND** **(2)** inside
**ICM's controllable feature space φ** (not pixels) **(3)** with a **leaky**
predictor so novelty regenerates instead of saturating. Everything else is support
machinery for stability and correctness.

---

## 2. The two insights that should drive exp_016

**Insight 1 — the leak is the real contribution, and it works.** Standard RND is a
one-way ratchet: once a state is learned its novelty dies *permanently*, so
exploration stalls. The leak (`θ_P ← (1−μ)θ_P + μ·θ_init` after each update) turns
RND from a *cumulative count* into a *recent-visitation rate* — abandoned states go
novel again. Three controlled experiments (exp_014_1/2/5) confirm this cleanly and
μ-ordered. **This is the one thing that behaves exactly as designed.**

**Insight 2 — but the leak never won a benchmark, because the binding constraint is
upstream.** On the hard cells the wall is **reward coverage** — the agent never
reaches the goal *even once* (`v_ext ≡ 0` for every method). Novelty saturation was
never the bottleneck there. Most of the supporting machinery (the entropy/value/γ
tuning ladder) was fighting **entropy collapse**, which is itself downstream of two
deeper bugs: a **marching-timer observation confound** and **φ never becoming
controllable**. **Fix the upstream causes and most of the machinery becomes
deletable.** That is the thesis of the simplification.

---

## 3. Keep / Simplify / Strip — the verdict at a glance

| verdict | techniques | why |
|---|---|---|
| **KEEP** (load-bearing) | the leak (§4.1); RND-on-φ vs pixels (§4.2); single intrinsic head + non-episodic GAE (§4.3); EMA-std normalizer + warm-up (§4.4); timer-row mask (§4.5) | each fixes a *measured* failure and the fix held up |
| **SIMPLIFY** | φ-freeze machinery (§4.6); raw-novelty clip; method-branching in the normalizer | keep the intent, drop the brittle/duplicated parts |
| **STRIP** | additive ICM+RND reward (dead — §4.7); the `c_value`/`γ`/`c_entropy`/`novelty_dead_eps` ladder (§4.8); on-policy freeze metric; φ-transfer + dual-stream + MCTS/disagreement scaffolding (§4.9) | dead, a band-aid for a deeper bug, or unused |

---

## 4. The techniques — what, why, verdict

### 4.1 The leak — KEEP (the thesis)
Pull the RND predictor back toward its random init each update, decoupled from Adam.
Distillation drives error *down* where you currently visit; the leak drags it *up*
everywhere — states you keep visiting win, abandoned states go novel again.
**Why:** without it RND saturates and exploration stalls (~80k steps on L1). **Note
for exp_016:** the leak rate `μ` is the headline knob, but it is *not yet
load-bearing for any solve* — converting it into a win is the goal. (See §6 for a
cadence bug to fix.)

### 4.2 RND on φ, not pixels — KEEP
Run RND over ICM's inverse-dynamics features φ instead of raw frames. **Why:**
pixel-RND was measured to have **no count resolution** (1 visit ≈ 2000) and a
**99.9% leak** to unseen states, because near-identical maze frames make the random
target smooth. φ must separate states (it has to name the action between them), which
restores resolution. **Risk:** φ never becomes a *trustworthy* ruler — see §4.6.

### 4.3 Single intrinsic head + non-episodic GAE — KEEP
No extrinsic value head (the first `+1` ends the run, so an extrinsic critic would
never train). One intrinsic head; non-episodic GAE so novelty value bootstraps
across death — the agent isn't deterred from life-costing deep exploration.
**Caveat:** non-episodic GAE *is* what inflates the return and triggers the value-lag
collapse (§4.8). The clean fix is **PopArt**, not the `γ`/`c_value` band-aids.

### 4.4 Reward normalizer: warm-up + EMA-std, no centering — KEEP (simplify)
Raw novelty → divide by a running **std of the intrinsic returns** (EMA, tracks the
*current* scale), never mean-centered (bonus ≥ 0); a short **warm-up** gives zero
reward while the predictor burns in. **Why:** ICM's untrained startup error is ~700×
its converged value; a cumulative std bakes that in and crushes the bonus for ~43k
steps — the warm-up + EMA fix took ls20-L1 from censored→18.8k (the clearest
harness-level win). **Simplify:** exp_016 is one method, so drop the per-method
branching — one normalizer.

### 4.5 Timer-row mask — KEEP (make it first-class)
The env feeds a step-timer that **marches every frame**, so every frame looks unique
(1073 "states" vs 43 real board states) and silently poisoned every novelty result.
Mask those rows on the φ/novelty path. **Why:** novelty must be scored on the true
board. This was "the most damaging bug." **Fix:** make it a clean preprocessing step,
not a monkey-patch on `.encode`, and make the rows per-game.

### 4.6 φ-freeze machinery — SIMPLIFY hard (or drop)
φ never self-stabilizes (drift plateaus at ~0.22–0.25 forever), so it's frozen once
it "separates states" to give RND a stationary ruler. The current code uses a
**held-out** inverse-accuracy trigger + a fallback + an uncontrollability guard —
all because the naive **on-policy** trigger is inflated by a narrowing policy and
froze a near-random φ. **The honest problem:** on the cells that matter (re86, ls20)
φ is *structurally uncontrollable*, so a **frozen-random φ is strictly better
there**. **Decision for exp_016:** prefer frozen-random φ (deletes the entire
freeze/holdout/guard subsystem) *unless* the now-fixed timer mask rescues
controllability — see §8.

### 4.7 Additive ICM+RND reward — STRIP (dead)
`w·norm(ICM-fwd) + (1−w)·norm(RND-on-φ)`. A free probe killed it: on ls20-L2 *both*
signals correlate **negatively** with true novelty (−0.46, −0.56), so no convex mix
can be positive. Combining two anti-informative rulers stays anti-informative.

### 4.8 The entropy-collapse tuning ladder — STRIP, replace with principled fixes
A cluster of hand-tuned knobs (`c_entropy` 0.01→0.10, `c_value` 0.5→1.0, `γ`
0.99→0.95, `novelty_dead_eps`, reward clip) added to fight entropy collapse. The
lineage proved collapse has **two mechanisms**: *value-lag/phantom advantage* (the
non-stationary return outruns the critic → a phantom advantage on the greedy action)
and *dead-field bleed* (reward decays to a trickle, entropy bleeds out). These knobs
are **band-aids** — "a big enough novelty burst overruns any fixed coefficient," and
the dual-head baseline collapsed *hardest*, so neither tuning nor architecture
protects you. **Replace with:** PopArt value-norm (mechanism 1) + an entropy floor /
KL-to-uniform (mechanism 2).

### 4.9 Unused scaffolding — STRIP
Cross-level φ-transfer (`init_phi_ckpt`), the `phi_mode` branch, the method-agnostic
dual-stream harness (`V_ext`, `make_bonus` dispatch), the MCTS-lookahead arm (benched
~7× slower than random) and the disagreement arm. None are part of leaky-RND proper.

---

## 5. How it actually trains (cadence & the update step)

Everything is **on-policy and update-synchronous — no replay.** All three nets reuse
the *same* 2048 transitions each update.

| net | optimizer | grad steps / update | trained on | notes |
|---|---|---|---|---|
| policy + value | PPO Adam, lr 3e-4 | epochs×mb = 4×4 = 16 | rollout | never touches φ |
| ICM (φ + inv + fwd) | own Adam, lr 1e-3 | 1×4 = 4 | non-done transitions | forward target stop-gradded; stops after freeze (1b) |
| RND predictor | own Adam, lr 1e-4 | 1×4 = 4 | cached φ(s') | target never trained; leak after each step |

**One update, in order:** collect rollout → score novelty on **φ snapshotted at the
start of the update** → normalize → PPO (policy+value only) → train φ (ICM) → train
RND predictor + leak → `global_step += 2048`.

**Time accounting:** 1 update = `128 steps × 16 envs = 2048` env steps. `global_step`
and all baselines are **summed over the 16 actors** (don't divide by 16). First-reward
step is a lockstep within-update interpolation.

---

## 6. Gotchas that cause real confusion (the short list)

The ones that make the design *look* inconsistent or that a rewrite would silently break:

1. **The leak fires per *minibatch*, not per update.** `apply_leak()` is inside the
   minibatch loop, so with mb=4 it runs **4×/update** → effective μ ≈ 4× nominal
   (~0.185 at μ=0.05), and is silently **coupled to the minibatch count.** All the
   docs say "once per update." → exp_016: move it outside the loop, leak once.
2. **The `RewardForwardFilter` is not the reward.** It's a forward discounted sum
   (persistent, never reset) used **only to estimate the divisor std.** The reward is
   `raw/std`. So `γ` is *not* applied twice — RFF-γ only shapes the normalizing scale.
3. **ICM and RND use different reductions → ~512× raw-scale gap** (ICM = `sum` over
   256 dims; RND = `½·mean`). That's why additive needs two normalizers and why their
   raw logs aren't eyeball-comparable.
4. **Non-episodic GAE removes *both* masks** (bootstrap *and* the λ-accumulator), not
   just the bootstrap — this double-unmasking is the direct cause of the return
   inflation behind the value-lag collapse.
5. **1b and the additive variant diverge after φ-freeze:** 1b halts the *entire* ICM
   update (and its held-out inv_acc logging goes stale); additive keeps training the
   ICM heads. Know which loop you're reading.
6. **"Frozen at update 100" isn't guaranteed** — the uncontrollability guard can
   block the freeze forever (re86), so φ may train the whole run.
7. **The RND target is seed-dependent** — each seed is a *different* random ruler;
   cross-seed comparisons compare different targets.

---

## 7. The organic design for exp_016 (the subtraction)

```
PPO (separate policy encoder; standard recipe)
  reward = normalize( leaky_RND_novelty( φ(masked s') ) )    # ONE signal
  φ      = frozen-random encoder        # no ICM/inverse/forward, NO freeze   (see fork below)
  leak   = once per update, μ≈0.05      # the thesis, cadence fixed
  norm   = warm-up → ÷EMA-std(returns), no centering
  mask   = zero UI/timer rows on the novelty path (per-game), first-class
  value  = single intrinsic head, non-episodic GAE, γ=0.99
  stable = PopArt value-norm + entropy floor / KL-to-uniform   # replaces the ladder
  stop   = first +1 (metric only)
```

**Deletes:** additive mix; ICM heads + freeze/holdout/guard; the
`c_value/γ/c_entropy/novelty_dead_eps` ladder; transfer/dual-stream/MCTS/disagreement
scaffolding. **Keeps:** leak, RND-on-φ, single head + non-episodic GAE, normalizer +
warm-up, timer mask.

**The one real fork:** **frozen-random φ vs learned ICM-φ.** Frozen-random φ deletes
the most machinery and was *strictly better where it mattered*; learned ICM-φ has a
higher ceiling *only if* controllability is fixed. The "leaky RND **on ICM**" brief
argues to keep ICM-φ — so exp_016 should first test whether the timer mask (a likely
cause of uncontrollability) rescues it.

---

## 8. What exp_016 must measure (decision gates)

1. **Does the timer mask rescue φ-controllability?** (held-out inv_acc on masked
   frames) → decides the §7 fork and whether the freeze machinery is justified.
2. **PopArt vs the γ/`c_value` band-aids** — does PopArt kill the value-lag collapse
   while keeping non-episodic GAE at γ=0.99?
3. **Entropy floor vs `c_entropy=0.10`** — does KL-to-uniform stop the dead-field
   bleed where a fixed coef can't?
4. **Does the leak finally convert?** μ>0 vs μ=0 on the ∞ cells, with *enough seeds*
   to clear variance (the lineage only had 2–4).
5. **Coverage is the real wall** — on L2/L3 `v_ext ≡ 0` for everyone. Decide whether
   exp_016 stays a pure-novelty study or adds Go-Explore-style frontier return (out
   of scope for the *simplification*, but the honest bottleneck).

---

## Appendix — hyperparameters & file map

**Backbone / PPO:** `n_envs=16`, `rollout_steps=128` (2048/update), `epochs=4`,
`minibatches=4`, `clip_eps=0.2`, `vf_clip_eps=0.2`, `grad_clip=0.5`, `lr=3e-4`,
`gae_λ=0.95`, `trunk_dim=256`, one-hot `(16,64,64)`, `max_episode_steps=200`.
Policy-head init gain 0.01 (near-uniform start), value-head 1.0.

**Intrinsic / value:** `γ=0.95` (shrunk from 0.99 as a band-aid — revisit),
`intrinsic_episodic=False`, `c_value=1.0`, `c_entropy=0.10` (laddered),
`int_norm_decay=0.99`, `norm_warmup_updates=2`, `reward_clip_k=5.0`,
`novelty_dead_eps=0.01`.

**ICM:** `β=0.2`, `icm_lr=1e-3`, `icm_epochs=1`, `icm_hidden=256`.

**RND + leak:** `rnd_feature_dim=256`, `rnd_hidden=256`, `rnd_lr=1e-4`,
`rnd_epochs=1`, predictor 3-layer / target 2-layer (frozen), `leak μ=0.05`.

**φ-freeze:** `phi_freeze_inverse_acc=0.70`, `patience=3`, `max_updates=100`,
`freeze_metric="holdout"`, `holdout_size=2000`, `phi_uncontrollable_factor=1.5`.
**Timer mask:** `mask_timer=True`, `timer_mask_rows=(60,63)` (ls20).

**File map**
- `exp_011_ls20_icm/shared/icm.py` — ICM (φ encoder, inverse/forward, raw error).
- `exp_012_ls20_rnd/shared/rnd.py` — RND nets, `RewardForwardFilter`, `RunningMeanStd`.
- `exp_013_…/exp_013_1b_leaky_rnd_on_icm_phi/` — **the headline variant** (rnd_phi=leak, trainer=full loop, config=all knobs).
- `exp_013_…/exp_013_2_additive_rnd_icm/` — additive variant (**dead**, §4.7).
- `exp_014_…/leaky_rnd_lineage_recap.html` — consolidated verdict & all diagnoses.
- `exp_014_…/exp_014_5_rnd_forget/` — cleanest controlled leak proof.
- `exp_015_kaggle_submission/leaky_rnd_agent.py` — an already-simplified online variant; useful reference for the rewrite.
