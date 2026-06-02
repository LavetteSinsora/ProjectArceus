# exp_013 — System Card: Optimistic Curiosity Control (OCC)

*The "ours" method for sparse-reward exploration. Goal metric: **env-steps to the
first positive extrinsic reward**. No reward to exploit during learning — the
entire problem is shaping an intrinsic signal that reaches a first success fast.*

---

## 0. TL;DR

Treat the whole run as a depth-limited UCT: **each episode = one MCTS iteration**,
the **intrinsic value `V_int` = the backed-up node value**, and **curiosity =
the optimism bonus**. Curiosity is **RND in a frozen controllable-feature space**
(kills the pixel "novelty leak" and gives count resolution), made **leaky** so it
measures a *visitation rate* and never permanently saturates (cures the
exploration stall). Novelty enters through the **reward channel** (one scale, no
hand-mixed bonuses) and is turned into a directional signal by a **dual-stream
PPO critic**. Two action-selection variants: **(B0)** a stochastic PPO actor
(robust, lagged optimism) and **(B1)** an actor-free **1-step lookahead softmax**
(immediate, decision-time optimism, but bets on a forward model). Stochastic
selection makes a deterministic env diffuse (random walk) instead of loop.

---

## 1. Motivation — why build anything new?

In these games the reward is terminal-only and a uniform-random policy reaches it
in **finite time in only 4 of 12 game×level cells** (tu93 L2 ~2k, ls20 L1 ~50k,
tu93 L1 ~500k, re86 L1 ~2M); **everything else is E = ∞** (see
`baseline_random_policy/SUMMARY.md`). So undirected exploration is hopeless on
most cells — we need *directed* exploration, and we need it to keep working long
enough to stumble onto the first reward.

The two standard intrinsic methods each fail in a way we measured:

| method | failure (measured) |
|---|---|
| **ICM** | curiosity *collapses* — the untrained-model startup transient poisons the normaliser, crushing the bonus to ~1e-4 for ~43k steps; even fixed, high seed-variance (some seeds worse than random). |
| **RND** | curiosity *saturates and never revives* (one-way ratchet → flat by ~80k on L1 → stall); on raw pixels it has **no count resolution** (1 visit ≈ 2000 ≈ floor) and **leaks to unseen states** (99.9%), because near-identical maze frames make the random target smooth and the predictor interpolates novelty everywhere. |
| **both** | the bonus is delivered at *reward time* and propagated by learning → **lagged**; with a deterministic policy it can also **loop** within an episode. |

OCC is the smallest design that fixes all four — **collapse, saturation,
resolution/leak, and lag** — while staying generalizable (no exact-frame hashing)
and honest about the one thing we can't afford (a trustworthy multi-step model).

The unifying lens is **optimism in the face of uncertainty / UCT**: act on
`Q + bonus`, where `Q` is learned value and `bonus` is a *shrinking confidence
width*. Curiosity is that width; the value function is the backup; the episode is
the iteration.

---

## 2. High-level run-through (read this first on any revisit)

```
for each episode (= one MCTS iteration):
    reset to the start state (= root)
    for each step until terminal / step-cap:
        z      = φ_frozen(s)                      # controllable features (stationary ruler)
        nov(s) = ½‖predictor(z) − target(z)‖²     # RND novelty, LEAKY (rate, not count)
        r_int  = normalize(nov)                   # one scale; warm-up zeros the burn-in
        choose action a:
            (B0 PPO)        a ~ π_θ(·|s)                              # stochastic actor
            (B1 lookahead)  a ~ softmax_a[ nov(ŝ'_a) + γ·V_int(ŝ'_a) ]/τ   # 1-step model
        step env -> s', extrinsic r_ext (≈0 until first success)
    BACKUP (one PPO update over the batch of episodes):
        V_int <- TD/GAE on r_int   (non-episodic)   # the curiosity backup
        V_ext <- TD/GAE on r_ext   (episodic)
        (B0) improve π by advantage A = c_e·A_ext + c_i·A_int
        predictor <- distill toward target; then θ ← (1-μ)θ + μ·θ_init   # the LEAK (decoupled shrink-to-init)
        (B1) forward model f <- predict next φ
    record first env-step where r_ext > 0  ->  STOP on first reward (or cap)
```

Everything below is the detail behind those ~12 lines.

---

## 3. Conceptual foundation

- **Episode = MCTS iteration.** Root = start state; the trajectory = selection +
  rollout; the post-episode PPO update = the backup; the persistent `V_int` + the
  (leaky) novelty statistics = the accumulated tree statistics over the state
  *graph* (states shared across paths — not a history tree).
- **`Q` is never a stored network.** PPO's advantage `A(s,a) = Q(s,a) − V(s)`
  already encodes the action-value comparison; the policy is improved toward
  high-advantage actions. In B1, a per-action `Q` is *constructed* on the fly from
  `V_int` + a 1-step model. So "act on Q" = "improve/derive π from `A_int`/the
  lookahead", not "argmax a Q-table".
- **Curiosity = the optimism bonus**, used in its native (count-free) form: RND
  error already shrinks with visitation like a `1/√N` confidence width; we do
  **not** reverse-engineer an integer count.
- **What we deliberately do NOT import from MCTS** (each is a place RL is better
  here): history-indexed counts (blow up), Monte-Carlo backups (use TD/GAE),
  argmax selection (loops on a graph → use stochastic), and a free simulation
  budget (we have no trustworthy model → one real episode per iteration).

---

## 4. Architecture (detailed)

### 4.1 Observation & action space
- Frame: 64×64 palette indices → one-hot `(16,64,64)`. UI rows masked by the game
  wrapper. Actions: 4 (ls20/tu93) or 5 (re86/g50t), auto-detected per game.

### 4.2 Encoders (three distinct roles — keep them straight)
| role | net | trained by | frozen? |
|---|---|---|---|
| **policy/critic encoder** | CNN (exp_010 `CNNEncoder`) | PPO (B0) / critic loss (B1) | no |
| **φ encoder (IDM)** | CNN | inverse dynamics (predict `a` from `φ(s),φ(s')`) | **frozen after warm-up** |
| **RND target / predictor** | CNN or MLP-on-φ | target: never; predictor: distillation | target frozen; predictor leaky |

### 4.3 Curiosity / novelty — RND on frozen φ
- Input to RND = **`z = φ_frozen(s')`**, the controllable features (not raw pixels).
  Rationale (measured): φ *separates* states that pixels conflate (the IDM must, to
  predict actions), which removes the pixel-smoothness leak and restores count
  resolution. φ is **frozen** once `inverse_acc` saturates so the ruler is
  stationary (φ never stabilizes on its own — drift ratio ~0.22–0.25 forever).
- `target T`: fixed random net. `predictor P`: trainable, distills `T`.
- raw novelty `nov(s) = ½·mean_j (P(z) − T(z))²_j`.
- Done-step transitions (reset frames) are zeroed.

### 4.4 The LEAK (forgetting) — **recommended: L2-to-init on the predictor**
RND has **no forgetting by default** (predictor error is a one-way ratchet → it
saturates and the agent stalls). To make novelty *revive* we must raise error back
up. Options, and the verdict:

| option | what | verdict |
|---|---|---|
| reset / drift the **target** | change `T` periodically/slowly | ❌ forgets **globally & uniformly** (frontier included), and destroys the stationary ruler RND depends on; drifting `T` = self-inflicted φ-drift |
| **shrink-to-init on the predictor** | after each step: `θ_P ← (1−μ)θ_P + μ·θ_P^init` (decoupled L2-to-init) | ✅ **recommended** (implemented) |
| periodic soft-reset of predictor | `θ_P ← (1−α)θ_P + α·θ_P^init` every K updates | ✓ simpler, but discontinuous → novelty jumps every K steps can jitter `V_int` |

**Why L2-to-init is best.** It creates a continuous **tug-of-war**: the
distillation gradient pulls error *down* on states currently being visited; the
`λ` pull drags weights back toward init *everywhere*. States you keep revisiting
win the tug-of-war (stay low-error = "still counted"); states you abandon lose it
and drift back up (= "novel again"). Net effect: **RND stops measuring cumulative
count and starts measuring *recent visitation rate*** — which never permanently
saturates, so exploration never stalls. It's one differentiable knob (`λ`),
integrates into the existing optimizer, and its forgetting is *recency-driven*
(not representation-churn-driven like φ-drift) and reproducible. In frozen-φ space
the forgetting is also reasonably *local* (per region) rather than global.
- Knob: `λ` (forget rate). `λ=0` recovers vanilla RND. Larger `λ` → faster
  forgetting → more sustained but noisier exploration. To be swept.

### 4.5 Reward → value pipeline (one scale)
- `r_int = normalize(nov)` via a running std of intrinsic returns
  (`RewardForwardFilter` + EMA or cumulative RMS), **no mean-centering** (bonus ≥ 0),
  with a short **warm-up** that delivers zero intrinsic reward while the predictor
  burns in (prevents the ICM-style transient from poisoning the normaliser).
- Novelty enters **only through the reward** — we do *not* hand-add heterogeneous
  bonuses (Q + RND + kNN live in different units; adding them needs fragile weight
  tuning). TD learns the scale.

### 4.6 Value critic — dual-stream
- Two heads on the shared policy/critic encoder: `V_ext` (episodic, `γ_ext=0.999`)
  and `V_int` (**non-episodic**, `γ_int=0.99`). Per-stream GAE.
- `V_int(s) ≈ Q_int(s,a) discounted future curiosity` — this **is** the backup.
- Combined advantage drives the policy (B0): `A = c_e·A_ext + c_i·A_int`. Before
  the first reward `A_ext ≈ 0`, so curiosity is the entire learning signal.

### 4.7 Action selection — two variants (A/B these)
- **B0 — PPO actor (robust default).** Stochastic `π_θ(a|s)`, clipped-surrogate
  PPO on the combined advantage. No model. Decision-time optimism is **lagged**
  (a novel state raises `A_int` → *next* update boosts the actions to it);
  entropy keeps it from looping. *This is the harness's current method and it
  solved ls20 L1.*
- **B1 — actor-free 1-step lookahead softmax (MCTS-faithful candidate).** Keep the
  critic `V_int`, RND, and a **latent forward model** `f(φ(s),a) → φ̂(s')`. Act by
  `Q(s,a) = nov(ŝ'_a) + γ·V_int(ŝ'_a)`, `π(a|s) = softmax(Q/τ)`, sample.
  - Gives **immediate, decision-time optimism** (cuts the lag/loop).
  - **Softmax, not argmax** → stays loop-free; `τ` is the explore knob.
  - **No actor network and no Q-network** — `Q` is *synthesized* from `V_int` + the
    model; `V_int` is still trained by TD on the generated trajectories.
  - Bets on the model → used **1-step, ranking-only** (the real `s'` is used for
    learning, so model error corrupts only the *choice*, not the *update*).
  - We do **not** distill a policy from this lookahead: a learned policy is worth it
    only if it acts (B0) or amortizes a *deep* search (AlphaZero). A policy distilled
    from a cheap 1-step/≤5-action lookahead that then acts by lookahead is pure
    redundancy.

### 4.8 Anti-loop
- A deterministic *environment* does not force looping; a deterministic *policy*
  does. Both variants select **stochastically** (B0 entropy, B1 softmax `τ`), so a
  would-be loop becomes a diffusing random walk biased outward by `V_int`. No
  within-episode counter is used at baseline; a **recurrent (GRU) policy** is the
  first count-free upgrade *if* measurement shows looping wastes budget.

### 4.9 exp_013_1 specialization (AS IMPLEMENTED — the "RND+ICM" run)
The general design above is dual-stream with two action-selection variants. The shipped
`exp_013_1b_leaky_rnd_on_icm_phi/` method is the **B0 (PPO actor)** variant with deliberate simplifications
(decided with the user; supersede the literal general-design wording where they differ):
- **Single value head, intrinsic-only — NO `V_ext` / dual-stream.** There is no extrinsic
  reward to optimise (the first +1 ENDS the run), so the env's +1 is used ONLY as the
  stop signal / metric, never fed to GAE. The `c_e,c_i` advantage mix is dropped.
- **Intrinsic GAE is NON-episodic** (`intrinsic_episodic=False`): novelty value bootstraps
  across death/reset (canonical RND), so the agent isn't deterred from life-costing deep
  exploration. (`True` = PPO-style episodic, for A/B.)
- **Leak = decoupled shrink-to-init** (§4.4): `θ_P ← (1−μ)θ_P + μ·θ_P^init` after each
  predictor step, μ=`leak`=0.01 — decoupled from Adam (not an additive loss term). Not swept.
- **φ-freeze:** adaptive (`inverse_acc ≥ 0.90` for 3 updates) + max-updates fallback (100),
  with a WARNING logged if the fallback fires with low `inverse_acc` (poor φ → §9).
- Reviewed by two critics (bug + faithfulness): no correctness bug; wiring verified.

---

## 5. Training loop / data flow (one update)

1. **Collect** a rollout of `T` steps × `N` envs (terminal-only `r_ext`).
2. **Novelty:** `z = φ_frozen(next_obs)` → `nov = ½‖P(z)−T(z)‖²`; zero done-steps.
3. **Normalize:** warm-up → 0; else `r_int = nov / running_std(returns)` (no center).
4. **GAE:** `A_ext, ret_ext` (episodic); `A_int, ret_int` (non-episodic).
5. **Policy/critic update (PPO):** clipped surrogate on `A = c_e A_ext + c_i A_int`;
   value-clip both heads. (B1: update only the critic + softmax has no policy loss.)
6. **Predictor update (the leak):** distill `‖P(z)−sg(T(z))‖²`, then shrink-to-init
   `θ_P ← (1−μ)θ_P + μ·θ_P^init` (decoupled, μ=`leak`) — NOT an additive loss term.
7. **(B1 only) forward-model update:** `‖f(φ(s),a) − sg(φ(s'))‖²`.
8. **Record** the first env-step with `r_ext > 0` → **stop-on-first-reward** (or cap).

---

## 6. Hyperparameters & knobs (the ones that matter)
| knob | role | default / note |
|---|---|---|
| `λ` (leak) | forget rate of the predictor | **the key new knob — sweep it**; 0 = vanilla RND |
| `int_norm_mode` / warm-up | normaliser stability | ema+warmup for moving signals; cumulative+0 for vanilla RND |
| `γ_int` / `γ_ext` | intrinsic / extrinsic horizon | 0.99 / 0.999 |
| `c_e`, `c_i` | advantage mix | 2.0 / 1.0 |
| `τ` (B1) | softmax temperature = explore rate | swept; replaces entropy coef |
| φ-freeze step | when to freeze the IDM encoder | when `inverse_acc` saturates (~early) |
| `max_env_steps` | censoring cap | per-cell, scaled to the random baseline |

---

## 7. Metric, stopping, baselines
- **Metric:** total env-steps (summed over actors) to the first `r_ext > 0`.
- **Stop:** the instant of first reward; right-censored at `max_env_steps`.
- **Variance:** ≥8 seeds per (method×game×level), report mean ± spread.
- **Baselines:** random-policy `E[steps]` per cell (`baseline_random_policy/`);
  only 4 cells finite, the rest "beat-the-impossible".

---

## 8. Design decisions & rejected alternatives (with reasons)
- **RND on frozen-φ, not raw pixels** — pixels gave no count resolution + 99.9%
  leak (measured); φ separates states; freeze it because φ never self-stabilizes.
- **Leak via predictor L2-to-init, not target reset/drift** — local, recency-based,
  preserves the stationary ruler (§4.4).
- **Novelty through the reward, not a decision-time additive bonus** — avoids
  mixing incommensurate scales (Q vs RND vs kNN).
- **No kNN / no exact-frame hashing for an episodic count (yet)** — adds a second
  scale and complexity; rely on stochasticity first, add a recurrent policy only if
  measurement demands.
- **No online-SGD-RND as an episodic counter** — measured: SGD-error saturates
  after ~1 visit and leaks to neighbours; it cannot count.
- **No deep MCTS / multi-step model planning** — the forward model is not exact
  enough (exp_006: ~99% pixel / 0% exact-frame); only 1-step ranking is safe.
- **No distilled-but-unused policy** (§4.7).

---

## 9. Open risks / what to measure
- Does `leak`>0 actually beat `leak`=0 on the ∞ cells (where RND stalls)?
- Does frozen-φ RND keep resolution during real training, or re-leak? (re-run the
  count probe in φ-space.)
- **Moving-φ-before-freeze window:** between the normaliser warm-up and φ-freeze,
  novelty is computed on a still-drifting φ (the stationary-ruler guarantee §4.3 does
  not hold there). Benign when freeze is early; watch the `φ FROZEN` log line — if it
  fires via the **FALLBACK** with low `inverse_acc` (the new WARNING), φ never became
  controllable and the whole run's RND ruler is suspect. (Mitigation if it bites:
  reset the predictor + normaliser at freeze, or raise `phi_freeze_max_updates`.)
- B1's dependence on `f`: does 1-step lookahead help or does model error hurt
  ranking? A/B vs B0 on the 4 finite cells first.
- Within-episode looping: measure revisit-rate / unique-states-per-episode; only
  add a recurrent policy if it's a real cost.

---

## 10. File map
- `shared/{config,intrinsic,ppo,trainer}.py`, `run.py` — the harness (B0 today).
- `baseline_random_policy/` — random `E[steps]` per cell + `SUMMARY.md`.
- `probes/` — φ-drift (`phi_drift_*`) and RND-count (`rnd_count_*`) findings.
- `design_narrative.html` — the one-page story behind this card.
- `colab_calibration.ipynb` — runs the multi-seed sweep on cloud GPU.
- *(to add)* `intrinsic.py`: `leak λ` on the predictor; a `LookaheadController` for B1.
