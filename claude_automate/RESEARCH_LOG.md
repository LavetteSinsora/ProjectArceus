# Research Log — claude_automate

A running, append-only log: proposals before code, results after runs,
findings that change the design. Newest entries at the bottom of each section.

---

## Environment facts (verified by `probe_env.py`, 2026-05-17)

- Frame: `(64, 64)` uint8, colour indices 0–15 (only 0–12 seen in L1).
- 4 actions (ACTION1–4 = directional moves). Deterministic environment in
  `OperationMode.OFFLINE` — identical layout every `reset()`.
- Episode length ~129–133 steps under random policy; **0/25 random completions.**
- UI rows 61–62 change every step (step counter) — must be masked in any
  frame-difference computation.
- Terminal: WIN / GAME_OVER, or `levels_completed >= 1`.

### LS20 L1 mechanics (decoded from `environment_files/ls20/.../ls20.py`)

- Player starts at sprite-grid (34, 45); goal cell at (34, 10).
- To clear the level the player must stand on the goal cell **with matching
  shape, colour, and rotation**. L1: start rotation 270°, goal rotation 0°,
  so the player must hit a rotation-modifier tile (at ~(19, 30)) once.
- 3 lives; 42 "energy" per life, −1 per action ⇒ ~126 action budget.
- This is a **sparse-reward hard-exploration** problem: navigate a maze to a
  modifier tile, then to the goal. We do NOT encode any of this in the reward.

---

## Proposal 001 — PPO + SimHash count-based exploration

### Problem framing
Sparse terminal reward (`level_completed`) with a ~130-step horizon. Random
policy never completes. Credit must propagate ~25–40 steps back from the goal.

### Why this design
- **PPO** (clipped on-policy actor-critic + GAE): generalizable, stable
  backbone. No replay-buffer staleness; works with a single SDK env instance.
- **SimHash count-based intrinsic reward**: the generalizable exploration
  driver. Hash each (UI-masked) frame to a k-bit code, keep a global visit
  count, add bonus `W_novel / sqrt(count)`. This densifies the reward so the
  agent systematically covers the maze until it finds the goal — without any
  hint of *where* the goal is.

### Reward (all terms generalizable — see README table)
```
r_t = -W_step
      - W_stuck          if masked frame unchanged (wasted action, any game)
      + W_novel/sqrt(N)  count-based novelty (any game)
      + W_complete       once, on the level-completion transition
```

### Architecture
- Input: frame → one-hot 16 channels → `(16, 64, 64)`.
- Encoder CNN: Conv(16→32,k8,s4) → Conv(32→64,k4,s2) → Conv(64→64,k3,s1)
  → flatten → FC(→512), ReLU throughout.
- Heads: policy logits (4) + value (1).
- PPO: GAE(γ=0.99, λ=0.95), clip 0.2, entropy bonus, 4 epochs/rollout,
  rollout = several full episodes.

### Success criterion
≥ 30% level-completion rate over a 20-episode greedy/stochastic eval.
Stretch: ≥ 80% (deterministic env ⇒ a converged policy should be ~100%).

### Tests written before implementation
See `tests/` — cover frame preprocessing, SimHash determinism + count decay,
reward composition (each term in isolation), network output shapes, GAE
correctness, and a tiny end-to-end PPO smoke step.

### Results — exp 001 (run_20260517_234844, SOLVED ✅)

PPO + `ExactFrameCounter` exploration, default `Config`. Single run, seed 0,
device MPS. 8 episodes/rollout, ~1k env steps/update.

| Milestone | Update | Env steps | Wall clock |
|---|---|---|---|
| First completion (train) | 16 | ~16.7k | ~75 s |
| Takeoff to 100% train completion | 21 | ~21k | ~100 s |
| First 100% **greedy eval** (20 ep) | 40 | ~27k | ~150 s |
| Converged 13-step solution | ~62 | ~30k | ~170 s |

**Final evaluation on `best.pt`:**
- Greedy: **50/50 completions (100%)**, every episode exactly 13 steps.
- Stochastic: **50/50 completions (100%)**, 13–14 steps.
- Baseline (random policy): 0/25. Converged policy is a clean, near-optimal
  13-action solve and is perfectly stable for 100+ updates afterward.

### What worked / findings

1. **Exact-match counting beats SimHash on discrete ARC frames.** The first
   build used random-projection SimHash and collapsed the whole maze into
   ~12 hash buckets — a tiny sprite moving barely perturbs the projection, so
   novelty was nearly dead. Switching to `ExactFrameCounter` (hash the exact
   UI-masked frame) gave ~133 distinct buckets and a sharp novelty gradient.
   This is *more* generalizable, not less: it relies only on observations
   being discrete and the environment deterministic — true for every ARC
   game — and never on LS20 geometry.
2. **Count-based novelty alone cracked the sparse reward.** No goal-direction
   shaping was needed. Coverage exploration reaches the rotation-modifier tile
   (which makes every subsequent frame novel, reinforcing further movement)
   and then the goal cell; the `+w_complete` terminal bonus then locks the
   route in via PPO. The agent was never told where the goal is.
3. **Convergence is fast and stable** (~27k env steps, <3 min wall clock),
   with no entropy collapse or value blow-up.

### Generalizability check (no level-specific shaping)

Every reward term and the counter are game-agnostic: completion flag, step
penalty, "frame unchanged" penalty, count-based novelty. `_MASKED_ROWS` is
read from the env wrapper, not hard-coded here. The same `train.py` should run
on tu93 / re86 / g50t by changing only `Config.game_id` (untested — left as
future work).

### Open / future work

- Validate the framework transfers to tu93 / re86 / g50t and to LS20 levels 2+.
- Per-episode (not just global) novelty counts could speed early exploration.
- Greedy eval is deterministic here because the env is; for stochastic games
  an ensemble or temperature schedule would matter.

---

## exp 002 — transfer test to LS20 Level 2

Goal: test whether the framework generalizes to a new level, in two senses —
(A) the *recipe* applied fresh (random init), (B) *continuing* from the L1
checkpoint (fine-tuning / curriculum). `LevelStartWrapper` (in `env_api.py`)
starts each episode on a chosen level via the engine `set_level` API + an
`ACTION5` no-op render; nothing outside `claude_automate/` is touched.

L2 facts (probe): random policy 0/20 completions; fixed 66-step episodes
(shorter horizon than L1; the agent can extend episodes by collecting energy
pickups).

### exp 002.A v1 — recipe fresh, default Config — FAILED ❌

Ran v1 recipe (global count-based novelty only) on L2. Result after 110
updates / 87k env steps (>3× the steps L1 needed): **0% completion**, entropy
collapsed 1.38 → 0.39, distinct states plateaued ~247, episode length grew
66 → ~140.

**Diagnosis.** The agent fell into a *survive-and-revisit* local optimum:
it learned to stay alive (collect energy pickups → longer episodes) and tour
its already-discovered region. Once global novelty counts saturate, the only
remaining reward gradient is the tiny step penalty — nothing pulls the policy
toward the still-undiscovered goal. With no exploration gradient, PPO's
entropy bonus is too weak to prevent collapse, and the policy froze before
ever reaching the goal. (On L1 the goal was shallow enough to be found at
~16k steps *before* entropy collapsed — L2's goal is deeper.)

### Proposal 002 — v2 recipe: episodic novelty + higher entropy

Two generalizable changes (still no level-specific shaping):

1. **Episodic novelty.** Add a second count-based bonus from a counter that
   **resets every episode**. Global novelty answers "have I *ever* seen this?"
   and dries up; episodic novelty answers "have I seen this *this episode*?"
   and never dries up — every episode is rewarded afresh for covering ground.
   This converts exploration from a one-off into a standing gradient, so the
   policy keeps probing instead of freezing. (Episodic memory is the same idea
   behind NGU / Agent57 — fully game-agnostic.)
   Reward gains a term `+ w_novel_episodic / sqrt(episodic_count)`.
2. **Higher entropy coefficient** (0.02 → 0.04) to further resist premature
   determinism.

Scale check: a survival episode (~150 distinct states) earns ≈150·0.05 ≈ 7.5
episodic novelty; a completing episode earns `w_complete` = 20. Completion
stays the dominant outcome. `w_complete` raised 10 → 20 for margin.

### Results — exp 002 v2 (episodic novelty) — FAILED ❌

Ran both A (fresh) and B (resume from L1 ckpt) on L2 with the v2 recipe.
After ~85k env steps each: **0% completion**. Entropy *did* stay healthy
(~1.0, no collapse — the episodic term worked for that). But distinct global
states **plateaued at ~80** (v1 had reached 247!).

**Diagnosis — episodic novelty caused complacency.** Because episodic counts
reset each episode, *re-touring* an already-known ~80-state region pays full
episodic novelty every episode. The agent settled into a fixed safe tour and
lost the pressure to expand globally. v2 fixed entropy collapse but explored
*less* than v1.

**Root cause (both v1 and v2) — detachment.** This is the well-known failure
of count-based exploration with a neural policy: once the easily reachable
region is covered, the policy has no reward gradient that *leads to* the
harder frontier, because frontier states have never been visited and so
appear in no trajectory PPO can reinforce. L2 needs the rotation modifier hit
**3×** then the goal reached — a deep, precise sequence that undirected
exploration almost never executes in one episode. PPO+novelty cracked L1
(1-hit, found at 16k steps before any freeze) but cannot crack L2.

### Proposal 003 — Go-Explore structured exploration + policy distillation

Switch the exploration front-end to **Go-Explore** (Ecoffet et al. 2021),
which is purpose-built for detachment and exploits our **deterministic**
environment. Fully generalizable — cells are just hashed observations; no
LS20 knowledge enters.

1. **Search.** Maintain an archive: distinct cell (UI-masked frame hash) →
   shortest action trajectory from reset that reaches it. Repeatedly: pick an
   archived cell (preferring less-chosen ones), **deterministically replay its
   trajectory to return there**, then take a burst of random actions; archive
   every new/shorter-reached cell. Returning-then-exploring means the frontier
   is always expanded from the frontier — detachment cannot occur. Stop when a
   `level_completed` transition is hit ⇒ a full completing trajectory.
2. **Distill.** Behavior-clone the completing trajectory into the ActorCritic
   policy (supervised cross-entropy on its (frame, action) pairs). On a
   deterministic env the greedy policy then reproduces the solve.

Still no level-specific shaping: completion flag + generic cell hashing only.
The PPO+novelty path is kept for shallow puzzles (it solved L1); Go-Explore is
the framework's tool for deep-exploration levels.

### Results — exp 003 (Go-Explore on L2) — SOLVED ✅

`solve.py` on LS20 Level 2 (`--level 1`):

| Stage | Result |
|---|---|
| Go-Explore search | completing trajectory found: **60 actions**, 332k env steps, 13 288 iterations, archive 552 cells, ~5 min |
| Distillation (1st try, stateless policy) | stuck at **90%** action-match — FAILED |
| Distillation (recurrent policy) | **100%** action-match by epoch ~70 |
| Distilled-policy eval | **30/30 completions (100%)**, every episode exactly 60 actions |

**The stateless-policy bug + fix.** The first distillation plateaued hard at
90% (54/60 actions). Diagnosis: the 60-frame solution has only **49 distinct
*player-states*** — the other 11 frames differ only in the step-counter UI
rows, which the downsampling CNN cannot read. The Go-Explore trajectory
**revisits player-states with different actions** (unavoidable — hitting the
rotation modifier 3× forces repeated corridor traversals). A stateless policy
`f(obs) -> action` provably cannot represent that.

Fix: a **`RecurrentActorCritic`** (CNN + GRU). The GRU hidden state lets the
policy act differently on identical observations depending on where it is in
the plan. Sequence behavior-cloning over the trajectory then fits 100%, and
because the env is deterministic the eval rollout reproduces the same
observation sequence → same hidden states → same actions → a 100% solve.
(Unit test `test_recurrent_distill_fits_revisited_states` pins this capability.)

### Transfer-test verdict (the original exp 002 question)

- **A — recipe applied fresh.** PPO+novelty (v1 and v2) does NOT generalize to
  L2 (detachment on a deep puzzle). The **Go-Explore + recurrent-distill**
  path DOES: L2 solved 100%. So the *framework* generalizes — but only once it
  carries a structured-exploration tool; a single flat recipe does not cover
  both shallow (L1) and deep (L2) levels.
- **B — continue from the L1 checkpoint.** Tested with PPO v2 (`--resume`):
  also 0% at ~85k steps — warm-starting the weights did not rescue the
  detaching recipe. Under Go-Explore the question is moot: the search phase
  uses no policy at all, so there is nothing to warm-start; transfer would
  only mean reusing the L1-distilled GRU as init, which is unnecessary (BC
  from scratch fits in ~70 epochs).

**Takeaway.** "Transfer" that works here is *method* transfer, not *weight*
transfer: the same generalizable framework (`solve.py`: Go-Explore search →
recurrent BC) solves a level it was never tuned for, with only `--level`
changed. No level-specific reward or geometry is used anywhere.

## Framework summary (final)

Two generalizable solving paths, both level-agnostic:
1. **PPO + count-based novelty** (`train.py`) — on-policy RL; solves shallow
   sparse-reward levels (LS20 L1: 100%).
2. **Go-Explore + recurrent distillation** (`solve.py`) — structured
   exploration that defeats detachment on deep puzzles, then behavior-clones
   the discovered solution into a recurrent policy. Recommended default for
   hard-exploration levels. Verified on **both** levels with only `--level`
   changed:
   - LS20 **L1**: solution found in 8.4k env steps / 7 s, eval 30/30 (100%).
   - LS20 **L2**: solution found in 332k env steps / 5 min, eval 30/30 (100%).

   (Go-Explore finds *a* completing trajectory, not the shortest — its L1
   solve is 33 actions vs PPO's optimized 13. PPO's step penalty optimizes
   length; Go-Explore optimizes only for *finding* a solution.)

Final status: **LS20 Level 1 and Level 2 both solved at 100%.**

---

## exp 004 — cross-game transfer to TU93 Level 1

Test: does the framework transfer to a *different game*, not just a different
level? TU93 is a graph-maze navigation game with moving obstacles — unrelated
to LS20's rotation-matching puzzle.

**Zero code changes.** TU93 was already registered in the shared
`env_wrapper.py` (`Tu93Env`, 4 actions, UI row 63). The only thing passed was
`solve.py --game-id tu93-0768757b --level 0`.

Probe: TU93 L1 random policy 0/20 completions, fixed 50-step episodes.

### Result — SOLVED ✅

| Stage | Result |
|---|---|
| Go-Explore search | 43-action solution, **1 941 env steps, 0.9 s**, 96 iterations, archive 61 cells |
| Recurrent distillation | 100% action-match by epoch ~60 |
| Distilled-policy eval | **30/30 completions (100%)**, every episode 43 actions |

TU93 L1 is a shallow puzzle — Go-Explore found the solve almost instantly
(<1 s). The framework transferred across games with no modification: same
`solve.py`, same Go-Explore + recurrent-distillation, same generic cell
hashing and `level_completed` signal.

**Confirmed solved at 100%: LS20 L1, LS20 L2, TU93 L1.**

---

## exp 005 — LS20 Levels 3 & 4

Same `solve.py` Go-Explore path, only `--level` changed (L3 = `--level 2`,
L4 = `--level 3`). Experiment directories are now auto-named
`solve_<game>_L<n>_<timestamp>` (n is the human 1-indexed level).

L3 puzzle: rotation 0°→180° (2 modifier hits) **plus** a colour change
(12→9). L4 puzzle: rotation already matches, but needs a **shape** change
(4→5) **and** a **colour** change (14→9) — two different modifiers, on the
long-horizon ~130-step budget. Both random baselines: 0%.

### Results — both SOLVED ✅

| Level | Go-Explore search | Solution | Distill | Eval |
|---|---|---|---|---|
| LS20 **L3** | 70.8k env steps, 74 s, 542 cells | 60 actions | 100% @ ep ~44 | **30/30 (100%)** |
| LS20 **L4** | 321.6k env steps, 11.5 min, 2 827 cells | 100 actions | 100% @ ep ~86 | **30/30 (100%)** |

L4 was the deepest level attempted — a 2 827-cell archive and a 100-action
solution — and still solved with no code or hyperparameter changes. Search
cost scales with puzzle depth (cells × horizon) exactly as expected; the
recurrent distillation fits every solution in <150 epochs regardless.

## Scoreboard (all 100%)

| Game/level | Method | Solution length |
|---|---|---|
| LS20 L1 | PPO+novelty / Go-Explore | 13 / 33 actions |
| LS20 L2 | Go-Explore + recurrent BC | 60 actions |
| LS20 L3 | Go-Explore + recurrent BC | 60 actions |
| LS20 L4 | Go-Explore + recurrent BC | 100 actions |
| TU93 L1 | Go-Explore + recurrent BC | 43 actions |

Five levels across two distinct games, one unchanged framework, zero
level-specific reward shaping.

---

## Proposal 006 — a transferable shared artifact: world model + planning

### Honest critique of exps 001–005
`solve.py` = **per-level search + per-level memorization**. Go-Explore brute-
forces a completing trajectory (~300k real env steps); the recurrent net just
memorizes that one trajectory. The distilled policy has **zero transfer** — the
L3 policy is useless on L4. Every level pays full search cost. Nothing the
framework learns on one level makes the next one cheaper.

### What is actually shared across LS20 levels
Not the solution — the **dynamics**: the avatar moves one cell per action,
walls block, modifier tiles transform the avatar (rotation / colour / shape),
energy drains, reaching the matching goal completes the level. Crucially the
maze layout, which differs per level, is **part of the observation** — so a
dynamics model that conditions on the current frame can be layout-agnostic.

### The shared artifact: a frame-level world model
Train one model `g(frame, action) → (next_frame, terminal, completed)` on
transitions pooled from already-solved levels (LS20 L1–L4). Because the layout
is an input, the learned rules ("the avatar is blocked by wall pixels", "this
tile recolours the avatar") should transfer to an unseen LS20 level.

### How it yields few-shot solving
On a held-out level, run Go-Explore **inside the world model** — imagined
rollouts cost zero real environment interaction. The search produces candidate
solution trajectories; each is then **verified once in the real env**. Real
env steps are spent only on (a) a seed transition probe and (b) verifying
candidates — target ≪ the ~300k of a from-scratch search. If a candidate fails
verification (model error), its real transitions correct the model and we
retry. "Few attempts" = few real episodes, not zero search; the search is
amortized into the (free) model.

### Make-or-break checkpoint
Before building the planner: train the model on L1–L4 and **measure next-frame
prediction accuracy on a held-out level**. If transfer accuracy is high, model-
based planning is viable; if not, report honestly.

### Tests
World-model forward/predict shapes; `ModelEnv` exposes the env interface and
is deterministic (replayable — required by Go-Explore); a trained model
overfits a tiny transition set.

### Results — exp 006 — PARTIAL / approach does NOT pay off ❌

World model trained on LS20 L1–L3 (34 912 transitions), evaluated on held-out
levels:

| Held-out | pixel acc | exact-frame acc | terminal acc | `completed` recall |
|---|---|---|---|---|
| L4 | 99.2% | **0.0%** | 99.7% | **0 / 20** |
| L5 | 99.2% | **0.0%** | 99.8% | (no completions in eval data) |
| L6 | 98.6% | **0.0%** | 100% | — |
| L7 (fog) | **39.2%** | 0.0% | 100% | — |

### Honest verdict

**The model-based-planning route does not work here.** Three concrete failures:

1. **`exact_frame_acc = 0.0%`.** 99% pixel accuracy = ~34 wrong pixels per
   frame, and those pixels are exactly the moving content. The model never
   predicts a fully-correct next frame. `solve_wm.py` does Go-Explore with
   *exact* frame hashing and *exact* imagined replay — with 0% exact frames,
   imagined cell hashes are wrong and long imagined rollouts diverge
   immediately. Model-based Go-Explore as designed is not viable with this
   model.
2. **`completed` head is dead** (0/20 on L4). Only ~3 distinct completion
   transitions exist in the L1–L3 training data — far too few to learn from.
   This was predicted in Proposal 006's "make-or-break" note and confirmed.
3. **L7 (fog) does not transfer** — visually out-of-distribution. Transfer is
   within-distribution only.

### What is genuinely real from exp 006

The model *did* learn **transferable aggregate dynamics**: trained only on
L1–L3, it predicts unseen L4/L5/L6 transitions at ~99% pixel accuracy and
~99.7% terminal accuracy. The hypothesis "dynamics are shared, layout is in
the observation" holds **as a representation result**. But ~99% pixel accuracy
is nowhere near "exact enough to plan with", so it does **not** enable
few-shot solving.

### Honest retrospective — over-engineering

Exp 006 jumped to the most ambitious answer (world model + planning) without
first checking whether the premise held. For a *deterministic* puzzle game,
exact-replay search (Go-Explore) is cheap (7 s – 11 min/level) and is the
right tool; a learned world model is not accurate enough to replace it
without far more data/compute. Simpler artifacts that were skipped — a shared
encoder, or simply accepting search — would have been more honest starting
points. `solve_wm.py` is kept in the repo but marked **not viable with the
current world model**; it is not run.

### Status of the framework (final, honest)

- **Solved (real):** LS20 L1–L4, TU93 L1 — completing trajectories found and
  reproduced. Caveat: the env is deterministic, so a greedy distilled policy
  produces one fixed episode; "30/30 eval" is that one solution replayed 30×,
  not 30 independent successes. The level is genuinely completed; the eval
  count is cosmetic.
- **PPO + count-based novelty** genuinely trains an RL policy (LS20 L1, 100%,
  stochastic eval shows real length variance — the most "RL" result).
- **Transfer:** *method/code* transfers (one framework, many levels/games);
  *dynamics prediction* transfers (exp 006, ~99% pixel acc). *Weight-level
  few-shot transfer was not achieved* — it would need meta-RL across many
  training levels, beyond this setup's data/compute.
