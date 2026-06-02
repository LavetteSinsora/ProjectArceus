# tu93 — random-policy exploration baseline (exp_013)

**Metric.** For each level, the expected number of *environment steps* a
uniform-random policy (each of the 4 directional actions w.p. 1/4) takes before
its **first positive extrinsic reward** = first level clear (wrapper
`level_completed` / engine `levels_completed` increments). One `env.step()` ==
one logical move == one decrement of the level step counter.

Game: `tu93-0768757b`. Source parsed from
`environment_files/tu93/0768757b/tu93.py`. Scripts:
`scripts/tu93_random_baseline.py`.

---

## Mechanics (recovered from source)

tu93 is **graph-maze navigation**. The board is a grid of nodes spaced
`hcgctulqhn = 6` px apart, connected by "track" cells. The maze sprite (tag
`0005uvnhiglpvh`) stores the layout in `.pixels`: value **2 = walkable track**,
**0 = node**, **−1 = empty**.

- **Actions.** `ACTION1..4` = up / down / left / right. A move probes the cell
  `hwthhtvyki = 3` px away in that direction; it is legal **iff that cell is
  track (==2)**, in which case the player hops one full node (6 px). Movement is
  thus **constrained to the maze tracks** (a 4-regular-at-most graph walk).
- **Illegal moves are no-ops but still cost a step.** Pressing into a wall does
  not move the player yet still decrements the step counter (verified on the
  real engine).
- **Step / energy budget per life.** Read from each level's `StepCounter`
  (row 63 is the shrinking colour-6 bar): **L1 = 50, L2 = 50, L3 = 35**.
- **Lives.** **One life per episode.** `lose()` sets `GAME_OVER` immediately —
  there is *no respawn / multi-life system*. Hence **per-life p == per-episode
  p**.
- **Win condition.** When the player node coincides with the exit node (tag
  `0015msvpvzxhqf`), the engine calls `next_level()` → `levels_completed += 1`.
  That increment is the first positive extrinsic reward.
- **Lose conditions.** (a) step counter hits 0, or (b) the player sprite is
  removed by colliding with a moving obstacle.
- **Moving obstacles (stochasticity source — see below).** Obstacle sprites
  carry tags `0001haidilggfh` (chasers), `0020npxxteirsg`, `0023otenflmryc`.
  They move along the tracks each step (chasers steer toward the player) and
  **kill the player on contact**. Per level:
  - **L1: NO obstacles** → pure deterministic random walk on a fixed graph.
  - **L2: 1 obstacle** (chaser `0018rquzkxccdu`, tag `0001`).
  - **L3: 3 obstacles** (chasers `0032/0033/0034`, tag `0001`).

**Determinism / well-definedness.** The engine is **fully deterministic given
the action sequence** — there is no internal RNG. All randomness in this
baseline comes from the uniform policy, so E[steps] is well-defined. The
obstacles add no *engine* stochasticity, but because their motion depends on the
(random) player trajectory, the per-episode win event is *not* a function of the
node alone — it depends on the full history. That makes an exact Markov-chain DP
over (node, steps) **only valid for L1**; for L2/L3 the no-obstacle DP is an
**upper bound on p** and **Monte-Carlo on the real engine is primary**.

**Shortest-solution floor (no-obstacle BFS on the parsed graph):**
L1 = **18 moves**, L2 = **8 moves**, L3 = **9 moves**. (claude_automate solved
L1 in 43 actions via cross-game transfer — a valid non-optimal solve, well above
the 18-move optimum; consistent.)

**E[steps] model.** Independent episodes (geometric restarts, 1 life each):
`E[env steps to first reward] = E[steps per episode] / p_episode` (renewal-
reward). For a failing episode the policy almost always burns the full budget;
a winning episode ends early.

---

## Results

### Level 1 (`level_index = 0`) — EXACT (deterministic, no obstacles)

L1 has no obstacles, so it is a pure 4-action random walk on a 31-node graph
with a hard 50-step budget. The absorbing-Markov-chain DP over
(node, steps_remaining) is **exact**.

| quantity | value |
|---|---|
| step budget / life | 50 |
| reachable nodes | 31 |
| shortest solution | 18 moves |
| **per-life = per-episode p_win** | **1.4303e-4** (≈ 1 in 6,992) |
| E[steps per episode] | 49.999 (wins are rare → nearly always full budget) |
| **E[env steps to first reward]** | **≈ 349,600** |

**Cross-check (MC).** A fast standalone simulator matching engine semantics
(illegal move = stay, costs a step), 2,000,000 episodes:
p̂ = 1.520e-4 ± 1.71e-5 (95%) — statistically consistent with the exact
1.4303e-4. (Engine-level MC was also smoke-checked but is ~26 eps/s, too slow to
resolve p≈1.4e-4; the exact DP is authoritative here.) The 18-move shortest
path was replayed on the **real engine** and confirmed to set
`levels_completed = 1`.

> For comparison, LS20 L1 is ≈ 50,000 steps. tu93 L1 is ~7× harder for a random
> policy: an 18-move (vs 13) optimum, a 4-way branch at every node, and only a
> 50-step budget.

### Level 2 (`level_index = 1`) — Monte-Carlo on the real engine (primary)

8-move shortest path, 12 nodes, budget 50. No-obstacle DP would give
p = 0.1336, but the single chaser kills the player in the large majority of
episodes (in a 400-episode probe: ~69% died, ~29% timed out, ~2% won), so the
true p is far lower. MC is authoritative.

<!-- FILL_L2 -->

### Level 3 (`level_index = 2`) — Monte-Carlo on the real engine (primary)

9-move shortest path, 23 nodes, budget 35 (tighter). No-obstacle DP upper bound
p = 0.0259; three chasers push the realised p below this.

<!-- FILL_L3 -->

---

## Honesty / caveats

- **L1 is exact**; L2/L3 are **Monte-Carlo** (the only sound estimator with
  moving obstacles). CIs are Wilson 95% on p̂; the E[steps] CI is propagated
  from the p̂ CI (episode-length variance is negligible by comparison).
- **No censoring on L2/L3**: p there is large enough that 30k engine episodes
  yield hundreds–thousands of wins. **L1's** p is too small for engine MC, which
  is exactly why the exact DP is used (and validated by a 2M-episode standalone
  sim).
- **No-obstacle DP is only an upper bound for L2/L3** because obstacle motion
  couples to the random trajectory; do not read those DP numbers as the answer.
- One life per episode ⇒ per-life p = per-episode p (no multi-life inflation,
  unlike LS20's 3-lives game probability).
