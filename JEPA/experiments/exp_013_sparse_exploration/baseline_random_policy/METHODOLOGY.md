# Random-policy baseline — methodology

Goal: for each game × level, compute **E[env steps to first positive extrinsic
reward] under a uniform-random policy** (each legal action chosen with prob
1/n_actions). This is the no-inductive-bias reference for exp_013.

This methodology is generalised from the LS20 Level-1 analysis (which got
E ≈ 50,000 env steps, p_win ≈ 8.63e-4 per life), cross-validated exact-DP vs
Monte-Carlo. Reproduce that style for every game/level.

## Two complementary estimators (do both; cross-validate)

1. **Monte-Carlo on the *real* engine (primary, always feasible).**
   Build the actual game with
   `claude_automate.framework.env_api.make_arc_env(game_id, level_index)`
   (works for any ARC game; `LevelStartWrapper` jumps to the level). Run a
   uniform-random policy for many episodes/lives, record env-steps until the
   first success (`levels_completed` increments / `GameState.WIN` /
   wrapper `level_completed`). Estimate p̂ (success prob per life/episode) and
   Ê[steps]. Use enough trials that p̂ has a tight CI; if p is too small to
   observe (< ~1e-4 → millions of steps needed), fall back to (2).

2. **Exact analytic model (for tiny p, and as a cross-check).**
   Read the game's source (`environment_files/<game>/<hash>/<game>.py`) and
   `metadata.json` to recover: action semantics, the energy/step budget per
   life, number of lives, win condition, level geometry, and any state that
   gates the win (e.g. LS20's rotation index). Then build a forward DP /
   absorbing-Markov-chain over the reachable state × budget to get the exact
   per-life win probability and E[steps]. Verify it matches the MC p̂.

Report **total env steps** (summed across actors) — this is invariant to the
number of parallel envs in expectation and is the quantity exp_013 compares
against. State per-life p, per-episode p, and E[steps] like the LS20 writeup.

## What to extract per game (read the .py first)

- Action set & whether actions can be no-ops / blocked (still cost a step?).
- Energy / step budget per life; number of lives; what reset restores.
- Win condition and any *ordering* constraints (LS20: must toggle a rotation
  tile an odd number of times before reaching the goal → 13-move minimum).
- Whether stepping on the goal in a wrong state is death or a free no-op.
- Shortest solution length (gives a sanity floor on p).

## Deliverables (write into this directory)

- `<game>_random_baseline.md` — mechanics summary, win condition, per-life p,
  per-episode p, **E[env steps to first reward]**, MC-vs-analytic cross-check,
  for **L1, L2, L3** (start with L1; do L2 & L3 once mechanics are understood).
- `scripts/<game>_random_baseline.py` — the reusable MC (+ DP if built) script,
  runnable with `uv run python ...`.

## Reference

LS20 L1 result to anchor against: 12×12 maze, start rot=3, must hit rotation
tile ≡1 mod 4 times then reach goal; 13-move min; energy 42/life, 3 lives;
p_life ≈ 8.63e-4, p_game(3 lives) ≈ 2.59e-3, **E ≈ 50,000 env steps**
(exact-DP, verified by 2M-life MC p̂=8.54e-4). Maze parsed directly from
`environment_files/ls20/9607627b/ls20.py`.
