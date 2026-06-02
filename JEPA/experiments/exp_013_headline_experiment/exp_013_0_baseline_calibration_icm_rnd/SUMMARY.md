# Random-policy baseline — consolidated summary

**E[total env steps to first positive reward] under a uniform-random policy**, per
game × level. This is the no-inductive-bias reference each intrinsic method (ICM,
RND, "ours") must beat in exp_013. Method = exact forward-DP and/or Monte-Carlo on
the real engine (see `METHODOLOGY.md`; per-game scripts in `scripts/`).

| game | actions | L1 | L2 | L3 | method |
|------|:------:|----|----|----|--------|
| **ls20** | 4 | **49,843** (p_life 8.63e-4) | **∞** (unreachable) | **∞** (unreachable) | exact DP, MC-verified |
| **tu93** | 4 | **≈500,000** (MC p̂ 1e-4; CI 324k–772k) | **≈2,173** (p̂ 1.5e-2, 8-move) | **∞** (0/200k, ≥1.2M) | MC 200k eps (+DP) |
| **re86** | 5 | **≈2,000,000** (p 5e-5, 3/60k wins) | **≥1.56M** (0/60k, censored) | **≥3.12M** (0/60k, censored) | MC (60k lives) |
| **g50t** | 5 | **∞** (0/30k, p≤1e-4) | **∞** (0/30k, p≤1e-4) | **∞** (0/30k, censored) | MC (30k lives) |

## Per-game notes

- **ls20** — 12×12 rotation-maze. L1: 13-move min, energy 42/life (43 win-chances),
  p_life 8.63e-4 → **E ≈ 49,843** (cross-validates the prior ~50k Markov analysis).
  **L2 & L3 are p=0 within the energy budget** (only ~22 win-opportunities/life vs a
  solution longer than the budget) → random *never* solves them, E = ∞. So on
  ls20 L2/L3 any method that ever solves beats random infinitely; expect these to be
  very hard for ICM/RND too (ICM scored 0% on L2 in exp_011).
- **tu93** — graph-maze, moving obstacles. **Difficulty is NON-monotonic in level:**
  L1 ≈ **500k** (18-move solution, budget 50; MC p̂ 1e-4 with live obstacles vs no-obstacle
  DP 1.43e-4 → obstacles make it ~1.4× harder); L2 ≈ **2,173** (only an 8-move solution,
  12 reachable nodes, p̂ 1.5e-2 → trivially reachable by random!); L3 = **∞** (9-move
  solution but budget only 35; 0/200k wins, E ≥ 1.2M). So tu93 L2 is the *easiest* cell in
  the whole suite and L1 is harder than L2 — watch for this when reading method results.
- **re86** — 5-action sliding puzzle (4 move + switch piece), budget 100/life. L1
  observed 3 wins in 60k lives → p ≈ 5e-5, **E ≈ 2.0M** (Wilson 95% CI on p
  [1.7e-5, 1.47e-4]). L2 (budget 100) and L3 (budget 200) saw **0 wins in 60k lives**
  → right-censored, E ≥ 1.56M / 3.12M respectively.
- **g50t** — 5-action Sokoban + undo. Very small p, **0 clears in 30k lives at every level**
  → all levels effectively **unreachable by random** (p ≤ 1e-4, E = ∞), like re86/ls20-L2+.
  (200k-life MC was ~3 h/level; 30k gives the rule-of-three bound.)

## Implication for exp_013 caps

**Only four cells are reachable by random at all** (finite E): **ls20 L1 (~50k), tu93 L2
(~2k), tu93 L1 (~500k), re86 L1 (~2M)** — these are where "beats random" is a meaningful,
quantitative comparison. **Everything else is E = ∞** (ls20 L2/L3, tu93 L3, re86 L2/L3, all
g50t) — there a method either solves (∞× better than random) or it doesn't. Cap guidance for
the notebook: ls20 L1 → 250k; tu93 L2 → 50k; tu93 L1 → 1M; re86 L1 → 3M; the ∞ cells →
whatever stretch budget compute allows (a solve is the result, censoring is expected).
