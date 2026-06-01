# Random-policy baseline — consolidated summary

**E[total env steps to first positive reward] under a uniform-random policy**, per
game × level. This is the no-inductive-bias reference each intrinsic method (ICM,
RND, "ours") must beat in exp_013. Method = exact forward-DP and/or Monte-Carlo on
the real engine (see `METHODOLOGY.md`; per-game scripts in `scripts/`).

| game | actions | L1 | L2 | L3 | method |
|------|:------:|----|----|----|--------|
| **ls20** | 4 | **49,843** (p_life 8.63e-4) | **∞** (unreachable) | **∞** (unreachable) | exact DP, MC-verified |
| **tu93** | 4 | **≈350,000** (p≈1.43e-4, 18-move) | *pending MC* | *pending MC* | no-obstacle DP (+MC) |
| **re86** | 5 | **≈2,000,000** (p 5e-5, 3/60k wins) | **≥1.56M** (0/60k, censored) | **≥3.12M** (0/60k, censored) | MC (60k lives) |
| **g50t** | 5 | *recomputing* | *recomputing* | *recomputing* | MC (30k lives) |

## Per-game notes

- **ls20** — 12×12 rotation-maze. L1: 13-move min, energy 42/life (43 win-chances),
  p_life 8.63e-4 → **E ≈ 49,843** (cross-validates the prior ~50k Markov analysis).
  **L2 & L3 are p=0 within the energy budget** (only ~22 win-opportunities/life vs a
  solution longer than the budget) → random *never* solves them, E = ∞. So on
  ls20 L2/L3 any method that ever solves beats random infinitely; expect these to be
  very hard for ICM/RND too (ICM scored 0% on L2 in exp_011).
- **tu93** — graph-maze, moving obstacles. L1 shortest (no-obstacle) solution 18 moves,
  budget 50/life, exact no-obstacle DP per-episode p ≈ 1.43e-4 → **E ≈ 350k**. MC
  cross-check (with live obstacles) + L2/L3 still running.
- **re86** — 5-action sliding puzzle (4 move + switch piece), budget 100/life. L1
  observed 3 wins in 60k lives → p ≈ 5e-5, **E ≈ 2.0M** (Wilson 95% CI on p
  [1.7e-5, 1.47e-4]). L2 (budget 100) and L3 (budget 200) saw **0 wins in 60k lives**
  → right-censored, E ≥ 1.56M / 3.12M respectively.
- **g50t** — 5-action Sokoban + undo, slow engine, very small p (0 wins early) →
  treated like re86 as a censored upper-bound; rerunning at 30k lives for a
  rule-of-three bound. (200k-life MC was ~3 h/level, killed.)

## Implication for exp_013 caps

Caps in the Colab notebook (`colab_calibration.ipynb`) scale to these: ls20 L1 150k;
re86 L1 2M; tu93 L1 ~ a few ×100k. The ls20 L2/L3 (and likely g50t/re86 L2/L3) cells
test "can the method do what random provably cannot," so a solve there is the
strongest possible result; budget them as stretch runs.
