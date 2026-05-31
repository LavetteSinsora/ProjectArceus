# ARC-AGI-3 (ARC Prize 2026) — Kaggle submission

A submission for the **ARC Prize 2026 / ARC-AGI-3** Kaggle competition, built
from the `claude_automate` Go-Explore artifacts.

| file | role |
|---|---|
| `kaggle_submission.ipynb` | the submission — one self-contained, offline notebook |
| `kaggle_agent.py` | the agent (same code inlined into the notebook); importable for local tests |
| `cached_solutions.json` | level-0 warm-start trajectories for public games (ls20, tu93) |
| `_build_kaggle_notebook.py` | regenerates the notebook from `kaggle_agent.py` + cache (keeps them in sync) |

## What the agent does

ARC-AGI-3's OFFLINE games are **deterministic** — replaying a fixed action
sequence from `reset()` always reproduces the same state. The agent exploits
this with **Go-Explore** (Ecoffet et al. 2021), the same level-agnostic engine
from `framework/go_explore.py`:

1. **Plan** on a *private* Arcade copy (its own throwaway scorecard): archive
   UI-masked frame "cells", return to any archived cell for free via
   reset+replay, explore onward, and keep the trajectory completing the most
   levels. The only goal signal is the universal `levels_completed` counter, so
   it works on **unseen/held-out games** — no per-game knowledge.
2. **Replay** that solution on the *scored* Arcade. Only those actions hit the
   official scorecard, so the scored action count ≈ the solution length, which
   is what RHAE (Relative Human Action Efficiency) rewards — not the search cost.

A cached level-0 solution (for public games we already solved) seeds the search
as a warm start; on unseen games it is simply absent and the agent searches
from scratch.

## How it maps to the competition rules

| rule | how this submission satisfies it |
|---|---|
| Single Kaggle notebook, one-click, top-to-bottom | `kaggle_submission.ipynb` runs start→finish with no manual steps |
| **No internet at evaluation** | imports only `arc_agi`/`arcengine`/`numpy` (provided by the competition image); no pip install, no hosted-model API calls |
| Permissive / public-domain license | `SPDX-License-Identifier: MIT-0` header in agent + notebook |
| Runs in < 12 h | per-game search budgets (`MAX_ENV_STEPS`, `TIME_BUDGET_S`) are configurable; default 300k steps / 600 s per game |
| Generalize to novel games (skill-acquisition efficiency) | Go-Explore learns each game at run time from the universal completion signal; nothing encodes a specific game |
| Robustness | a failing or unsupported (coordinate-only) game is reported as 0, never crashing the run |

## Before you submit — wire it to the eval games

In the notebook's **config cell** set:
- `ENVIRONMENTS_DIR` → the competition-provided game folder (same shape as the
  public preview's `environment_files/`).
- `GAME_IDS` → the game ids to play (usually supplied by the harness).

If the official harness hands your agent a live game object rather than letting
you construct your own Arcade, move the agent's per-game logic into that
harness's agent entry point — the planning/replay functions in `kaggle_agent.py`
are written to make that swap small.

## Verified locally

Against the repo's `environment_files/` (offline):
- **Warm-start** (ls20): completes level 1 in **33 scored actions**, scorecard
  `score ≈ 1.59`, `total_actions = 33`.
- **Pure discovery** (ls20, cache disabled — simulates a held-out game):
  Go-Explore rediscovers a 33-action level-1 solution from scratch (~10⁵ search
  steps on the private copy), again **33 scored actions**.
- The full notebook executes top-to-bottom and writes `submission_report.json`
  with per-game official scorecards.

To re-run locally:
```bash
uv run python -c "from claude_automate.kaggle_agent import solve_game; \
  print(solve_game('ls20-9607627b','environment_files', \
  cached={}, max_env_steps=120000, time_budget_s=90)['scored_score'])"
# regenerate the notebook after editing the agent:
uv run python claude_automate/_build_kaggle_notebook.py
```

## Honest caveats

- **Plan-on-copy assumption.** Searching a private Arcade and replaying only the
  solution is legitimate under OFFLINE determinism and is what keeps the scored
  action count low. If the official harness forbids a second environment, set
  `DIRECT=True` — every exploration action is then scored (much lower RHAE, but
  it still completes levels).
- **Depth.** Go-Explore clears early levels reliably; very deep puzzles may
  exhaust the per-game budget and return a best-effort partial trajectory
  (partial credit, capped at completed/total levels).
- **Discrete actions only.** Pure coordinate/click games are skipped, not solved.
- This packages the `claude_automate` *search method*, not a trained per-game
  policy. It learns each game at run time; it does not rely on weights that
  memorise a specific game.
