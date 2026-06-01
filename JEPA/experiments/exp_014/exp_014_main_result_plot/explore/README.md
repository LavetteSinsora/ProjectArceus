# Main-result visualization — candidate comparison

Goal: present, for 4 ARC games × 3 levels, **env-steps to first reward** per method, with seed
spread, the random-policy reference, and the ∞/unreachable frontier. Spans 1k–2M steps.

## Candidates (all from real data)
| id | file | paradigm |
|---|---|---|
| **v1** | `../v1/staircase.py` → `../v1/figures/staircase_*.png` | cumulative level-attainment **staircase** (x=cum steps, y=level; one fig per game) |
| **c1** | `c1_forest.py` → `figures/c1_forest.png` | **forest / dot-and-band**, faceted by game (rows=L1/L2/L3; ICM/RND dots + p25–p75; random diamond; log-x; "∞ never" zone) |
| **c2** | `c2_heatmap.py` → `figures/c2_heatmap.png` | **heatmap**, methods × all 12 cells (color=log steps; hatched=∞; n/N labels) |

## Pros / cons
- **v1 staircase** — ✅ tells the *narrative* (cumulative "journey" across levels; novel framing; shows when each method gives up); ✅ random reference + ∞ break built in. ❌ wastes vertical space when few levels are cleared; ❌ tu93 cluttered (L2 costs only ~2k → risers pile up); ❌ relies on the additive-independent-runs caveat.
- **c1 forest** — ✅ most rigorous & complete: shows median, **seed spread (p25–p75)**, **solve fraction (n/N)**, random reference, all on **log-x** (handles 1k–2M cleanly); ✅ per-cell, no additive assumption. ❌ 4 facets = more space; ❌ doesn't show the cross-level "progression" story.
- **c2 heatmap** — ✅ most compact (whole 3×12 grid at a glance); ✅ great as a summary/appendix table; ✅ ∞ unmistakable (hatched). ❌ color encodes magnitude coarsely; ❌ no seed-spread; ❌ less precise than the forest.

## Recommendation
- **Main quantitative result figure → c1 (forest).** It's the most honest and complete single view:
  median + spread + solve-rate + random reference + ∞, on a log axis that fits the dynamic range.
  No additive assumption to defend.
- **Conceptual / narrative figure → v1 (staircase).** Keep it for the intro/story ("methods clear L1
  but stall at the L2/L3 frontier; random never leaves L1 on g50t"). It's the more memorable framing.
- **Summary table → c2 (heatmap).** Appendix / at-a-glance.

→ **Promote c1 to v2** as the result figure; keep v1 as the conceptual companion.

## Not yet built (viz-explore agent died mid-run)
- Pareto scatter (total steps vs levels-cleared) and horizontal segmented bars — optional; the three
  above already cover detail (forest), overview (heatmap), and narrative (staircase).
