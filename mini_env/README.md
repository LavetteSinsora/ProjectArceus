# mini_env

A pure-numpy 32×32-pixel mini version of LS20 Level 1. Zero arcengine
dependency, JSON-defined levels, faithful to LS20's locked-door mechanics
(walls, gated goal with rotation matching, +90° rotation crosses, energy
counter, denial flash). Designed to be **plug-compatible** with
`JEPA/shared/env_wrapper.BaseArcEnv` so the existing PPO / Go-Explore / JEPA
trainers run unchanged.

## Attribute surface

Matches `BaseArcEnv` minus the arcengine internals:

```python
env = MiniLS20Env("mini_env/configs/level_01.json")
env.reset()           -> np.ndarray (32, 32) uint8
env.step(action_idx)  -> (np.ndarray, bool)   # action_idx in 0..3
env.n_actions          # 4
env.available_actions  # [1, 2, 3, 4]
env.level_completed    # bool
env.won                # bool
env._MASKED_ROWS       # slice(28, 32) — the UI strip
env.frame_diff(f0, f1) -> (32, 32) float32 — UI rows zeroed
env.patch_weights(f0, f1) -> (16,) float32 in [0, 1]   # 4×4 grid of 8×8 patches
env.detect_moved_cell(f0, f1) -> int | None             # 0..63 cell index, 8×8 grid of 4×4 px cells
```

## Action mapping

```
0  = ACTION1 = up      (Δrow = −1)
1  = ACTION2 = down    (Δrow = +1)
2  = ACTION3 = left    (Δcol = −1)
3  = ACTION4 = right   (Δcol = +1)
```

## Game rules (faithful to LS20 L1)

`step(action_idx)`, in order:

1. Decrement `step_counter`. If it hits 0 → terminal=True (won stays False).
2. Out-of-bounds destination → blocked (decrement `denial_frames` if > 0).
3. Wall destination → blocked.
4. Goal destination:
   - If `goal_gated` and `player_rotation != goal_rotation` → set
     `denial_frames=5`, terminal=False. (Mirrors LS20's `akoadfsur=5`.)
   - Else → step in, `won=True`, terminal=True.
5. Cross destination → step in, `player_rotation = (player_rotation + 90) % 360`.
6. Empty cell → step in.

## Level JSON schema (LOCKED)

```json
{
  "name": "level_01",
  "grid_cells": [8, 8],
  "tile_px": 4,
  "step_limit": 42,
  "palette": {
    "bg": 3, "wall": 4, "player": 9, "goal_frame": 5, "cross": 5,
    "preview_bg": 0, "highlight": 14, "energy": 6, "denial_flash": 0
  },
  "walls": [[col, row], ...],
  "player_start": {"cell": [col, row], "rotation": 270},
  "goal":         {"cell": [col, row], "rotation": 0},
  "cross":        {"cell": [col, row]},
  "goal_gated":   true,
  "show_match_cue": true
}
```

- `grid_cells * tile_px` **must** equal 32×32.
- Coordinates are `(col, row)`, zero-indexed cells. The bottom row of cells
  is reserved for the UI strip and is not addressable by walls/player/goal/cross.
- Rotations must be one of `{0, 90, 180, 270}`.
- The loader runs a BFS reachability check from `player_start` and rejects
  any layout that boxes in the cross or goal.

## Frame layout (32×32 uint8)

```
rows  0-27   playfield (7 cell-rows × 8 cell-cols, each cell 4×4 px)
rows 28-31   UI strip  — masked in frame_diff / patch_weights / detect_moved_cell
              cols  0-15  energy bar (row 29 only; fill grows left → right)
              cols 16-19  L-mark pattern preview (rotates with player_rotation)
              cols 20-31  preview background
```

## CLI

```
uv run python -m mini_env.cli render mini_env/configs/level_01.json --out /tmp/level_01.png
```

PIL/Pillow is optional and gated by try/except: omit `--out` to just sanity-
check that the env loads, or install Pillow with `uv pip install pillow` to
write a 16×-upscaled PNG.

## Regenerating the golden frame

The renderer is locked in place by
`mini_env/tests/test_render_matches_golden.py`, which compares the level_01
reset frame to `mini_env/tests/golden/level_01_frame.npy`. To intentionally
re-bake the golden after a renderer change:

```bash
uv run python -c "
from mini_env.env import MiniLS20Env
import numpy as np
np.save('mini_env/tests/golden/level_01_frame.npy',
        MiniLS20Env('mini_env/configs/level_01.json').reset())
"
```

## Test suite

```bash
uv run pytest mini_env/tests -q
```

## Integration with claude_automate

```python
from claude_automate.framework.env_api import make_mini_env
env = make_mini_env("mini_env/configs/level_01.json")
```

`claude_automate/solve.py` and `claude_automate/train.py` both accept a
`--mini-env <path>` flag that routes to `make_mini_env(...)` instead of
`make_arc_env(...)`. The rest of the training stack is unchanged because the
mini env exposes the same attribute surface.
