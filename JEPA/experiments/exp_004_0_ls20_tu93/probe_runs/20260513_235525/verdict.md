# TU93 life-end probe verdict

- Attempts run: 8
- Steps per attempt cap: 800
- Episode length distribution: min=50, mean=50.0, max=50
- Terminal states observed: {'GameState.GAME_OVER': 8}

## Candidate raw env attributes
- None (no `lives`/`energy`/`score`/etc. attribute on raw)

## No intra-game lives attribute found

TU93 appears to terminate via `state in (WIN, GAME_OVER)` or `levels_completed >= 1` only. The conservative implementation is:

```python
def is_end_of_life_tu93(frame, next_frame, is_terminal):
    return is_terminal
```

If the terminal-only pixel-region scan above identified a stable region, consider using it as a more sensitive per-life detector. Otherwise, treat each game-over as the only life-end signal.