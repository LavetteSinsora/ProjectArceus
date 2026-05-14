"""
TU93 life-end detection probe.

Goal: empirically determine how a single life ends in TU93 (graph-maze navigation),
so we can implement `is_end_of_life_tu93` for the multi-env training loop.

Strategy:
  1. Roll N random-action attempts in TU93 OFFLINE.
  2. Per step, capture: frame, raw.state, raw.available_actions, raw.levels_completed,
     and (once per attempt) a vars(raw) dump so we know what fields the raw env exposes.
  3. Look for any obvious `lives`-like attribute on `raw`.
  4. If absent, find pixel regions that monotonically change only around terminal events.
  5. Print a verdict.

Outputs:
  - probe_runs/<timestamp>/raw_fields.txt       — vars(raw) for one episode
  - probe_runs/<timestamp>/transitions.npz      — per-step (frame, state, lvls, avail)
  - probe_runs/<timestamp>/around_terminal_*.png — frames around death events
  - probe_runs/<timestamp>/verdict.md            — conclusion
"""

from __future__ import annotations

import datetime
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.shared.env_wrapper import Tu93Env  # noqa: E402

GAME_ID = "tu93-0768757b"
N_ATTEMPTS = 8
MAX_STEPS_PER_ATTEMPT = 800
RNG_SEED = 0


def save_frame_png(frame: np.ndarray, path: Path) -> None:
    """Save a (64,64) uint8 frame as a small PNG with a palette."""
    try:
        from PIL import Image
    except ImportError:
        # PIL not available — write as raw .npy instead so we at least keep data.
        np.save(path.with_suffix(".npy"), frame)
        return
    palette = np.zeros((16, 3), dtype=np.uint8)
    # ARC-AGI-3 default palette approximation (good enough for human inspection)
    palette[:16] = np.array([
        [0, 0, 0], [0, 116, 217], [255, 65, 54], [46, 204, 64],
        [255, 220, 0], [170, 170, 170], [240, 18, 190], [255, 133, 27],
        [127, 219, 255], [135, 12, 37], [255, 255, 255], [1, 255, 1],
        [255, 109, 182], [136, 91, 23], [101, 67, 33], [192, 192, 192],
    ], dtype=np.uint8)
    rgb = palette[frame]
    img = Image.fromarray(rgb, mode="RGB").resize((256, 256), Image.NEAREST)
    img.save(path)


def main() -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "probe_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe_tu93] writing to {out_dir}")

    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    raw_env = arc.make(GAME_ID)
    env = Tu93Env(raw_env)

    rng = np.random.default_rng(RNG_SEED)

    # Per-attempt logs
    all_frames: list[np.ndarray] = []      # list of (T, 64, 64) per attempt
    all_states: list[list[str]] = []
    all_lvls: list[list[int]] = []
    all_avail: list[list[list[int]]] = []
    all_raw_extra: list[list[dict]] = []   # any non-standard attrs per step

    raw_fields_dumped = False

    for attempt in range(N_ATTEMPTS):
        frame = env.reset()
        per_step_frames: list[np.ndarray] = [frame.copy()]
        per_step_states: list[str] = []
        per_step_lvls: list[int] = []
        per_step_avail: list[list[int]] = []
        per_step_extra: list[dict] = []

        # Dump vars(raw) for the very first step of the first attempt.
        raw0 = env._latest_raw
        if not raw_fields_dumped:
            with open(out_dir / "raw_fields.txt", "w") as f:
                try:
                    f.write("vars(raw) keys:\n")
                    for k, v in vars(raw0).items():
                        f.write(f"  {k!r}: {type(v).__name__} = {v!r}\n")
                except Exception as e:
                    f.write(f"vars(raw) failed: {e}\n")
                # Also try dir()
                f.write("\ndir(raw):\n")
                for name in dir(raw0):
                    if name.startswith("_"):
                        continue
                    try:
                        v = getattr(raw0, name)
                    except Exception as e:
                        v = f"<getattr error: {e}>"
                    if callable(v):
                        continue
                    f.write(f"  {name}: {type(v).__name__} = {v!r}\n")
            raw_fields_dumped = True

        for t in range(MAX_STEPS_PER_ATTEMPT):
            avail = env.available_actions
            # avail is 1-indexed GameAction values; convert to 0-indexed action_idx in {0..3}.
            if not avail:
                avail_idx = list(range(4))
            else:
                # The env's _ACTIONS list maps idx -> GameAction; available_actions returns
                # the GameAction.value ints. Take any legal one.
                avail_idx = [i for i, a in enumerate(env._ACTIONS) if a.value in avail]
                if not avail_idx:
                    avail_idx = list(range(4))
            action_idx = int(rng.choice(avail_idx))

            next_frame, is_terminal = env.step(action_idx)
            raw = env._latest_raw

            per_step_frames.append(next_frame.copy())
            per_step_states.append(str(getattr(raw, "state", "?")))
            per_step_lvls.append(int(getattr(raw, "levels_completed", -1)))
            per_step_avail.append(list(getattr(raw, "available_actions", []) or []))

            # Grab any non-standard attributes for diagnostic.
            extra = {}
            for cand in ("lives", "remaining_lives", "energy", "score",
                         "step_count", "time_left", "level"):
                v = getattr(raw, cand, None)
                if v is not None and not callable(v):
                    extra[cand] = v
            per_step_extra.append(extra)

            if is_terminal:
                break

        all_frames.append(np.stack(per_step_frames))
        all_states.append(per_step_states)
        all_lvls.append(per_step_lvls)
        all_avail.append(per_step_avail)
        all_raw_extra.append(per_step_extra)
        print(f"[probe_tu93] attempt {attempt}: steps={len(per_step_states)}  "
              f"final_state={per_step_states[-1] if per_step_states else '?'}  "
              f"lvls={per_step_lvls[-1] if per_step_lvls else '?'}")

    # ── Analysis ──────────────────────────────────────────────────────────────

    # 1. Did any candidate 'lives' attribute show up?
    candidates_seen = Counter()
    for ep in all_raw_extra:
        for step_extra in ep:
            for k in step_extra:
                candidates_seen[k] += 1
    print("\n[probe_tu93] candidate raw attributes (count of steps where present):")
    for k, n in candidates_seen.most_common():
        print(f"  {k}: {n}")

    # If a 'lives' attribute exists, track its trajectory across an attempt.
    lives_attr = None
    for cand in ("lives", "remaining_lives"):
        if candidates_seen.get(cand, 0) > 0:
            lives_attr = cand
            break

    if lives_attr is not None:
        print(f"\n[probe_tu93] Found candidate lives attribute: '{lives_attr}'")
        for i, ep in enumerate(all_raw_extra):
            traj = [s.get(lives_attr) for s in ep]
            unique = sorted(set(traj))
            print(f"  attempt {i}: {lives_attr} unique values = {unique}; final = {traj[-1]}")

    # 2. Terminal state breakdown.
    term_counts = Counter()
    for ep in all_states:
        if ep:
            term_counts[ep[-1]] += 1
    print("\n[probe_tu93] terminal state breakdown:")
    for s, n in term_counts.most_common():
        print(f"  {s}: {n}")

    # 3. Episode length distribution.
    lengths = [len(ep) for ep in all_states]
    print(f"\n[probe_tu93] episode lengths: "
          f"min={min(lengths)} mean={sum(lengths)/len(lengths):.1f} max={max(lengths)}")

    # 4. Save frames around terminal events.
    around = 4
    for i, frames in enumerate(all_frames):
        T = frames.shape[0]
        if T < 2:
            continue
        start = max(0, T - around - 1)
        for j in range(start, T):
            png = out_dir / f"attempt{i:02d}_step{j:04d}_state{all_states[i][min(j, len(all_states[i])-1)] if all_states[i] else '?'}.png"
            save_frame_png(frames[j], png)

    # 5. Save first frames of each attempt (to compare resets — does each attempt look
    #    the same? does reset return the same start state?).
    for i, frames in enumerate(all_frames):
        save_frame_png(frames[0], out_dir / f"attempt{i:02d}_initial_t0000.png")

    # 6. Persist per-attempt arrays for later inspection.
    npz_path = out_dir / "transitions.npz"
    np.savez_compressed(
        npz_path,
        **{
            f"attempt_{i}_frames": frames for i, frames in enumerate(all_frames)
        },
        **{
            f"attempt_{i}_states": np.array(all_states[i], dtype="<U16") for i in range(len(all_states))
        },
        **{
            f"attempt_{i}_lvls": np.array(all_lvls[i], dtype=np.int32) for i in range(len(all_lvls))
        },
    )
    print(f"\n[probe_tu93] persisted transitions → {npz_path}")

    # 7. Pixel-level scan: find pixels that change ONLY on the terminal step.
    #    Strategy: for each attempt, compute frame_diff between consecutive frames.
    #    Mark any pixel that changes ≥1 time during the attempt. Then, separately,
    #    record pixels that change during the LAST 2 steps. The set difference
    #    (changes-near-terminal MINUS changes-throughout) is candidate life-indicator.
    print("\n[probe_tu93] scanning for terminal-only pixel regions...")
    all_attempt_terminal_only: list[np.ndarray] = []
    for i, frames in enumerate(all_frames):
        T = frames.shape[0]
        if T < 4:
            continue
        diffs = (frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)) != 0
        changes_throughout = diffs[:-2].any(axis=0)   # (64,64) bool
        changes_near_end   = diffs[-2:].any(axis=0)
        terminal_only      = changes_near_end & ~changes_throughout
        all_attempt_terminal_only.append(terminal_only)
        n_terminal_only = int(terminal_only.sum())
        rows = np.where(terminal_only.any(axis=1))[0]
        cols = np.where(terminal_only.any(axis=0))[0]
        print(f"  attempt {i}: terminal_only_pixels={n_terminal_only}  "
              f"rows={list(rows) if len(rows)<10 else f'{len(rows)} rows in [{rows.min()},{rows.max()}]'}  "
              f"cols={list(cols) if len(cols)<10 else f'{len(cols)} cols in [{cols.min()},{cols.max()}]'}")

    # Intersection of terminal_only across attempts: stable life-indicator region.
    if all_attempt_terminal_only:
        common = all_attempt_terminal_only[0].copy()
        for m in all_attempt_terminal_only[1:]:
            common &= m
        n_common = int(common.sum())
        if n_common > 0:
            rows = np.where(common.any(axis=1))[0]
            cols = np.where(common.any(axis=0))[0]
            print(f"\n[probe_tu93] STABLE terminal-only region across all attempts:")
            print(f"  total pixels: {n_common}")
            print(f"  rows: {rows.tolist()}  cols: {cols.tolist()}")
        else:
            print("\n[probe_tu93] No pixels changed at-and-only-at terminal across ALL attempts.")

    # ── Write a markdown verdict ──────────────────────────────────────────────
    verdict_lines = [
        "# TU93 life-end probe verdict",
        "",
        f"- Attempts run: {N_ATTEMPTS}",
        f"- Steps per attempt cap: {MAX_STEPS_PER_ATTEMPT}",
        f"- Episode length distribution: "
        f"min={min(lengths)}, mean={sum(lengths)/len(lengths):.1f}, max={max(lengths)}",
        f"- Terminal states observed: {dict(term_counts)}",
        "",
        "## Candidate raw env attributes",
    ]
    if candidates_seen:
        for k, n in candidates_seen.most_common():
            verdict_lines.append(f"- `{k}` ({n} steps)")
    else:
        verdict_lines.append("- None (no `lives`/`energy`/`score`/etc. attribute on raw)")
    verdict_lines.append("")

    if lives_attr is not None:
        verdict_lines += [
            f"## Lives attribute: `{lives_attr}`",
            "",
            "Use this in `is_end_of_life_tu93`:",
            "",
            "```python",
            f"def is_end_of_life_tu93(prev_lives, curr_lives, is_terminal):",
            f"    return is_terminal or (curr_lives < prev_lives)",
            "```",
            "",
        ]
    else:
        verdict_lines += [
            "## No intra-game lives attribute found",
            "",
            "TU93 appears to terminate via `state in (WIN, GAME_OVER)` or "
            "`levels_completed >= 1` only. The conservative implementation is:",
            "",
            "```python",
            "def is_end_of_life_tu93(frame, next_frame, is_terminal):",
            "    return is_terminal",
            "```",
            "",
            "If the terminal-only pixel-region scan above identified a stable region, "
            "consider using it as a more sensitive per-life detector. Otherwise, "
            "treat each game-over as the only life-end signal.",
        ]

    (out_dir / "verdict.md").write_text("\n".join(verdict_lines))
    print(f"\n[probe_tu93] verdict → {out_dir / 'verdict.md'}")


if __name__ == "__main__":
    main()
