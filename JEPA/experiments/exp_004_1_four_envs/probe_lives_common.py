"""
Shared life-end probe protocol.

Mirrors exp_004_0_ls20_tu93/probe_tu93_lives.py exactly, but parameterised so
the per-env probes (probe_re86_lives.py, probe_g50t_lives.py) are thin wrappers.

Protocol per probe:
  1. Roll N random-action attempts in the target env (OFFLINE arc engine).
  2. Per step, capture: frame, raw.state, raw.available_actions,
     raw.levels_completed, and (once per attempt) a vars(raw) dump.
  3. Look for any obvious `lives`-like attribute on `raw`.
  4. If absent, find pixel regions that monotonically change only around terminal.
  5. Persist artefacts + verdict.md.
"""

from __future__ import annotations

import datetime
from collections import Counter
from pathlib import Path
from typing import Type

import numpy as np


def save_frame_png(frame: np.ndarray, path: Path) -> None:
    """Save a (64, 64) uint8 frame as a small PNG with the ARC-AGI-3 palette."""
    try:
        from PIL import Image
    except ImportError:
        np.save(path.with_suffix(".npy"), frame)
        return
    palette = np.array([
        [0, 0, 0], [0, 116, 217], [255, 65, 54], [46, 204, 64],
        [255, 220, 0], [170, 170, 170], [240, 18, 190], [255, 133, 27],
        [127, 219, 255], [135, 12, 37], [255, 255, 255], [1, 255, 1],
        [255, 109, 182], [136, 91, 23], [101, 67, 33], [192, 192, 192],
    ], dtype=np.uint8)
    rgb = palette[frame]
    img = Image.fromarray(rgb, mode="RGB").resize((256, 256), Image.NEAREST)
    img.save(path)


def run_probe(
    *,
    env_name: str,
    game_id: str,
    wrapper_cls: Type,
    out_root: Path,
    n_attempts: int = 8,
    max_steps_per_attempt: int = 800,
    rng_seed: int = 0,
    repo_root: Path,
) -> Path:
    """
    Run the standard life-end probe. Returns the per-run output directory.

    Args:
      env_name: short env name (used as a tag in stdout and filenames).
      game_id: full game id under environment_files/.
      wrapper_cls: BaseArcEnv subclass to wrap the raw arc env with.
      out_root: parent directory under which `<timestamp>/` is created.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"probe_{env_name}"
    print(f"[{tag}] writing to {out_dir}")

    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root / "environment_files"),
    )
    raw_env = arc.make(game_id)
    env = wrapper_cls(raw_env)
    n_actions = env.n_actions

    rng = np.random.default_rng(rng_seed)

    all_frames: list[np.ndarray] = []
    all_states: list[list[str]] = []
    all_lvls: list[list[int]] = []
    all_avail: list[list[list[int]]] = []
    all_raw_extra: list[list[dict]] = []
    raw_fields_dumped = False

    for attempt in range(n_attempts):
        frame = env.reset()
        per_step_frames: list[np.ndarray] = [frame.copy()]
        per_step_states: list[str] = []
        per_step_lvls: list[int] = []
        per_step_avail: list[list[int]] = []
        per_step_extra: list[dict] = []

        raw0 = env._latest_raw
        if not raw_fields_dumped:
            with open(out_dir / "raw_fields.txt", "w") as f:
                try:
                    f.write("vars(raw) keys:\n")
                    for k, v in vars(raw0).items():
                        f.write(f"  {k!r}: {type(v).__name__} = {v!r}\n")
                except Exception as e:
                    f.write(f"vars(raw) failed: {e}\n")
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

        for _ in range(max_steps_per_attempt):
            avail = env.available_actions
            if not avail:
                avail_idx = list(range(n_actions))
            else:
                # avail returns 1-indexed GameAction.values. The wrapper's _ACTIONS
                # list maps action_idx (0-indexed) -> GameAction.
                avail_idx = [i for i, a in enumerate(env._ACTIONS) if a.value in avail]
                if not avail_idx:
                    avail_idx = list(range(n_actions))
            action_idx = int(rng.choice(avail_idx))

            next_frame, is_terminal = env.step(action_idx)
            raw = env._latest_raw

            per_step_frames.append(next_frame.copy())
            per_step_states.append(str(getattr(raw, "state", "?")))
            per_step_lvls.append(int(getattr(raw, "levels_completed", -1)))
            per_step_avail.append(list(getattr(raw, "available_actions", []) or []))

            extra = {}
            for cand in ("lives", "remaining_lives", "energy", "hp", "score",
                         "step_count", "steps_remaining", "time_left", "level"):
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
        print(f"[{tag}] attempt {attempt}: steps={len(per_step_states)}  "
              f"final_state={per_step_states[-1] if per_step_states else '?'}  "
              f"lvls={per_step_lvls[-1] if per_step_lvls else '?'}")

    # ── Analysis ────────────────────────────────────────────────────────────
    candidates_seen: Counter = Counter()
    for ep in all_raw_extra:
        for step_extra in ep:
            for k in step_extra:
                candidates_seen[k] += 1
    print(f"\n[{tag}] candidate raw attributes (count of steps where present):")
    for k, n in candidates_seen.most_common():
        print(f"  {k}: {n}")

    lives_attr = None
    for cand in ("lives", "remaining_lives", "hp", "energy"):
        if candidates_seen.get(cand, 0) > 0:
            lives_attr = cand
            break

    if lives_attr is not None:
        print(f"\n[{tag}] Found candidate lives attribute: '{lives_attr}'")
        for i, ep in enumerate(all_raw_extra):
            traj = [s.get(lives_attr) for s in ep]
            unique = sorted({t for t in traj if t is not None}, key=lambda x: str(x))
            print(f"  attempt {i}: {lives_attr} unique values = {unique}; "
                  f"final = {traj[-1] if traj else None}")

    term_counts: Counter = Counter()
    for ep in all_states:
        if ep:
            term_counts[ep[-1]] += 1
    print(f"\n[{tag}] terminal state breakdown:")
    for s, n in term_counts.most_common():
        print(f"  {s}: {n}")

    lengths = [len(ep) for ep in all_states]
    print(f"\n[{tag}] episode lengths: "
          f"min={min(lengths)} mean={sum(lengths)/len(lengths):.1f} max={max(lengths)}")

    around = 4
    for i, frames in enumerate(all_frames):
        T = frames.shape[0]
        if T < 2:
            continue
        start = max(0, T - around - 1)
        for j in range(start, T):
            state_lbl = (
                all_states[i][min(j, len(all_states[i]) - 1)]
                if all_states[i] else "?"
            )
            png = out_dir / f"attempt{i:02d}_step{j:04d}_state{state_lbl}.png"
            save_frame_png(frames[j], png)

    for i, frames in enumerate(all_frames):
        save_frame_png(frames[0], out_dir / f"attempt{i:02d}_initial_t0000.png")

    npz_path = out_dir / "transitions.npz"
    np.savez_compressed(
        npz_path,
        **{f"attempt_{i}_frames": frames for i, frames in enumerate(all_frames)},
        **{f"attempt_{i}_states": np.array(all_states[i], dtype="<U16")
           for i in range(len(all_states))},
        **{f"attempt_{i}_lvls": np.array(all_lvls[i], dtype=np.int32)
           for i in range(len(all_lvls))},
    )
    print(f"\n[{tag}] persisted transitions → {npz_path}")

    print(f"\n[{tag}] scanning for terminal-only pixel regions...")
    all_attempt_terminal_only: list[np.ndarray] = []
    for i, frames in enumerate(all_frames):
        T = frames.shape[0]
        if T < 4:
            continue
        diffs = (frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)) != 0
        changes_throughout = diffs[:-2].any(axis=0)
        changes_near_end   = diffs[-2:].any(axis=0)
        terminal_only      = changes_near_end & ~changes_throughout
        all_attempt_terminal_only.append(terminal_only)
        n_terminal_only = int(terminal_only.sum())
        rows = np.where(terminal_only.any(axis=1))[0]
        cols = np.where(terminal_only.any(axis=0))[0]
        print(f"  attempt {i}: terminal_only_pixels={n_terminal_only}  "
              f"rows={list(rows) if len(rows)<10 else f'{len(rows)} rows in [{rows.min()},{rows.max()}]'}  "
              f"cols={list(cols) if len(cols)<10 else f'{len(cols)} cols in [{cols.min()},{cols.max()}]'}")

    if all_attempt_terminal_only:
        common = all_attempt_terminal_only[0].copy()
        for m in all_attempt_terminal_only[1:]:
            common &= m
        n_common = int(common.sum())
        if n_common > 0:
            rows = np.where(common.any(axis=1))[0]
            cols = np.where(common.any(axis=0))[0]
            print(f"\n[{tag}] STABLE terminal-only region across all attempts:")
            print(f"  total pixels: {n_common}")
            print(f"  rows: {rows.tolist()}  cols: {cols.tolist()}")
        else:
            print(f"\n[{tag}] No pixels changed at-and-only-at terminal across ALL attempts.")

    # ── Write a markdown verdict ────────────────────────────────────────────
    verdict_lines = [
        f"# {env_name.upper()} life-end probe verdict",
        "",
        f"- Game id: `{game_id}`",
        f"- Attempts run: {n_attempts}",
        f"- Steps per attempt cap: {max_steps_per_attempt}",
        f"- Episode length distribution: min={min(lengths)}, "
        f"mean={sum(lengths)/len(lengths):.1f}, max={max(lengths)}",
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
            f"Use this in `is_end_of_life_{env_name}`:",
            "",
            "```python",
            f"def is_end_of_life_{env_name}(prev_lives, curr_lives, is_terminal):",
            f"    return is_terminal or (curr_lives < prev_lives)",
            "```",
            "",
        ]
    else:
        verdict_lines += [
            "## No intra-game lives attribute found",
            "",
            f"{env_name.upper()} appears to terminate via "
            "`state in (WIN, GAME_OVER)` or `levels_completed >= 1` only. "
            "The conservative implementation is:",
            "",
            "```python",
            f"def is_end_of_life_{env_name}(frame, next_frame, is_terminal):",
            "    return is_terminal",
            "```",
            "",
            "If the terminal-only pixel-region scan above identified a stable "
            "region, consider using it as a per-life detector. Otherwise treat "
            "each game-over as the only life-end signal.",
        ]

    (out_dir / "verdict.md").write_text("\n".join(verdict_lines))
    print(f"\n[{tag}] verdict → {out_dir / 'verdict.md'}")
    return out_dir
