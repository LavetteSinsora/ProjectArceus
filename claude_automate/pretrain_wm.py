"""pretrain_wm.py — train the shared world model and measure transfer.

Trains a `FrameWorldModel` on transitions from LS20 Levels 1-3 (the "seen"
levels, with their Go-Explore solutions for modifier/completion coverage),
then measures prediction accuracy on **held-out** Level 4 (full — including
completion transitions) and Levels 5-7 (dynamics only). Saves the model.

    cd "Code Repo"
    uv run python claude_automate/pretrain_wm.py

The held-out accuracy is the make-or-break number for model-based planning.
"""

from __future__ import annotations

import datetime
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from claude_automate.framework.env_api import make_arc_env
from claude_automate.framework.world_model import FrameWorldModel
from claude_automate.framework.wm_train import (
    collect_transitions, evaluate_world_model, train_world_model,
)

_EXP_DIR = Path(__file__).resolve().parent / "experiments"
_GAME = "ls20-9607627b"
_TRAIN_LEVELS = [0, 1, 2]              # L1, L2, L3  (0-indexed)
_HELDOUT_FULL = 3                      # L4 — held out, solution known
_HELDOUT_DYN = [4, 5, 6]               # L5-L7 — held out, dynamics only


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_solution(level_index: int) -> list[int] | None:
    """Load a Go-Explore solution trajectory for an LS20 level, if one exists."""
    hits = sorted(glob.glob(str(
        _EXP_DIR / f"solve_ls20_L{level_index + 1}_*/solution.json")))
    if not hits:
        return None
    return json.loads(Path(hits[-1]).read_text())["trajectory"]


def main():
    device = pick_device()
    rng = np.random.default_rng(0)
    print(f"[pretrain-wm] device={device}")

    # ── training transitions: LS20 L1-L3 (with solutions) ────────────────────
    train_tr = []
    for lvl in _TRAIN_LEVELS:
        env = make_arc_env(_GAME, lvl)
        sol = load_solution(lvl)
        tr = collect_transitions(env, rng, n_random_episodes=110,
                                 solution=sol, n_solution_replays=40)
        train_tr += tr
        print(f"[pretrain-wm] L{lvl+1}: {len(tr)} transitions "
              f"(solution {'yes' if sol else 'no'})")
    print(f"[pretrain-wm] total training transitions: {len(train_tr)}")

    n_actions = make_arc_env(_GAME, 0).n_actions
    model = FrameWorldModel(n_colors=16, n_actions=n_actions).to(device)
    train_world_model(model, train_tr, device, epochs=15, batch_size=64)

    # ── held-out evaluation ──────────────────────────────────────────────────
    report = {"train_levels": [l + 1 for l in _TRAIN_LEVELS],
              "n_train_transitions": len(train_tr), "heldout": {}}

    env4 = make_arc_env(_GAME, _HELDOUT_FULL)
    sol4 = load_solution(_HELDOUT_FULL)
    tr4 = collect_transitions(env4, rng, n_random_episodes=40,
                              solution=sol4, n_solution_replays=20)
    m4 = evaluate_world_model(model, tr4, device)
    report["heldout"][f"L{_HELDOUT_FULL+1}"] = m4
    print(f"[pretrain-wm] HELD-OUT L{_HELDOUT_FULL+1} (full): {m4}")

    for lvl in _HELDOUT_DYN:
        env = make_arc_env(_GAME, lvl)
        tr = collect_transitions(env, rng, n_random_episodes=30)
        m = evaluate_world_model(model, tr, device)
        report["heldout"][f"L{lvl+1}"] = m
        print(f"[pretrain-wm] HELD-OUT L{lvl+1} (dynamics): {m}")

    # ── save ─────────────────────────────────────────────────────────────────
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _EXP_DIR / f"wm_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "n_actions": n_actions,
                "n_colors": 16, "game_id": _GAME,
                "train_levels": _TRAIN_LEVELS},
               run_dir / "wm.pt")
    (run_dir / "transfer_report.json").write_text(json.dumps(report, indent=2))
    print(f"[pretrain-wm] saved → {run_dir}/wm.pt")


if __name__ == "__main__":
    main()
