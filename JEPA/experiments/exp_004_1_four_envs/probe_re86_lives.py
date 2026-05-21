"""
RE86 life-end detection probe.

Goal: empirically determine how a single life ends in RE86 (5-action sliding
puzzle with piece-switch), so we can implement `is_end_of_life_re86` for the
multi-env training loop.

Outputs go to `probe_runs/<timestamp>/` next to this script:
  - raw_fields.txt
  - transitions.npz
  - around-terminal frame PNGs
  - verdict.md
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_004_1_four_envs.probe_lives_common import run_probe  # noqa: E402
from JEPA.shared.env_wrapper import Re86Env  # noqa: E402


GAME_ID = "re86-8af5384d"
N_ATTEMPTS = 8
MAX_STEPS_PER_ATTEMPT = 800
RNG_SEED = 0


def main() -> None:
    out_root = Path(__file__).parent / "probe_runs" / "re86"
    run_probe(
        env_name="re86",
        game_id=GAME_ID,
        wrapper_cls=Re86Env,
        out_root=out_root,
        n_attempts=N_ATTEMPTS,
        max_steps_per_attempt=MAX_STEPS_PER_ATTEMPT,
        rng_seed=RNG_SEED,
        repo_root=_repo_root,
    )


if __name__ == "__main__":
    main()
