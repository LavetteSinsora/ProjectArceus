"""exp_013_1 — Leaky RND ("A"): RND novelty + leaky predictor on a FIXED random encoder φ
(no ICM inverse-dynamics training). This is the *leak-only* change to standard RND.

A is the `phi_mode='frozen'` special case of the shared leaky-RND module
(`exp_013_1b_leaky_rnd_on_icm_phi`), so this is a thin wrapper that forces that mode rather than
duplicating ~600 lines. The implementation, config, and trainer all live in exp_013_1b.

    uv run python -m JEPA.experiments.exp_013_headline_experiment.exp_013_1_leaky_rnd.run \
        --game ls20 --level 0 --seed 0          # (--phi-mode frozen is injected automatically)
"""

from __future__ import annotations

import sys

from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.run import main

if __name__ == "__main__":
    if "--phi-mode" not in sys.argv:          # A := the frozen-random-φ mode of exp_013_1b
        sys.argv += ["--phi-mode", "frozen"]
    main()
