"""Training entry point for exp_016_0 — alias of run.py.

Exists under this name so the JEPA dashboard (JEPA/dashboard/server.py) discovers
the experiment (its heuristic requires config.py + train.py|eval.py) and so the
dashboard "start training" button can launch `python -m ...train`. The real CLI
lives in run.py; this just forwards to it.
"""
from __future__ import annotations

from .run import main

if __name__ == "__main__":
    main()
