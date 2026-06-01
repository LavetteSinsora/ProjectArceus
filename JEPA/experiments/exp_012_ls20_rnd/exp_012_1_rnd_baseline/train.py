"""Train exp_012_1 (RND exploration baseline on real LS20).

    uv run python -m JEPA.experiments.exp_012_ls20_intrinsic_exploration.exp_012_1_rnd_baseline.train
    uv run python -m JEPA.experiments.exp_012_ls20_intrinsic_exploration.exp_012_1_rnd_baseline.train --smoke
"""

# Let any MPS op without a native kernel fall back to CPU instead of crashing.
# Must be set before torch initialises the MPS backend.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

from JEPA.experiments.exp_012_ls20_rnd.shared.trainer import train
from .config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train(Config(), smoke=args.smoke)


if __name__ == "__main__":
    main()
