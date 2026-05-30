"""Collect uniform-random (s, a, s') transitions on real LS20.

    uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_2_jepa_random_pretrain.collect
    uv run python -m ....exp_010_2_jepa_random_pretrain.collect --smoke
"""

import argparse
import json
from pathlib import Path

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.pretrain import collect_random, _repo_root
from .config import Config


def buffer_path(cfg) -> Path:
    return _repo_root() / cfg.exp_dir / "data" / "random_buffer.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    cfg = Config()
    n = args.n or cfg.n_random_transitions
    meta = collect_random(cfg, n, buffer_path(cfg), smoke=args.smoke)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
