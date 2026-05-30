"""Pretrain the JEPA encoder on the random buffer until plateau.

    uv run python -m ....exp_010_2_jepa_random_pretrain.train_jepa
    uv run python -m ....exp_010_2_jepa_random_pretrain.train_jepa --smoke

Writes <exp_dir>/jepa_pretrained/encoder_final.pt. Run collect.py first.
"""

import argparse
import json

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.pretrain import pretrain_jepa
from .collect import buffer_path
from .config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    bp = buffer_path(cfg)
    if not bp.exists():
        raise SystemExit(f"No random buffer at {bp}. Run `collect` first "
                         f"(add --smoke for the tiny version).")
    meta = pretrain_jepa(cfg, bp, smoke=args.smoke)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
