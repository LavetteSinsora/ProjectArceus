"""PPO from the random-data-pretrained encoder (UNFROZEN) on real LS20.

    uv run python -m ....exp_010_2_jepa_random_pretrain.train_ppo
    uv run python -m ....exp_010_2_jepa_random_pretrain.train_ppo --smoke

If --encoder_ckpt is not given, uses <exp_dir>/jepa_pretrained/encoder_final.pt
(produced by train_jepa.py). Run collect.py + train_jepa.py first.
"""

import argparse
import dataclasses

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.trainer import train, _repo_root
from .config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--encoder_ckpt", default=None)
    ap.add_argument("--freeze", action="store_true",
                    help="freeze the pretrained encoder during PPO (default: unfrozen)")
    args = ap.parse_args()

    cfg = Config()
    enc = args.encoder_ckpt
    if enc is None:
        default_enc = _repo_root() / cfg.exp_dir / "jepa_pretrained" / "encoder_final.pt"
        if default_enc.exists():
            enc = str(default_enc)
        elif not args.smoke:
            raise SystemExit(f"No pretrained encoder at {default_enc}. Run collect + "
                             f"train_jepa first, or pass --encoder_ckpt.")
    cfg = dataclasses.replace(cfg, init_encoder_ckpt=enc, freeze_encoder=args.freeze)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
