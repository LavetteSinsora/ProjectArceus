"""mini_env CLI — render a level config to a PNG (upscaled 16x).

Usage:
    python -m mini_env.cli render mini_env/configs/level_01.json --out /tmp/level_01.png

PIL/Pillow is optional. If not installed, the command prints a clear message
explaining how to install it and exits non-zero, without crashing imports.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from mini_env.env import MiniLS20Env


# Canonical 16-color ARC palette (mirrors JEPA/experiments/exp_003_0_normalized_latent_jepa/
# debug_runner.py:ARC_COLORS_RGB). The web editor at mini_env/editor/editor.js uses the
# same palette, so editor previews and CLI PNG exports match pixel-for-pixel.
_COLOR_TABLE_RGB = {
    0:  (0,   0,   0),    # 0  black
    1:  (0,   116, 217),  # 1  blue
    2:  (255, 65,  54),   # 2  red
    3:  (46,  204, 64),   # 3  green   (LS20 bg)
    4:  (255, 220, 0),    # 4  yellow  (walls)
    5:  (170, 170, 170),  # 5  grey    (goal_frame, cross)
    6:  (240, 18,  190),  # 6  magenta (energy bar)
    7:  (255, 133, 27),   # 7  orange
    8:  (127, 219, 255),  # 8  azure
    9:  (135, 12,  37),   # 9  maroon  (player body)
    10: (61,  153, 112),  # 10 teal-green
    11: (255, 255, 255),  # 11 white
    12: (0,   31,  63),   # 12 navy    (player top band)
    13: (1,   255, 112),  # 13 lime
    14: (133, 20,  75),   # 14 burgundy (match-cue highlight)
    15: (1,   75,  101),  # 15 dark teal
}


def _frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, col in _COLOR_TABLE_RGB.items():
        mask = frame == idx
        rgb[mask] = col
    return rgb


def _render_cmd(args: argparse.Namespace) -> int:
    env = MiniLS20Env(args.config)
    frame = env.reset()
    print(f"[mini-env] rendered level '{env.config.name}' "
          f"(player @ {(env.player_c, env.player_r)} rot={env.player_rotation}); "
          f"frame.shape={frame.shape} dtype={frame.dtype}")

    if args.out is None:
        return 0

    try:
        from PIL import Image
    except ImportError:
        print("[mini-env] Pillow is not installed; cannot write PNG. "
              "Install with `uv pip install pillow` and re-run.",
              file=sys.stderr)
        return 2

    rgb = _frame_to_rgb(frame)
    img = Image.fromarray(rgb, mode="RGB")
    # Upscale 16x with nearest-neighbour for crisp pixel-art appearance.
    img = img.resize((rgb.shape[1] * 16, rgb.shape[0] * 16), Image.NEAREST)
    img.save(args.out)
    print(f"[mini-env] wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mini_env",
                                 description="MiniLS20 level utilities.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("render", help="Render a level config to PNG.")
    rp.add_argument("config", type=str, help="Path to a level JSON.")
    rp.add_argument("--out", type=str, default=None,
                    help="Output PNG path (omit to just sanity-check the env).")
    rp.set_defaults(func=_render_cmd)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
