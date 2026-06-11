"""exp_014_7 — single-panel Distillation-Error figure (probe leakage, first 5 updates).

One axes, x = environment steps (update index × rollout_env_steps, updates 0→5),
y = Distillation Error (= raw RND novelty ½‖P(φ)−T(φ)‖² on the probe states).

Three encoders × two line styles (exp_014_6 masked/unmasked convention):
    pixel  (blue),  random φ (orange),  idm φ^w (green)
    solid  = a single PROBE  state  — never counted   (labelled "..., masked")
    dashed = the mean DRIVER state  — visited/counted  (labelled "..., unmasked")

The SOLID−DASHED gap is the leak measure: the dashed (driver) line is distilled
DOWN; the solid (probe) line moves only via leak. A wide gap = the probe keeps its
novelty while a neighbour is counted = NO leak (idm). A narrow gap = counting the
driver also killed the probe = leak (pixel / random).

Reads one seed's results/encoder_leak_series_<tag>_seed<s>.npz.

Run:
    uv run python -m JEPA.experiments.exp_014_figures_and_results.\
exp_014_7_encoder_leak_comparison.plot_distill_error --seed 0 --probe 0 --updates 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# blue = pixel (vanilla-RND blue, exp_014_6), green = idm (seaborn-deep / main result),
# orange = random encoder.
COLOR = {"pixel": "#4C72B0", "idm": "#55A868", "random": "#DD8452"}
# label name per encoder (mathtext); idm = φ^w (our warm-trained φ).
NAME = {"pixel": r"pixel $\phi$", "random": r"random $\phi$", "idm": r"$\phi^w$"}
# draw order = legend order (green first, as requested)
ORDER = ["idm", "pixel", "random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20"); ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", type=int, default=0,
                    help="which probe state to draw as the SOLID (masked) line (0-indexed)")
    ap.add_argument("--probe-mean", action="store_true",
                    help="use the mean over probes for the solid line instead of one probe")
    ap.add_argument("--updates", type=int, default=5, help="x-axis range in updates (0→N)")
    ap.add_argument("--logy", action="store_true", default=True)
    ap.add_argument("--linear", dest="logy", action="store_false")
    ap.add_argument("--out-dir", default=None, help="figure output dir (default: ./figures)")
    args = ap.parse_args()

    HERE = Path(__file__).resolve().parent
    tag = f"{args.game}_L{args.level + 1}_seed{args.seed}"
    npz = HERE / "results" / f"encoder_leak_series_{tag}.npz"
    d = dict(np.load(npz, allow_pickle=True))
    names = [str(x) for x in d["encoder_names"]]
    nov = d["nov"]                                   # (n_enc, n_mon, n_updates+1)
    is_probe = d["is_probe"]
    env_per_update = int(d["rollout_env_steps"])
    probe_idx = np.where(is_probe == 1)[0]            # never-counted (masked)
    driver_idx = np.where(is_probe == 0)[0]           # visited / counted (unmasked)
    if args.probe >= len(probe_idx):
        raise SystemExit(f"--probe {args.probe} but only {len(probe_idx)} probes")
    one_probe = probe_idx[args.probe]

    n = args.updates + 1                              # measurement points 0..updates
    x = np.arange(n) * env_per_update                # env steps

    # draw idm on TOP (highest z); green dashed (idm, unmasked) is the very topmost line.
    ZBASE = {"idm": 5.0, "random": 3.0, "pixel": 2.0}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for enc in ORDER:
        e = names.index(enc); c = COLOR[enc]; z = ZBASE[enc]
        # masked = probe (never counted); unmasked = driver (visited, distilled down)
        masked = (nov[e, probe_idx, :n].mean(0) if args.probe_mean
                  else nov[e, one_probe, :n])
        unmasked = nov[e, driver_idx, :n].mean(0)
        ax.plot(x, masked, color=c, lw=2.4, ls="-", label=f"{NAME[enc]}, masked", zorder=z)
        ax.plot(x, unmasked, color=c, lw=2.4, ls="--", label=f"{NAME[enc]}, unmasked",
                zorder=z + 0.5)               # dashed above its own solid; idm dashed = top

    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("Number of Environment Steps", fontsize=12)
    ax.set_ylabel("Distillation Error", fontsize=12)
    ax.set_xlim(0, x[-1])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, frameon=False, ncol=1)
    fig.tight_layout()
    out_dir = Path(args.out_dir) if args.out_dir else HERE / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"distill_error_probe_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")   # vector PDF for the writeup
    plt.close(fig)
    print(f"[exp_014_7] wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
