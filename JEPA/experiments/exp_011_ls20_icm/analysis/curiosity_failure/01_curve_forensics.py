"""01 — Training-curve forensics from metrics.jsonl (no env / no checkpoint needed).

Contrast L1 (ICM solved) vs L2 (ICM failed) on the quantities that define the
"curiosity collapse" hypothesis: intrinsic_reward_mean, forward_error_mean,
inverse_acc, policy_entropy, and the first-reward / success timeline.

Outputs: figures/fig1_timeline.png and prints the key numbers used in captions.
"""
from __future__ import annotations
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

L1 = "JEPA/experiments/exp_011_ls20_icm/exp_011_0_icm_baseline/runs/*/metrics.jsonl"
L2 = "JEPA/experiments/exp_011_ls20_icm/exp_011_2_icm_ls20_l2/runs/*/metrics.jsonl"


def load(pattern):
    runs = []
    for f in sorted(glob.glob(pattern)):
        rows = [json.loads(l) for l in open(f)]
        runs.append((f, rows))
    return runs


def series(rows, key):
    s = np.array([r["step"] for r in rows], dtype=float)
    v = np.array([(r.get(key) if r.get(key) is not None else np.nan) for r in rows], dtype=float)
    return s, v


def main():
    l1 = load(L1)
    l2 = load(L2)
    print(f"L1 runs: {len(l1)}, L2 runs: {len(l2)}")

    # ---- key scalar facts ----
    def report(name, runs):
        print(f"\n=== {name} ===")
        for f, rows in runs:
            seed = os.path.basename(os.path.dirname(f)).split("_")[3]
            ir0 = rows[0]["intrinsic_reward_mean"]
            ir_last = rows[-1]["intrinsic_reward_mean"]
            fe0 = rows[0]["forward_error_mean"]
            fe_last = rows[-1]["forward_error_mean"]
            ia_last = rows[-1]["inverse_acc"]
            # first reward
            fr = None
            for r in rows:
                if r.get("first_reward_step") is not None:
                    fr = r["first_reward_step"]; break
            # final eval success
            succ = [r.get("success_rate") for r in rows if r.get("success_rate") is not None]
            final_succ = succ[-1] if succ else None
            last_step = rows[-1]["step"]
            # update at which ir falls below 10% of initial (1e-3)
            collapse_step = None
            for r in rows:
                if r["intrinsic_reward_mean"] < 0.1 * 0.01:
                    collapse_step = r["step"]; break
            print(f"  {seed}: ir {ir0:.4f}->{ir_last:.2e}  fe {fe0:.2f}->{fe_last:.3f}  "
                  f"inv_acc_last={ia_last:.3f}  first_reward={fr}  final_succ={final_succ}  "
                  f"ir<1e-3 @ step={collapse_step}  last_step={last_step}")

    report("L1 (solved)", l1)
    report("L2 (failed)", l2)

    # ---- figure: 2x3 grid, L1 vs L2 overlaid ----
    keys = [
        ("intrinsic_reward_mean", "mean intrinsic reward r^i", True),
        ("forward_error_mean", "forward error  ||phi_hat-phi(s')||^2", True),
        ("inverse_acc", "inverse-model action accuracy", False),
        ("policy_entropy", "policy entropy", False),
        ("intrinsic_reward_std", "intrinsic reward std", True),
        ("train_success_rate", "train success rate", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, logy) in zip(axes.flat, keys):
        for f, rows in l1:
            s, v = series(rows, key)
            ax.plot(s, v, color="tab:green", alpha=0.55, lw=1.3)
        for f, rows in l2:
            s, v = series(rows, key)
            ax.plot(s, v, color="tab:red", alpha=0.55, lw=1.3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("env steps")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    # reference line: PPO entropy bonus scale on intrinsic-reward panel
    axes.flat[0].axhline(0.01, color="k", ls=":", lw=1, label="calibration target 0.01")
    axes.flat[0].axhline(1.0, color="purple", ls="--", lw=1, label="terminal reward +1")
    axes.flat[0].legend(fontsize=8)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color="tab:green",lw=2,label="L1 (solved, 3 seeds)"),
                        Line2D([0],[0],color="tab:red",lw=2,label="L2 (failed, 3 seeds)")],
               loc="upper center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, "fig1_timeline.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
