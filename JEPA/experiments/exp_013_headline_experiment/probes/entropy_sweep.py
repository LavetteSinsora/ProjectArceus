"""H5 (c_entropy threshold) + H6 (episodic vs non-episodic) sweeps.

Reuses run() from reward_mode_control.py so the loop is byte-identical to the probe
used for H1 (only the knob under study changes). All runs use the REAL novelty reward.

H5: c_entropy in {0.01, 0.02, 0.05, 0.1}, 1 seed, ~50 updates (~100k steps).
H6: intrinsic_episodic True vs False at the collapsing c_entropy=0.01.

Run: uv run python -m JEPA.experiments.exp_013_headline_experiment.probes.entropy_sweep --which h5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from JEPA.experiments.exp_013_headline_experiment.probes.reward_mode_control import run

P = Path(__file__).resolve().parent / "results"


def h5(updates: int, seed: int):
    out = []
    for ce in (0.01, 0.02, 0.05, 0.1):
        s = run("novelty", ce, updates, seed, leak=0.01, intrinsic_episodic=False,
                noise_sigma=0.1, out_path=P / f"h5_ce{ce}_s{seed}.json")
        out.append(dict(c_entropy=ce, ent_min=s["ent_min"], ent_last=s["ent_last"],
                        ent_lastq_mean=s["ent_lastq_mean"], freeze_upd=s["freeze_upd"]))
    (P / f"h5_summary_s{seed}.json").write_text(json.dumps(out, indent=2))
    print("\nH5 SUMMARY:")
    for r in out:
        print(f"  c_entropy={r['c_entropy']:<5} ent_min={r['ent_min']:.3f} "
              f"ent_last={r['ent_last']:.3f} lastQ={r['ent_lastq_mean']:.3f}")


def h6(updates: int, seed: int):
    out = []
    for epi in (False, True):
        s = run("novelty", 0.01, updates, seed, leak=0.01, intrinsic_episodic=epi,
                noise_sigma=0.1, out_path=P / f"h6_epi{epi}_s{seed}.json")
        out.append(dict(episodic=epi, ent_min=s["ent_min"], ent_last=s["ent_last"],
                        ent_lastq_mean=s["ent_lastq_mean"], freeze_upd=s["freeze_upd"]))
    (P / f"h6_summary_s{seed}.json").write_text(json.dumps(out, indent=2))
    print("\nH6 SUMMARY:")
    for r in out:
        print(f"  episodic={str(r['episodic']):<5} ent_min={r['ent_min']:.3f} "
              f"ent_last={r['ent_last']:.3f} lastQ={r['ent_lastq_mean']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["h5", "h6"])
    ap.add_argument("--updates", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.which == "h5":
        h5(args.updates, args.seed)
    else:
        h6(args.updates, args.seed)
