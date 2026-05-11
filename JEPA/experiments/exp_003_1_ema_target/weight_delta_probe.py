"""
How much do parameters actually change between consecutive checkpoints?

For each pair of consecutive checkpoints (5K apart), compute for every
named encoder parameter:
  ||theta_new - theta_old||  (absolute change)
  ||theta_new - theta_old|| / ||theta_old||  (relative change)

Group by sub-module and report.

Hypothesis: round0.cross has huge gradient norm but tiny actual weight delta
because of (a) gradient clipping, (b) Adam normalization, (c) weight decay
balance at the collapse fixed point.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

CKPT_DIR = Path(__file__).parent / "checkpoints"

GROUPS = [
    ("encoder.color_embed",     lambda n: n.startswith("color_embed")),
    ("encoder.patch_proj",      lambda n: n.startswith("patch_proj")),
    ("encoder.sa_blocks",       lambda n: n.startswith("sa_blocks")),
    ("encoder.sa_norm",         lambda n: n.startswith("sa_norm")),
    ("perceiver.placeholders",  lambda n: n == "perceiver.placeholders"),
    ("perceiver.r0.cross",      lambda n: n.startswith("perceiver.rounds.0.cross_attn")),
    ("perceiver.r0.self",       lambda n: n.startswith("perceiver.rounds.0.self_attn")),
    ("perceiver.r1.cross",      lambda n: n.startswith("perceiver.rounds.1.cross_attn")),
    ("perceiver.r1.self",       lambda n: n.startswith("perceiver.rounds.1.self_attn")),
    ("perceiver.output_norm",   lambda n: n.startswith("perceiver.output_norm")),
]


def load_encoder_state(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt["encoder"], ckpt.get("step")


def compute_deltas(prev_state, curr_state):
    rows = []
    for k in curr_state:
        if k not in prev_state:
            continue
        p_old = prev_state[k].float()
        p_new = curr_state[k].float()
        if p_old.shape != p_new.shape:
            continue
        delta = (p_new - p_old).norm().item()
        norm_old = p_old.norm().item()
        rows.append({
            "name": k, "n": p_old.numel(),
            "p_norm": norm_old, "delta": delta,
            "rel_delta": delta / max(norm_old, 1e-12),
        })
    return rows


def group(rows):
    out = {}
    for label, pred in GROUPS:
        matched = [r for r in rows if pred(r["name"])]
        if not matched:
            continue
        # aggregate as L2 (delta) and weighted relative change
        sq_delta = sum(r["delta"] ** 2 for r in matched)
        sq_pnorm = sum(r["p_norm"] ** 2 for r in matched)
        out[label] = {
            "delta_l2":  sq_delta ** 0.5,
            "p_l2":      sq_pnorm ** 0.5,
            "rel_delta": (sq_delta ** 0.5) / max(sq_pnorm ** 0.5, 1e-12),
            "n_params":  sum(r["n"] for r in matched),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=[
        "step_005000.pt:step_010000.pt",
        "step_035000.pt:step_040000.pt",
        "step_040000.pt:step_045000.pt",
        "step_055000.pt:step_060000.pt",
        "step_075000.pt:step_080000.pt",
    ])
    args = ap.parse_args()

    print(f"{'pair':30s}  "
          f"{'r0.cross Δ':>11s}  {'r0.cross rel Δ':>15s}  "
          f"{'r1.cross Δ':>11s}  {'sa_blocks Δ':>11s}  "
          f"{'placeh. Δ':>10s}")
    print("-" * 100)

    summaries = []
    for pair in args.pairs:
        a, b = pair.split(":")
        sa, _ = load_encoder_state(CKPT_DIR / a)
        sb, _ = load_encoder_state(CKPT_DIR / b)
        rows = compute_deltas(sa, sb)
        g = group(rows)
        s = {"pair": pair, "groups": g}
        summaries.append(s)
        print(f"{pair:30s}  "
              f"{g['perceiver.r0.cross']['delta_l2']:11.4f}  "
              f"{g['perceiver.r0.cross']['rel_delta']*100:13.3f}%  "
              f"{g['perceiver.r1.cross']['delta_l2']:11.4f}  "
              f"{g['encoder.sa_blocks']['delta_l2']:11.4f}  "
              f"{g['perceiver.placeholders']['delta_l2']:10.6f}")

    # detailed last pair
    print(f"\n--- detailed: {summaries[-1]['pair']} ---")
    print(f"{'group':28s}  {'#params':>10s}  {'||θ||':>10s}  "
          f"{'||Δ||':>10s}  {'rel Δ':>10s}")
    for label, _ in GROUPS:
        if label in summaries[-1]["groups"]:
            d = summaries[-1]["groups"][label]
            print(f"  {label:26s}  {d['n_params']:10d}  {d['p_l2']:10.3f}  "
                  f"{d['delta_l2']:10.4f}  {d['rel_delta']*100:9.4f}%")

    out_path = Path(__file__).parent / "results" / "weight_delta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
