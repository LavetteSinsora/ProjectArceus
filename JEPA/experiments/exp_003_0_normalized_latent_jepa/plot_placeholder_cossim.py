"""
Plot average pairwise cosine similarity of perceiver placeholder tokens
across all checkpoints in exp_003.

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.plot_placeholder_cossim
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

CKPT_DIR = Path(__file__).parent / "checkpoints"
OUT_PATH = Path(__file__).parent / "results" / "placeholder_cossim.png"


def avg_pairwise_cossim(placeholders: torch.Tensor) -> float:
    """Mean of all unique pairwise cosine similarities among n placeholder vectors."""
    n = placeholders.shape[0]
    normed = F.normalize(placeholders.float(), dim=-1)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(normed[i].dot(normed[j]).item())
    return sum(sims) / len(sims)


def main():
    ckpts = sorted(CKPT_DIR.glob("step_*.pt"))
    steps, cossims = [], []

    for path in ckpts:
        # Parse step number; skip non-numeric suffixes like _final, _critical
        m = re.match(r"step_(\d+)(?:_\w+)?\.pt$", path.name)
        if not m:
            continue
        step = int(m.group(1))

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        enc_sd = ckpt.get("encoder", ckpt)
        placeholders = enc_sd["perceiver.placeholders"]  # (n_placeholders, d_model)

        sim = avg_pairwise_cossim(placeholders)
        steps.append(step)
        cossims.append(sim)
        print(f"  step={step:7d}  avg_pairwise_cossim={sim:.4f}  "
              f"placeholder_shape={tuple(placeholders.shape)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, cossims, marker="o", markersize=3, linewidth=1.2, color="steelblue",
            label=f"avg pairwise cos-sim (all = {cossims[0]:.4f})" if len(set(cossims)) == 1 else "avg pairwise cos-sim")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", label="cos-sim = 0 (orthogonal)")
    ax.axhline(1.0, color="red",  linewidth=0.8, linestyle="--", label="cos-sim = 1 (collapse)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Avg pairwise cosine similarity")
    ax.set_title("Exp-003: Perceiver placeholder token diversity over training\n"
                 "(low = diverse; high ≈ 1 means all placeholders collapsed to the same direction)")

    # Annotate if the placeholders are frozen (all identical values)
    if len(set(round(c, 6) for c in cossims)) == 1:
        mid_step = steps[len(steps) // 2]
        ax.annotate(
            "⚠ Placeholders FROZEN at init value\n"
            "(placeholders.grad = None throughout training;\n"
            "batch.h_queries is stored as numpy→tensor,\n"
            "severing the grad path to placeholders)",
            xy=(mid_step, cossims[0]),
            xytext=(mid_step, cossims[0] + 0.15),
            fontsize=8,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="orange"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange"),
        )

    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved figure → {OUT_PATH}")

    # Also print summary for the most recent checkpoint
    if steps:
        last_idx = steps.index(max(steps))
        print(f"\nMost recent checkpoint (step {steps[last_idx]}):")
        print(f"  avg pairwise cosine similarity = {cossims[last_idx]:.4f}")


if __name__ == "__main__":
    main()
