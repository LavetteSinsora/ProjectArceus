# exp_018 — essay figures (PDFs) + the scripts that made them

Vector PDFs used in the writeup, plus archival copies of every generating script so the
essay's figures are reproducible from one place.

## Figures (`figures/*.pdf`)

| figure | generator (live) | archived copy | data source | code-gen |
|---|---|---|---|---|
| `headline_ablation_grid.pdf` | `headline_ablation.py` (runs HERE) | — | exp_017 ablation `data/` | **final** |
| `main_results_grid.pdf` | exp_014_0 `staircase.py` | `scripts/staircase__main_results_grid.py` | exp_014 `data/` archive | old (pre-mask) |
| `organic_forget_ls20_L2_combined.pdf` | exp_014_6 `plot.py` | `scripts/plot__organic_forget.py` | exp_014_6 `results/*.npz` | — |
| `distill_error_probe_ls20_L2_seed0.pdf` | exp_014_7 `plot_distill_error.py` | `scripts/plot_distill_error.py` | exp_014_7 `results/*.npz` | — |

Also `scripts/make_table.py` — generates the ablation `table.tex` (lives in exp_017).

## Two headline figures, on purpose
- **`headline_ablation_grid`** — built ENTIRELY from the final-code exp_017 ablation (L1,
  4 games): Full vs − leak vs − φ (random/pixel) vs − harness. Self-consistent; this is the
  one that matches the ablation **table**.
- **`main_results_grid`** — the original multi-level (L1→L3) staircase vs ICM/RND/random/
  goose. All **old (pre-timer-mask)** code, so internally consistent but a *different
  generation* than the ablation. Do NOT cross-reference its exact numbers with the table.

## Regenerate
```bash
# the consistent headline (final-code, from ablation data)
uv run python -m JEPA.experiments.exp_018_essay_figures.headline_ablation
# the others (run from their home dirs; --out-dir copies PDFs here)
uv run python -m JEPA.experiments.exp_014_figures_and_results.exp_014_0_main_result_plot.staircase --out-dir JEPA/experiments/exp_018_essay_figures/figures
uv run python -m JEPA.experiments.exp_014_figures_and_results.exp_014_6_organic_forget.plot --out-dir JEPA/experiments/exp_018_essay_figures/figures
uv run python -m JEPA.experiments.exp_014_figures_and_results.exp_014_7_encoder_leak_comparison.plot_distill_error --seed 0 --level 1 --out-dir JEPA/experiments/exp_018_essay_figures/figures
```
The `scripts/` copies are **archival snapshots** (paths point at their original homes); the
commands above run the live versions.
