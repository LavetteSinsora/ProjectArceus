# exp_013_1 — Leaky RND (frozen-random φ, "A")

**The leak-only change to standard RND.** RND novelty (`½‖P(φ)−T(φ)‖²`) on a **fixed random
encoder** φ (no ICM inverse-dynamics training), with the **leaky predictor** (`θ←(1−μ)θ+μθ₀`,
μ=0.05) so novelty regenerates by recency instead of saturating to zero.

This experiment shares its implementation with **`exp_013_1b_leaky_rnd_on_icm_phi`** — A is just
that module run with `--phi-mode frozen` (no `ICM`, no φ-freeze gate). `run.py` here is a thin
wrapper that injects `--phi-mode frozen`; the config/trainer/RND code live in `exp_013_1b/`.

Run:
```bash
uv run python -m JEPA.experiments.exp_013_headline_experiment.exp_013_1_leaky_rnd.run --game ls20 --level 0
```

Contrast:
- **A (here):** leaky RND on a *frozen-random* φ — needs no controllable representation.
- **B (`exp_013_1b`):** leaky RND on the *learned ICM* φ — `--phi-mode icm` (the default there).

See `exp_014_figures_and_results/data/leaky_rnd/` for A's results and the headline staircase.
