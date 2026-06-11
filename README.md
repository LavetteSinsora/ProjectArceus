# Leaky RND: Regenerating Novelty in Sparse-Reward Environments

**CSE 190 Deep Reinforcement Learning — UCSD, Spring 2026**

This repository contains the full code, experiments, and writeup for *Leaky RND*, a method that prevents curiosity saturation in Random Network Distillation (RND) by introducing a small, controlled amount of forgetting into the distillation network. We test it on four ARC-AGI card-game environments and show that it consistently reaches first reward faster than standard RND.

**Paper:** [`Final Writeup/LeakyRND_Paper.pdf`](Final%20Writeup/LeakyRND_Paper.pdf)

---

## The core idea in one paragraph

Standard RND saturates: once the predictor network has seen a state, its error on that state drops to near zero and never recovers — even if the agent hasn't visited that region in thousands of steps. Leaky RND adds a small weight-decay (leak) to the **predictor** network only, so old states gradually become novel again. The result is that the curiosity signal regenerates over time instead of collapsing, which helps on hard-exploration tasks where the agent must revisit areas many times to make progress.

---

## Repository layout

```
Final Writeup/           # LaTeX source + compiled PDF of the paper
  LeakyRND_Paper.pdf     #   compiled paper (start here)
  main.tex               #   LaTeX source
  figures/               #   paper figures (PDFs + PNGs)

JEPA/experiments/
  exp_013_headline_experiment/   # Main result: steps-to-first-reward across 4 games
    exp_013_0_rnd_baseline/      #   Vanilla RND baseline
    exp_013_1_leaky_rnd/         #   Leaky RND (our method)
    exp_013_2_additive_rnd_icm/  #   Leaky RND + ICM combo
    ablation_2x2.py              #   Local driver for encoder × leak ablation (exp_017)

  exp_014_figures_and_results/   # Diagnostic experiments supporting the paper
    exp_014_0_main_result_plot/  #   Main results figure (4-game grid)
    exp_014_1_rnd_saturation/    #   Single-state saturation probe
    exp_014_2_leak_recency/      #   Leak cadence analysis
    exp_014_3_unique_states/     #   Unique-state visitation curves
    exp_014_4_mechanism_diagnosis/   # Why leak works: causal diagram
    exp_014_5_rnd_forget/        #   Controlled forget probe (std vs. leaky RND)
    exp_014_6_organic_forget/    #   Organic forget probe (real rollouts)
    exp_014_7_encoder_leak_comparison/  # Encoder choice × leak interaction

  exp_015_kaggle_submission/     # Kaggle competition submission (Go-Explore + Leaky RND)
  exp_016_organic_leaky_rnd_icm/ # Organic ICM + Leaky RND integration runs
  exp_019_exact_delta_wm/        # Exact-delta world model exploration

replication/
  card_stochastic_goose/         # Stochastic Goose baseline replication

environment_files/
  ls20/ g50t/ re86/ tu93/        # Offline ARC-AGI environment bundles
```

---

## Setup

We use [`uv`](https://docs.astral.sh/uv/) for hermetic dependency management (auto-installs Python 3.13, no manual venv needed).

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone
git clone https://github.com/LavetteSinsora/ProjectArceus.git
cd ProjectArceus

# 3. Install all dependencies
uv sync

# 4. Add ARC API key (needed to talk to the ARC-AGI game server)
echo "ARC_API_KEY=your_key_here" > .env
#  Get a key at https://arcprize.org
```

---

## Reproducing the main results (exp_013)

The headline result is **steps to first reward** on four ARC-AGI card games, comparing vanilla RND, Leaky RND, and Leaky RND + ICM.

**Run on Colab (recommended):** open `JEPA/experiments/exp_013_headline_experiment/colab_exp013.ipynb` in Google Colab. The notebook installs dependencies, runs all three agents in parallel across all four games, and saves results + figures to Google Drive.

**Run locally (single game):**
```bash
# Leaky RND on LS20 Level 1, seed 0
uv run python JEPA/experiments/exp_013_headline_experiment/exp_013_1_leaky_rnd/run.py \
    --env ls20 --level 1 --seed 0

# Vanilla RND baseline
uv run python JEPA/experiments/exp_013_headline_experiment/exp_013_0_rnd_baseline/run.py \
    --env ls20 --level 1 --seed 0
```

Results are written to `results/metrics.jsonl` under each experiment directory.

**Ablation table (encoder × leak):**
```bash
uv run python JEPA/experiments/exp_013_headline_experiment/ablation_2x2.py
```

---

## Reproducing the diagnostic figures (exp_014)

Each `exp_014_*` sub-directory is self-contained. Run its `probe.py` or `diagnose.py` script to regenerate the figure:

```bash
# Example: single-state saturation probe (Figure 3 in the paper)
uv run python JEPA/experiments/exp_014_figures_and_results/exp_014_1_rnd_saturation/single_state_recency_probe/probe.py

# Controlled forget probe (Figure 4)
uv run python JEPA/experiments/exp_014_figures_and_results/exp_014_5_rnd_forget/probe.py

# Organic forget probe (Figure 5)
uv run python JEPA/experiments/exp_014_figures_and_results/exp_014_6_organic_forget/probe.py
```

Output PNGs land in each directory's `figures/` subfolder.

---

## Environments

We use four ARC-AGI card-game environments from the [ARC Prize](https://arcprize.org) platform, accessed via the official `arc-agi` Python SDK:

| ID | Game | Levels used |
|----|------|-------------|
| `ls20` | LevelScript 20 | L1–L4 |
| `g50t` | Game 50T | L1 |
| `re86` | Reorder 86 | L1 |
| `tu93` | TU-93 | L1, L3 |

Environment bundles are in `environment_files/` and are loaded offline — no internet connection required after setup.

---

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `leak_alpha` | 0.01 | Weight-decay coefficient on the RND predictor (the core Leaky RND parameter) |
| `rnd_lr` | 1e-4 | RND predictor learning rate |
| `ppo_lr` | 2.5e-4 | PPO policy learning rate |
| `n_envs` | 8 | Parallel environments |
| `rollout_steps` | 128 | Steps per rollout before a PPO update |
| `intrinsic_coef` | auto-calibrated | η is set so the first-batch intrinsic reward matches the extrinsic scale |

Setting `leak_alpha=0` recovers standard RND exactly.

---

## Citation

If you use this code or build on this work, please cite:

```
@techreport{leakyrnd2026,
  title  = {Leaky RND: Regenerating Novelty in Sparse-Reward Environments},
  author = {[authors]},
  year   = {2026},
  institution = {UC San Diego, CSE 190},
}
```

---

## Contact

Questions about the code? Open an issue on GitHub.
