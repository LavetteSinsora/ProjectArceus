# exp_010_3 — "Phantom Advantages": why a JEPA encoder fails sparse-reward RL

Self-contained investigation + write-up for the exp_010 phenomenon: on real LS20 with
terminal-only reward, a **random-init CNN encoder** lets PPO solve Level 1, but a
**JEPA-pretrained encoder** gets **0%**. This folder holds the whole story — the polished
narrative, the full report, and **every** script/datum/figure used to produce them
(including figures that did not make it into the final web page).

## Start here

| file | what it is |
|---|---|
| **`index.html`** | the polished **one-page web narrative** (no background assumed). Open this first. |
| **`report.html`** | the **full investigation report** (10 figures, all sections, hedges, limits). |
| `README.md` | this map. |

Both HTML files are **self-contained** (figures embedded as base64) — just open in a browser.

## Directory map

```
exp_010_3/
├── index.html              <- polished web narrative   (built by scripts/19)
├── report.html             <- full detailed report     (built by scripts/07)
├── README.md
├── data/                   <- every measured result (JSON / txt)
├── figures/                <- every figure (report + web)
│   ├── fig1..fig10*.png       . figures used in report.html
│   └── web_*.png              . the 3 figures used in index.html
└── scripts/                <- every generating script, numbered in run order
```

### `scripts/` — what each one produces

| # | script | produces |
|---|---|---|
| 01 | `01_phantom_advantage_probe.py` | `data/phantom.json` — V(s)-spread, 1-update entropy drop per encoder |
| 02 | `02_quiet_frozen_control.py` | falsifies the norm (H2) & freeze (H3) hypotheses |
| 03 | `03_nobaseline_rescue.py` | `data/rescue_seed0.json` — seed-0 critic-removal rescue |
| 04 | `04_parallel_sweep.py` | `data/sweep_results.txt` — seeds 1–2 robustness + L2 transfer probe |
| 05 | `05_l2_transfer_probe.py` | standalone L2 transfer probe (also embedded in 04) |
| 06 | `06_figure_curves.py` | `data/curves.json` — full per-update hero curves (Fig 2) |
| 07 | `07_make_report.py` | **`report.html`** + report figures (`figures/fig1..fig10`) |
| 08 | `08_value_histogram.py` | `figures/fig5_value_hist.png`, `data/value_hist.json` |
| 09 | `09_gradient_consistency.py` | split-half gradient-cosine / state-consistency table |
| 10 | `10_phantom_policy_shaping.py` | `data/policy_shaping.json` — early-training policy drift (Fig 6) |
| 11 | `11_value_spread_matched_norm.py` | `figures/fig7_*`, `data/value_spread_matched.json` (keystone) |
| 12 | `12_advantage_distribution.py` | `figures/fig8_*`, `data/advantage_distribution.json` |
| 13 | `13_exploit_trajectory.py` | `data/exploit_trajectory.json` — L1 reinforce-a-win + credit localization |
| 14 | `14_credit_localization.py` | `figures/fig9_*`, `data/credit_localization.json` |
| 15 | `15_l2_exploit.py` | `data/l2_exploit.json` — L2 frozen-encoder exploitation (JEPA 5/5 vs random 0/5) |
| 16 | `16_baseline_differentiation.py` | `figures/fig10_*` — encoder differentiates as reward arrives |
| 17 | `17_policy_decomposition.py` | `data/policy_decomposition.json` — entropy = log4 − KL − I(S;A) |
| 18 | `18_web_figures.py` | the 3 **clean web figures** `figures/web_{value_spread,rescue,advantage}.png` |
| 19 | `19_build_page.py` | **`index.html`** (embeds the 3 web figures) |
| – | `_run1_single.py` | helper used by 04's sweep |

## Reproduce

From the repo root (`Code Repo/`). Heavy steps (rollouts/training) use the real LS20 env;
`07`, `17`, `19` are fast (read data + plot).

```bash
R=JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts
uv run python $R/01_phantom_advantage_probe.py
uv run python $R/04_parallel_sweep.py            # seeds 1-2 + L2 probe (parallel)
uv run python $R/06_figure_curves.py             # hero curves
uv run python $R/10_phantom_policy_shaping.py
uv run python $R/11_value_spread_matched_norm.py
uv run python $R/12_advantage_distribution.py
uv run python $R/17_policy_decomposition.py
uv run python $R/07_make_report.py               # -> report.html
uv run python $R/18_web_figures.py               # -> figures/web_*.png
uv run python $R/19_build_page.py                # -> index.html
```

Paths resolve relative to each script (`DBG = parents[1]` = this folder, `EXP = parents[2]`
= the exp_010 dir, `ROOT = parents[5]` = repo root), so the folder can be moved as a unit as
long as it stays one level under `exp_010_ls20_cnn_ppo_jepa/`.

## The result in one sentence

The failure is not "directed exploration" — it is the **value critic manufacturing a
content-free direction out of an informative representation before any reward exists**;
remove the critic (Monte-Carlo returns) and the "bad" representation works fine. Conclusions
rest on the continuous, norm-controlled measurements; single game, levels 1–2, ≤3 seeds —
treat broader generalization as a hypothesis.
