# exp_005_dreamer_v3 — Dreamer V3 replication on LS20 Level 1

PyTorch replication of Hafner et al. 2023 ([arXiv:2301.04104](https://arxiv.org/abs/2301.04104)).
Algorithm core (RSSM + actor-critic + Plan2Explore) lives in `shared/`; each
sub-experiment varies one knob (reward source / exploration policy).

## Sub-experiment matrix

| Sub-exp | Extrinsic reward | Acting policy in env | Purpose |
|---|---|---|---|
| `exp_005_0_sparse_goal/`        | `+1` on level complete | π_e (first 100K) → π_t | Canonical DV3 baseline. |
| `exp_005_1_step_penalty/`       | `+1 − 0.01/step`       | π_e → π_t              | Tests reward-scale robustness with mild dense shaping. |
| `exp_005_2_curiosity/`          | `+1` on level complete | π_e → π_t              | Future hook for intrinsic added to task λ-returns. |
| `exp_005_3_plan2explore_only/`  | 0                       | π_e (always)           | Pure unsupervised exploration. |

`shared/` is identical across sub-exps. All four train and eval the *same*
world model (RSSM + ensemble) and the *same* P2E exploration actor π_e; they
diverge only in (a) what extrinsic reward gets stored in the buffer, and
(b) which actor takes env actions.

## Run

```
cd "Code Repo"

# Sub-exp A — canonical DV3
uv run python -m JEPA.experiments.exp_005_dreamer_v3.exp_005_0_sparse_goal.train

# Sub-exp D — pure P2E exploration sanity test
uv run python -m JEPA.experiments.exp_005_dreamer_v3.exp_005_3_plan2explore_only.train

# Smoke test (no training; prints shapes + per-step latency)
uv run python -m JEPA.experiments.exp_005_dreamer_v3.shared.debug_runner

# Eval
uv run python -m JEPA.experiments.exp_005_dreamer_v3.exp_005_0_sparse_goal.eval \
    --checkpoint JEPA/experiments/exp_005_dreamer_v3/runs/run_*/checkpoints/final.pt
```

Run artifacts (training.log, metrics.jsonl, checkpoints/) land in
`shared/runs/<timestamp>/` by default. Sub-exp `config.run_name` is reflected
in the log filename for cross-comparison.

## Hyperparameters — nano config (sized for Apple M3 Pro MPS, ~2.6 h)

See [`shared/config_base.py`](shared/config_base.py). Key choices vs the paper:

| Knob | Paper XS | Nano (this repo) |
|---|---:|---:|
| GRU `deter` | 512 | **256** |
| Categorical `n_groups × n_classes` | 32 × 32 | **16 × 16** |
| Hidden width | 512 | **256** |
| CNN depth | 32 | **24** |
| Encoder embed_dim | 1024 | **512** |
| Twohot bins | 255 | **127** |
| Batch length | 64 | **32** |
| Train ratio (transitions/env step) | 512 | **32** |
| Replay capacity | 1e6 | **250 000** |
| World-model params | ~24 M | **~6.1 M** |

All DV3 robustness tricks are kept (symlog, twohot, KL balancing + 1-nat free
bits, unimix 0.01, percentile return scaling, critic EMA 0.98). Deviations:
standard PyTorch GRU instead of Block GRU; Adam instead of LaProp.

**Measured latency** on Apple M3 Pro (MPS): **~300 ms / gradient step** for
the nano config (B=16, T=32, H=15). Projected 500K env steps × train_ratio=32
→ 31250 updates → **~2.6 h**.

If a strict 2 h is required, reduce `max_env_steps` to 380K (~2.0 h) or
`train_ratio` to 24 (~2.0 h but fewer gradient steps per env step).

[`shared/debug_runner.py`](shared/debug_runner.py) measures actual per-step
latency and prints an extrapolated ETA before any long run.

## Plan reference

Full design rationale: [`/Users/chrishe/.claude/plans/please-research-and-look-velvety-aurora.md`](/Users/chrishe/.claude/plans/please-research-and-look-velvety-aurora.md).
