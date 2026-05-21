# exp_005_0_sparse_goal

**Algorithm:** Dreamer V3 (XS) on LS20 Level 1.
**Reward:** `r_t = 1` if `env.level_completed` else `0`. No shaping.
**Exploration:** P2E actor for first 100K env steps → task actor thereafter.

## What this tests

The honest DV3 replication. If the paper's claim that DV3 + sparse reward
generalises to small puzzle environments is true, this should solve L1 within
the 500K-env-step / 2-h budget. If it stalls at 0%, the bottleneck is
exploration, not algorithm — and 005_3 (pure P2E) tells us whether P2E alone
is strong enough to even visit the goal.

## Hyperparameters

All defaults from `shared/config_base.py` (XS). See plan §3.

## Logged metrics

`metrics.jsonl` records per-`log_every` step:
- `L_wm`, `L_pred`, `L_dyn`, `L_rep`, `L_ensemble`
- `L_actor`, `L_critic`, `L_actor_p2e`, `L_critic_p2e`
- `policy_entropy` (task actor)
- `ret_scale_task` — running EMA of Per95 - Per5
- `recent20_compl` — fraction of last 20 episodes that completed L1
- `env_step`, `grad_step`, `env_sps`, `buffer_size`

## How to run

```
cd "Code Repo"
uv run python -m JEPA.experiments.exp_005_dreamer_v3.exp_005_0_sparse_goal.train
uv run python -m JEPA.experiments.exp_005_dreamer_v3.exp_005_0_sparse_goal.eval --checkpoint runs/run_*/checkpoints/final.pt
```
