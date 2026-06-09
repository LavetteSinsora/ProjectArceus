# exp_017 — encoder × leak ablation study (paper §4.4)

Isolates the two components of the Leaky-RND harness — the **representation φ** novelty
is measured in, and the **leak μ** that regenerates it — by flipping one component at a
time around the full method. All rows run through the **same** `exp_013_1b` harness
(agent = CNN+PPO, all hypers fixed); only `--phi-mode` and `--leak` change. Metric:
env-steps to first extrinsic reward at **L1** (lower is better). L2/L3 are ∞ for every
method except tu93 L2, so the table is an L1 comparison.

## Rows

| id | --phi-mode | --leak | meaning |
|----|-----------|--------|---------|
| **R1** | `icm`    | 0.05 | **full method**: leaky RND on the ICM/IDM φ (ICM warm-up on) |
| **R2** | `icm`    | 0.0  | − leak: vanilla RND on φ |
| **R3** | `frozen` | 0.05 | − φ: random-encoder leaky RND |
| **R4** | `frozen` | 0.0  | − φ, − leak: true vanilla RND |
| **R5** | `pixel`  | 0.05 | − φ: raw-pixel RND (no encoder; the floor) |

R1/R2 use the **ICM warm-up** (20 random-policy episodes pretraining φ; warm-up env-steps
counted into the budget). All rows use the timer-mask + freeze-guard ("final code").
**All five rows must come from this same code** — the pre-existing μ=0.01 / pre-mask
archive data is NOT commensurable and is deliberately excluded (`collect.py` only ingests
runs whose `result.json` carries `phi_mode`, i.e. final-code runs).

## Files

- `ablation_2x2.py` — driver. `--rows R1..R5 --games ... --seeds ...`. Runs land in the
  exp_013_1b harness `runs/` dir.
- `collect.py` — ingest raw runs (harness `runs/` + any `--extra` unzipped Colab archive)
  into `data/<ROW>/<game>_L1_seed<n>/`. De-dups, keeps the real run, idempotent.
- `aggregate.py` — build `table.md` / `table.csv` (median steps, solved n/N per row×game).
- `data/` — the curated archive (the single source for the table).

## Workflow

```bash
# run a shard (local or Colab)
python -m JEPA.experiments.exp_017_ablation_study.ablation_2x2 --rows R1 R3 R4 --games re86 tu93 --seeds 0 1 2
# pull results in (add --extra <dir> for an unzipped Colab zip)
python -m JEPA.experiments.exp_017_ablation_study.collect --extra /tmp/colab_unzipped
# rebuild the table
python -m JEPA.experiments.exp_017_ablation_study.aggregate
```

## Status (2026-06-08)

- **R2** (vanilla RND on φ): complete, 4 games × 3 seeds. Solves all four; ls20 ~92k
  (slower than random's ~50k) → removing leak hurts.
- **R5** (raw pixels): ls20 1/3, tu93 3/3, **re86 0/3 ✗**, g50t (s1/s2 finishing). Pixels
  are the weak ruler — fails re86, flaky on ls20.
- **R1, R3, R4**: NOT YET RUN under final code → the remaining work. Per-game caps:
  ls20 200k, tu93 600k, re86 1M, g50t 300k. re86/tu93 are the long poles → run on Colab
  (CUDA ≈ 10× MPS); keep ls20/g50t local (R1 solves fast).

See `table.md` for the current numbers.
