#!/usr/bin/env bash
# Parallel multi-seed sweep: PAR cells at a time via xargs -P.
# Skips already-completed cells (see _run_one.sh), so it resumes the
# partially-finished sequential sweep without redoing seed 0.
set -u
cd "$(git rev-parse --show-toplevel)"
EXP=JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_3_encoder_transfer
M=JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer
PAR=${PAR:-6}
SEEDS=${SEEDS:-"0 1 2"}

CELLS=(
  "jepa --freeze"      "jepa --no-freeze"
  "ppo_early --freeze" "ppo_early --no-freeze"
  "ppo_final --freeze" "ppo_final --no-freeze"
  "scratch --no-freeze"
)

jobs="$EXP/logs/_par_jobs.txt"
: > "$jobs"
for seed in $SEEDS; do
  for env in hard1 hard2; do
    for cell in "${CELLS[@]}"; do
      echo "$cell $env $seed" >> "$jobs"
    done
  done
done
echo ">>> $(wc -l < "$jobs") cells queued; running $PAR-wide"

xargs -P "$PAR" -L1 bash "$EXP/_run_one.sh" < "$jobs"

echo ">>> all cells done; building aggregate + figure"
uv run python -m $M.aggregate_seeds --seeds $SEEDS > "$EXP/logs/par_aggregate_final.log" 2>&1
echo ">>> parallel multiseed complete"
