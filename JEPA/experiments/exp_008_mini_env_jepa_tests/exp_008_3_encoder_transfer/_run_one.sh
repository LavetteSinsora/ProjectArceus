#!/usr/bin/env bash
# One transfer cell: skip if already completed (a run dir with stop.json
# exists for this source/freeze/env/seed), else train it.
# args: <source> <--freeze|--no-freeze> <env> <seed>
set -u
cd "$(git rev-parse --show-toplevel)"
M=JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer
RUNS=JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_3_encoder_transfer/ppo_runs
LOG=JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_3_encoder_transfer/logs
src=$1 frz=$2 env=$3 seed=$4
[ "$frz" = "--freeze" ] && dirtag="${src}_frozen" || dirtag="${src}_unfrozen"
cell="${dirtag}__${env}_s${seed}"

for d in "$RUNS"/008_3_transfer__${dirtag}__${env}_s${seed}_*; do
  if [ -e "$d/stop.json" ]; then
    echo "SKIP (done) $cell"
    exit 0
  fi
done

echo ">>> [$(date +%H:%M:%S)] START $cell"
uv run python -m $M.train_ppo --source "$src" "$frz" --env "$env" \
  --seed "$seed" --updates "${UPDATES:-488}" > "$LOG/par_${cell}.log" 2>&1
rc=$?
echo "<<< [$(date +%H:%M:%S)] DONE  $cell (exit $rc)"
