#!/usr/bin/env bash
# Run the full exp_008_3 transfer sweep (14 runs) sequentially.
set -u
cd "$(git rev-parse --show-toplevel)"
M=JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo
LOG=JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_3_encoder_transfer/logs
UPDATES=${UPDATES:-488}

run() {  # source freeze_flag env
  local src=$1 frz=$2 env=$3
  local tag="${src}_${frz#--}__${env}"
  echo ">>> [$(date +%H:%M:%S)] START $tag"
  uv run python -m $M --source "$src" "$frz" --env "$env" --updates "$UPDATES" \
    > "$LOG/${tag}.log" 2>&1
  echo "<<< [$(date +%H:%M:%S)] DONE  $tag (exit $?)"
}

for env in hard1 hard2; do
  for src in jepa ppo_early ppo_final; do
    run "$src" --freeze    "$env"
    run "$src" --no-freeze  "$env"
  done
  run scratch --no-freeze "$env"
done

echo ">>> all runs complete; building plots"
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.plot \
  > "$LOG/plot.log" 2>&1
echo ">>> plot done"
