#!/usr/bin/env bash
# Multi-seed exp_008_3 sweep: SEEDS x 2 envs x 7 cells = 42 runs.
# Seed-outer so each finished seed tightens error bars across the whole matrix.
# eval_every=10 + early-stop are config defaults; unfrozen solvers bail fast.
set -u
cd "$(git rev-parse --show-toplevel)"
M=JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer
LOG=JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_3_encoder_transfer/logs
SEEDS=${SEEDS:-"0 1 2"}
UPDATES=${UPDATES:-488}

# cell = "source freeze_flag"
CELLS=(
  "jepa --freeze"      "jepa --no-freeze"
  "ppo_early --freeze" "ppo_early --no-freeze"
  "ppo_final --freeze" "ppo_final --no-freeze"
  "scratch --no-freeze"
)

run() {  # source freeze_flag env seed
  local src=$1 frz=$2 env=$3 seed=$4
  local tag="${src}_${frz#--}__${env}_s${seed}"
  echo ">>> [$(date +%H:%M:%S)] START $tag"
  uv run python -m $M.train_ppo --source "$src" "$frz" --env "$env" \
    --seed "$seed" --updates "$UPDATES" > "$LOG/ms_${tag}.log" 2>&1
  echo "<<< [$(date +%H:%M:%S)] DONE  $tag (exit $?)"
}

for seed in $SEEDS; do
  for env in hard1 hard2; do
    for cell in "${CELLS[@]}"; do
      run $cell "$env" "$seed"
    done
  done
  echo ">>> seed $seed complete; refreshing aggregate"
  uv run python -m $M.aggregate_seeds --seeds $SEEDS --no_fig \
    > "$LOG/ms_aggregate_seed${seed}.log" 2>&1
done

echo ">>> all seeds done; building final multi-seed aggregate + figure"
uv run python -m $M.aggregate_seeds --seeds $SEEDS > "$LOG/ms_aggregate_final.log" 2>&1
echo ">>> multi-seed sweep complete"
