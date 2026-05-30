#!/usr/bin/env bash
# Re-run the hard1 cells with fine-grained eval (eval_every=10), AFTER the
# main sweep (pid passed as $1) exits, to avoid MPS contention. Resolves the
# censored 51.2K-step tie among the fast unfrozen cells.
set -u
cd "$(git rev-parse --show-toplevel)"
M=JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.train_ppo
LOG=JEPA/experiments/exp_008_mini_env_jepa_tests/exp_008_3_encoder_transfer/logs
SWEEP_PID=${1:-}

if [ -n "$SWEEP_PID" ]; then
  echo ">>> waiting for main sweep (pid $SWEEP_PID) to finish..."
  while kill -0 "$SWEEP_PID" 2>/dev/null; do sleep 30; done
  echo ">>> main sweep done; starting hard1 fine-eval re-runs"
fi

run() {  # source freeze_flag
  local src=$1 frz=$2
  local tag="${src}_${frz#--}__hard1_fine"
  echo ">>> [$(date +%H:%M:%S)] START $tag"
  uv run python -m $M --source "$src" "$frz" --env hard1 --updates 488 --eval_every 10 \
    > "$LOG/${tag}.log" 2>&1
  echo "<<< [$(date +%H:%M:%S)] DONE  $tag"
}

# Unfrozen cells (the censored tie) + jepa_frozen (real sub-solve curve worth resolving).
run jepa      --no-freeze
run ppo_early --no-freeze
run ppo_final --no-freeze
run scratch   --no-freeze
run jepa      --freeze

echo ">>> rebuilding plots"
uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_3_encoder_transfer.plot \
  > "$LOG/plot_final.log" 2>&1
echo ">>> all done"
