#!/bin/zsh
# Sequential driver (ONE python process at a time -> no MPS contention).
# Waits for the already-running zero control, then runs the rest.
set -e
cd "/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo"
P=JEPA/experiments/exp_013_sparse_exploration/probes/results
export PYTHONUNBUFFERED=1
RMC="JEPA.experiments.exp_013_sparse_exploration.probes.reward_mode_control"

# 1) wait for the in-flight zero run
while [ ! -f $P/h1_zero_ce01_s0.json ]; do sleep 5; done
echo "=== zero done; starting noise ==="

# 2) H1b noise control (mean-zero N(0,0.1) reward)
uv run python -m $RMC --mode noise --c_entropy 0.01 --updates 60 --seed 0 \
    --noise_sigma 0.1 --out $P/h1_noise_ce01_s0.json > $P/log_h1_noise_ce01_s0.txt 2>&1
echo "=== noise done; starting novelty baseline ==="

# 3) novelty baseline replica @ ce=0.01 (the collapsing config) for the H3 table + H1 contrast
uv run python -m $RMC --mode novelty --c_entropy 0.01 --updates 60 --seed 0 \
    --out $P/h1_novelty_ce01_s0.json > $P/log_h1_novelty_ce01_s0.txt 2>&1
echo "=== novelty done; starting H5 sweep ==="

# 4) H5 c_entropy sweep (0.01 already covered by #3; run 0.02, 0.05, 0.1)
for ce in 0.02 0.05 0.1; do
  uv run python -m $RMC --mode novelty --c_entropy $ce --updates 50 --seed 0 \
      --out $P/h5_ce${ce}_s0.json > $P/log_h5_ce${ce}_s0.txt 2>&1
  echo "=== H5 ce=$ce done ==="
done

# 5) H6 episodic vs non-episodic @ ce=0.01 (non-episodic already = #3; run episodic)
uv run python -m $RMC --mode novelty --c_entropy 0.01 --updates 60 --seed 0 --episodic \
    --out $P/h6_episodic_ce01_s0.json > $P/log_h6_episodic_ce01_s0.txt 2>&1
echo "=== H6 episodic done ==="

echo "ALL_SEQUENTIAL_DONE"
