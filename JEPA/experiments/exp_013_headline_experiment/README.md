# exp_013 — Sparse-Reward Exploration: how fast can we reach the first reward?

**This is the main experiment.** We study *exploration* in reward-sparse
environments. The single quantity we care about is:

> **`env_steps_to_first_reward`** — the total number of environment interactions
> (summed across all parallel actors) an agent needs before it receives its
> first positive *extrinsic* reward (= its first successful trajectory).

We do **not** care how good the agent becomes afterwards. This reframes the RL
problem: there is effectively **no extrinsic reward to optimise** during
learning — the question is purely *how to shape the intrinsic reward so the
agent stumbles onto a success as fast as possible.*

## Environments (4 games × early levels)

`environment_files/{ls20,tu93,re86,g50t}`. Driven through
`JEPA.shared.env_wrapper` / `claude_automate.framework.env_api.make_arc_env`.
Levels selected with the engine's `set_level(idx)` (0-indexed). Action counts:
ls20 = 4, tu93 = 4, re86 = 5, g50t = 5.

## Methods compared

1. **Random policy** — the baseline. Max-entropy policy with no inductive bias;
   the expected steps-to-first-reward under uniform-random action selection.
   Computed analytically / by Monte-Carlo per game×level (see
   `baseline_random_policy/`).
2. **ICM** (Pathak 2017) — forward-dynamics prediction error as the bonus.
   Existing impl: `JEPA/experiments/exp_011_ls20_icm/`.
3. **RND** (Burda 2018) — frozen-target distillation error as the bonus.
   Existing impl: `JEPA/experiments/exp_012_ls20_rnd/`.
4. **(ours)** — combination / improvement, TBD.

## Protocol

- **High variance**: every (method × game × level) is run over **multiple random
  seeds in parallel**; report mean ± spread of `env_steps_to_first_reward`.
- **Stop on first reward** (+ a hard step-budget cap for the censored
  "never solved within budget" case). We do not run to convergence.
- Random baseline gives the no-inductive-bias reference each method must beat.

See `baseline_random_policy/METHODOLOGY.md` for how the random reference is
computed.
