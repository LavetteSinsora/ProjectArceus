# claude_automate — a generalizable RL training framework for ARC-AGI games

A self-contained reinforcement-learning framework that solves ARC-AGI
**LS20 Levels 1–4 and TU93 Level 1** (100% each) without any reward signal
tied to a specific level or game. Everything here lives under
`claude_automate/`; no file outside this directory is modified (the shared
`JEPA/shared/env_wrapper.py` and the `environment_files/` bundle are
*imported*, never edited).

Solved (all 100%, 30/30 greedy eval): LS20 L1 · L2 · L3 · L4 · TU93 L1 —
five levels across two games, one unchanged framework.

## Why "generalizable"

The hard constraint driving every design decision: **no level-specific reward
shaping**. We never reward "move up", "approach pixel (34, 10)", or "touch the
tile at (19, 30)". Those would solve LS20 L1 and nothing else.

Instead the reward is built only from signals that exist in *every* ARC-AGI
game:

| Signal | Generalizable rationale |
|---|---|
| `+W_complete` on `level_completed` | every level has a completion flag |
| `-W_step` per action | every game rewards solving in fewer steps |
| `-W_stuck` when the (UI-masked) frame did not change | an action with no visible effect is wasted in any game |
| `+W_novel · 1/√count(state)` intrinsic bonus | count-based novelty drives exploration in any sparse-reward task |

The intrinsic bonus is what actually cracks a sparse-reward level: it turns
"systematically explore until you stumble onto the goal" into a dense gradient,
without telling the agent *where* the goal is.

## Two solving paths

The framework offers two generalizable, level-agnostic ways to solve a level.

**1. PPO + count-based exploration (`train.py`)** — clipped actor-critic with
GAE over a small CNN encoder; exploration driven by count-based novelty
(global `ExactFrameCounter` + episodic counts). On-policy RL. Solves *shallow*
sparse-reward levels. **LS20 Level 1: 100%, 13-step solve.**

**2. Go-Explore + recurrent distillation (`solve.py`)** — for *deep*
exploration puzzles where a neural policy "detaches" (covers the easy region,
then has no gradient to the hard frontier). Go-Explore (Ecoffet et al. 2021)
archives distinct states, deterministically *returns* to them, and explores
onward — defeating detachment. The discovered solution trajectory is then
behavior-cloned into a **recurrent (GRU) policy**, whose memory lets it
reproduce a plan that revisits states. **LS20 Level 2: 100%, 30/30 episodes.**

Both are generalizable: only `--level` / `--game-id` changes between levels;
no reward term or search heuristic encodes level geometry.

> **Honest caveat.** The ARC OFFLINE environment is *deterministic*. A greedy
> distilled policy therefore produces one fixed episode — "30/30 eval" means
> one found solution replayed 30×, not 30 independent successes. The level is
> genuinely completed; the eval count is cosmetic. `solve.py` is best read as
> **search + memorize**: Go-Explore finds a completing trajectory and the
> recurrent net memorizes it. It does not transfer between levels.

## Attempted transfer (exp 006) — partial result

`world_model.py` + `pretrain_wm.py` train a shared frame-prediction model
across levels, intended to enable few-shot solving by planning in imagination
(`solve_wm.py`). **Outcome: the planning route does not work.** The model
transfers *aggregate* dynamics well (~99% next-frame pixel accuracy on unseen
levels) but reaches **0% exact-frame accuracy** — not exact enough for the
exact-replay search planning needs. `solve_wm.py` is kept but marked
not-viable. See `RESEARCH_LOG.md` exp 006 for the honest write-up.

## Layout

```
claude_automate/
  README.md            this file
  RESEARCH_LOG.md      proposals, experiment results, findings (running log)
  probe_env.py         read-only environment reconnaissance
  framework/
    config.py          hyperparameter dataclass
    env_api.py         env construction, frame preprocessing, level targeting
    networks.py        CNN encoder; feedforward + recurrent actor-critic
    exploration.py     count-based novelty (exact / SimHash counters)
    rewards.py         generalizable reward composition (global+episodic)
    ppo.py             PPO rollout collection + update
    go_explore.py      Go-Explore structured-exploration search
    distill.py         behavior-cloning distillation (feedforward + recurrent)
    world_model.py     shared frame-prediction model + ModelEnv (exp 006)
    wm_train.py        world-model transition collection + training (exp 006)
  train.py             PPO training entrypoint  (path 1)
  eval.py              greedy/stochastic evaluation of a PPO checkpoint
  solve.py             Go-Explore search + distillation entrypoint  (path 2)
  pretrain_wm.py       train the shared world model (exp 006)
  solve_wm.py          model-based solve — NOT viable with current WM (exp 006)
  tests/               pytest unit tests (27) for every framework module
  experiments/         run outputs (metrics, checkpoints) — created at runtime
```

## Run

```bash
cd "Code Repo"
uv run python claude_automate/probe_env.py            # sanity check the env
uv run python -m pytest claude_automate/tests -q      # 27 unit tests

# Path 1 — PPO (LS20 Level 1)
uv run python claude_automate/train.py                # solves L1 in ~3 min
uv run python claude_automate/eval.py --checkpoint <run>/best.pt

# Path 2 — Go-Explore (LS20 Level 2, or any level via --level)
uv run python claude_automate/solve.py --level 1      # solves L2
uv run python claude_automate/solve.py --level 0      # also solves L1
```
