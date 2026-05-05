# Project Arceus — JEPA World Model for LS20

CSE 190 Deep RL group project. A JEPA-based world model + policy that learns to play the ARC-AGI LS20 environment.

## Setup

```bash
uv sync
```

Requires Python ≥ 3.13 and a valid `ARC_API_KEY` in a `.env` file.

## Training

```bash
uv run python JEPA/train.py
```

Checkpoints are saved to `JEPA/checkpoints/` every 5 000 steps.

## Debug Dashboard

The dashboard runs a single episode with a chosen checkpoint and visualizes per-timestep internals: patch embeddings, encoder attention, policy attention weights, JEPA prediction error, action probabilities, and reasoning token stats.

**Launch:**
```bash
uv run python JEPA/dashboard/server.py
```

Then open [http://localhost:8787](http://localhost:8787) in your browser.

The UI lists all available checkpoints from `JEPA/checkpoints/`. Select one, set `max_steps`, and click **Run Episode** to stream the results.

**Standalone episode runner (no server):**
```bash
uv run python JEPA/dashboard/debug_runner.py JEPA/checkpoints/step_1055000.pt
```

## Evaluation

```bash
uv run python JEPA/eval.py
```

## Project layout

```
JEPA/
  train.py            # JEPA + policy training loop
  encoder.py          # patch encoder (ViT-style)
  predictor.py        # latent-space transition predictor
  policy.py           # cross-attention policy network
  action_embed.py     # action embedding
  buffer.py           # replay buffer
  env_wrapper.py      # LS20Env wrapper
  reward_shaping.py   # auxiliary reward utilities
  ema.py              # exponential moving average for target encoder
  config.py           # hyperparameter dataclass
  eval.py             # evaluation script
  run_online.py       # online rollout runner
  inspect_policy.py   # policy weight inspector
  dashboard/          # debug visualization server (FastAPI + HTML)
  checkpoints/        # saved model weights (step_*.pt)

replication/
  card_stochastic_goose/   # baseline replication agent

environment_files/
  ls20/               # offline ARC-AGI environment bundle
```
