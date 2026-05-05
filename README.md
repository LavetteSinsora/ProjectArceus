# Project Arceus — JEPA World Model for LS20

CSE 190 Deep RL group project. A JEPA-based world model + policy that learns to play the ARC-AGI LS20 environment.

---

## Quick start (zero setup)

This guide walks you from a fresh machine to a running dashboard in ~5 minutes.

### 1. Install `uv` (Python package manager)

`uv` manages both the Python version and all dependencies for you — no manual `pip install` or virtual environment setup needed.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installing, restart your terminal so the `uv` command is available.

> `uv` will automatically download Python 3.13 the first time you run `uv sync` — you do not need to install Python separately.

### 2. Clone the repo

```bash
git clone https://github.com/LavetteSinsora/ProjectArceus.git
cd ProjectArceus
```

### 3. Install dependencies

```bash
uv sync
```

This creates a `.venv/` folder and installs every package listed below. Takes ~1–2 minutes on first run.

### 4. Add your ARC API key

Create a file named `.env` in the repo root:

```
ARC_API_KEY=your_key_here
```

You can get a key from the [ARC Prize platform](https://arcprize.org). The key is needed for the `arc-agi` SDK to communicate with the ARC-AGI game server.

### 5. Launch the dashboard

```bash
uv run python JEPA/dashboard/server.py
```

Open **[http://localhost:8787](http://localhost:8787)** in your browser.

- Select a checkpoint from the dropdown (e.g. `step_1055000.pt`)
- Set `max_steps` (default 200)
- Click **Run Episode**

The dashboard visualizes: grid frames, patch embeddings, encoder attention maps, policy attention weights, per-patch JEPA prediction error, action probabilities, and the reasoning token over each timestep.

**No browser needed — standalone episode runner:**
```bash
uv run python JEPA/dashboard/debug_runner.py JEPA/checkpoints/step_1055000.pt
```
Prints a JSON summary of the episode to stdout.

---

## Package reference

These are installed automatically by `uv sync`. Here's what each one does:

| Package | Purpose |
|---|---|
| `torch` | PyTorch — the core deep learning framework. Used for all neural network layers (encoder, predictor, policy), training, and inference. |
| `numpy` | Numerical arrays. Used for all non-gradient computations in the dashboard (statistics, attention maps, embedding summaries). |
| `arc-agi` | Official ARC-AGI Python SDK. Provides `Arcade` and `OperationMode` to load and run ARC-AGI levels offline from `environment_files/`. |
| `arcengine` | Low-level ARC game engine that `arc-agi` builds on. Handles grid logic, action dispatch, and level loading. |
| `fastapi` | Web framework for the dashboard server. Serves the UI and exposes `/api/checkpoints` and `/api/run_episode` endpoints. |
| `uvicorn[standard]` | ASGI server that runs the FastAPI app. The `[standard]` extra adds WebSocket and HTTP/2 support. |
| `pydantic` | Data validation (pulled in by `arc-agi` and `fastapi`). Used to validate the `/api/run_episode` request body. |
| `python-dotenv` | Loads `ARC_API_KEY` from your `.env` file into the environment at startup (pulled in by `arc-agi`). |
| `flask`, `matplotlib`, `pillow`, `requests` | Pulled in by `arc-agi` for its own rendering and HTTP utilities. Not used directly by this project. |

**Dev-only** (installed with `uv sync --extra dev`):

| Package | Purpose |
|---|---|
| `pytest` | Test runner for the dashboard unit and integration tests in `JEPA/dashboard/tests/`. |
| `httpx` | HTTP client required by FastAPI's `TestClient` for integration tests. |

---

## Training from scratch

```bash
uv run python JEPA/train.py
```

Checkpoints are saved to `JEPA/checkpoints/` every 5 000 steps. Training runs on MPS (Apple Silicon), CUDA, or CPU automatically.

## Evaluation

```bash
uv run python JEPA/eval.py
```

---

## Project layout

```
JEPA/
  train.py            # JEPA + policy training loop
  encoder.py          # patch encoder (ViT-style transformer)
  predictor.py        # latent-space transition predictor
  policy.py           # cross-attention recurrent policy network
  action_embed.py     # learned action embedding
  buffer.py           # replay buffer
  env_wrapper.py      # LS20Env wrapper around arc-agi
  reward_shaping.py   # auxiliary reward utilities
  ema.py              # exponential moving average for target encoder
  config.py           # hyperparameter dataclass
  eval.py             # evaluation script
  run_online.py       # online rollout runner
  inspect_policy.py   # policy weight inspector
  dashboard/          # debug visualization server (FastAPI + HTML)
    server.py         #   FastAPI app — launch this
    debug_runner.py   #   runs one episode and returns serialized data
    static/index.html #   single-page dashboard UI
    tests/            #   pytest suite
  checkpoints/        # saved model weights (step_*.pt)
    step_1055000.pt   # latest checkpoint

replication/
  card_stochastic_goose/   # baseline replication agent

environment_files/
  ls20/               # offline ARC-AGI environment bundle
```
