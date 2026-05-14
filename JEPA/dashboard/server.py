"""
JEPA Debug Dashboard — FastAPI server.

Launch:
    cd "Code Repo"
    uv run python JEPA/dashboard/server.py

Then open http://localhost:8787
"""

import importlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure repo root is importable so JEPA package resolves
_repo_root = Path(__file__).parent.parent.parent  # dashboard → JEPA → Code Repo
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _get_run_debug_episode():
    """Always return the freshest run_debug_episode by reloading the module."""
    mod_name = "JEPA.dashboard.debug_runner"
    if mod_name in sys.modules:
        mod = importlib.reload(sys.modules[mod_name])
    else:
        mod = importlib.import_module(mod_name)
    return mod.run_debug_episode

JEPA_ROOT       = Path(__file__).parent.parent
EXPERIMENTS_DIR = JEPA_ROOT / "experiments"
STATIC_DIR      = Path(__file__).parent / "static"

app = FastAPI(title="JEPA Debug Dashboard")

# ── Training process state ────────────────────────────────────────────────────
_train_proc: subprocess.Popen | None = None
_train_experiment: str = ""
_train_run_dir: str = ""


class RunEpisodeRequest(BaseModel):
    experiment: str        # e.g. "exp_001_vit_jepa_baseline"
    checkpoint: str        # filename only, e.g. "step_235000.pt"
    max_steps: int = 200
    env: str | None = None  # short env name (e.g. "ls20", "tu93"); default = experiment's first env


class TrainingStartRequest(BaseModel):
    experiment: str = "exp_003_0_normalized_latent_jepa"
    resume_checkpoint: str | None = None   # filename under checkpoints/, or None for fresh


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/exp/{experiment}/panel.js")
def serve_panel_js(experiment: str):
    path = EXPERIMENTS_DIR / experiment / "panel.js"
    if not path.exists():
        # Fall back to a sibling experiment's panel.js if the package declares
        # `PANEL_EXPERIMENT = "<other_experiment>"` in its __init__.py. This lets
        # follow-up experiments that share the same architecture (e.g. exp_003_1
        # extending exp_003_0) reuse the upstream dashboard panels without
        # duplicating ~1k lines of JS.
        try:
            pkg = importlib.import_module(f"JEPA.experiments.{experiment}")
            alias = getattr(pkg, "PANEL_EXPERIMENT", None)
        except Exception:
            alias = None
        if alias:
            alias_path = EXPERIMENTS_DIR / alias / "panel.js"
            if alias_path.exists():
                return FileResponse(alias_path, media_type="application/javascript")
        raise HTTPException(status_code=404, detail=f"No panel.js for {experiment}")
    return FileResponse(path, media_type="application/javascript")


@app.get("/api/experiments")
def list_experiments():
    """List all experiment directories under JEPA/experiments/."""
    if not EXPERIMENTS_DIR.exists():
        return {"experiments": []}
    exps = sorted(
        d.name for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("__")
    )
    return {"experiments": exps}


@app.get("/api/checkpoints")
def list_checkpoints(experiment: str):
    """List checkpoints for a given experiment (newest first)."""
    ckpt_dir = EXPERIMENTS_DIR / experiment / "checkpoints"
    if not ckpt_dir.exists():
        return {"checkpoints": []}
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), reverse=True)
    return {"checkpoints": [p.name for p in ckpts]}


def _experiment_trained_envs(experiment: str) -> tuple[list[str], str]:
    """
    Inspect the experiment's Config and return (trained_envs, default_env).

    Resolution order:
      1. `cfg.env_names` exists and is non-empty   → multi-env (e.g. exp_004).
      2. `cfg.game_id` exists                       → single-env (exp_001..exp_003_*).
      3. Fallback                                   → ["ls20"].
    """
    try:
        cfg_mod = importlib.import_module(f"JEPA.experiments.{experiment}.config")
        cfg = cfg_mod.Config()
    except Exception:
        return (["ls20"], "ls20")

    env_names = getattr(cfg, "env_names", None)
    if env_names:
        envs = [str(e) for e in env_names]
        return (envs, envs[0])

    game_id = getattr(cfg, "game_id", None)
    if game_id:
        from JEPA.shared.env_wrapper import short_env_name
        short = short_env_name(game_id)
        return ([short], short)

    return (["ls20"], "ls20")


@app.get("/api/experiment_envs")
def list_experiment_envs(experiment: str):
    """
    Report which envs an experiment was trained on, plus all known envs.

    Response:
      {
        "experiment": "<name>",
        "default": "ls20",
        "envs": [
          {"name": "ls20", "trained": true},
          {"name": "tu93", "trained": false},
          ...
        ]
      }

    UI uses `trained` to flag a cross-env play attempt and show a warning banner.
    """
    from JEPA.shared.env_wrapper import SHORT_TO_FULL_GAME_ID

    trained, default = _experiment_trained_envs(experiment)
    trained_set = set(trained)
    envs = []
    # Stable order: trained envs first (in their declared order), then the rest.
    seen: set = set()
    for e in trained:
        envs.append({"name": e, "trained": True})
        seen.add(e)
    for e in sorted(SHORT_TO_FULL_GAME_ID):
        if e in seen:
            continue
        envs.append({"name": e, "trained": False})
    return {"experiment": experiment, "default": default, "envs": envs}


@app.post("/api/run_episode")
def run_episode(req: RunEpisodeRequest):
    ckpt_path = EXPERIMENTS_DIR / req.experiment / "checkpoints" / req.checkpoint
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {req.experiment}/{req.checkpoint}",
        )
    # Default the env if not provided.
    env_name = req.env
    if env_name is None:
        _trained, env_name = _experiment_trained_envs(req.experiment)
    try:
        run_debug_episode = _get_run_debug_episode()
        data = run_debug_episode(
            str(ckpt_path),
            experiment=req.experiment,
            env=env_name,
            max_steps=req.max_steps,
        )
        return JSONResponse(content=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/training")
def serve_training_ui():
    return FileResponse(STATIC_DIR / "training.html")


@app.get("/api/training/runs")
def list_training_runs(experiment: str = Query("exp_003_0_normalized_latent_jepa")):
    """List all run directories for an experiment, newest first."""
    runs_dir = EXPERIMENTS_DIR / experiment / "runs"
    if not runs_dir.exists():
        return {"runs": []}
    runs = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        m_path = d / "metrics.jsonl"
        last_step = None
        if m_path.exists():
            # Read last line efficiently
            try:
                with open(m_path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 512))
                    tail = f.read().decode("utf-8", errors="replace")
                last_line = [l for l in tail.splitlines() if l.strip()][-1]
                last_step = json.loads(last_line).get("step")
            except Exception:
                pass
        runs.append({
            "name": d.name,
            "has_metrics": m_path.exists(),
            "last_step": last_step,
        })
    return {"experiment": experiment, "runs": runs}


@app.get("/api/training/metrics")
def get_training_metrics(
    experiment: str = Query("exp_003_0_normalized_latent_jepa"),
    run: str = Query(...),
    since_step: int = Query(0),
):
    """Return all metrics records from a run's metrics.jsonl, optionally filtered by step."""
    m_path = EXPERIMENTS_DIR / experiment / "runs" / run / "metrics.jsonl"
    if not m_path.exists():
        raise HTTPException(status_code=404, detail=f"No metrics.jsonl for {experiment}/{run}")
    records = []
    try:
        with open(m_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("step", 0) > since_step:
                    records.append(rec)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"experiment": experiment, "run": run, "records": records}


@app.post("/api/training/start")
def start_training(req: TrainingStartRequest):
    global _train_proc, _train_experiment, _train_run_dir
    if _train_proc is not None and _train_proc.poll() is None:
        raise HTTPException(status_code=409, detail="Training already running")
    module = f"JEPA.experiments.{req.experiment}.train"
    cmd = [sys.executable, "-m", module]
    if req.resume_checkpoint:
        ckpt_path = EXPERIMENTS_DIR / req.experiment / "checkpoints" / req.resume_checkpoint
        cmd += ["--resume", str(ckpt_path)]
    _train_proc = subprocess.Popen(
        cmd,
        cwd=str(_repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _train_experiment = req.experiment
    return {"pid": _train_proc.pid, "experiment": req.experiment, "cmd": " ".join(cmd)}


@app.post("/api/training/stop")
def stop_training():
    global _train_proc
    if _train_proc is None or _train_proc.poll() is not None:
        return {"status": "not_running"}
    try:
        os.kill(_train_proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _train_proc = None
    return {"status": "stopped"}


@app.get("/api/training/status")
def training_status():
    if _train_proc is None:
        return {"running": False}
    rc = _train_proc.poll()
    if rc is not None:
        return {"running": False, "exit_code": rc, "experiment": _train_experiment}
    return {"running": True, "pid": _train_proc.pid, "experiment": _train_experiment}


def main():
    uvicorn.run(app, host="127.0.0.1", port=8787, reload=False)


if __name__ == "__main__":
    main()
