"""exp_007 dashboard — standalone FastAPI server.

Launch (from `Code Repo/`):
    uv run python JEPA/experiments/exp_007_mini_env_cnn_ppo_baseline/dashboard/server.py

Then open http://localhost:8789
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure repo root on sys.path so mini_env + JEPA packages resolve.
_REPO_ROOT = Path(__file__).resolve().parents[4]  # dashboard → exp_007 → experiments → JEPA → Code Repo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parents[1]      # exp_007_mini_env_cnn_ppo_baseline/
RUNS_DIR = EXP_DIR / "runs"
STATIC_DIR = Path(__file__).parent / "static"

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.dashboard.inspector import (  # noqa: E402
    run_episode, run_many,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.dashboard.dynamics import (  # noqa: E402
    run_debug_update,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.dashboard.stepper import (  # noqa: E402
    create_session, get_session, delete_session, list_sessions,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.dashboard import probe as probe_mod  # noqa: E402


app = FastAPI(title="exp_007 CNN+PPO Dashboard")


# ── Static + index ────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/{name}")
def serve_static(name: str):
    p = STATIC_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail=name)
    return FileResponse(p)


# ── Runs + checkpoints + metrics ──────────────────────────────────────────

def _list_run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], reverse=True)


@app.get("/api/runs")
def list_runs():
    runs = []
    for d in _list_run_dirs():
        m = d / "metrics.jsonl"
        cfg = d / "config.json"
        n_lines = 0
        last_update = None
        if m.exists():
            try:
                with m.open() as f:
                    for line in f:
                        if line.strip():
                            n_lines += 1
                            last = line
                last_update = json.loads(last).get("update")
            except Exception:
                pass
        cfg_data = {}
        if cfg.exists():
            try:
                cfg_data = json.loads(cfg.read_text())
            except Exception:
                pass
        # Checkpoints
        ckpts = []
        ckpt_dir = d / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted([p.name for p in ckpt_dir.glob("*.pt")])
        runs.append({
            "name": d.name,
            "exp_name": cfg_data.get("exp_name"),
            "reward_mode": cfg_data.get("reward_mode"),
            "total_env_steps": cfg_data.get("total_env_steps"),
            "n_metric_records": n_lines,
            "last_update": last_update,
            "checkpoints": ckpts,
            "trained_on": cfg_data.get("level_path"),
        })
    return {"runs": runs}


# ── Available level configs ───────────────────────────────────────────────

_CONFIGS_ROOT = _REPO_ROOT / "mini_env" / "configs"


def _level_label(rel: Path) -> str:
    parts = list(rel.with_suffix("").parts)
    return " / ".join(parts)


@app.get("/api/levels")
def list_levels():
    levels = []
    if _CONFIGS_ROOT.exists():
        for p in sorted(_CONFIGS_ROOT.rglob("*.json")):
            rel = p.relative_to(_REPO_ROOT)
            levels.append({
                "path": str(rel).replace("\\", "/"),
                "label": _level_label(p.relative_to(_CONFIGS_ROOT)),
            })
    return {"levels": levels}


def _scrub_nan(obj):
    """Recursively replace NaN/Inf floats with None for strict-JSON compliance."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _scrub_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_nan(v) for v in obj]
    return obj


@app.get("/api/metrics")
def get_metrics(run: str = Query(...)):
    d = RUNS_DIR / run
    m = d / "metrics.jsonl"
    if not m.exists():
        raise HTTPException(status_code=404, detail=f"no metrics for {run}")
    records = []
    with m.open() as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return JSONResponse({"run": run, "records": _scrub_nan(records)})


# ── Episode inspector ─────────────────────────────────────────────────────

class EpisodeRequest(BaseModel):
    run: str
    checkpoint: str = "final.pt"
    seed: int = 0
    greedy: bool = False
    level_path: str | None = None


class BatchRequest(BaseModel):
    run: str
    checkpoint: str = "final.pt"
    n: int = 16
    greedy: bool = False
    level_path: str | None = None


@app.post("/api/episode")
def episode(req: EpisodeRequest):
    ckpt = RUNS_DIR / req.run / "checkpoints" / req.checkpoint
    if not ckpt.exists():
        raise HTTPException(status_code=404, detail=f"no checkpoint {ckpt}")
    try:
        data = run_episode(str(ckpt), level_path=req.level_path,
                           seed=req.seed, sample=not req.greedy)
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/api/batch")
def batch(req: BatchRequest):
    ckpt = RUNS_DIR / req.run / "checkpoints" / req.checkpoint
    if not ckpt.exists():
        raise HTTPException(status_code=404, detail=f"no checkpoint {ckpt}")
    try:
        return run_many(str(ckpt), n=req.n, sample=not req.greedy,
                        level_path=req.level_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


class DebugUpdateRequest(BaseModel):
    run: str
    checkpoint: str = "final.pt"
    seed: int = 0
    epochs: int = 2
    level_path: str | None = None


@app.post("/api/debug_update")
def debug_update(req: DebugUpdateRequest):
    ckpt = RUNS_DIR / req.run / "checkpoints" / req.checkpoint
    if not ckpt.exists():
        raise HTTPException(status_code=404, detail=f"no checkpoint {ckpt}")
    try:
        return JSONResponse(_scrub_nan(
            run_debug_update(str(ckpt), seed=req.seed, epochs=req.epochs,
                             level_path=req.level_path)
        ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.get("/dynamics")
def serve_dynamics():
    return FileResponse(STATIC_DIR / "dynamics.html")


# ── Interactive stepper sessions ──────────────────────────────────────────

class StepperStartRequest(BaseModel):
    run: str
    checkpoint: str = "final.pt"
    lr: float | None = None
    level_path: str | None = None


class StepperEpisodeRequest(BaseModel):
    seed: int = 0
    greedy: bool = False


class StepperUpdateRequest(BaseModel):
    n_episodes: int = 8
    epochs: int = 2
    minibatches: int | None = None


class StepperEvalRequest(BaseModel):
    frame: list[list[int]]


@app.get("/api/stepper/sessions")
def stepper_list():
    return {"sessions": list_sessions()}


@app.post("/api/stepper/start")
def stepper_start(req: StepperStartRequest):
    ckpt = RUNS_DIR / req.run / "checkpoints" / req.checkpoint
    if not ckpt.exists():
        raise HTTPException(status_code=404, detail=f"no checkpoint {ckpt}")
    try:
        s = create_session(str(ckpt), lr=req.lr, level_path=req.level_path)
        return JSONResponse(s.summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/api/stepper/{sid}/episode")
def stepper_episode(sid: str, req: StepperEpisodeRequest):
    try:
        s = get_session(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no session {sid}")
    try:
        return JSONResponse(_scrub_nan(
            s.rollout_episode(seed=req.seed, sample=not req.greedy)
        ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/api/stepper/{sid}/update")
def stepper_update(sid: str, req: StepperUpdateRequest):
    try:
        s = get_session(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no session {sid}")
    try:
        return JSONResponse(_scrub_nan(
            s.apply_update(n_episodes=req.n_episodes, epochs=req.epochs,
                           minibatches=req.minibatches)
        ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/api/stepper/{sid}/eval_state")
def stepper_eval_state(sid: str, req: StepperEvalRequest):
    try:
        s = get_session(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no session {sid}")
    try:
        return JSONResponse(_scrub_nan(s.eval_state(req.frame)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/api/stepper/{sid}/reset")
def stepper_reset(sid: str):
    try:
        s = get_session(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no session {sid}")
    return JSONResponse(s.reset())


@app.delete("/api/stepper/{sid}")
def stepper_delete(sid: str):
    delete_session(sid)
    return {"ok": True, "session_id": sid}


# ── State probe (manual state inspector) ──────────────────────────────────

class ProbeEvalRequest(BaseModel):
    run: str
    checkpoint: str = "final.pt"
    level_path: str | None = None
    player_c: int
    player_r: int
    player_rotation: int
    step_counter: int
    denial_frames: int = 0


class ProbeStepRequest(ProbeEvalRequest):
    action: int


class ProbeResetRequest(BaseModel):
    run: str
    checkpoint: str = "final.pt"
    level_path: str | None = None


def _resolve_ckpt(run: str, checkpoint: str) -> Path:
    ckpt = RUNS_DIR / run / "checkpoints" / checkpoint
    if not ckpt.exists():
        raise HTTPException(status_code=404, detail=f"no checkpoint {ckpt}")
    return ckpt


@app.get("/probe")
def serve_probe():
    return FileResponse(STATIC_DIR / "probe.html")


@app.get("/api/probe/level")
def probe_level(path: str = Query(...)):
    try:
        return JSONResponse(probe_mod.get_level_info(path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=repr(e))


@app.post("/api/probe/eval")
def probe_eval(req: ProbeEvalRequest):
    ckpt = _resolve_ckpt(req.run, req.checkpoint)
    try:
        return JSONResponse(_scrub_nan(probe_mod.eval_state(
            str(ckpt), req.level_path,
            req.player_c, req.player_r, req.player_rotation,
            req.step_counter, req.denial_frames,
        )))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/api/probe/step")
def probe_step(req: ProbeStepRequest):
    ckpt = _resolve_ckpt(req.run, req.checkpoint)
    try:
        return JSONResponse(_scrub_nan(probe_mod.step_at(
            str(ckpt), req.level_path,
            req.player_c, req.player_r, req.player_rotation,
            req.step_counter, req.denial_frames,
            req.action,
        )))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/api/probe/reset")
def probe_reset(req: ProbeResetRequest):
    ckpt = _resolve_ckpt(req.run, req.checkpoint)
    try:
        return JSONResponse(_scrub_nan(probe_mod.reset_state(str(ckpt), req.level_path)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


def main():
    uvicorn.run(app, host="127.0.0.1", port=8789, reload=False, log_level="info")


if __name__ == "__main__":
    main()
