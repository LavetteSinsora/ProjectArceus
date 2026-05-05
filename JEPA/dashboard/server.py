"""
JEPA Debug Dashboard — FastAPI server.

Launch:
    cd "Code Repo"
    uv run python JEPA/dashboard/server.py

Then open http://localhost:8787
"""

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Put JEPA/ on the path so debug_runner can import from it
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.debug_runner import run_debug_episode

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="JEPA Debug Dashboard")


class RunEpisodeRequest(BaseModel):
    checkpoint: str        # filename only, e.g. "step_235000.pt"
    max_steps: int = 200


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/checkpoints")
def list_checkpoints():
    if not CKPT_DIR.exists():
        return {"checkpoints": []}
    ckpts = sorted(CKPT_DIR.glob("step_*.pt"), reverse=True)
    return {"checkpoints": [p.name for p in ckpts]}


@app.post("/api/run_episode")
def run_episode(req: RunEpisodeRequest):
    ckpt_path = CKPT_DIR / req.checkpoint
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {req.checkpoint}")
    try:
        data = run_debug_episode(str(ckpt_path), max_steps=req.max_steps)
        return JSONResponse(content=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8787, reload=False)
