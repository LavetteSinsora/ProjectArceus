"""Encoder source resolution + loading for exp_008_3.

Every source resolves to a state-dict that fits `CNNEncoder` verbatim
(shared/model.py), so the JEPA-vs-PPO comparison differs only in initial
weights. Two checkpoint formats are handled:

    - JEPA (008_2):  bare ``encoder_state_dict`` → used directly.
    - PPO  (007_0):  full ActorCritic ``model_state_dict`` → the
      ``encoder.``-prefixed keys are extracted and the prefix stripped.

`ppo_early` additionally needs to *find* the first checkpoint that solves
simple_1_rotation; see `select_first_solve_ckpt`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import (
    EXP_007_RUNS,
    EXP_008_2_JEPA_RUNS,
    JEPA_SOURCE_GLOB,
    PPO_SOURCE_RUN_GLOB,
)


# ── path resolution ──────────────────────────────────────────────────────


def _latest(base: Path, glob_pat: str) -> Path:
    matches = sorted(base.glob(glob_pat))
    if not matches:
        raise FileNotFoundError(f"no match for {base}/{glob_pat}")
    return matches[-1]


def resolve_jepa_encoder() -> Path:
    """Latest 008_2 offline JEPA encoder trained on simple_1_rotation (1rot)."""
    return _latest(EXP_008_2_JEPA_RUNS, JEPA_SOURCE_GLOB)


def resolve_ppo_run() -> Path:
    """Latest GAE-fixed exp_007_0_naive run (the one that actually solves)."""
    return _latest(EXP_007_RUNS, PPO_SOURCE_RUN_GLOB)


def select_first_solve_ckpt(run_dir: Path, thr: float = 0.99) -> Path:
    """First saved checkpoint at/after the env first reaches `thr` eval success.

    Reads metrics.jsonl, finds the earliest ``update`` whose
    ``eval_success_rate`` crosses `thr`, then snaps to the nearest saved
    checkpoint at or after that update. Falls back to the last available
    checkpoint if `thr` is never reached (with a warning).
    """
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics.jsonl in {run_dir}")
    records = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    solve_update = None
    for r in records:
        if r.get("eval_success_rate", 0.0) >= thr:
            solve_update = int(r["update"])
            break

    ckpt_dir = run_dir / "checkpoints"
    saved = sorted(
        (int(p.stem.split("_")[1]), p) for p in ckpt_dir.glob("update_*.pt")
    )
    if not saved:
        raise FileNotFoundError(f"no update_*.pt checkpoints in {ckpt_dir}")

    if solve_update is None:
        print(f"[encoders] WARNING: {run_dir.name} never reached "
              f"eval_success_rate>={thr}; falling back to last checkpoint.")
        return saved[-1][1]

    for upd, path in saved:
        if upd >= solve_update:
            return path
    # solve happened after the last saved checkpoint — use the last one.
    return saved[-1][1]


# ── state-dict extraction ────────────────────────────────────────────────


def _extract_encoder_from_actorcritic(model_state_dict: dict) -> dict:
    """Pull the ``encoder.``-prefixed keys out of a full ActorCritic state dict."""
    prefix = "encoder."
    enc_sd = {
        k[len(prefix):]: v
        for k, v in model_state_dict.items()
        if k.startswith(prefix)
    }
    if not enc_sd:
        raise RuntimeError(
            "no 'encoder.'-prefixed keys found in model_state_dict; "
            f"available prefixes: {sorted({k.split('.')[0] for k in model_state_dict})}"
        )
    return enc_sd


def resolve_encoder(
    source: str,
    override_ckpt: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[dict | None, Path | None]:
    """Return ``(encoder_state_dict, source_ckpt_path)`` for the given source.

    `scratch` returns ``(None, None)`` — caller keeps the fresh random init.
    `override_ckpt`, if given, is used directly (JEPA format assumed unless the
    checkpoint stores a full ``model_state_dict``).
    """
    if source == "scratch":
        return None, None

    if override_ckpt is not None:
        ckpt_path = Path(override_ckpt)
    elif source == "jepa":
        ckpt_path = resolve_jepa_encoder()
    elif source == "ppo_final":
        ckpt_path = resolve_ppo_run() / "checkpoints" / "final.pt"
    elif source == "ppo_early":
        ckpt_path = select_first_solve_ckpt(resolve_ppo_run())
    else:
        raise ValueError(f"unknown source {source!r}")

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if "encoder_state_dict" in ckpt:                 # JEPA format
        enc_sd = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:                 # full ActorCritic (PPO)
        enc_sd = _extract_encoder_from_actorcritic(ckpt["model_state_dict"])
    else:
        raise RuntimeError(
            f"checkpoint {ckpt_path} has neither 'encoder_state_dict' nor "
            f"'model_state_dict'; keys: {sorted(ckpt)[:8]}"
        )
    return enc_sd, ckpt_path
