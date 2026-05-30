"""Collect uniform-random-agent transitions for offline JEPA pretraining.

Usage:
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.collect --env 1rot
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.collect --env 2rot
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.collect --env 1rot --smoke

Output: data/random_<env_tag>_seed<seed>.pt — a torch.save'd dict of
    obs       (M, 32, 32) uint8
    actions   (M,)        int64
    next_obs  (M, 32, 32) uint8
    dones     (M,)        bool      (True ⇒ next_obs is a reset frame)
    meta      dict with env_tag, level_path, seed, n_transitions, ...

Transitions where `done=True` are kept; downstream training filters them out
with `valid = ~dones`. Matches the rule used by exp_008_1 and exp_007_4.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.vec_env import VecMiniEnv

from .config import (
    CollectConfig,
    DATA_DIR,
    ENV_TAG_TO_LEVEL,
    level_path_for,
)


def buffer_path(env_tag: str, seed: int) -> Path:
    return DATA_DIR / f"random_{env_tag}_seed{seed}.pt"


def collect_random_transitions(cfg: CollectConfig) -> dict:
    """Run uniform-random actions until `cfg.n_transitions` valid transitions
    accumulate. Returns the dict written to disk."""
    level = level_path_for(cfg.env_tag)
    envs = VecMiniEnv(level, n_envs=cfg.n_envs, seed=cfg.seed)
    envs.reset_all()

    rng = np.random.default_rng(cfg.seed)
    obs = envs.current_obs()  # (N, 32, 32) uint8

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    nxt_buf: list[np.ndarray] = []
    don_buf: list[np.ndarray] = []

    n_valid = 0
    t0 = time.time()
    while n_valid < cfg.n_transitions:
        actions = rng.integers(0, 4, size=cfg.n_envs, dtype=np.int64)
        next_obs, _rew, dones, _infos = envs.step(actions)

        obs_buf.append(obs)
        act_buf.append(actions.astype(np.int64))
        nxt_buf.append(next_obs)
        don_buf.append(dones.astype(bool))

        n_valid += int((~dones).sum())
        envs.drain_completed_episodes()  # avoid unbounded growth

        obs = next_obs

        if len(obs_buf) % 200 == 0:
            print(
                f"[collect/{cfg.env_tag}] step={len(obs_buf) * cfg.n_envs:7d} "
                f"valid={n_valid:7d}/{cfg.n_transitions} "
                f"({time.time() - t0:.1f}s)"
            )

    out_obs = np.concatenate(obs_buf, axis=0)
    out_act = np.concatenate(act_buf, axis=0)
    out_nxt = np.concatenate(nxt_buf, axis=0)
    out_don = np.concatenate(don_buf, axis=0)

    return {
        "obs": torch.from_numpy(out_obs),         # (M, 32, 32) uint8
        "actions": torch.from_numpy(out_act),     # (M,)        int64
        "next_obs": torch.from_numpy(out_nxt),    # (M, 32, 32) uint8
        "dones": torch.from_numpy(out_don),       # (M,)        bool
        "meta": {
            **asdict(cfg),
            "level_path": level,
            "n_total": int(out_obs.shape[0]),
            "n_valid": int((~out_don).sum()),
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_clock_s": float(time.time() - t0),
        },
    }


def save_buffer(buf: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(buf, path)
    # Also dump a human-readable meta sidecar.
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(buf["meta"], indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=sorted(ENV_TAG_TO_LEVEL), required=True,
                   help="env tag (1rot or 2rot)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=None,
                   help="override n_transitions (default = config)")
    p.add_argument("--n_envs", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny buffer (2048 transitions) for plumbing tests")
    p.add_argument("--force", action="store_true",
                   help="re-collect even if buffer already exists")
    args = p.parse_args()

    cfg = CollectConfig(env_tag=args.env, seed=args.seed)
    if args.n is not None:
        cfg.n_transitions = args.n
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.smoke:
        cfg.n_transitions = 2048

    out_path = buffer_path(cfg.env_tag, cfg.seed)
    if out_path.exists() and not args.force:
        print(f"[collect] buffer already exists: {out_path}  (use --force to overwrite)")
        return

    print(f"[collect] env={cfg.env_tag}  level={level_path_for(cfg.env_tag)}  "
          f"n_transitions={cfg.n_transitions}  n_envs={cfg.n_envs}  seed={cfg.seed}")
    buf = collect_random_transitions(cfg)
    save_buffer(buf, out_path)
    print(f"[collect] saved {buf['meta']['n_total']} total "
          f"({buf['meta']['n_valid']} valid) to {out_path}")


if __name__ == "__main__":
    main()
