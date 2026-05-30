"""Metric collection: feature-stability + periodic eval rollouts."""

from __future__ import annotations

import numpy as np
import torch

from mini_env.env import MiniLS20Env

from .model import ActorCritic
from .rollout import Rollout
from .vec_env import VecMiniEnv, EpisodeStats


# ── per-update: cosine similarity of consecutive trunk features ──────────

def mean_feature_cosine(rollout: Rollout) -> float:
    """Mean cosine(h_t, h_{t+1}) over *same-episode* consecutive timesteps.

    h_{t+1} starts a new episode whenever ep_starts[t+1] is True; those
    pairs are dropped.
    """
    feats = rollout.features  # (T, N, D)
    ep_starts = rollout.ep_starts  # (T, N) bool — True if this step BEGAN a new ep
    T, N, D = feats.shape
    if T < 2:
        return float("nan")

    h_t = feats[:-1]                       # (T-1, N, D)
    h_next = feats[1:]                     # (T-1, N, D)
    # h_next starts a new episode at indices where ep_starts[t+1] is True.
    boundary = ep_starts[1:]               # (T-1, N) — drop these
    keep = ~boundary

    if not bool(keep.any()):
        return float("nan")

    # Cosine over feature axis.
    h_t_n = torch.nn.functional.normalize(h_t, dim=-1, eps=1e-8)
    h_next_n = torch.nn.functional.normalize(h_next, dim=-1, eps=1e-8)
    cos = (h_t_n * h_next_n).sum(-1)       # (T-1, N)
    return float(cos[keep].mean().item())


# ── periodic: full eval rollouts ─────────────────────────────────────────

def run_eval_episodes(level_path: str, model: ActorCritic, device: torch.device,
                       n_episodes: int = 32, sample: bool = True) -> dict[str, float]:
    """Run `n_episodes` fresh episodes IN PARALLEL and return summary metrics.

    Builds n_episodes independent MiniLS20Env instances and steps them
    together using a single batched forward pass per step. ~20x faster on
    MPS than the previous single-env loop (which paid per-step host↔device
    overhead).
    """
    envs = [MiniLS20Env(level_path) for _ in range(n_episodes)]
    goal_rot = envs[0].goal_rotation
    n_walkable = int((envs[0].grid != 1).sum())

    # All envs start with reset() called inside __init__ already.
    obs_list = [e._render() for e in envs]
    done_mask = [False] * n_episodes
    visited = [{(e.player_c, e.player_r)} for e in envs]
    wall_hits = [0] * n_episodes
    steps_taken = [0] * n_episodes

    model.eval()
    with torch.no_grad():
        max_iters = envs[0].config.step_limit + 2  # safety cap
        for _ in range(max_iters):
            active_idx = [i for i, d in enumerate(done_mask) if not d]
            if not active_idx:
                break
            batch = np.stack([obs_list[i] for i in active_idx], axis=0)
            obs_t = torch.from_numpy(batch).to(device)
            logits, _, _ = model.forward(obs_t)
            if sample:
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample().cpu().numpy()
            else:
                actions = logits.argmax(-1).cpu().numpy()
            for k, i in enumerate(active_idx):
                e = envs[i]
                pre = (e.player_c, e.player_r)
                frame, done = e.step(int(actions[k]))
                post = (e.player_c, e.player_r)
                steps_taken[i] += 1
                won = bool(e.won)
                if (post == pre) and not won:
                    wall_hits[i] += 1
                visited[i].add(post)
                obs_list[i] = frame
                if done:
                    done_mask[i] = True
    model.train()

    success_count = sum(1 for e in envs if e.won)
    matched_at_end_count = sum(1 for e in envs if e.player_rotation == goal_rot)
    successful_step_counts = [steps_taken[i] for i, e in enumerate(envs) if e.won]
    wall_hits_per_ep = wall_hits
    coverage_per_ep = [len(visited[i]) / max(1, n_walkable) for i in range(n_episodes)]
    returns_unshaped = [1.0 if e.won else 0.0 for e in envs]

    return {
        "eval_success_rate": success_count / n_episodes,
        "eval_min_steps_to_solve": (min(successful_step_counts) if successful_step_counts else float("nan")),
        "eval_avg_steps_to_solve": (sum(successful_step_counts) / len(successful_step_counts) if successful_step_counts else float("nan")),
        "eval_pattern_matched_at_end_rate": matched_at_end_count / n_episodes,
        "eval_coverage_rate": float(np.mean(coverage_per_ep)),
        "eval_wall_hit_rate": float(np.mean(wall_hits_per_ep)),
        "eval_mean_episode_return": float(np.mean(returns_unshaped)),
        "eval_n_episodes": n_episodes,
    }


# ── rolling stats from completed train episodes ─────────────────────────

def summarise_completed(episodes: list[EpisodeStats]) -> dict[str, float]:
    """Light summary of train-time finished episodes (between two updates)."""
    if not episodes:
        return {}
    won = [int(e.won) for e in episodes]
    steps = [e.steps for e in episodes]
    wall_hits = [e.wall_hits for e in episodes]
    matched = [int(e.rotation_matched_at_end) for e in episodes]
    cov = [len(e.visited) for e in episodes]
    return {
        "train_completed_episodes": len(episodes),
        "train_success_rate": float(np.mean(won)),
        "train_mean_steps": float(np.mean(steps)),
        "train_mean_wall_hits": float(np.mean(wall_hits)),
        "train_mean_matched_at_end": float(np.mean(matched)),
        "train_mean_unique_cells": float(np.mean(cov)),
    }
