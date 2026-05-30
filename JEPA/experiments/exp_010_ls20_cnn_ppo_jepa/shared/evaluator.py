"""Periodic evaluation: run fresh episodes with the current policy and report
performance metrics. Uses a *separate* eval vec env so it never disturbs the
training env's episode state."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, eval_envs, device, n_episodes: int = 32,
             greedy: bool = False) -> dict:
    """Roll `n_episodes` episodes across the eval vec env and aggregate.

    Returns success_rate, mean_episode_steps, avg_steps_to_solve (successful
    episodes only), and min_steps_to_solve.
    """
    eval_envs.reset_all()
    obs_np = eval_envs.current_obs()
    collected: list = []
    # Run until we have at least n_episodes completed episodes.
    max_iters = n_episodes * eval_envs.max_episode_steps + 1000
    it = 0
    while len(collected) < n_episodes and it < max_iters:
        it += 1
        obs_t = torch.from_numpy(obs_np).to(device)
        logits, _, _ = model.forward(obs_t)
        if greedy:
            action = logits.argmax(-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()
        obs_np, _, _, infos = eval_envs.step(action.cpu().numpy().astype(np.int64))
        collected.extend(eval_envs.drain_completed_episodes())

    collected = collected[:n_episodes]
    n = max(1, len(collected))
    successes = [e for e in collected if e.success]
    succ_steps = [e.steps for e in successes]
    return {
        "eval_episodes": len(collected),
        "success_rate": len(successes) / n,
        "mean_episode_steps": float(np.mean([e.steps for e in collected])) if collected else float("nan"),
        "avg_steps_to_solve": float(np.mean(succ_steps)) if succ_steps else float("nan"),
        "min_steps_to_solve": float(np.min(succ_steps)) if succ_steps else float("nan"),
        "truncation_rate": float(np.mean([e.truncated for e in collected])) if collected else float("nan"),
    }
