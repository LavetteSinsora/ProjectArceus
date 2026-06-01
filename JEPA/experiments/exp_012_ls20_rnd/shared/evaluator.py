"""Periodic evaluation for exp_012 (dual-head model).

Identical to exp_010's evaluator except it unpacks the 4-tuple forward
(logits, v_ext, v_int, feat) and uses a separate eval vec env so eval never
perturbs training episode state. Eval reports the *extrinsic* task metrics only
(success rate, steps-to-solve) — the intrinsic reward is a training signal.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, eval_envs, device, n_episodes: int = 32,
             greedy: bool = False) -> dict:
    eval_envs.reset_all()
    obs_np = eval_envs.current_obs()
    collected: list = []
    max_iters = n_episodes * eval_envs.max_episode_steps + 1000
    it = 0
    while len(collected) < n_episodes and it < max_iters:
        it += 1
        obs_t = torch.from_numpy(obs_np).to(device)
        logits = model.forward(obs_t)[0]
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
