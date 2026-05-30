"""Training-dynamics inspection: from a given checkpoint, simulate what one
PPO update would do to V(s) for each state in an episode.

Procedure (mirrors actual training, scaled down for one episode):
  1. Roll out one full episode with stochastic actions using the checkpoint's
     policy.
  2. Compute per-step rewards (terminal-only, matching variant 0's reward).
     We do *not* re-apply shaping here — the goal is to see the raw value
     function evolve under the training signal that variant 0 actually used.
     For other variants (wallpen / match) we pass the run's reward_mode to
     reconstruct the shaped reward stream so V_target is faithful.
  3. Compute V_target via GAE: A_t = δ_t + γλA_{t+1}, V_target = A + V.
  4. Clone the model. Apply ONE PPO update step (single epoch, single
     minibatch over the episode's transitions) to the clone.
  5. Re-evaluate V on the same states using the updated clone → V_new.

For each step we return: frame, state info, action, π(a|s), V(s),
V_target(s), V_new(s), δ_t (TD error), A_t (advantage), reward.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_env.env import MiniLS20Env

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.model import ActorCritic, one_hot_frame
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.dashboard.inspector import ARC_COLORS_RGB, _frame_to_rgb_list


def _reward_for_step(
    mode: str, pre_pos, post_pos, won: bool, pre_rot: int, post_rot: int,
    goal_rot: int, wall_penalty: float, match_bonus: float, unmatch_penalty: float,
) -> float:
    r = 1.0 if won else 0.0
    if mode == "terminal_only":
        return r
    wall_hit = (post_pos == pre_pos) and not won
    if wall_hit:
        r += wall_penalty
    if mode == "wall+match":
        was_match = (pre_rot == goal_rot)
        is_match = (post_rot == goal_rot)
        if not was_match and is_match: r += match_bonus
        elif was_match and not is_match: r += unmatch_penalty
    return r


def run_debug_update(
    checkpoint_path: str,
    seed: int = 0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_clip_eps: float | None = 0.2,
    c_value: float = 0.5,
    c_entropy: float = 0.01,
    grad_clip: float = 0.5,
    learning_rate: float = 3e-4,
    epochs: int = 2,
    level_path: str | None = None,
) -> dict:
    """One-episode debug rollout + PPO update on a clone.

    Returns a dict shaped for the dashboard with per-step records including
    V(s), V_target(s), V_new(s), advantage, TD error.
    """
    device = torch.device("cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    reward_mode = cfg.get("reward_mode", "terminal_only")
    wall_penalty = cfg.get("wall_penalty", -0.05)
    match_bonus = cfg.get("match_bonus", 0.1)
    unmatch_penalty = cfg.get("unmatch_penalty", -0.1)
    if level_path is None:
        level_path = cfg.get("level_path", "mini_env/configs/level_01/simple_1_rotation.json")

    model = ActorCritic().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    env = MiniLS20Env(level_path)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 1. roll out one full episode ──────────────────────────────────────
    obs = env.reset()
    obs_buf, action_buf, logp_buf, value_buf = [], [], [], []
    reward_buf, done_buf, frame_buf = [], [], []
    state_meta = []  # per-step metadata

    action_names = ["up", "down", "left", "right"]
    max_steps = env.config.step_limit + 2
    for _ in range(max_steps):
        pre_pos = (env.player_c, env.player_r)
        pre_rot = env.player_rotation
        pre_step_counter = env.step_counter
        pre_denial = env.denial_frames

        obs_t = torch.from_numpy(obs[None, ...])
        with torch.no_grad():
            logits, value, feat = model.forward(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)

        a_int = int(a.item())
        v_int = float(value.item())
        logp_int = float(logp.item())

        next_obs, done = env.step(a_int)
        post_pos = (env.player_c, env.player_r)
        post_rot = env.player_rotation
        won = bool(env.won)
        r = _reward_for_step(
            reward_mode, pre_pos, post_pos, won, pre_rot, post_rot,
            env.goal_rotation, wall_penalty, match_bonus, unmatch_penalty,
        )
        wall_hit = (post_pos == pre_pos) and not won

        obs_buf.append(obs.copy())
        frame_buf.append(_frame_to_rgb_list(obs))
        action_buf.append(a_int)
        logp_buf.append(logp_int)
        value_buf.append(v_int)
        reward_buf.append(r)
        done_buf.append(bool(done))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        state_meta.append({
            "player_c": pre_pos[0], "player_r": pre_pos[1],
            "player_rotation": pre_rot,
            "step_counter": pre_step_counter,
            "denial_frames": pre_denial,
            "wall_hit": wall_hit,
            "won": won,
            "rotation_matched": (pre_rot == env.goal_rotation),
            "action": a_int,
            "action_name": action_names[a_int],
            "probs": probs,
            "logits": [float(x) for x in logits.cpu().numpy()[0]],
        })

        obs = next_obs
        if done:
            break

    T = len(obs_buf)
    obs_arr = np.stack(obs_buf, axis=0)
    actions = np.array(action_buf, dtype=np.int64)
    logp_old = np.array(logp_buf, dtype=np.float32)
    values = np.array(value_buf, dtype=np.float32)
    rewards = np.array(reward_buf, dtype=np.float32)
    dones = np.array(done_buf, dtype=bool)

    # bootstrap value at last state (if terminal, 0; else V(s_T))
    with torch.no_grad():
        last_v = float(model.forward(torch.from_numpy(obs[None, ...]))[1].item())
    bootstrap = 0.0 if (T > 0 and dones[-1]) else last_v

    # ── 2. compute GAE → V_target ────────────────────────────────────────
    # In our convention, dones[t]=True means "action at step t terminated the
    # episode". So at step t, V(s_{t+1}) is zero iff dones[t] is True. The
    # mask for delta_t is therefore (1 - dones[t]); same mask used for GAE
    # propagation. NB: an earlier version of this loop took the mask from the
    # *previous* iteration (effectively (1 - dones[t+1])), which had the
    # off-by-one effect of zeroing the bootstrap for the step BEFORE a
    # terminal — breaking propagation of terminal reward backward through the
    # trajectory. The production code in shared/rollout.py has the same bug;
    # fix is queued separately so production re-training is opt-in.
    advantages = np.zeros(T, dtype=np.float32)
    deltas = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = bootstrap
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        deltas[t] = delta
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    v_target = advantages + values

    # ── 3. clone model + apply ONE PPO update ─────────────────────────────
    clone = copy.deepcopy(model)
    clone.train()
    optimizer = torch.optim.Adam(clone.parameters(), lr=learning_rate)

    obs_t = torch.from_numpy(obs_arr)        # (T, 32, 32) uint8
    actions_t = torch.from_numpy(actions)
    logp_old_t = torch.from_numpy(logp_old)
    advantages_t = torch.from_numpy(advantages)
    v_target_t = torch.from_numpy(v_target)
    values_old_t = torch.from_numpy(values)

    # Normalise advantages.
    adv_mean = advantages_t.mean()
    adv_std = advantages_t.std() + 1e-8
    advantages_norm = (advantages_t - adv_mean) / adv_std

    pre_value_loss, pre_policy_loss, pre_entropy = None, None, None
    post_value_loss, post_policy_loss, post_entropy = None, None, None

    for ep in range(epochs):
        logp_new, ent, v_new, _ = clone.evaluate(obs_t, actions_t)
        ratio = (logp_new - logp_old_t).exp()
        surr1 = ratio * advantages_norm
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_norm
        policy_loss = -torch.min(surr1, surr2).mean()
        if vf_clip_eps is not None:
            v_clipped = values_old_t + torch.clamp(
                v_new - values_old_t, -vf_clip_eps, vf_clip_eps
            )
            vl_unclipped = (v_new - v_target_t).pow(2)
            vl_clipped = (v_clipped - v_target_t).pow(2)
            value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
        else:
            value_loss = 0.5 * (v_new - v_target_t).pow(2).mean()
        entropy = ent.mean()
        loss = policy_loss + c_value * value_loss - c_entropy * entropy
        if ep == 0:
            pre_value_loss = float(value_loss.item())
            pre_policy_loss = float(policy_loss.item())
            pre_entropy = float(entropy.item())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(clone.parameters(), grad_clip)
        optimizer.step()

    # Post-update stats.
    clone.eval()
    with torch.no_grad():
        logp_post, ent_post, v_post, _ = clone.evaluate(obs_t, actions_t)
        post_value_loss = float(0.5 * (v_post - v_target_t).pow(2).mean().item())
        post_entropy = float(ent_post.mean().item())
        # Per-state new value + new probs
        logits_post, v_post_full, _ = clone.forward(obs_t)
        probs_post = torch.softmax(logits_post, dim=-1).cpu().numpy().tolist()
        v_post_per_step = v_post_full.cpu().numpy().tolist()
        # post-update policy log-prob of the taken action, for KL probe
        logp_post_taken = clone.evaluate(obs_t, actions_t)[0].cpu().numpy().tolist()

    # ── 3b. compute per-step GAE breakdown ───────────────────────────────
    # GAE: A_t = sum_{k=0..T-t-1} (γλ)^k · δ_{t+k}  (truncated at the next
    # done-flag, since GAE zeros the bootstrap on terminals).
    # We expose the leading `n_leading` terms verbatim + an aggregated "tail"
    # so the dashboard can show how each future TD error contributed to the
    # advantage at this step.
    n_leading = 5
    gl = gamma * gae_lambda
    contribs_by_step: list[list[dict]] = []
    tail_by_step: list[dict] = []
    for t in range(T):
        leading = []
        tail_sum = 0.0
        tail_n = 0
        weight = 1.0
        # Walk forward through future TD errors until we hit a terminal (which
        # truncates the GAE sum) or run out of transitions.
        k_max = T - t
        for k in range(k_max):
            d = float(deltas[t + k])
            contribution = float(weight * d)
            if k < n_leading:
                leading.append({
                    "k": k,
                    "delta": d,
                    "weight": float(weight),
                    "contribution": contribution,
                })
            else:
                tail_sum += contribution
                tail_n += 1
            # Stop accumulating past a terminal at t+k.
            if dones[t + k]:
                break
            weight *= gl
        contribs_by_step.append(leading)
        tail_by_step.append({"n": tail_n, "sum": float(tail_sum)})

    # ── 4. assemble per-step records ─────────────────────────────────────
    steps = []
    for t in range(T):
        steps.append({
            "t": t,
            "frame": frame_buf[t],
            **state_meta[t],
            "reward": float(rewards[t]),
            "value": float(values[t]),
            "value_target": float(v_target[t]),
            "value_new": float(v_post_per_step[t]),
            "td_error": float(deltas[t]),
            "advantage": float(advantages[t]),
            "advantage_norm": float(advantages_norm[t].item()),
            "gae_contributors": contribs_by_step[t],
            "gae_tail": tail_by_step[t],
            "log_prob_old": float(logp_old[t]),
            "log_prob_new": float(logp_post_taken[t]),
            "probs_new": probs_post[t],
            "done": bool(dones[t]),
        })

    won_at_end = bool(state_meta[-1]["won"]) if state_meta else False
    return {
        "checkpoint": checkpoint_path,
        "reward_mode": reward_mode,
        "seed": seed,
        "n_steps": T,
        "won": won_at_end,
        "goal_rotation": int(env.goal_rotation),
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "epochs_applied": epochs,
        "rollout_summary": {
            "total_reward": float(rewards.sum()),
            "advantage_mean": float(adv_mean.item()),     # μ used for A_norm
            "advantage_std": float(adv_std.item() - 1e-8), # σ used for A_norm
            "td_error_mean_abs": float(np.abs(deltas).mean()),
            "pre_value_loss": pre_value_loss,
            "post_value_loss": post_value_loss,
            "pre_policy_loss": pre_policy_loss,
            "pre_entropy": pre_entropy,
            "post_entropy": post_entropy,
            "v_pre_mean": float(values.mean()),
            "v_target_mean": float(v_target.mean()),
            "v_post_mean": float(np.mean(v_post_per_step)),
            "value_loss_reduction": (None if pre_value_loss is None
                                     else pre_value_loss - post_value_loss),
        },
        "steps": steps,
    }
