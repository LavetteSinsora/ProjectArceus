"""Interactive PPO stepper sessions.

A `StepperSession` keeps a model + optimizer alive across HTTP calls so the
dashboard can alternate rollouts and PPO updates and watch the policy drift.

Lifecycle:
    s = StepperSession(checkpoint_path)
    s.rollout_episode(seed=0)          # appends to s.buffer
    s.rollout_episode(seed=1)
    s.apply_update(n_episodes=8)       # PPO update on s.model using buffer tail
    s.rollout_episode(seed=2)          # now uses the updated weights
    s.eval_state(frame)                # π(·|s_pinned) under current weights
"""

from __future__ import annotations

import copy
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mini_env.env import MiniLS20Env

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.model import ActorCritic
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.dashboard.inspector import (
    _frame_to_rgb_list,
    ARC_COLORS_RGB,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.dashboard.dynamics import (
    _reward_for_step,
)


ACTION_NAMES = ["up", "down", "left", "right"]


@dataclass
class _Rollout:
    """Single episode rollout buffer for PPO."""
    obs: np.ndarray              # (T, 32, 32) uint8
    actions: np.ndarray          # (T,) int64
    logp_old: np.ndarray         # (T,) float32
    values: np.ndarray           # (T,) float32
    rewards: np.ndarray          # (T,) float32
    dones: np.ndarray            # (T,) bool
    bootstrap_value: float       # V(s_T) or 0 if terminal


class StepperSession:
    def __init__(self, checkpoint_path: str, lr: float | None = None,
                 level_path: str | None = None):
        self.session_id = uuid.uuid4().hex[:8]
        self.checkpoint_path = checkpoint_path
        self.created_at = time.time()
        self.device = torch.device("cpu")

        ck = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.cfg: dict = ck.get("config", {}) or {}
        self.source_update = int(ck.get("update", 0))

        self.model = ActorCritic().to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.train()

        self.lr = float(lr if lr is not None else self.cfg.get("learning_rate", 3e-4))
        self.gamma = float(self.cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(self.cfg.get("clip_eps", 0.2))
        _vf = self.cfg.get("vf_clip_eps", 0.2)
        self.vf_clip_eps: float | None = float(_vf) if _vf is not None else None
        self.c_value = float(self.cfg.get("c_value", 0.5))
        self.c_entropy = float(self.cfg.get("c_entropy", 0.01))
        self.grad_clip = float(self.cfg.get("grad_clip", 0.5))
        self.reward_mode = self.cfg.get("reward_mode", "terminal_only")
        self.wall_penalty = float(self.cfg.get("wall_penalty", -0.05))
        self.match_bonus = float(self.cfg.get("match_bonus", 0.1))
        self.unmatch_penalty = float(self.cfg.get("unmatch_penalty", -0.1))
        self.level_path = level_path or self.cfg.get(
            "level_path", "mini_env/configs/level_01/simple_1_rotation.json"
        )
        self.trained_level_path = self.cfg.get(
            "level_path", "mini_env/configs/level_01/simple_1_rotation.json"
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.buffer: deque[_Rollout] = deque(maxlen=64)
        self.history: list[dict] = []  # event log of episodes/updates
        self.n_episodes = 0
        self.n_updates = 0

    # ── public state summary ─────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "checkpoint_path": self.checkpoint_path,
            "source_update": self.source_update,
            "n_episodes": self.n_episodes,
            "n_updates": self.n_updates,
            "buffer_len": len(self.buffer),
            "reward_mode": self.reward_mode,
            "level_path": self.level_path,
            "lr": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_eps,
            "vf_clip_eps": self.vf_clip_eps,
            "c_value": self.c_value,
            "c_entropy": self.c_entropy,
            "created_at": self.created_at,
        }

    # ── rollout ─────────────────────────────────────────────────────────

    def rollout_episode(self, seed: int = 0, sample: bool = True) -> dict:
        """Run one episode with the current weights. Append to buffer."""
        env = MiniLS20Env(self.level_path)
        torch.manual_seed(seed)
        np.random.seed(seed)

        obs = env.reset()
        obs_buf: list[np.ndarray] = []
        action_buf: list[int] = []
        logp_buf: list[float] = []
        value_buf: list[float] = []
        reward_buf: list[float] = []
        done_buf: list[bool] = []
        steps_data: list[dict] = []

        max_steps = env.config.step_limit + 2
        self.model.eval()
        for t in range(max_steps):
            pre_pos = (env.player_c, env.player_r)
            pre_rot = env.player_rotation
            pre_step_counter = env.step_counter
            pre_denial = env.denial_frames

            obs_t = torch.from_numpy(obs[None, ...]).to(self.device)
            with torch.no_grad():
                logits, value, feat = self.model.forward(obs_t)
            if sample:
                dist = torch.distributions.Categorical(logits=logits)
                a_tensor = dist.sample()
                a_int = int(a_tensor.item())
                logp_val = float(dist.log_prob(a_tensor).item())
            else:
                a_int = int(torch.argmax(logits, dim=-1).item())
                # log_prob under categorical(logits)
                logp_all = torch.log_softmax(logits, dim=-1)
                logp_val = float(logp_all[0, a_int].item())

            probs_np = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
            logits_np = logits.cpu().numpy()[0].tolist()
            v_int = float(value.item())
            fnorm = float(torch.linalg.norm(feat).item())

            next_obs, done = env.step(a_int)
            post_pos = (env.player_c, env.player_r)
            post_rot = env.player_rotation
            won = bool(env.won)
            r = _reward_for_step(
                self.reward_mode, pre_pos, post_pos, won, pre_rot, post_rot,
                env.goal_rotation, self.wall_penalty, self.match_bonus, self.unmatch_penalty,
            )
            wall_hit = (post_pos == pre_pos) and not won

            obs_buf.append(obs.copy())
            action_buf.append(a_int)
            logp_buf.append(logp_val)
            value_buf.append(v_int)
            reward_buf.append(r)
            done_buf.append(bool(done))

            steps_data.append({
                "t": t,
                "frame": _frame_to_rgb_list(obs),
                "obs": obs.astype(np.uint8).tolist(),  # for pinning
                "player_c": int(pre_pos[0]),
                "player_r": int(pre_pos[1]),
                "player_rotation": int(pre_rot),
                "step_counter": int(pre_step_counter),
                "denial_frames": int(pre_denial),
                "rotation_matched": bool(pre_rot == env.goal_rotation),
                "logits": [float(x) for x in logits_np],
                "probs": probs_np,
                "value": v_int,
                "feature_norm": fnorm,
                "action": a_int,
                "action_name": ACTION_NAMES[a_int],
                "reward": float(r),
                "wall_hit": wall_hit,
                "won": won,
                "done": bool(done),
            })

            obs = next_obs
            if done:
                break

        T = len(obs_buf)
        with torch.no_grad():
            last_v = float(self.model.forward(
                torch.from_numpy(obs[None, ...]).to(self.device)
            )[1].item())
        bootstrap = 0.0 if (T > 0 and done_buf[-1]) else last_v

        # Terminal-state snapshot (no action), so the UI can show the final frame.
        with torch.no_grad():
            obs_t = torch.from_numpy(obs[None, ...]).to(self.device)
            logits, value, feat = self.model.forward(obs_t)
        steps_data.append({
            "t": T,
            "frame": _frame_to_rgb_list(obs),
            "obs": obs.astype(np.uint8).tolist(),
            "player_c": int(env.player_c),
            "player_r": int(env.player_r),
            "player_rotation": int(env.player_rotation),
            "step_counter": int(env.step_counter),
            "denial_frames": int(env.denial_frames),
            "rotation_matched": bool(env.player_rotation == env.goal_rotation),
            "logits": [float(x) for x in logits.cpu().numpy()[0]],
            "probs": [float(x) for x in torch.softmax(logits, -1).cpu().numpy()[0]],
            "value": float(value.item()),
            "feature_norm": float(torch.linalg.norm(feat).item()),
            "action": None,
            "action_name": None,
            "reward": 0.0,
            "wall_hit": False,
            "won": bool(env.won),
            "done": True,
            "terminal": True,
        })

        rollout = _Rollout(
            obs=np.stack(obs_buf, axis=0),
            actions=np.array(action_buf, dtype=np.int64),
            logp_old=np.array(logp_buf, dtype=np.float32),
            values=np.array(value_buf, dtype=np.float32),
            rewards=np.array(reward_buf, dtype=np.float32),
            dones=np.array(done_buf, dtype=bool),
            bootstrap_value=bootstrap,
        )
        self.buffer.append(rollout)
        self.n_episodes += 1

        episode_idx = self.n_episodes  # 1-based
        won_at_end = bool(steps_data[-1].get("won", False))
        total_reward = float(rollout.rewards.sum())
        event = {
            "kind": "episode",
            "episode_idx": episode_idx,
            "update_idx": self.n_updates,  # which update this episode followed
            "seed": int(seed),
            "sample": bool(sample),
            "n_steps": int(T),
            "won": won_at_end,
            "total_reward": total_reward,
        }
        self.history.append(event)

        return {
            "session_id": self.session_id,
            "episode_idx": episode_idx,
            "after_update": self.n_updates,
            "seed": int(seed),
            "sample": bool(sample),
            "n_steps": int(T),
            "won": won_at_end,
            "total_reward": total_reward,
            "goal_rotation": int(env.goal_rotation),
            "grid_cols": int(env.cols),
            "grid_play_rows": int(env.play_rows),
            "step_limit": int(env.config.step_limit),
            "palette_rgb": ARC_COLORS_RGB,
            "steps": steps_data,
            "summary": self.summary(),
        }

    # ── PPO update ──────────────────────────────────────────────────────

    def apply_update(self, n_episodes: int = 8, epochs: int = 2,
                     minibatches: int | None = None) -> dict:
        if not self.buffer:
            raise RuntimeError("buffer is empty — run at least one episode first")
        n_use = min(n_episodes, len(self.buffer))
        recent = list(self.buffer)[-n_use:]

        # Compute GAE per rollout, then concatenate.
        all_obs, all_actions, all_logp_old = [], [], []
        all_values, all_adv, all_v_target = [], [], []
        per_rollout_stats = []
        for r in recent:
            T = len(r.actions)
            advantages = np.zeros(T, dtype=np.float32)
            last_gae = 0.0
            next_value = r.bootstrap_value
            # NB: mask uses CURRENT step's done flag (see shared/rollout.py for
            # the explanation of the previous off-by-one bug we removed).
            for t in reversed(range(T)):
                nonterm = 0.0 if r.dones[t] else 1.0
                delta = r.rewards[t] + self.gamma * next_value * nonterm - r.values[t]
                last_gae = delta + self.gamma * self.gae_lambda * nonterm * last_gae
                advantages[t] = last_gae
                next_value = r.values[t]
            v_target = advantages + r.values
            all_obs.append(r.obs)
            all_actions.append(r.actions)
            all_logp_old.append(r.logp_old)
            all_values.append(r.values)
            all_adv.append(advantages)
            all_v_target.append(v_target)
            per_rollout_stats.append({
                "n_steps": int(T),
                "total_reward": float(r.rewards.sum()),
                "adv_mean": float(advantages.mean()),
                "adv_std": float(advantages.std()),
            })

        obs_arr = np.concatenate(all_obs, axis=0)
        actions_arr = np.concatenate(all_actions, axis=0)
        logp_old_arr = np.concatenate(all_logp_old, axis=0)
        adv_arr = np.concatenate(all_adv, axis=0)
        v_target_arr = np.concatenate(all_v_target, axis=0)
        values_arr = np.concatenate(all_values, axis=0)

        adv_mean = float(adv_arr.mean())
        adv_std = float(adv_arr.std()) + 1e-8
        adv_norm = (adv_arr - adv_mean) / adv_std

        obs_t = torch.from_numpy(obs_arr).to(self.device)
        actions_t = torch.from_numpy(actions_arr).to(self.device)
        logp_old_t = torch.from_numpy(logp_old_arr).to(self.device)
        adv_t = torch.from_numpy(adv_norm).to(self.device)
        v_target_t = torch.from_numpy(v_target_arr).to(self.device)
        values_old_t = torch.from_numpy(values_arr).to(self.device)

        N = obs_t.shape[0]
        if minibatches is None:
            minibatches = int(self.cfg.get("minibatches", 4))
        minibatches = max(1, min(minibatches, N))
        mb_size = max(1, N // minibatches)

        self.model.train()

        # Pre-update snapshot loss on the full batch.
        with torch.no_grad():
            logp_pre, ent_pre, v_pre, _ = self.model.evaluate(obs_t, actions_t)
            pre_policy_loss = float(-(adv_t * 1.0).mean().item())  # unweighted approx
            pre_value_loss = float(0.5 * (v_pre - v_target_t).pow(2).mean().item())
            pre_entropy = float(ent_pre.mean().item())
            pre_logp_taken = logp_pre.cpu().numpy().copy()

        # Multi-epoch / multi-minibatch PPO update.
        for ep in range(epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb_size):
                idx = perm[start:start + mb_size]
                if idx.numel() == 0:
                    continue
                logp_new, ent, v_new, _ = self.model.evaluate(obs_t[idx], actions_t[idx])
                ratio = (logp_new - logp_old_t[idx]).exp()
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                if self.vf_clip_eps is not None:
                    v_old_mb = values_old_t[idx]
                    v_clipped = v_old_mb + torch.clamp(
                        v_new - v_old_mb, -self.vf_clip_eps, self.vf_clip_eps
                    )
                    vl_unclipped = (v_new - v_target_t[idx]).pow(2)
                    vl_clipped = (v_clipped - v_target_t[idx]).pow(2)
                    value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
                else:
                    value_loss = 0.5 * (v_new - v_target_t[idx]).pow(2).mean()
                entropy = ent.mean()
                loss = policy_loss + self.c_value * value_loss - self.c_entropy * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

        # Post-update snapshot.
        self.model.eval()
        with torch.no_grad():
            logp_post, ent_post, v_post, _ = self.model.evaluate(obs_t, actions_t)
            post_value_loss = float(0.5 * (v_post - v_target_t).pow(2).mean().item())
            post_entropy = float(ent_post.mean().item())
            # Approx KL on taken actions: mean(logp_old - logp_new).
            approx_kl = float((logp_old_t - logp_post).mean().item())
            # Mean change in logp on taken action: positive means update increased prob of taken action.
            mean_logp_delta = float((logp_post - torch.from_numpy(pre_logp_taken).to(self.device)).mean().item())

        self.n_updates += 1
        stats = {
            "kind": "update",
            "update_idx": self.n_updates,
            "n_episodes_used": n_use,
            "n_transitions": int(N),
            "epochs": int(epochs),
            "minibatches": int(minibatches),
            "pre_value_loss": pre_value_loss,
            "post_value_loss": post_value_loss,
            "value_loss_reduction": pre_value_loss - post_value_loss,
            "pre_entropy": pre_entropy,
            "post_entropy": post_entropy,
            "approx_kl": approx_kl,
            "mean_logp_delta_on_taken": mean_logp_delta,
            "adv_mean": adv_mean,
            "adv_std": adv_std - 1e-8,
            "per_rollout": per_rollout_stats,
        }
        self.history.append(stats)
        return {
            "session_id": self.session_id,
            "summary": self.summary(),
            **stats,
        }

    # ── pinned-state evaluation ─────────────────────────────────────────

    def eval_state(self, frame: list[list[int]] | np.ndarray) -> dict:
        """Run the current model on a single frame; return logits/probs/value."""
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.shape != (32, 32):
            raise ValueError(f"expected (32, 32) frame, got {arr.shape}")
        self.model.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(arr[None, ...]).to(self.device)
            logits, value, feat = self.model.forward(obs_t)
        return {
            "logits": [float(x) for x in logits.cpu().numpy()[0]],
            "probs": [float(x) for x in torch.softmax(logits, -1).cpu().numpy()[0]],
            "value": float(value.item()),
            "feature_norm": float(torch.linalg.norm(feat).item()),
            "n_updates": self.n_updates,
            "n_episodes": self.n_episodes,
        }

    # ── reset ───────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reload from the original checkpoint, clear buffer + counters."""
        ck = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.buffer.clear()
        self.history.clear()
        self.n_episodes = 0
        self.n_updates = 0
        return self.summary()


# ── module-level session registry ──────────────────────────────────────

_SESSIONS: dict[str, StepperSession] = {}


def create_session(checkpoint_path: str, lr: float | None = None,
                   level_path: str | None = None) -> StepperSession:
    s = StepperSession(checkpoint_path, lr=lr, level_path=level_path)
    _SESSIONS[s.session_id] = s
    return s


def get_session(session_id: str) -> StepperSession:
    if session_id not in _SESSIONS:
        raise KeyError(f"no session {session_id}")
    return _SESSIONS[session_id]


def delete_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


def list_sessions() -> list[dict]:
    return [s.summary() for s in _SESSIONS.values()]
