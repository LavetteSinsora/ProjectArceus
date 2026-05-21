"""PPO: rollout collection, GAE, and the clipped-objective update.

Game-agnostic. Episodes are collected to termination, so every trajectory ends
with `done=True` and no value bootstrap is needed at the buffer boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from claude_automate.framework.env_api import frame_to_tensor


def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """Generalized Advantage Estimation over concatenated episodes.

    `dones[t]` is True on the last step of an episode. Because every collected
    episode terminates, the post-terminal value is always 0.

    Returns (advantages, returns) as float32 numpy arrays.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        nonterminal = 0.0 if dones[t] else 1.0
        next_value = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


@dataclass
class EpisodeStats:
    length: int
    completed: bool
    raw_return: float          # sum of composed reward
    novelty_sum: float
    stuck_steps: int


@dataclass
class Rollout:
    frames: list = field(default_factory=list)       # list[(64,64) uint8]
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    episodes: list = field(default_factory=list)      # list[EpisodeStats]

    def __len__(self):
        return len(self.actions)


def collect_episodes(env, model, reward_computer, cfg, device,
                     n_episodes: int, greedy: bool = False) -> Rollout:
    """Run `n_episodes` full episodes; return a Rollout of all transitions."""
    roll = Rollout()
    model.eval()
    for _ in range(n_episodes):
        frame = env.reset()
        reward_computer.reset_episode()        # fresh episodic novelty counts
        ep_len = 0
        ep_return = 0.0
        ep_novelty = 0.0
        ep_stuck = 0
        completed = False
        for _ in range(cfg.max_episode_steps):
            obs = frame_to_tensor(frame, cfg.n_colors).unsqueeze(0).to(device)
            logits, value = model.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(-1) if greedy else dist.sample()
            log_prob = dist.log_prob(action)

            next_frame, terminal = env.step(int(action.item()))
            level_done = env.level_completed
            br = reward_computer.compute(env, frame, next_frame, level_done)
            done = bool(terminal or level_done)

            roll.frames.append(frame)
            roll.actions.append(int(action.item()))
            roll.log_probs.append(float(log_prob.item()))
            roll.values.append(float(value.item()))
            roll.rewards.append(br.total)
            roll.dones.append(done)

            ep_len += 1
            ep_return += br.total
            ep_novelty += br.novelty
            ep_stuck += int(not br.frame_changed)
            completed = completed or level_done
            frame = next_frame
            if done:
                break
        roll.episodes.append(EpisodeStats(
            length=ep_len, completed=completed, raw_return=ep_return,
            novelty_sum=ep_novelty, stuck_steps=ep_stuck,
        ))
    return roll


class PPO:
    """Clipped-objective PPO updater wrapping an ActorCritic model."""

    def __init__(self, model: nn.Module, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    def update(self, roll: Rollout) -> dict:
        cfg = self.cfg
        adv, returns = compute_gae(
            roll.rewards, roll.values, roll.dones, cfg.gamma, cfg.gae_lambda
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        frames = torch.stack(
            [frame_to_tensor(f, cfg.n_colors) for f in roll.frames]
        )                                                # (N, C, H, W) on CPU
        actions = torch.as_tensor(roll.actions, dtype=torch.long)
        old_log_probs = torch.as_tensor(roll.log_probs, dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)

        n = len(roll)
        self.model.train()
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "clip_frac": 0.0, "approx_kl": 0.0}
        n_batches = 0
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                mb_obs = frames[idx].to(self.device)
                mb_act = actions[idx].to(self.device)
                mb_old_lp = old_log_probs[idx].to(self.device)
                mb_adv = adv_t[idx].to(self.device)
                mb_ret = ret_t[idx].to(self.device)

                log_probs, entropy, values = self.model.evaluate(mb_obs, mb_act)
                ratio = torch.exp(log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps,
                                    1 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (values - mb_ret).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = (policy_loss
                        + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy_mean)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
                    approx_kl = (mb_old_lp - log_probs).mean()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_mean.item())
                stats["clip_frac"] += float(clip_frac.item())
                stats["approx_kl"] += float(approx_kl.item())
                n_batches += 1

        for k in stats:
            stats[k] /= max(n_batches, 1)
        return stats
