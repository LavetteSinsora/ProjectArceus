"""Unit tests for the claude_automate framework.

Run: uv run python -m pytest claude_automate/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from claude_automate.framework.config import Config
from claude_automate.framework.env_api import frame_to_tensor, frames_to_batch
from claude_automate.framework.exploration import (
    ExactFrameCounter, SimHashCounter, make_counter,
)
from claude_automate.framework.networks import ActorCritic, RecurrentActorCritic
from claude_automate.framework.ppo import compute_gae, PPO, Rollout
from claude_automate.framework.rewards import RewardComputer
from claude_automate.framework.go_explore import (
    GoExplore, collect_trajectory_frames,
)
from claude_automate.framework.distill import (
    distill_trajectory, distill_trajectory_recurrent,
)


# ── stub env ────────────────────────────────────────────────────────────────

class StubEnv:
    """Minimal env exposing only what RewardComputer needs: frame_diff."""

    def frame_diff(self, f0, f1):
        d = np.abs(f1.astype(np.float32) - f0.astype(np.float32))
        d[61:63, :] = 0.0          # mask LS20 UI rows
        return d


def rand_frame(seed=0):
    return np.random.default_rng(seed).integers(0, 13, (64, 64), dtype=np.uint8)


# ── frame preprocessing ─────────────────────────────────────────────────────

def test_frame_to_tensor_shape_and_onehot():
    f = rand_frame(1)
    t = frame_to_tensor(f, n_colors=16)
    assert t.shape == (16, 64, 64)
    # exactly one channel hot per pixel
    assert torch.all(t.sum(dim=0) == 1.0)
    # the hot channel matches the colour index
    assert int(t[:, 3, 7].argmax()) == int(f[3, 7])


def test_frames_to_batch():
    b = frames_to_batch([rand_frame(i) for i in range(5)], n_colors=16)
    assert b.shape == (5, 16, 64, 64)


# ── SimHash exploration ─────────────────────────────────────────────────────

def test_simhash_deterministic():
    c1 = SimHashCounter(hash_bits=24, seed=0)
    c2 = SimHashCounter(hash_bits=24, seed=0)
    f = rand_frame(2)
    assert c1.code(f) == c2.code(f)              # same seed ⇒ same code
    assert c1.code(f) == c1.code(f.copy())       # stable across calls


def test_simhash_count_and_novelty_decay():
    c = SimHashCounter(hash_bits=24, seed=0)
    f = rand_frame(3)
    n1 = c.visit(f)
    n2 = c.visit(f)
    assert (n1, n2) == (1, 2)
    assert c.novelty(1) == pytest.approx(1.0)
    assert c.novelty(4) == pytest.approx(0.5)
    assert c.novelty(2) < c.novelty(1)           # strictly decaying


def test_simhash_masks_ui_rows():
    """Frames differing only in the masked UI rows hash identically."""
    c = SimHashCounter(hash_bits=24, seed=0, masked_rows=slice(61, 63))
    f = rand_frame(4)
    g = f.copy()
    g[61:63, :] = (g[61:63, :] + 7) % 13         # change only UI rows
    assert c.code(f) == c.code(g)


def test_exact_counter_distinguishes_distinct_frames():
    """ExactFrameCounter gives one bucket per genuinely distinct screen."""
    c = ExactFrameCounter(masked_rows=slice(61, 63))
    fa, fb = rand_frame(20), rand_frame(21)
    assert c.code(fa) == c.code(fa.copy())       # identical ⇒ same code
    assert c.code(fa) != c.code(fb)              # different ⇒ different code
    assert (c.visit(fa), c.visit(fa), c.visit(fb)) == (1, 2, 1)
    assert c.n_distinct == 2


def test_exact_counter_masks_ui_rows():
    c = ExactFrameCounter(masked_rows=slice(61, 63))
    f = rand_frame(22)
    g = f.copy()
    g[61:63, :] = (g[61:63, :] + 5) % 13         # change only UI rows
    assert c.code(f) == c.code(g)


def test_make_counter_modes():
    assert isinstance(make_counter("exact"), ExactFrameCounter)
    assert isinstance(make_counter("simhash"), SimHashCounter)
    with pytest.raises(ValueError):
        make_counter("bogus")


# ── reward composition ──────────────────────────────────────────────────────

def test_reward_completion_term():
    cfg = Config()
    rc = RewardComputer(cfg)
    f = rand_frame(5)
    br_done = rc.compute(StubEnv(), f, rand_frame(6), level_completed=True)
    assert br_done.completion == cfg.w_complete
    br_not = RewardComputer(cfg).compute(StubEnv(), f, rand_frame(6),
                                         level_completed=False)
    assert br_not.completion == 0.0


def test_reward_stuck_penalty_when_frame_unchanged():
    cfg = Config()
    rc = RewardComputer(cfg)
    f = rand_frame(7)
    # identical frame ⇒ stuck penalty applied
    br = rc.compute(StubEnv(), f, f.copy(), level_completed=False)
    assert br.stuck == -cfg.w_stuck
    assert br.frame_changed is False


def test_reward_no_stuck_penalty_when_frame_changes():
    cfg = Config()
    rc = RewardComputer(cfg)
    br = rc.compute(StubEnv(), rand_frame(8), rand_frame(9),
                    level_completed=False)
    assert br.stuck == 0.0
    assert br.frame_changed is True


def test_reward_novelty_positive_and_decays():
    cfg = Config()
    rc = RewardComputer(cfg)
    env, f0, f1 = StubEnv(), rand_frame(10), rand_frame(11)
    first = rc.compute(env, f0, f1, level_completed=False).novelty
    second = rc.compute(env, f0, f1, level_completed=False).novelty
    assert first > 0.0
    assert second < first                        # revisiting ⇒ less novel
    # novelty = global + episodic, both present
    br = rc.compute(env, f0, f1, level_completed=False)
    assert br.novelty == pytest.approx(br.novelty_global + br.novelty_episodic)
    assert br.novelty_global > 0.0 and br.novelty_episodic > 0.0


def test_reward_episodic_novelty_resets_per_episode():
    """reset_episode() restores episodic novelty; global novelty keeps decaying."""
    cfg = Config()
    rc = RewardComputer(cfg)
    env, f0, f1 = StubEnv(), rand_frame(12), rand_frame(13)
    first = rc.compute(env, f0, f1, level_completed=False)
    rc.compute(env, f0, f1, level_completed=False)   # second visit, decays both
    rc.reset_episode()
    after = rc.compute(env, f0, f1, level_completed=False)
    # episodic counter reset ⇒ episodic novelty back to its first-visit value
    assert after.novelty_episodic == pytest.approx(first.novelty_episodic)
    # global counter NOT reset ⇒ global novelty strictly lower than first visit
    assert after.novelty_global < first.novelty_global


# ── networks ────────────────────────────────────────────────────────────────

def test_actor_critic_shapes():
    model = ActorCritic(n_actions=4, n_colors=16, hidden_dim=128)
    obs = torch.zeros(3, 16, 64, 64)
    logits, value = model(obs)
    assert logits.shape == (3, 4)
    assert value.shape == (3,)
    action, log_prob, v = model.act(obs)
    assert action.shape == log_prob.shape == v.shape == (3,)
    lp, ent, val = model.evaluate(obs, action)
    assert lp.shape == ent.shape == val.shape == (3,)


# ── GAE ─────────────────────────────────────────────────────────────────────

def test_gae_single_step_episode():
    # one-step episode: adv = r - V, return = r
    adv, ret = compute_gae([1.0], [0.4], [True], gamma=0.99, lam=0.95)
    assert adv[0] == pytest.approx(0.6)
    assert ret[0] == pytest.approx(1.0)


def test_gae_two_step_episode_matches_manual():
    rewards = [0.0, 1.0]
    values = [0.5, 0.7]
    dones = [False, True]
    g, lam = 0.99, 0.95
    adv, ret = compute_gae(rewards, values, dones, g, lam)
    # t=1 (terminal): delta = 1.0 + 0 - 0.7 = 0.3 ; gae = 0.3
    # t=0: delta = 0 + g*0.7 - 0.5 = 0.193 ; gae = delta + g*lam*0.3
    d1 = 0.3
    d0 = 0.0 + g * 0.7 - 0.5
    expected0 = d0 + g * lam * d1
    assert adv[1] == pytest.approx(d1, abs=1e-5)
    assert adv[0] == pytest.approx(expected0, abs=1e-5)
    assert ret[0] == pytest.approx(expected0 + 0.5, abs=1e-5)


def test_gae_resets_across_episode_boundary():
    # two separate one-step episodes — advantages must not leak across
    adv, _ = compute_gae([1.0, 5.0], [0.0, 0.0], [True, True], 0.99, 0.95)
    assert adv[0] == pytest.approx(1.0)
    assert adv[1] == pytest.approx(5.0)


# ── PPO update smoke test ───────────────────────────────────────────────────

# ── Go-Explore ──────────────────────────────────────────────────────────────

class LineWorld:
    """Deterministic 1-D corridor mock env: action 0 = +1, 1 = -1, 2/3 noop.

    Row 63 is a step-counter UI row that changes every step (must be masked).
    """
    _MASKED_ROWS = slice(63, 64)
    n_actions = 4

    def __init__(self, goal=8, max_steps=40):
        self.goal = goal
        self.max_steps = max_steps
        self.pos = 0
        self.steps = 0

    def reset(self):
        self.pos = 0
        self.steps = 0
        return self._frame()

    def step(self, a):
        self.steps += 1
        if a == 0:
            self.pos = min(self.pos + 1, self.goal)
        elif a == 1:
            self.pos = max(self.pos - 1, 0)
        terminal = self.level_completed or self.steps >= self.max_steps
        return self._frame(), terminal

    @property
    def level_completed(self):
        return self.pos >= self.goal

    def _frame(self):
        f = np.zeros((64, 64), dtype=np.uint8)
        f[self.pos, :] = self.pos + 1          # distinct, deterministic per pos
        f[63, :] = self.steps % 16             # UI row — changes every step
        return f


def test_go_explore_cell_code_masks_ui_row():
    ge = GoExplore(LineWorld(), masked_rows=slice(63, 64))
    w = LineWorld()
    w.reset()
    f_a = w._frame()
    w.steps += 1                               # only the UI row changes
    f_b = w._frame()
    assert ge.cell_code(f_a) == ge.cell_code(f_b)


def test_go_explore_finds_and_replays_solution():
    ge = GoExplore(LineWorld(goal=8, max_steps=40),
                   masked_rows=slice(63, 64), explore_steps=20, seed=0)
    result = ge.search(max_env_steps=50_000, verbose=False)
    assert result.solution is not None         # a completing trajectory exists
    # replaying the solution must actually complete the level
    env = LineWorld(goal=8, max_steps=40)
    env.reset()
    for a in result.solution:
        env.step(a)
    assert env.level_completed
    assert result.archive_size >= 8            # archived the corridor cells


def test_collect_trajectory_frames_length():
    traj = [0, 0, 0, 2, 1, 0]
    frames, actions = collect_trajectory_frames(LineWorld(), traj)
    assert actions == traj
    assert len(frames) == len(traj)
    assert frames[0].shape == (64, 64)


# ── distillation ────────────────────────────────────────────────────────────

def test_distill_reproduces_trajectory():
    cfg = Config(hidden_dim=64)
    model = ActorCritic(n_actions=4, n_colors=16, hidden_dim=64)
    frames = [rand_frame(200 + i) for i in range(8)]
    actions = [i % 4 for i in range(8)]
    stats = distill_trajectory(model, frames, actions, cfg,
                               device=torch.device("cpu"),
                               epochs=400, verbose=False)
    assert stats["train_acc"] == 1.0           # policy reproduces every action


def test_recurrent_actor_critic_shapes():
    model = RecurrentActorCritic(n_actions=4, n_colors=16, hidden_dim=64,
                                 gru_dim=32)
    h = model.initial_state(batch=2)
    assert h.shape == (2, 32)
    logits, value, h2 = model.step(torch.zeros(2, 16, 64, 64), h)
    assert logits.shape == (2, 4) and value.shape == (2,) and h2.shape == (2, 32)
    seq_logits = model.forward_sequence(torch.zeros(5, 16, 64, 64))
    assert seq_logits.shape == (5, 4)


def test_recurrent_distill_fits_revisited_states():
    """The capability a stateless policy lacks: a trajectory that revisits the
    SAME observation with DIFFERENT actions. Recurrent BC must still reach 100%."""
    cfg = Config(hidden_dim=64)
    model = RecurrentActorCritic(n_actions=4, n_colors=16, hidden_dim=64,
                                 gru_dim=32)
    a, b, c = rand_frame(1), rand_frame(2), rand_frame(3)
    frames = [a, b, a, c, a]                   # frame `a` appears 3×
    actions = [0, 1, 2, 3, 1]                  # ...with different actions
    # a stateless policy provably cannot fit this; the recurrent one can
    stats = distill_trajectory_recurrent(model, frames, actions, cfg,
                                         device=torch.device("cpu"),
                                         epochs=2000, verbose=False)
    assert stats["train_acc"] == 1.0


# ── world model ─────────────────────────────────────────────────────────────

def test_world_model_shapes_and_predict():
    from claude_automate.framework.world_model import FrameWorldModel
    wm = FrameWorldModel(n_colors=16, n_actions=4, base_ch=16)
    obs = torch.zeros(3, 16, 64, 64)
    act = torch.tensor([0, 1, 2])
    logits, term, comp = wm(obs, act)
    assert logits.shape == (3, 16, 64, 64)
    assert term.shape == (3,) and comp.shape == (3,)
    nf, t, c = wm.predict(rand_frame(1), 2, device=torch.device("cpu"))
    assert nf.shape == (64, 64) and nf.dtype == np.uint8
    assert isinstance(t, bool) and isinstance(c, bool)


def test_model_env_interface_and_determinism():
    """ModelEnv must expose the Go-Explore env interface and replay exactly."""
    from claude_automate.framework.world_model import FrameWorldModel, ModelEnv
    wm = FrameWorldModel(n_colors=16, n_actions=4, base_ch=16)
    init = rand_frame(7)
    env = ModelEnv(wm, init, n_actions=4, device=torch.device("cpu"),
                   masked_rows=slice(63, 64), max_steps=20)
    assert env.n_actions == 4
    f0 = env.reset()
    assert np.array_equal(f0, init)
    traj = [0, 1, 2, 3, 0, 1]
    seq_a = [env.step(a)[0] for a in traj]
    env.reset()
    seq_b = [env.step(a)[0] for a in traj]            # replay must be identical
    assert all(np.array_equal(x, y) for x, y in zip(seq_a, seq_b))


def _structured_frame(seed):
    """A frame like an ARC observation: mostly background, a few small blocks."""
    rng = np.random.default_rng(seed)
    f = np.zeros((64, 64), dtype=np.uint8)
    for _ in range(4):
        r, c = rng.integers(0, 58, 2)
        f[r:r + 5, c:c + 5] = rng.integers(1, 13)
    return f


def test_world_model_overfits_tiny_set():
    from claude_automate.framework.world_model import FrameWorldModel
    from claude_automate.framework.wm_train import train_world_model
    wm = FrameWorldModel(n_colors=16, n_actions=4, base_ch=16)
    # 4 transitions: next_frame is a small fixed edit of an ARC-like frame
    transitions = []
    for i in range(4):
        f = _structured_frame(50 + i)
        nf = f.copy()
        nf[20:24, 20:24] = (i % 12) + 1
        transitions.append((f, i % 4, nf, False, False))
    before = evaluate(wm, transitions)
    train_world_model(wm, transitions, device=torch.device("cpu"),
                      epochs=120, batch_size=4, verbose=False)
    after = evaluate(wm, transitions)
    assert after > before and after > 0.95           # fits the tiny set


def evaluate(wm, transitions):
    from claude_automate.framework.wm_train import evaluate_world_model
    return evaluate_world_model(wm, transitions, torch.device("cpu"))["pixel_acc"]


def test_ppo_update_runs_and_changes_params():
    cfg = Config(hidden_dim=64, minibatch_size=8, ppo_epochs=2)
    model = ActorCritic(n_actions=4, n_colors=16, hidden_dim=64)
    ppo = PPO(model, cfg, device=torch.device("cpu"))

    roll = Rollout()
    for i in range(16):
        roll.frames.append(rand_frame(100 + i))
        roll.actions.append(i % 4)
        roll.log_probs.append(-1.386)            # log(1/4)
        roll.values.append(0.0)
        roll.rewards.append(1.0 if i % 5 == 0 else -0.1)
        roll.dones.append((i + 1) % 8 == 0)

    before = model.policy_head.weight.detach().clone()
    stats = ppo.update(roll)
    after = model.policy_head.weight.detach()
    assert not torch.equal(before, after)        # an update actually happened
    assert set(stats) == {"policy_loss", "value_loss", "entropy",
                          "clip_frac", "approx_kl"}
    assert np.isfinite(stats["policy_loss"])
