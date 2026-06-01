"""Level-selectable vec env for exp_011.

exp_010's `VecLS20Env` always starts every episode on level 0 (LS20 Level 1).
To test ICM on a *harder* level (the user's request after ICM solved L1), we
need each episode to start on an arbitrary 0-indexed level. The ARC engine
exposes `game.set_level(idx)` for exactly this; `claude_automate` used it to
solve LS20 L2-L4. We reuse that mechanism here without touching exp_010.

`VecLS20EnvLevel` subclasses exp_010's `VecLS20Env` and, when `level_index > 0`,
wraps each underlying env so its `reset()` jumps to the chosen level. Everything
else — the synchronous step loop, terminal/truncation logic, per-episode
success tracking — is inherited unchanged, so the only thing that differs from
the L1 runs is which level the agent is dropped into.

Terminal logic is unchanged: the LS20 wrapper stops at `levels_completed >= 1`,
which (because `set_level` leaves the completed-counter at 0) means "complete
the level you were dropped into" — i.e. solve L2 when started on L2.
"""

from __future__ import annotations

from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env


class _LevelStartWrapper:
    """Make `reset()` start the episode on a chosen 0-indexed level.

    Mirrors claude_automate.framework.env_api.LevelStartWrapper (kept local to
    avoid a JEPA->claude_automate import dependency). After the normal reset
    (which lands on level 0) it calls `set_level(level_index)` and renders that
    level's first frame with a no-op action — the first GameAction *beyond* this
    game's action set (LS20 has 4 actions -> ACTION5), which advances no state
    and costs no energy. All other calls delegate to the wrapped env.
    """

    def __init__(self, base_env, level_index: int):
        self._base = base_env
        self._level_index = level_index
        self._render_action = f"ACTION{base_env.n_actions + 1}"

    def reset(self):
        self._base.reset()
        if self._level_index > 0:
            from arcengine import ActionInput, GameAction
            game = self._base._env._game
            game.set_level(self._level_index)
            raw = game.perform_action(
                ActionInput(id=getattr(GameAction, self._render_action)),
                raw=True,
            )
            self._base._latest_raw = raw
        return self._base._extract(self._base._latest_raw)

    def __getattr__(self, name):
        # Only reached for attributes not defined on the wrapper itself
        # (step, level_completed, n_actions, _extract, _env, ...).
        return getattr(self._base, name)


class VecLS20EnvLevel(VecLS20Env):
    """N synchronous real-LS20 instances, every episode starting on `level_index`
    (0-indexed). `level_index=0` is byte-for-byte the exp_010 behaviour."""

    def __init__(self, env_name: str = "ls20", n_envs: int = 8,
                 max_episode_steps: int = 200, seed: int = 0,
                 environments_dir: str | None = None, level_index: int = 0):
        # Parent builds N envs (one shared offline Arcade) and resets them at
        # level 0.
        super().__init__(env_name=env_name, n_envs=n_envs,
                         max_episode_steps=max_episode_steps, seed=seed,
                         environments_dir=environments_dir)
        self.level_index = level_index
        if level_index > 0:
            # Wrap each env so reset() (here and inside step() on episode end)
            # starts the agent on the chosen level, then re-reset.
            self.envs = [_LevelStartWrapper(e, level_index) for e in self.envs]
            self.reset_all()
