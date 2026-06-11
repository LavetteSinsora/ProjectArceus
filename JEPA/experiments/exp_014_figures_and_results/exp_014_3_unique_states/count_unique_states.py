"""exp_014_3 — Count distinct masked states per game × level under a random policy.

For each game/level, we roam with 16 envs for MAX_UPDATES rollouts (128 steps × 16 envs each),
hash the timer-masked board of every observation, and count unique hashes. Converges quickly —
most game/level combos saturate within the first few hundred updates.

    uv run python -m JEPA.experiments.exp_014_figures_and_results.exp_014_3_unique_states.count_unique_states

Writes results/unique_states.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel

N_ENVS = 16
ROLLOUT_STEPS = 128
MAX_UPDATES = 400           # 400 * 2048 = 819k env-steps — enough for any L1
SEED = 0
TIMER_ROW0 = 60             # rows 60-63 = step-timer/energy UI → mask out

GAMES_LEVELS = [
    ("ls20", 0),   # L1
    ("tu93", 0),
    ("re86", 0),
    ("g50t", 0),
]

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(parents=True, exist_ok=True)


def mask_board(frame: np.ndarray) -> np.ndarray:
    m = frame.copy()
    m[..., TIMER_ROW0:, :] = 0
    return m


def state_key(masked_frame: np.ndarray) -> bytes:
    return masked_frame.tobytes()


def count_states(game: str, level_index: int) -> dict:
    rng = np.random.default_rng(SEED)
    envs = VecLS20EnvLevel(env_name=game, n_envs=N_ENVS,
                           max_episode_steps=200, seed=SEED, level_index=level_index)
    seen: set[bytes] = set()
    history: list[int] = []     # unique count after each update

    for u in range(MAX_UPDATES):
        for _ in range(ROLLOUT_STEPS):
            a = rng.integers(0, envs.n_actions, size=envs.n_envs)
            nobs, _r, dones, _i = envs.step(a)
            m = mask_board(nobs)
            for i in range(envs.n_envs):
                if not dones[i]:
                    seen.add(state_key(m[i]))
        history.append(len(seen))
        if u % 50 == 0 or (u > 10 and history[-1] == history[-6]):
            print(f"  [{game} L{level_index+1}] u{u:>4}: {len(seen)} distinct states", flush=True)
        # early-stop: no new states in last 20 updates
        if u >= 20 and history[-1] == history[-21]:
            print(f"  [{game} L{level_index+1}] saturated at {len(seen)} states (u={u})")
            break

    envs.close() if hasattr(envs, "close") else None
    return {"game": game, "level_index": level_index, "n_distinct_states": len(seen),
            "n_updates": len(history), "history": history}


def main():
    results = {}
    for game, level in GAMES_LEVELS:
        tag = f"{game}_L{level+1}"
        print(f"\n=== {tag} ===")
        r = count_states(game, level)
        results[tag] = r
        print(f"  → {r['n_distinct_states']} distinct masked states")

    out = RES / "unique_states.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nWrote {out}")
    print("\nSummary:")
    for tag, r in results.items():
        print(f"  {tag}: {r['n_distinct_states']} states")


if __name__ == "__main__":
    main()
