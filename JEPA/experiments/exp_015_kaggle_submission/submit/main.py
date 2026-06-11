#!/usr/bin/env python3
"""
ARC Prize 2026 — ARC-AGI-3 submission entry point.

Usage (offline smoke-test):
    python main.py --agent leakyrnd --game ls20 --offline --max-actions 500

Usage (online Kaggle eval):
    python main.py --agent leakyrnd --game <game_id>

The Kaggle harness runs this file as:
    uv run main.py --agent leakyrnd --game <id>
"""
from __future__ import annotations

import argparse
import sys

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from agents.leaky_rnd_agent import LeakyRNDAgent

AVAILABLE_AGENTS = {
    "leakyrnd": LeakyRNDAgent,
}


def main() -> None:
    p = argparse.ArgumentParser(description="ARC-AGI-3 agent runner")
    p.add_argument("--agent",       required=True,
                   help="agent name (choices: " + ", ".join(AVAILABLE_AGENTS) + ")")
    p.add_argument("--game",        required=True,
                   help="game id, e.g. 'ls20' or 'ls20-9607627b'")
    p.add_argument("--offline",     action="store_true",
                   help="use OFFLINE mode (local env_files) instead of Kaggle ONLINE")
    p.add_argument("--max-actions", type=int, default=None,
                   help="action cap (default: agent's MAX_ACTIONS)")
    a = p.parse_args()

    name = a.agent.lower()
    if name not in AVAILABLE_AGENTS:
        print(f"Unknown agent '{name}'. Available: {list(AVAILABLE_AGENTS)}", file=sys.stderr)
        sys.exit(1)

    mode = OperationMode.OFFLINE if a.offline else OperationMode.ONLINE
    arc  = Arcade(operation_mode=mode)

    # Resolve full versioned game_id (e.g. "ls20" → "ls20-9607627b")
    try:
        envs = arc.get_environments()
        gid  = next((e.game_id for e in envs if e.game_id.startswith(a.game)), a.game)
    except Exception:
        gid = a.game

    env   = arc.make(gid)
    agent = AVAILABLE_AGENTS[name](game_id=gid)

    print(f"[main] agent={name}  game={gid}  mode={mode.name}", flush=True)
    result = agent.run_episode(env, max_actions=a.max_actions)
    print(f"[main] DONE  levels={result['levels_completed']}  "
          f"steps={result['total_steps']}  wall={result['wall_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
