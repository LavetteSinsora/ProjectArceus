"""
Run the Stochastic Goose replication on one ARC-AGI-3 puzzle.

Usage:
    uv run replication/card_stochastic_goose/run.py
    uv run replication/card_stochastic_goose/run.py --game ls20 --max-minutes 30

The script will:
  1. Open a scorecard on the ARC-AGI-3 server (ONLINE mode)
  2. Run the Stochastic Goose CNN-RL agent on the chosen game
  3. Close the scorecard and print the replay URL

Replay viewable at: https://three.arcprize.org/scorecards/<card_id>

Requirements:
  - ARC_API_KEY must be set in .env (already present in this repo)
  - torch must be installed:  uv add torch
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Make sure project root is on path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from agent import Action

BASE_URL = "https://three.arcprize.org"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stochastic Goose replication runner")
    p.add_argument(
        "--game",
        default="ls20",
        help="ARC-AGI-3 game ID to run (default: ls20)",
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        default=30.0,
        help="Agent time limit in minutes (default: 30). Original used 480 (8h).",
    )
    p.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Hard cap on total actions taken (optional; useful for quick smoke tests).",
    )
    p.add_argument(
        "--save-dir",
        default=None,
        help="Directory to auto-save model checkpoints when a level is completed.",
    )
    p.add_argument(
        "--load-model",
        default=None,
        help="Path to a .pt checkpoint to load before playing (runs inference-only, no training).",
    )
    return p.parse_args()


def run(
    game_id: str,
    max_minutes: float,
    max_actions: Optional[int],
    save_dir: Optional[str] = None,
    load_model: Optional[str] = None,
) -> None:
    arc = Arcade(operation_mode=OperationMode.ONLINE)

    # List available games so we can validate the game_id
    games = arc.get_environments()
    game_ids = [e.game_id for e in games]
    print(f"[Runner] {len(game_ids)} games available from API.")

    # Normalise: accept short IDs like "ls20" even if server returns "ls20-abcd1234"
    matched = [g for g in game_ids if g == game_id or g.startswith(f"{game_id}-")]
    if not matched:
        print(f"[Runner] WARNING: game '{game_id}' not found. Available: {sorted(game_ids)}")
        print(f"[Runner] Proceeding anyway — the server may accept it.")
    else:
        game_id = matched[0]
        print(f"[Runner] Using game: {game_id}")

    # Open scorecard (server-side — creates the replay record)
    card_id = arc.open_scorecard(tags=["stochastic_goose_replication"])
    replay_url = f"{BASE_URL}/scorecards/{card_id}"
    print(f"\n[Runner] Scorecard opened: {card_id}")
    print(f"[Runner] Replay URL (available after close): {replay_url}\n")

    # Ensure scorecard is always closed on exit
    closed = False

    def _close(sig=None, frame=None):
        nonlocal closed
        if not closed:
            closed = True
            print("\n[Runner] Closing scorecard…")
            try:
                result = arc.close_scorecard(card_id)
                if result is not None:
                    print(f"[Runner] Score: {result.score:.2f}%  levels: {result.total_levels_completed}")
            except Exception as e:
                print(f"[Runner] close_scorecard error: {e}")
            print(f"\n{'='*60}")
            print(f"  card_id  : {card_id}")
            print(f"  Replay   : {replay_url}")
            print(f"{'='*60}\n")

    signal.signal(signal.SIGINT, _close)
    signal.signal(signal.SIGTERM, _close)

    # Create environment (RemoteEnvironmentWrapper in ONLINE mode)
    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        print(f"[Runner] Could not create environment for '{game_id}'. Exiting.")
        _close()
        return

    # Initialise agent
    inference_only = load_model is not None
    agent = Action(
        game_id=game_id,
        max_minutes=max_minutes,
        save_dir=save_dir,
        inference_only=inference_only,
    )
    if load_model is not None:
        agent.load_model(load_model)

    # arc.make() auto-resets internally; use that initial frame directly
    frame = env.observation_space
    if frame is None:
        print("[Runner] No initial frame from env; sending explicit RESET…")
        frame = env.step(GameAction.RESET)
    if frame is None:
        print("[Runner] Failed to get initial frame. Exiting.")
        _close()
        return
    frames = [frame]

    total_actions = 0
    t0 = time.time()

    print(f"[Runner] Starting agent loop (limit: {max_minutes:.0f} min)…")
    try:
        while True:
            # Time / action budget checks
            if agent.is_done(frame):
                reason = "WIN" if frame.state is GameState.WIN else "time limit reached"
                print(f"[Runner] Done ({reason}) after {total_actions} actions.")
                break
            if max_actions is not None and total_actions >= max_actions:
                print(f"[Runner] Action cap ({max_actions}) reached.")
                break

            # Agent chooses action
            action = agent.choose_action(frames, frame)

            # Execute — pass coordinate data for ACTION6 clicks
            if action == GameAction.ACTION6:
                ad = action.action_data
                step_frame = env.step(action, data={"x": int(ad.x), "y": int(ad.y)})
            else:
                step_frame = env.step(action)

            total_actions += 1

            # Guard: skip None responses (transient API errors) without breaking the loop
            if step_frame is None:
                print(f"[Runner] WARNING: step returned None at action {total_actions}, retrying…")
                continue

            frame = step_frame
            frames.append(frame)

            # Progress log every 100 steps
            if total_actions % 100 == 0:
                elapsed = time.time() - t0
                fps = total_actions / elapsed if elapsed > 0 else 0
                lvls = getattr(frame, "levels_completed", "?")
                print(
                    f"[Runner] step={total_actions}  levels={lvls}  "
                    f"state={frame.state.name}  fps={fps:.1f}"
                )

    except Exception as exc:
        print(f"[Runner] Agent error: {exc}")
        import traceback
        traceback.print_exc()

    _close()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.game,
        args.max_minutes,
        args.max_actions,
        save_dir=args.save_dir,
        load_model=args.load_model,
    )
