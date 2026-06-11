"""exp_019 Gate C — in-model Go-Explore with real-env verification + correction.

Loop (resumable across 45 s sandbox calls via state pickle):
  1. Go-Explore expands an archive of cells *inside the DeltaWorldModel*
     (batched model steps; zero real env interaction).
  2. Periodically, the most promising unverified archive trajectories are
     replayed in the REAL env ("verification"). Every real step is counted.
     - If the real rollout completes the level: done — report total real steps.
     - Wherever the model's imagined frame diverges from the real frame, the
       real transitions are appended to a correction buffer.
  3. The model is fine-tuned on correction transitions (+ replay of original
     train data to prevent forgetting), and the archive is re-seeded.
Honest accounting: report real_steps (resets count as 1 obs, steps as 1 each),
model_steps, and wall time. Baseline: from-scratch Go-Explore on L4 = 321.6k
real steps (claude_automate exp 005).

Verification priority: archive cells are scored by trajectory length (deeper =
more informative) mixed with novelty (rarely-chosen cells); completion cannot
be detected in-model (the completion head has ~11 positive examples — known
dead end from exp_006), so verification doubles as completion probe and as
model correction. This is the honest budget: we pay real steps only for what
the model cannot know.

  python3 plan_in_model.py --run runs/delta_v2 --level 3 --seconds 33 \
      --state runs/plan_L4/state.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))          # repo root
sys.path.insert(0, str(HERE.parents[2] / "claude_automate"))

from model import DeltaWorldModel  # noqa: E402

MASK_ROWS = slice(61, 63)


def mask(frame: np.ndarray) -> np.ndarray:
    f = frame.copy()
    f[MASK_ROWS, :] = 0
    return f


def cell_code(frame: np.ndarray) -> int:
    return hash(frame.tobytes())


class PlannerState:
    def __init__(self, seed_frame):
        self.archive = {}           # code -> dict(traj, frame, chosen, verified_ok)
        self.add(seed_frame, [])
        self.real_steps = 0
        self.model_steps = 0
        self.verifications = 0
        self.corrections = []       # (frame, action, next_frame, terminal)
        self.solved = False
        self.solution = None
        self.history = []           # event log

    def add(self, frame, traj):
        code = cell_code(frame)
        if code not in self.archive or len(traj) < len(self.archive[code]["traj"]):
            self.archive[code] = {"traj": list(traj), "frame": frame.copy(),
                                  "chosen": 0, "verified": False}
            return True
        return False


def model_burst(model, st: PlannerState, rng, n_select=24, burst=12):
    """Select archive cells, batch-roll random bursts in the model."""
    cells = list(st.archive.values())
    # preference: less-chosen, then longer trajectories (frontier bias)
    weights = np.array([1.0 / np.sqrt(1 + c["chosen"]) *
                        (1 + 0.1 * len(c["traj"])) for c in cells])
    weights /= weights.sum()
    idx = rng.choice(len(cells), size=min(n_select, len(cells)),
                     p=weights, replace=False)
    sel = [cells[i] for i in idx]
    for c in sel:
        c["chosen"] += 1
    frames = np.stack([c["frame"] for c in sel])
    trajs = [list(c["traj"]) for c in sel]
    new_cells = 0
    for t in range(burst):
        acts = rng.integers(0, 4, size=len(sel))
        nxt, term_p = model.predict_batch(frames, acts)
        nxt[:, MASK_ROWS, :] = 0
        st.model_steps += len(sel)
        for i in range(len(sel)):
            trajs[i].append(int(acts[i]))
            if st.add(nxt[i], trajs[i]):
                new_cells += 1
        frames = nxt
    return new_cells


def verify(env_factory, model, st: PlannerState, max_verify=3, retrain_buf=512):
    """Replay top-priority unverified trajectories in the real env."""
    env = env_factory()
    cells = [c for c in st.archive.values() if not c["verified"] and c["traj"]]
    cells.sort(key=lambda c: -len(c["traj"]))  # deepest first
    for c in cells[:max_verify]:
        c["verified"] = True
        obs = env.reset()
        st.real_steps += 1
        st.verifications += 1
        model_frame = mask(obs)
        diverged_at = None
        for t, a in enumerate(c["traj"]):
            nxt, term = env.step(a)
            st.real_steps += 1
            pred, _ = model.predict_batch(model_frame[None], np.array([a]))
            pred = pred[0]; pred[MASK_ROWS, :] = 0
            real_next = mask(nxt)
            if not (pred == real_next).all():
                st.corrections.append((model_frame.copy(), a, real_next.copy(),
                                       bool(term)))
                if diverged_at is None:
                    diverged_at = t
            if env.level_completed:
                st.solved = True
                st.solution = c["traj"][:t + 1]
                st.history.append({"event": "SOLVED",
                                   "real_steps": st.real_steps,
                                   "solution_len": t + 1})
                return
            if term:
                break
            model_frame = pred if diverged_at is None else real_next
            obs = nxt
        st.history.append({"event": "verify", "traj_len": len(c["traj"]),
                           "diverged_at": diverged_at,
                           "real_steps": st.real_steps})


def finetune(model, st: PlannerState, opt, steps=40, batch=16, pos_weight=10.0):
    """Fine-tune on correction transitions (with train-data replay)."""
    if not st.corrections:
        return 0.0
    d = np.load(HERE / "data/ls20_L1.npz")  # replay source (train levels)
    rf, ra, rn = d["frames"], d["actions"], d["next_frames"]
    rf = rf.copy(); rf[:, MASK_ROWS, :] = 0
    rn = rn.copy(); rn[:, MASK_ROWS, :] = 0
    cf = np.stack([c[0] for c in st.corrections])
    ca = np.array([c[1] for c in st.corrections])
    cn = np.stack([c[2] for c in st.corrections])
    losses = []
    pw = torch.tensor(pos_weight)
    for _ in range(steps):
        # half corrections, half replay
        ic = np.random.randint(0, len(cf), batch // 2)
        ir = np.random.randint(0, len(rf), batch // 2)
        f = torch.from_numpy(np.concatenate([cf[ic], rf[ir]])).long()
        a = torch.from_numpy(np.concatenate([ca[ic], ra[ir]])).long()
        nf = torch.from_numpy(np.concatenate([cn[ic], rn[ir]])).long()
        oh = F.one_hot(f, 16).permute(0, 3, 1, 2).float()
        chg, col, _, noop = model(oh, a)
        changed = (f != nf)
        moved = changed.flatten(1).any(1)
        loss = F.binary_cross_entropy_with_logits(
            noop, (~moved).float())
        if moved.any():
            loss = loss + F.binary_cross_entropy_with_logits(
                chg[moved], changed[moved].float(), pos_weight=pw)
            loss = loss + F.cross_entropy(
                col.permute(0, 2, 3, 1)[changed], nf[changed])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/delta_v2")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--state", default="runs/plan_L4/state.pkl")
    ap.add_argument("--seconds", type=float, default=33.0)
    ap.add_argument("--verify-every", type=int, default=6,
                    help="model-burst rounds between verification batches")
    args = ap.parse_args()

    torch.set_num_threads(4)
    state_path = HERE / args.state
    state_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    from framework.env_api import make_arc_env  # noqa: E402
    env_factory = lambda: make_arc_env("ls20-9607627b", level_index=args.level)

    # model (+ planner-local fine-tuned weights if present)
    ck = torch.load(HERE / args.run / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    model = DeltaWorldModel()
    ft_path = state_path.parent / "model_ft.pt"
    model.load_state_dict(torch.load(ft_path, weights_only=False)
                          if ft_path.exists() else ck["model"])
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    if state_path.exists():
        st = pickle.load(open(state_path, "rb"))
    else:
        env = env_factory()
        seed = mask(env.reset())
        st = PlannerState(seed)
        st.real_steps += 1   # the seed reset
        st.history.append({"event": "seed", "real_steps": 1})

    t0 = time.time()
    rounds = 0
    while time.time() - t0 < args.seconds and not st.solved:
        new = model_burst(model, st, rng)
        rounds += 1
        if rounds % args.verify_every == 0:
            n_corr_before = len(st.corrections)
            verify(env_factory, model, st)
            if len(st.corrections) > n_corr_before:
                fl = finetune(model, st, opt)
                st.history.append({"event": "finetune", "loss": fl,
                                   "n_corrections": len(st.corrections)})
                torch.save(model.state_dict(), ft_path)

    pickle.dump(st, open(state_path, "wb"))
    print(json.dumps({
        "solved": st.solved,
        "solution_len": len(st.solution) if st.solution else None,
        "real_steps": st.real_steps, "model_steps": st.model_steps,
        "archive": len(st.archive), "verifications": st.verifications,
        "corrections": len(st.corrections), "rounds": rounds,
        "secs": round(time.time() - t0, 1),
    }))


if __name__ == "__main__":
    main()
