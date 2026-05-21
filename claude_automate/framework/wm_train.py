"""Transition collection and training for the shared world model.

`collect_transitions` gathers `(frame, action, next_frame, terminal,
completed)` tuples from an env — random rollouts for movement/wall/death
coverage, plus optional replays of a known solution for modifier-tile and
level-completion transitions.

`train_world_model` fits a `FrameWorldModel`; `evaluate_world_model` measures
next-frame / terminal / completed accuracy on a held-out transition set — the
make-or-break transfer metric.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from claude_automate.framework.env_api import frame_to_tensor


# ── transition collection ────────────────────────────────────────────────────

def collect_transitions(env, rng, n_random_episodes: int = 80,
                        solution: list[int] | None = None,
                        n_solution_replays: int = 0,
                        max_steps: int = 220) -> list:
    """Return a list of (frame, action, next_frame, terminal, completed)."""
    transitions = []

    def rollout(action_source):
        frame = env.reset()
        for t in range(max_steps):
            a = action_source(t)
            if a is None:
                break
            nxt, terminal = env.step(a)
            transitions.append((np.asarray(frame, np.uint8), int(a),
                                np.asarray(nxt, np.uint8), bool(terminal),
                                bool(env.level_completed)))
            frame = nxt
            if terminal:
                break

    for _ in range(n_random_episodes):
        rollout(lambda t: int(rng.integers(env.n_actions)))

    if solution:
        for _ in range(n_solution_replays):
            rollout(lambda t: solution[t] if t < len(solution) else None)

    return transitions


# ── dataset packing ──────────────────────────────────────────────────────────

def _pack(transitions):
    frames = np.stack([t[0] for t in transitions])
    actions = np.array([t[1] for t in transitions], dtype=np.int64)
    next_frames = np.stack([t[2] for t in transitions])
    terms = np.array([t[3] for t in transitions], dtype=np.float32)
    comps = np.array([t[4] for t in transitions], dtype=np.float32)
    return frames, actions, next_frames, terms, comps


def _batch_onehot(frames_u8, n_colors, device):
    return torch.stack([frame_to_tensor(f, n_colors) for f in frames_u8]
                       ).to(device)


# ── training ─────────────────────────────────────────────────────────────────

def train_world_model(model, transitions, device, epochs: int = 12,
                      batch_size: int = 64, lr: float = 1e-3,
                      n_colors: int = 16, verbose: bool = True) -> dict:
    """Fit the world model. Returns final-epoch metrics."""
    frames, actions, next_frames, terms, comps = _pack(transitions)
    n = len(transitions)
    # completion transitions are rare — up-weight them in the BCE.
    n_comp = max(float(comps.sum()), 1.0)
    comp_pos_weight = torch.tensor([(n - n_comp) / n_comp], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = {}
    for ep in range(epochs):
        perm = np.random.permutation(n)
        tot = {"frame_ce": 0.0, "pix_acc": 0.0, "term_acc": 0.0,
               "comp_recall": 0.0, "nb": 0}
        comp_seen = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            obs = _batch_onehot(frames[idx], n_colors, device)
            tgt = torch.as_tensor(next_frames[idx], dtype=torch.long,
                                  device=device)               # (B,64,64)
            act = torch.as_tensor(actions[idx], device=device)
            term = torch.as_tensor(terms[idx], device=device)
            comp = torch.as_tensor(comps[idx], device=device)

            logits, term_l, comp_l = model(obs, act)
            frame_ce = F.cross_entropy(logits, tgt)
            term_loss = F.binary_cross_entropy_with_logits(term_l, term)
            comp_loss = F.binary_cross_entropy_with_logits(
                comp_l, comp, pos_weight=comp_pos_weight)
            loss = frame_ce + term_loss + comp_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                tot["frame_ce"] += float(frame_ce)
                tot["pix_acc"] += float((logits.argmax(1) == tgt).float().mean())
                tot["term_acc"] += float(
                    ((term_l > 0).float() == term).float().mean())
                if comp.sum() > 0:
                    rec = (((comp_l > 0).float() * comp).sum() / comp.sum())
                    tot["comp_recall"] += float(rec)
                    comp_seen += 1
                tot["nb"] += 1
        nb = tot["nb"]
        last = {
            "epoch": ep + 1,
            "frame_ce": round(tot["frame_ce"] / nb, 4),
            "pixel_acc": round(tot["pix_acc"] / nb, 4),
            "terminal_acc": round(tot["term_acc"] / nb, 4),
            "completed_recall": round(tot["comp_recall"] / max(comp_seen, 1), 3),
        }
        if verbose:
            print(f"[wm-train] ep{last['epoch']:2d} "
                  f"frame_ce={last['frame_ce']:.4f} "
                  f"pixel_acc={last['pixel_acc']:.4f} "
                  f"term_acc={last['terminal_acc']:.3f} "
                  f"comp_recall={last['completed_recall']:.3f}")
    return last


@torch.no_grad()
def evaluate_world_model(model, transitions, device, n_colors: int = 16,
                         batch_size: int = 64) -> dict:
    """Held-out metrics — the transfer test. Reports next-frame pixel accuracy,
    whole-frame exact-match rate, terminal accuracy, completion recall."""
    frames, actions, next_frames, terms, comps = _pack(transitions)
    n = len(transitions)
    model.eval()
    pix_ok = exact_ok = term_ok = 0
    comp_tp = comp_fp = comp_fn = 0
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        obs = _batch_onehot(frames[sl], n_colors, device)
        tgt = torch.as_tensor(next_frames[sl], dtype=torch.long, device=device)
        act = torch.as_tensor(actions[sl], device=device)
        term = torch.as_tensor(terms[sl], device=device)
        comp = torch.as_tensor(comps[sl], device=device)
        logits, term_l, comp_l = model(obs, act)
        pred = logits.argmax(1)
        pix_ok += float((pred == tgt).float().sum())
        exact_ok += int((pred == tgt).flatten(1).all(1).sum())
        term_ok += int(((term_l > 0).float() == term).sum())
        cp = (comp_l > 0).float()
        comp_tp += int((cp * comp).sum())
        comp_fp += int((cp * (1 - comp)).sum())
        comp_fn += int(((1 - cp) * comp).sum())
    return {
        "n": n,
        "pixel_acc": round(pix_ok / (n * 64 * 64), 5),
        "exact_frame_acc": round(exact_ok / n, 4),
        "terminal_acc": round(term_ok / n, 4),
        "completed_tp": comp_tp, "completed_fp": comp_fp,
        "completed_fn": comp_fn,
    }
