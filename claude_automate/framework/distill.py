"""Behavior-cloning distillation of a solution trajectory into a policy.

Go-Explore returns a completing action sequence. `distill_trajectory` bakes it
into the `ActorCritic` policy network by supervised cross-entropy on the
trajectory's (frame, action) pairs. On a deterministic environment the greedy
policy then reproduces the solve.

Game-agnostic: it only consumes (frame, action) pairs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from claude_automate.framework.env_api import frame_to_tensor


def distill_trajectory(model, frames, actions, cfg, device,
                       epochs: int = 4000, lr: float = 2e-3,
                       verbose: bool = True) -> dict:
    """Behavior-clone (frames -> actions) into `model`'s policy head.

    On a deterministic environment the eval frames exactly equal the trajectory
    frames, so the goal is simply 100% train action-match — at which point the
    greedy policy reproduces the solve exactly. Training therefore runs until
    100% match (then a short consolidation) or `epochs` is exhausted.

    Returns a dict of final-epoch metrics.
    """
    obs = torch.stack(
        [frame_to_tensor(f, cfg.n_colors) for f in frames]
    ).to(device)
    targets = torch.as_tensor(actions, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    last = {}
    first_perfect = None
    for ep in range(epochs):
        logits, _ = model.forward(obs)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            acc = float((logits.argmax(-1) == targets).float().mean())
        last = {"epoch": ep + 1, "loss": float(loss.item()), "train_acc": acc}
        if verbose and ((ep + 1) % 200 == 0 or acc == 1.0):
            print(f"[distill] epoch {ep+1:4d} | loss {loss.item():.4f} "
                  f"| action-match {acc:.1%}")
        # Run a little past the first perfect epoch to widen the logit margin
        # (keeps argmax robust), then stop.
        if acc == 1.0:
            if first_perfect is None:
                first_perfect = ep
            elif ep - first_perfect >= 50:
                break
    return last


def distill_trajectory_recurrent(model, frames, actions, cfg, device,
                                 epochs: int = 4000, lr: float = 2e-3,
                                 verbose: bool = True) -> dict:
    """Sequence behavior-cloning into a `RecurrentActorCritic`.

    The GRU is unrolled over the whole trajectory, so it can fit a plan that
    revisits observations with different actions — which a stateless policy
    cannot. At eval the deterministic env reproduces the same observation
    sequence, hence the same hidden states, hence the same actions: 100%
    sequence-match ⇒ a 100% reproduction of the solve.

    Returns a dict of final-epoch metrics.
    """
    obs = torch.stack(
        [frame_to_tensor(f, cfg.n_colors) for f in frames]
    ).to(device)                                     # (T, C, H, W)
    targets = torch.as_tensor(actions, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    last, first_perfect = {}, None
    for ep in range(epochs):
        logits = model.forward_sequence(obs)         # (T, n_actions)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            acc = float((logits.argmax(-1) == targets).float().mean())
        last = {"epoch": ep + 1, "loss": float(loss.item()), "train_acc": acc}
        if verbose and ((ep + 1) % 200 == 0 or acc == 1.0):
            print(f"[distill-rnn] epoch {ep+1:4d} | loss {loss.item():.4f} "
                  f"| action-match {acc:.1%}")
        if acc == 1.0:
            if first_perfect is None:
                first_perfect = ep
            elif ep - first_perfect >= 50:
                break
    return last
