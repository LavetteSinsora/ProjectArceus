"""exp_010_1 — joint online JEPA + PPO on real LS20.

Policy and encoder are both randomly initialised. PPO runs from scratch, and
on every update the encoder additionally receives a JEPA (+IDM) gradient from
the agent's *own* (on-policy) rollout transitions. There is no separate
pretraining phase — the world-model objective and the policy objective shape
the shared encoder simultaneously.
"""
