"""Shared library for exp_011 (real-LS20 exploration methods).

Deliberately thin: the environment, rollout buffer, GAE, PPO update, evaluator,
metrics, device selection and CNN encoder are all *imported unchanged* from the
exp_010 shared library, so any difference in outcome is attributable to the
exploration mechanism (ICM) and never to a divergent PPO/backbone impl. The only
new code here is `icm.py` (the Intrinsic Curiosity Module) and a `trainer.py`
that inserts the intrinsic reward into the otherwise-identical exp_010 loop.
"""
