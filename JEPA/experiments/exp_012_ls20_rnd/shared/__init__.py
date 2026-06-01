"""Shared library for exp_012 (real-LS20 PPO + intrinsic-reward exploration).

Unchanged infrastructure (vec env, metrics, device selection) is imported
directly from the exp_010 shared library to avoid divergence; only the modules
that change for the dual-stream RND recipe live here:

    model.py    ActorCritic with TWO value heads (V_E extrinsic, V_I intrinsic)
    rnd.py      frozen random target + trainable predictor + reward normalisers
    rollout.py  dual-stream rollout buffer + per-stream GAE (intrinsic = non-episodic)
    ppo.py      combined-advantage PPO update + RND predictor distillation loss
    trainer.py  the RND training loop (intrinsic-reward compute + normalisation)
    evaluator.py / debug_runner.py  thin adaptations for the 4-tuple forward
"""
