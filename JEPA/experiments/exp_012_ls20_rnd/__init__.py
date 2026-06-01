"""exp_012 — intrinsic-reward exploration baselines on the *real* LS20 game.

Container package. Two sub-experiments, both on top of the exp_010 real-LS20
CNN+PPO infrastructure so the *only* change versus exp_010_0 is the exploration
mechanism:

    exp_012_0_icm_baseline   Intrinsic Curiosity Module (Pathak et al. 2017)
    exp_012_1_rnd_baseline   Random Network Distillation (Burda et al. 2018)

The headline metric is the number of environment steps to the *first* extrinsic
reward (first level completion), measured against the ~50k-step uniform-random
baseline (memory/finding_random_policy_ls20_l1.md).

See exp_012_1_rnd_baseline/SYSTEM_CARD.md for the full design.
"""
