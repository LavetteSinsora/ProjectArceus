"""exp_010_2 — random-policy JEPA pretraining, then unfrozen PPO on real LS20.

Pipeline:
    1. collect.py     — uniform-random agent gathers (s, a, s') transitions
    2. train_jepa.py  — pretrain encoder+predictor+IDM until JEPA loss plateaus
                        (reports the number of environment steps the data used)
    3. train_ppo.py   — PPO from the pretrained encoder as initial weights,
                        encoder UNFROZEN (receives PPO gradients)
"""
