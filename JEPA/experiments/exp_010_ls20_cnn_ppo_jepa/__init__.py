"""exp_010 — CNN+PPO and JEPA-encoder baselines on the *real* LS20 game.

Container package. The three sub-experiments are:

    exp_010_0_cnn_ppo_baseline      vanilla CNN + PPO (the 7_0 recipe, on real LS20)
    exp_010_1_jepa_joint_online     JEPA encoder trained online on the PPO agent's
                                    own (on-policy) transitions, jointly with PPO
    exp_010_2_jepa_random_pretrain  JEPA encoder pretrained on random-policy data
                                    until plateau, then PPO (unfrozen) from that init

All three share `shared/` (env adapter, model, PPO, JEPA modules, trainer).
Everything runs against the 64x64 ARC-AGI-3 LS20 environment via
`JEPA.shared.env_wrapper.LS20Env`, so results surface on the main JEPA
dashboard (port 8787) exactly like exp_001-004.
"""
