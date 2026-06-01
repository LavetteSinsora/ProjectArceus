"""exp_011_2 — the exp_011_0 ICM recipe on the HARDER LS20 Level 2.

Identical ICM+PPO setup to exp_011_0 (same code path), only `level_index=1`.
exp_011_0 SOLVED L1 (100% in 3/3 seeds); L2 is the deep 3-rotation puzzle that
plain PPO+novelty scored 0% on (only Go-Explore cleared it). This run asks
whether ICM's curiosity survives long enough to crack it, or collapses first.
See SYSTEM_CARD.md.
"""
