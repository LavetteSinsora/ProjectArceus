# exp_010_4 — JEPA recipe research

**[report.html](report.html)** is the deliverable. **[SYSTEM_CARD.md](SYSTEM_CARD.md)** is the original
research map. Code: [study.py](study.py) (train + eval) → [make_report.py](make_report.py) (figures + report).

## Study 1 — anti-collapse recipe comparison (real LS20, executed)

Action-conditioned forward-JEPA on the **real** LS20 env (60k random transitions), **fixed** exp_010 CNN
trunk (256-d post-ReLU) + 128-d projector; prediction in projector space; eval frozen on the trunk. Recipes:
stop-grad-only / VICReg / SIGReg (LeJEPA) / EMA. References: random-init trunk, old exp_010_2 encoder.
Eval harness: effective rank, per-dim std, anisotropy (mean cosine), ‖h‖, frozen linear IDM
(action-decodability), frame-diff agent-loc probe. 1000 steps each, MPS, shared init + data order.

**Verified results (final):**

| recipe | eff.rank/256 | per-dim std | anisotropy cos | ‖h‖ | IDM acc | loc R² |
|---|---|---|---|---|---|---|
| random init | 18.9 | 0.013 | 0.985 | 2.3 | 0.706 | 0.41 |
| exp_010_2 (old) | 6.1 | 3.63 | 0.598 | 147.8 | 0.816 | 0.41 |
| stop-grad only | 11.3 | 0.002 | 1.000 | 20.0 | 0.456 | 0.36 |
| **VICReg** ✅ | **37.4** | **0.72** | **0.939** | 65.2 | **0.813** | 0.40 |
| SIGReg (mine) | 18.2 | 0.0002 | 1.000 | 1.0 | 0.279 | 0.00 |
| EMA target | 1.6 | 0.12 | 0.995 | 68.5 | 0.722 | 0.39 |

- A frozen **random** CNN trunk is already a strong baseline (IDM 0.71). Any recipe must beat it.
- **Naive stop-grad JEPA degrades** the rep (anisotropic, IDM 0.46 < random) — the exp_010_2 failure.
- **VICReg wins** and is the only recipe to beat random init (rank 37, IDM 0.81). → `artifacts/encoder_best_vicreg.pt`.
- **My SIGReg variant collapsed** (near-zero embeddings, IDM ≈ chance) under λ=1 in the prediction space with
  un-standardised slices — likely an implementation/weighting issue, not a refutation of LeJEPA. Fixing SIGReg
  (stronger λ, separate projector, standardised slices) is the top follow-up.
- Effective rank is scale-free → it can look fine for a near-zero collapsed rep (SIGReg rank 18 but std 2e-4);
  always pair it with per-dim std and a probe.

**Next ceiling = data:** random data reaches the goal ~1/1159 lives, so no recipe learns goal dynamics it
never sees — goal-aware data is the highest-leverage follow-up.

*(Note: an earlier crashed run produced fabricated numbers that were retracted; the table above is from the
verified `/tmp/jepa4/metrics.json`.)*
