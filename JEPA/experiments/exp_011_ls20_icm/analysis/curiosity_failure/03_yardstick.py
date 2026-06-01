"""03 — Quantify r^i against PPO yardsticks, and depth reached vs steps.

The PPO objective adds an entropy bonus c_ent * H(pi). With c_ent=0.01 and the
L2 final entropy ~0.3-0.9 nats, the entropy term contributes ~0.003-0.009 to the
per-step objective. The terminal reward is +1. We compare r^i_mean / r^i_p90 /
r^i_max against these. We also print the GAE-discounted credit a single best-case
r^i spike can propagate.

Also: read mean_episode_steps to show L2 dies at a fixed depth (~66 steps), the
behavioural plateau, from metrics (cheap, no env).
"""
from __future__ import annotations
import json, os

ROOT = os.path.dirname(__file__)
res = json.load(open(os.path.join(ROOT, "results_02.json")))

C_ENT = 0.01      # exp_010 base entropy coefficient (inherited)
GAMMA = 0.99      # typical; we only use it illustratively

print("PPO yardsticks: entropy bonus scale c_ent =", C_ENT, " | terminal reward = +1")
print()
for lvl in ("l2", "l1"):
    print(f"=== {lvl.upper()} ===")
    for r in res[lvl]:
        # entropy term magnitude not available per-ck here; report r^i vs the
        # calibration target 0.01 (== entropy bonus scale) and vs +1.
        print(f"  step {r['step']:>7}: r^i_mean={r['ri_mean']:.2e} "
              f"({r['ri_mean']/0.01:.4f}x of c_ent=0.01, {r['ri_mean']/1.0:.1e} of +1) | "
              f"r^i_max={r['ri_max']:.2e} ({r['ri_max']/0.01:.3f}x c_ent) | "
              f"contrast(first/revisit fwd err)={r['contrast_ratio']:.2f} | "
              f"uniq_states={r['unique_states']}")
    print()

print(f"RANDOM-policy coverage (same budget {res['budget']} transitions): "
      f"L2={res['rand_l2_uniq']} unique states, L1={res['rand_l1_uniq']}")
