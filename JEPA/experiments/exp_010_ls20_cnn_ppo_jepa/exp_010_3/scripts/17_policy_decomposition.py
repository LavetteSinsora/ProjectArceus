"""Reproduce the entropy decomposition / state-dependence numbers from
policy_shaping.json (closes a reproducibility gap flagged in review).

Identity (uniform weighting over the fixed probe states):
    mean_s H(pi(.|s)) = log4 - KL(marginal-policy || uniform) - I(S;A)
=>  I(S;A) = log4 - KL(marginal||uniform) - mean_s H

I(S;A) = state-action mutual information = how state-DEPENDENT the policy became
(0 = identical action distribution in every state). log4 = 1.386 is the ceiling.

Run: uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/17_policy_decomposition.py
"""
import json, math
from pathlib import Path
import numpy as np
DBG=Path(__file__).resolve().parents[1]
d=json.load(open(DBG/"data/policy_shaping.json")); log4=math.log(4); out={}
print(f"{'enc':>8} {'per_state_H':>12} {'KL(marg||unif)':>14} {'I(S;A)':>8} {'I as % of log4':>14}")
for k in ["random","raw","resc"]:
    rs=[v for v in d.values() if v["kind"]==k]
    H=float(np.mean([v["H"][-1] for v in rs])); KL=float(np.mean([v["KL"][-1] for v in rs])); I=log4-KL-H
    out[k]={"per_state_entropy":round(H,3),"KL_marginal_uniform":round(KL,3),
            "I_state_action":round(I,3),"I_pct_of_log4":round(100*I/log4,1)}
    print(f"{k:>8} {H:>12.3f} {KL:>14.3f} {I:>8.3f} {100*I/log4:>13.1f}%")
out["_note"]="uniform weighting over the fixed probe set; measured after 25 zero-reward updates, mean over 3 seeds"
json.dump(out,open(DBG/"data/policy_decomposition.json","w"),indent=1)
print("wrote policy_decomposition.json")
