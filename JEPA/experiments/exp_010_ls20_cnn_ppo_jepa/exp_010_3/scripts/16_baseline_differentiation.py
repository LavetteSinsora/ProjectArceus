"""Fig 10: the random-init baseline encoder DIFFERENTIATES states over training,
driven by reward. Reads the logged 10_0 metrics (mean_feature_cosine = state
similarity, eval success_rate). Shows cosine 0.99->0.73 as success consolidates.

Run: uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/16_baseline_differentiation.py
"""
import json, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EXP=Path(__file__).resolve().parents[1].parent; DBG=Path(__file__).resolve().parents[1]
r=sorted(glob.glob(str(EXP/'exp_010_0_cnn_ppo_baseline/runs/*/metrics.jsonl')))
r=[x for x in r if sum(1 for _ in open(x))>=50][-1]
recs=[json.loads(l) for l in open(r)]
fc=[(x['step'],x['mean_feature_cosine']) for x in recs if x.get('mean_feature_cosine') is not None]
sr=[(x['step'],x['success_rate']) for x in recs if x.get('success_rate') is not None]
fig,ax=plt.subplots(figsize=(7.5,4.0)); ax2=ax.twinx()
ax.plot([s for s,_ in fc],[c for _,c in fc],color="#7570b3",lw=2,label="feature mean-cosine (state similarity)")
ax2.plot([s for s,_ in sr],[v for _,v in sr],color="#1b9e77",lw=2,marker="o",ms=3,label="eval success rate")
ax.axvline(3072,color="grey",ls=":",lw=1); ax.text(3072,0.96,"  first reward\n  (step 3072)",fontsize=8,color="grey")
ax.set_xlabel("env steps"); ax.set_ylabel("feature mean-cosine  (1.0 = states identical)",color="#7570b3")
ax2.set_ylabel("eval success rate",color="#1b9e77"); ax2.set_ylim(-0.05,1.05)
ax.set_title("Fig 10 — Baseline (random encoder): reward drives the encoder to DIFFERENTIATE states\n(cosine 0.99->0.73 as the policy learns to exploit the found reward)",fontsize=10,weight="bold")
ax.legend(loc="lower left",fontsize=8); ax2.legend(loc="center right",fontsize=8)
fig.savefig(DBG/"figures/fig10_baseline_differentiation.png",bbox_inches="tight",dpi=120)
print("saved fig10; cosine",round(fc[0][1],3),"->",round(fc[-1][1],3))
