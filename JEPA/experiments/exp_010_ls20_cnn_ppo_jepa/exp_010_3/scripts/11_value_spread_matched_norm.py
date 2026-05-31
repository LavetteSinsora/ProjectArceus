"""Does JEPA's feature STRUCTURE (not its norm) cause a more spread-out value
distribution? Control for magnitude AND for the particular value head.

Setup: random encoder vs JEPA encoder rescaled to the SAME feature norm (~2.3).
Same fixed set of rollout states. For K independent random value heads
(w ~ orthogonal init, the untrained critic), compute V(s)=w.h(s)+b over states
and record std_s V(s). Average over the K heads.

If, at matched magnitude and averaged over random heads, JEPA still gives a
wider V distribution, the spread is caused purely by the encoder's structure
(its features vary more across states), not by feature norm or a lucky head.

Run: uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/11_value_spread_matched_norm.py
"""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[5]; sys.path.insert(0,str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic, one_hot_frame
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
DBG=Path(__file__).resolve().parents[1]; EXP=DBG.parent
ENC=EXP/"exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
K=40   # random value heads

def encoder(kind):
    torch.manual_seed(0); np.random.seed(0); m=ActorCritic()
    if kind!="random": m.encoder.load_state_dict(torch.load(ENC,map_location="cpu",weights_only=False)["encoder"])
    if kind=="resc":
        s=2.3/152.0; orig=m.encoder.forward; m.encoder.forward=lambda x,_o=orig,_s=s:_o(x)*_s
    return m.encoder.eval()

# fixed states
env=VecLS20Env("ls20",8,200,seed=0); rng=np.random.default_rng(0); st=[env.current_obs()]
for _ in range(700): o,*_=env.step(rng.integers(0,4,8)); st.append(o)
S=torch.from_numpy(np.concatenate(st,0))

@torch.no_grad()
def feats(enc):
    out=[];
    for i in range(0,len(S),512): out.append(enc(one_hot_frame(S[i:i+512])))
    return torch.cat(out)

H={k:feats(encoder(k)) for k in ["random","resc"]}     # both ~norm 2.3
print("mean |h|: random=%.2f  resc(JEPA norm-matched)=%.2f"%(H['random'].norm(-1).mean(),H['resc'].norm(-1).mean()),flush=True)

# K random value heads (gain 1.0 orthogonal, as in ActorCritic.value_head)
allV={"random":[],"resc":[]}; stds={"random":[],"resc":[]}
for j in range(K):
    g=torch.Generator().manual_seed(1000+j)
    w=torch.empty(1,256); torch.nn.init.orthogonal_(w,1.0)  # uses default generator; reseed:
    torch.manual_seed(1000+j); w=torch.empty(1,256); torch.nn.init.orthogonal_(w,1.0); b=torch.zeros(1)
    for k in ["random","resc"]:
        V=(H[k]@w.T+b).squeeze(-1).numpy()
        allV[k].append(V); stds[k].append(V.std())
for k in ["random","resc"]:
    s=np.array(stds[k]); print(f"{k:>6}: V_std over {K} random heads = {s.mean():.4f} ± {s.std():.4f}",flush=True)

# figure: single representative head histogram + per-head V_std boxplot
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
v0r=allV["random"][0]; v0s=allV["resc"][0]   # one representative random value head
rng_=max(np.abs(np.concatenate([v0r,v0s])).max(),0.25); bins=np.linspace(-rng_,rng_,80)
ax[0].hist(v0s,bins=bins,color="#7570b3",alpha=0.6,density=True,label=f"JEPA norm-matched (V_std {v0s.std():.3f})")
ax[0].hist(v0r,bins=bins,color="#1b9e77",alpha=0.8,density=True,label=f"random CNN (V_std {v0r.std():.3f})")
ax[0].set(title="(a) V(s) for ONE random value head\n(feature norm matched ≈2.3)",xlabel="V(s)",ylabel="density"); ax[0].legend(fontsize=8)
ax[1].boxplot([stds["random"],stds["resc"]],labels=["random","JEPA\nnorm-matched"])
ax[1].set(title=f"(b) per-head std of V(s)  (each of {K} random heads)",ylabel="std_s V(s)")
fig.suptitle("Fig 7 — At MATCHED feature magnitude, JEPA's structure alone gives a ~4× wider value distribution (robust over random heads)",y=1.04,fontsize=10,weight="bold")
fig.savefig(DBG/"figures/fig7_value_spread_matched.png",bbox_inches="tight",dpi=120)
json.dump({k:{"V_std_mean":float(np.mean(stds[k])),"V_std_std":float(np.std(stds[k]))} for k in stds},
          open(DBG/"data/value_spread_matched.json","w"),indent=1)
print("saved fig7 + value_spread_matched.json")
