"""Regenerate the 3 core figures with a clean, consistent, web-friendly style
(no baked-in titles — captions live in the HTML; larger fonts; tight layout).
Run: uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/18_web_figures.py
"""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parents[1]      # the exp_010_3 dir (scripts/ -> exp_010_3)
ROOT=Path(__file__).resolve().parents[5]      # scripts/exp_010_3/exp_010_ls20../experiments/JEPA -> Code Repo
sys.path.insert(0,str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic, one_hot_frame
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
EXP=ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa"; ENC=EXP/"exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
FIG=HERE/"figures"; DATA=HERE/"data"; FIG.mkdir(parents=True, exist_ok=True)
RND="#2a9d8f"; JEP="#e76f51"; JEPN="#6c5ce7"; GREY="#8a8f98"
plt.rcParams.update({"figure.dpi":150,"font.size":13,"axes.titlesize":13,"axes.titleweight":"bold",
    "axes.grid":True,"grid.alpha":0.18,"axes.spines.top":False,"axes.spines.right":False,
    "axes.facecolor":"white","figure.facecolor":"white","font.family":"DejaVu Sans","legend.frameon":False})

def enc(kind):
    torch.manual_seed(0); np.random.seed(0); m=ActorCritic()
    if kind!="random": m.encoder.load_state_dict(torch.load(ENC,map_location="cpu",weights_only=False)["encoder"])
    if kind=="resc":
        s=2.3/152.0; o=m.encoder.forward; m.encoder.forward=lambda x,_o=o,_s=s:_o(x)*_s
    return m

# ---------- FIG A: value-landscape spread at matched scale ----------
e=VecLS20Env("ls20",8,200,seed=0); rng=np.random.default_rng(0); st=[e.current_obs()]
for _ in range(700): o,*_=e.step(rng.integers(0,4,8)); st.append(o)
S=torch.from_numpy(np.concatenate(st,0))
def feats(m):
    out=[]
    with torch.no_grad():
        for i in range(0,len(S),512): out.append(m.encoder(one_hot_frame(S[i:i+512])))
    return torch.cat(out)
HR=feats(enc("random")); HJ=feats(enc("resc"))
stds={"random":[],"jepa":[]}; v0={}
for j in range(40):
    torch.manual_seed(1000+j); w=torch.empty(1,256); torch.nn.init.orthogonal_(w,1.0)
    vr=(HR@w.T).squeeze(-1).numpy(); vj=(HJ@w.T).squeeze(-1).numpy()
    stds["random"].append(vr.std()); stds["jepa"].append(vj.std())
    if j==0: v0["random"]=vr; v0["jepa"]=vj
fig,ax=plt.subplots(1,2,figsize=(9.2,3.7))
m=4*v0["random"].std(); b=np.linspace(-max(m,0.25),max(m,0.25),64)
ax[0].hist(v0["jepa"],bins=b,color=JEPN,alpha=.55,density=True,label=f"JEPA  (spread {v0['jepa'].std():.3f})")
ax[0].hist(v0["random"],bins=b,color=RND,alpha=.8,density=True,label=f"random  (spread {v0['random'].std():.3f})")
ax[0].set(xlabel="value V(s) across states",ylabel="density",title="One untrained critic"); ax[0].legend(fontsize=11)
bp=ax[1].boxplot([stds["random"],stds["jepa"]],labels=["random","JEPA"],patch_artist=True,widths=.5)
for patch,c in zip(bp['boxes'],[RND,JEPN]): patch.set_facecolor(c); patch.set_alpha(.6)
for med in bp['medians']: med.set_color("#222")
ax[1].set(ylabel="spread of V(s)",title="Across 40 random critics")
fig.tight_layout(); fig.savefig(FIG/"web_value_spread.png",bbox_inches="tight"); plt.close(fig)
print("fig A done")

# ---------- FIG B: critic-removal rescue ----------
cv=json.load(open(DATA/"curves.json"))
lab={"random_base":("random encoder + critic",RND,"-"),
     "resc_base":("JEPA encoder + critic",JEPN,"-"),
     "resc_nobase":("JEPA encoder, NO critic",JEP,"--")}
fig,ax=plt.subplots(1,2,figsize=(9.2,3.7))
for k,(nm,c,ls) in lab.items():
    e_=cv[k]["entropy"]; cs=cv[k]["cum_success"]; x=range(1,len(e_)+1)
    ax[0].plot(x,e_,color=c,lw=2.4,ls=ls,label=nm); ax[1].plot(x,cs,color=c,lw=2.4,ls=ls,label=nm)
ax[0].axhline(1.386,color=GREY,ls=":",lw=1); ax[0].set(xlabel="training update",ylabel="policy entropy",title="Exploration (entropy)")
ax[0].legend(fontsize=10.5,loc="lower left")
ax[1].set(xlabel="training update",ylabel="cumulative successes",title="Solving",yscale="symlog")
fig.tight_layout(); fig.savefig(FIG/"web_rescue.png",bbox_inches="tight"); plt.close(fig)
print("fig B done")

# ---------- FIG C: advantage through normalization ----------
def adv_stats(kind,seeds=(0,1,2,3)):
    raws=[]; norms=[]
    m=enc(kind)
    for sd in seeds:
        envs=VecLS20Env("ls20",8,200,seed=sd); roll=collect_rollout(envs,m,torch.device("cpu"),128); compute_gae(roll,0.99,0.95)
        a=roll.advantages.reshape(-1).numpy(); raws.append(a.std()); norms.append((a-a.mean())/(a.std()+1e-8))
    return float(np.mean(raws)), np.concatenate(norms)
RS={}; NM={}
for k in ["random","resc","raw"]: RS[k],NM[k]=adv_stats(k)
dec=json.load(open(DATA/"policy_decomposition.json"))
fig,ax=plt.subplots(1,3,figsize=(11.5,3.7))
ks=["random","resc","raw"]; cmap={"random":RND,"resc":JEPN,"raw":JEP}; disp={"random":"random","resc":"JEPA\n(matched)","raw":"JEPA\n(raw)"}
ax[0].bar(range(3),[RS[k] for k in ks],color=[cmap[k] for k in ks]); ax[0].set_yscale("log")
ax[0].set(xticks=range(3),xticklabels=[disp[k] for k in ks],ylabel="advantage spread (log)",title="Before normalization")
for i,k in enumerate(ks): ax[0].text(i,RS[k],f"{RS[k]:.3f}",ha="center",va="bottom",fontsize=10)
bb=np.linspace(-4,5,80)
for k in ks: ax[1].hist(NM[k],bins=bb,density=True,histtype="step",lw=2.2,color=cmap[k],label=disp[k].replace("\n"," "))
ax[1].set_yscale("log"); ax[1].set(xlabel="normalized advantage",ylabel="density (log)",title="After normalization"); ax[1].legend(fontsize=10)
iv={"random":dec["random"]["I_state_action"],"resc":dec["resc"]["I_state_action"],"raw":dec["raw"]["I_state_action"]}
ax[2].bar(range(3),[iv[k] for k in ks],color=[cmap[k] for k in ks])
ax[2].set(xticks=range(3),xticklabels=[disp[k] for k in ks],ylabel="state-dependence  I(S;A)",title="What survives")
for i,k in enumerate(ks): ax[2].text(i,iv[k],f"{iv[k]:.3f}",ha="center",va="bottom",fontsize=10)
fig.tight_layout(); fig.savefig(FIG/"web_advantage.png",bbox_inches="tight"); plt.close(fig)
print("fig C done")
