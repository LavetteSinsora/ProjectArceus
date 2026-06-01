"""Fig 8: distribution of GAE's estimated advantage, BEFORE and AFTER PPO's
advantage normalization, for random / JEPA(raw) / JEPA(norm-matched).

Point: normalization sets mean 0 / std 1 for EVERY encoder by construction, so
it erases the *scale* difference (a). But it does NOT erase the *shape*: the
JEPA-driven normalized advantage stays right-skewed & heavy-tailed — a minority
of state-actions the critic strongly favours (the phantom 'reward') — while the
random encoder's is a symmetric Gaussian (pure noise). That surviving tail is
what reshapes the policy after normalization.

Pools zero-reward rollouts over several seeds. Run from repo root:
  uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/12_advantage_distribution.py
"""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[5]; sys.path.insert(0,str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
DBG=Path(__file__).resolve().parents[1]; EXP=DBG.parent
ENC=EXP/"exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
SEEDS=list(range(8))
col={"random":"#1b9e77","raw":"#d95f02","resc":"#7570b3"}
name={"random":"random CNN","raw":"JEPA (raw)","resc":"JEPA (norm-matched)"}

def build(kind,seed):
    torch.manual_seed(seed); np.random.seed(seed); m=ActorCritic()
    if kind!="random": m.encoder.load_state_dict(torch.load(ENC,map_location="cpu",weights_only=False)["encoder"])
    if kind=="resc":
        s=2.3/152.0; o=m.encoder.forward; m.encoder.forward=lambda x,_o=o,_s=s:_o(x)*_s
    return m
def skew(x): z=(x-x.mean())/x.std(); return float((z**3).mean())
def kurt(x): z=(x-x.mean())/x.std(); return float((z**4).mean()-3)

RAW={}; NORM={}; rawstd={}
for kind in ["random","raw","resc"]:
    raws=[]; norms=[]; stds=[]
    for seed in SEEDS:
        m=build(kind,seed); envs=VecLS20Env("ls20",8,200,seed=seed)
        roll=collect_rollout(envs,m,torch.device("cpu"),128); compute_gae(roll,0.99,0.95)
        adv=roll.advantages.reshape(-1).numpy()
        raws.append(adv); stds.append(adv.std())
        norms.append((adv-adv.mean())/(adv.std()+1e-8))   # PPO normalization, per rollout
    RAW[kind]=np.concatenate(raws); NORM[kind]=np.concatenate(norms); rawstd[kind]=float(np.mean(stds))
    print(f"{kind:>7}: raw_adv_std={rawstd[kind]:.3f}  norm skew={skew(NORM[kind]):+.2f}  excess_kurt={kurt(NORM[kind]):+.2f}",flush=True)

decomp=json.load(open(DBG/"data/policy_decomposition.json")) if (DBG/"data/policy_decomposition.json").exists() else None
fig,ax=plt.subplots(1,3,figsize=(14,3.9))
# (a) RAW advantage std (scale) — log axis, three live on wildly different scales
ks=["random","resc","raw"]
ax[0].bar(range(3),[rawstd[k] for k in ks],color=[col[k] for k in ks]); ax[0].set_yscale("log")
ax[0].set(title="(a) RAW GAE advantage: std\n(before normalization — scale differs ~250×)",xticks=range(3),
          xticklabels=[name[k].replace(' ','\n') for k in ks],ylabel="std of advantage (log)")
for i,k in enumerate(ks): ax[0].text(i,rawstd[k],f"{rawstd[k]:.3f}",ha="center",va="bottom",fontsize=8)
# (b) NORMALIZED advantage marginal — ~identical for all (normalization erases scale AND shape)
bins=np.linspace(-4,5,90)
for k in ["random","raw","resc"]:
    ax[1].hist(NORM[k],bins=bins,density=True,histtype="step",lw=2,color=col[k],
               label=f"{name[k]}\nskew{skew(NORM[k]):+.2f} kurt{kurt(NORM[k]):+.2f}")
ax[1].set_yscale("log"); ax[1].set(title="(b) NORMALIZED advantage marginal (mean0,std1 ∀)\nmarginal ≈ identical — difference NOT here",
          xlabel="normalized advantage", ylabel="density (log)"); ax[1].legend(fontsize=7)
# (c) what survives normalization: state-dependence the policy inherits, I(S;A)
if decomp:
    iv=[decomp[k]["I_state_action"] for k in ks]
    ax[2].bar(range(3),iv,color=[col[k] for k in ks])
    ax[2].set(title="(c) what SURVIVES: state-structure\nI(S;A) of policy after 25 updates",xticks=range(3),
              xticklabels=[name[k].replace(' ','\n') for k in ks],ylabel="I(S;A)  [nats]")
    for i,v in enumerate(iv): ax[2].text(i,v,f"{v:.3f}",ha="center",va="bottom",fontsize=8)
fig.suptitle("Fig 8 — Normalization erases the advantage's SCALE (a) AND its marginal shape (b); the surviving difference is STATE-STRUCTURE (c), not the marginal distribution",
             y=1.05,fontsize=9.6,weight="bold")
fig.savefig(DBG/"figures/fig8_advantage_dist.png",bbox_inches="tight",dpi=120)
json.dump({k:{"raw_adv_std":rawstd[k],"norm_skew":skew(NORM[k]),"norm_excess_kurtosis":kurt(NORM[k])} for k in NORM},
          open(DBG/"data/advantage_distribution.json","w"),indent=1)
print("saved fig8 + advantage_distribution.json")
