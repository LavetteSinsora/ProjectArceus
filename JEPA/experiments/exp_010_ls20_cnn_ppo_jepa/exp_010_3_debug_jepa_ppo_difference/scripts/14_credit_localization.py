"""The mechanism behind the objection 'how can the same advantage reinforce
differently?': it is not the advantage at s, it is how the UPDATE GENERALIZES
to OTHER states through the shared representation.

Test: take a fresh ~uniform policy head on a FROZEN encoder, reinforce one
action a* at ONE target state s* with a single gradient step, and measure the
induced change Δπ(a*|s_j) at every OTHER state s_j (a DIVERSE probe set), as a
function of representational similarity cos(h_j, h*).

Prediction: JEPA (structured h) -> Δπ concentrated at states similar to s*
(localized credit; strong corr with similarity). Random (h≈const) -> Δπ ~uniform
across all states (smeared credit; the update can only move the global policy).

Run: uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/14_credit_localization.py
"""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[5]; sys.path.insert(0,str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
DBG=Path(__file__).resolve().parents[1]; EXP=DBG.parent
ENC=EXP/"exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
col={"random":"#1b9e77","jepa":"#d95f02","resc":"#7570b3"}; name={"random":"random CNN","jepa":"JEPA (raw)","resc":"JEPA (norm-matched)"}

# diverse probe states
env=VecLS20Env("ls20",8,200,seed=0); rng=np.random.default_rng(0); st=[env.current_obs()]
for _ in range(120): o,*_=env.step(rng.integers(0,4,8)); st.append(o)
S=torch.from_numpy(np.concatenate(st,0))[:800]

def build(kind):
    torch.manual_seed(0); np.random.seed(0); m=ActorCritic()
    if kind!="random": m.encoder.load_state_dict(torch.load(ENC,map_location="cpu",weights_only=False)["encoder"])
    if kind=="resc":
        s=2.3/152.0; o=m.encoder.forward; m.encoder.forward=lambda x,_o=o,_s=s:_o(x)*_s
    for p in m.encoder.parameters(): p.requires_grad_(False)
    torch.manual_seed(123); m.policy_head=torch.nn.Linear(256,4)
    torch.nn.init.orthogonal_(m.policy_head.weight,0.01); torch.nn.init.zeros_(m.policy_head.bias)
    return m

out={}; PLOT=["random","resc"]; fig,ax=plt.subplots(1,2,figsize=(10,3.9)); axi=0
for kind in ["random","jepa","resc"]:
    m=build(kind)
    with torch.no_grad():
        H=m.features(S); Hn=torch.nn.functional.normalize(H,dim=-1)
        p0=torch.softmax(m.forward(S)[0],-1)
    # similarity of every state to the target state 0
    sim=(Hn@Hn[0]).numpy()
    # reinforce action 0 at state 0 (one gradient step)
    opt=torch.optim.SGD(m.policy_head.parameters(),lr=0.1)
    lg,_,_=m.forward(S[:1]); loss=torch.nn.functional.cross_entropy(lg,torch.tensor([0])); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): p1=torch.softmax(m.forward(S)[0],-1)
    dpi=(p1[:,0]-p0[:,0]).numpy()      # change in pi(a*=0 | s_j)
    corr=float(np.corrcoef(sim,dpi)[0,1])
    cv=float(dpi.std()/ (abs(dpi.mean())+1e-9))
    loc=float(dpi[0]/ (dpi.mean()+1e-9))
    out[kind]=dict(sim_dpi_corr=round(corr,3),localization_ratio=round(loc,2),CV=round(cv,2),
                   mean_dpi=float(dpi.mean()),target_dpi=float(dpi[0]))
    print(f"{kind:>6}: corr(sim,Δπ)={corr:+.3f}  Δπ@target/mean={loc:.1f}  CV(Δπ)={cv:.3f}",flush=True)
    if kind in PLOT:
        ax[axi].scatter(sim,dpi,s=8,alpha=0.5,color=col[kind])
        ax[axi].set(title=f"{name[kind]}  (norm≈2.3)\ncos-sim spans {sim.min():.2f}–{sim.max():.2f}; CV(Δπ)={cv:.2f}",
                    xlabel="cos sim of h(s_j) to target h(s*)",ylabel="Δ π(a*|s_j) after 1 update",xlim=(0.15,1.02))
        ax[axi].axhline(0,color="grey",lw=.5); axi+=1
fig.suptitle("Fig 9 — Credit generalization (matched feature norm): reinforcing ONE action at ONE state.\nrandom smears the change uniformly (all states look alike); JEPA localizes it to similar states",y=1.07,fontsize=10,weight="bold")
fig.savefig(DBG/"figures/fig9_credit_localization.png",bbox_inches="tight",dpi=120)
json.dump(out,open(DBG/"data/credit_localization.json","w"),indent=1)
print("saved fig9 + credit_localization.json")
