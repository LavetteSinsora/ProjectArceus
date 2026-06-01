"""One PPO run, single-threaded, for the parallel sweep.
argv: kind(random|raw|resc)  baseline(1|0)  seed  n_updates  out_tag
Appends one JSON line to /tmp/exp010_sweep/<out_tag>.json
"""
import sys, os, json, numpy as np, torch
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
ROOT = Path("/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo")
sys.path.insert(0, str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update

ENC = ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
DEV = torch.device("cpu")
kind, baseline, seed, nupd, tag = sys.argv[1], sys.argv[2]=="1", int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]

def build():
    torch.manual_seed(seed); np.random.seed(seed)
    m=ActorCritic()
    if kind!="random":
        m.encoder.load_state_dict(torch.load(ENC,map_location="cpu",weights_only=False)["encoder"])
    if kind=="resc":
        with torch.no_grad():
            env=VecLS20Env("ls20",n_envs=8,max_episode_steps=200,seed=999)
            o=env.current_obs()
            for _ in range(20): o,*_=env.step(np.random.randint(0,4,size=8))
            cur=m.features(torch.from_numpy(o)).norm(dim=-1).mean().item()
        s=2.3/cur; orig=m.encoder.forward
        m.encoder.forward=lambda x,_o=orig,_s=s:_o(x)*_s
    return m

def mc(roll,g):
    T,N=roll.rewards.shape; ret=torch.zeros(T,N); nxt=torch.zeros(N)
    for t in reversed(range(T)):
        nt=(~roll.dones[t]).float(); nxt=roll.rewards[t]+g*nxt*nt; ret[t]=nxt
    return ret

m=build(); opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],3e-4)
envs=VecLS20Env("ls20",n_envs=8,max_episode_steps=200,seed=seed)
first=None; tot=0; ent_at30=None
for u in range(1,nupd+1):
    roll=collect_rollout(envs,m,DEV,128); compute_gae(roll,0.99,0.95)
    if not baseline:
        r=mc(roll,0.99); roll.advantages=r; roll.returns=r
    st=ppo_update(m,opt,roll,PPOConfig(),DEV)
    n=int((roll.rewards>0).sum().item()); tot+=n
    if n>0 and first is None: first=u
    if u==30: ent_at30=st.entropy
out=dict(kind=kind,baseline=baseline,seed=seed,nupd=nupd,first_succ=first,total_succ=tot,ent_at30=ent_at30)
od=Path("/tmp/exp010_sweep"); od.mkdir(exist_ok=True)
with open(od/f"{tag}.json","w") as f: f.write(json.dumps(out)+"\n")
print(tag, out, flush=True)
