"""L2 representation-transfer probe.

Question (the user's): are JEPA-pretrained representations more *transferable*
to a new, visually-related task (LS20 Level 2) than task-specific PPO
representations? We isolate REPRESENTATION QUALITY from the exploration
problem by probing FROZEN encoders on self-supervised tasks on L2 data.

Encoders compared (all frozen):
  random : random-init CNN (generic random projection)
  jepa   : JEPA-pretrained on L1 random-policy data (dynamics objective)
  ppo_l1 : the encoder of the PPO agent that SOLVED L1 (task-specific)

Probes on L2 (held-out test split), features standardised per-encoder so the
comparison is about INFORMATION, not scale:
  idm_acc  : MLP predicts action a_t from (h_t, h_{t+1})  -> action/dynamics info
  fwd_r2   : MLP predicts h_{t+1} from (h_t, a_t)         -> dynamics predictability
  eff_rank : participation-ratio effective rank of features on L2 -> richness
We also report the same probes on L1 (in-domain reference).
"""
import sys, os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
ROOT = Path("/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo")
sys.path.insert(0, str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic, one_hot_frame
from JEPA.shared.env_wrapper import make_env, full_game_id
from arc_agi import Arcade, OperationMode

DEV = torch.device("cpu")
JEPA_ENC = ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
PPO_CKPT = ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_0_cnn_ppo_baseline/checkpoints/step_00307200.pt"

# ---- collect transitions at a given level ----
def collect(level, n=6000, seed=0):
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT/"environment_files"))
    gid = full_game_id("ls20")
    from arcengine import ActionInput, GameAction
    def fresh():
        e = make_env(arc.make(gid), gid); e.reset()
        if level>0:
            g=e._env._game; g.set_level(level)
            raw=g.perform_action(ActionInput(id=getattr(GameAction,f"ACTION{e.n_actions+1}")),raw=True)
            e._latest_raw=raw
        return e
    rng=np.random.default_rng(seed); e=fresh()
    O,A,NO=[],[],[]; obs=e._extract(e._latest_raw); steps=0
    while len(O)<n:
        a=int(rng.integers(0,e.n_actions)); nobs,term=e.step(a); steps+=1
        trunc = steps>=200
        if not (term or trunc):
            O.append(obs); A.append(a); NO.append(nobs)
        obs=nobs
        if term or trunc:
            e=fresh(); obs=e._extract(e._latest_raw); steps=0
    return (np.stack(O).astype(np.uint8), np.array(A), np.stack(NO).astype(np.uint8))

def encoder(kind):
    torch.manual_seed(0)
    m=ActorCritic()
    if kind=="jepa":
        m.encoder.load_state_dict(torch.load(JEPA_ENC,map_location="cpu",weights_only=False)["encoder"])
    elif kind=="ppo_l1":
        ck=torch.load(PPO_CKPT,map_location="cpu",weights_only=False)
        m.encoder.load_state_dict(ck["encoder"])
    return m.encoder.eval()

@torch.no_grad()
def feats(enc, frames, bs=512):
    out=[]
    for i in range(0,len(frames),bs):
        x=one_hot_frame(torch.from_numpy(frames[i:i+bs]))
        out.append(enc(x))
    return torch.cat(out)

def eff_rank(H):
    Hc=H-H.mean(0); s=torch.linalg.svdvals(Hc); p=s/s.sum()
    return float(torch.exp(-(p*torch.log(p+1e-12)).sum()))

def standardize(H):
    return (H-H.mean(0))/(H.std(0)+1e-6)

def train_idm(Ht,Hn,A,split):
    X=torch.cat([Ht,Hn],-1); y=torch.as_tensor(A)
    tr,te=split
    net=nn.Sequential(nn.Linear(X.shape[1],128),nn.ReLU(),nn.Linear(128,4))
    opt=torch.optim.Adam(net.parameters(),1e-3)
    for ep in range(300):
        idx=tr[torch.randperm(len(tr))[:1024]]
        opt.zero_grad(); loss=nn.functional.cross_entropy(net(X[idx]),y[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        acc=(net(X[te]).argmax(-1)==y[te]).float().mean().item()
    return acc

def train_fwd(Ht,Hn,A,split):
    A=torch.as_tensor(A); emb=nn.Embedding(4,16); tr,te=split; Y=Hn
    net=nn.Sequential(nn.Linear(Ht.shape[1]+16,256),nn.ReLU(),nn.Linear(256,Hn.shape[1]))
    opt=torch.optim.Adam(list(net.parameters())+list(emb.parameters()),1e-3)
    for ep in range(300):
        idx=tr[torch.randperm(len(tr))[:1024]]
        opt.zero_grad(); xb=torch.cat([Ht[idx],emb(A[idx])],-1)
        loss=nn.functional.mse_loss(net(xb),Y[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        xte=torch.cat([Ht[te],emb(A[te])],-1); pred=net(xte)
        sse=((pred-Y[te])**2).sum(); sst=((Y[te]-Y[te].mean(0))**2).sum()
        r2=1-(sse/sst).item()
    return r2

def run_level(level, name):
    O,A,NO=collect(level, n=6000, seed=level)
    nperm=torch.randperm(len(A)); cut=int(0.8*len(A))
    split=(nperm[:cut], nperm[cut:])
    print(f"\n### LS20 {name}  (n={len(A)} transitions) ###", flush=True)
    print(f"{'encoder':>8} {'eff_rank':>8} {'idm_acc':>8} {'fwd_r2':>8}", flush=True)
    for kind in ["random","jepa","ppo_l1"]:
        enc=encoder(kind)
        Ht=standardize(feats(enc,O)); Hn=standardize(feats(enc,NO))
        er=eff_rank(Ht)
        acc=train_idm(Ht,Hn,A,split); r2=train_fwd(Ht,Hn,A,split)
        print(f"{kind:>8} {er:>8.2f} {acc:>8.3f} {r2:>8.3f}  (chance acc=0.25)", flush=True)

if __name__=="__main__":
    run_level(0,"Level 1 (in-domain reference)")
    run_level(1,"Level 2 (transfer target)")
