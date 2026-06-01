"""Parallel sweep via multiprocessing (one uv-run process, N single-threaded
workers). Robust: parent waits on the pool; children inherit nothing fragile.

Tasks:
  - 6 PPO runs: {resc,random} x {baseline 1/0} x seeds {1,2}, 70 updates
  - 1 L2 transfer probe (random/jepa/ppo_l1 frozen encoders on L1 & L2)
Results printed as they complete and dumped to /tmp/exp010_sweep/results.txt
"""
import os, sys, json
from pathlib import Path
import multiprocessing as mp

ROOT = Path("/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo")

def _setup():
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(ROOT))

def ppo_task(args):
    kind, baseline, seed, nupd = args
    _setup()
    import numpy as np, torch
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
    ENC = ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
    DEV = torch.device("cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    m = ActorCritic()
    if kind != "random":
        m.encoder.load_state_dict(torch.load(ENC, map_location="cpu", weights_only=False)["encoder"])
    if kind == "resc":
        with torch.no_grad():
            env = VecLS20Env("ls20", n_envs=8, max_episode_steps=200, seed=999)
            o = env.current_obs()
            for _ in range(20): o,*_ = env.step(np.random.randint(0,4,size=8))
            cur = m.features(torch.from_numpy(o)).norm(dim=-1).mean().item()
        s = 2.3/cur; orig = m.encoder.forward
        m.encoder.forward = lambda x,_o=orig,_s=s: _o(x)*_s
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], 3e-4)
    envs = VecLS20Env("ls20", n_envs=8, max_episode_steps=200, seed=seed)
    def mc(roll,g):
        T,N=roll.rewards.shape; ret=torch.zeros(T,N); nxt=torch.zeros(N)
        for t in reversed(range(T)):
            nt=(~roll.dones[t]).float(); nxt=roll.rewards[t]+g*nxt*nt; ret[t]=nxt
        return ret
    first=None; tot=0; ent30=None
    for u in range(1,nupd+1):
        roll=collect_rollout(envs,m,DEV,128); compute_gae(roll,0.99,0.95)
        if not baseline:
            r=mc(roll,0.99); roll.advantages=r; roll.returns=r
        st=ppo_update(m,opt,roll,PPOConfig(),DEV)
        n=int((roll.rewards>0).sum().item()); tot+=n
        if n>0 and first is None: first=u
        if u==30: ent30=round(st.entropy,3)
    tag=f"{kind}_{'base' if baseline else 'nobase'}_s{seed}"
    return ("ppo", tag, dict(kind=kind,baseline=baseline,seed=seed,first_succ=first,total_succ=tot,ent_at30=ent30))

def probe_task(_):
    _setup()
    import numpy as np, torch, torch.nn as nn
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic, one_hot_frame
    from JEPA.shared.env_wrapper import make_env, full_game_id
    from arc_agi import Arcade, OperationMode
    JEPA_ENC = ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
    PPO_CKPT = ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_0_cnn_ppo_baseline/checkpoints/step_00307200.pt"
    DEV=torch.device("cpu")
    def collect(level,n=6000,seed=0):
        arc=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(ROOT/"environment_files"))
        gid=full_game_id("ls20"); from arcengine import ActionInput, GameAction
        def fresh():
            e=make_env(arc.make(gid),gid); e.reset()
            if level>0:
                g=e._env._game; g.set_level(level)
                raw=g.perform_action(ActionInput(id=getattr(GameAction,f"ACTION{e.n_actions+1}")),raw=True); e._latest_raw=raw
            return e
        rng=np.random.default_rng(seed); e=fresh(); O,A,NO=[],[],[]; obs=e._extract(e._latest_raw); steps=0
        while len(O)<n:
            a=int(rng.integers(0,e.n_actions)); nobs,term=e.step(a); steps+=1; trunc=steps>=200
            if not (term or trunc): O.append(obs); A.append(a); NO.append(nobs)
            obs=nobs
            if term or trunc: e=fresh(); obs=e._extract(e._latest_raw); steps=0
        return np.stack(O).astype(np.uint8), np.array(A), np.stack(NO).astype(np.uint8)
    def encoder(kind):
        torch.manual_seed(0); m=ActorCritic()
        if kind=="jepa": m.encoder.load_state_dict(torch.load(JEPA_ENC,map_location="cpu",weights_only=False)["encoder"])
        elif kind=="ppo_l1": m.encoder.load_state_dict(torch.load(PPO_CKPT,map_location="cpu",weights_only=False)["encoder"])
        return m.encoder.eval()
    @torch.no_grad()
    def feats(enc,fr,bs=512):
        out=[]
        for i in range(0,len(fr),bs): out.append(enc(one_hot_frame(torch.from_numpy(fr[i:i+bs]))))
        return torch.cat(out)
    def eff_rank(H):
        Hc=H-H.mean(0); s=torch.linalg.svdvals(Hc); p=s/s.sum(); return float(torch.exp(-(p*torch.log(p+1e-12)).sum()))
    def std(H): return (H-H.mean(0))/(H.std(0)+1e-6)
    def idm(Ht,Hn,A,tr,te):
        X=torch.cat([Ht,Hn],-1); y=torch.as_tensor(A)
        net=nn.Sequential(nn.Linear(X.shape[1],128),nn.ReLU(),nn.Linear(128,4)); opt=torch.optim.Adam(net.parameters(),1e-3)
        for _ in range(300):
            idx=tr[torch.randperm(len(tr))[:1024]]; opt.zero_grad()
            nn.functional.cross_entropy(net(X[idx]),y[idx]).backward(); opt.step()
        with torch.no_grad(): return (net(X[te]).argmax(-1)==y[te]).float().mean().item()
    def fwd(Ht,Hn,A,tr,te):
        A=torch.as_tensor(A); emb=nn.Embedding(4,16)
        net=nn.Sequential(nn.Linear(Ht.shape[1]+16,256),nn.ReLU(),nn.Linear(256,Hn.shape[1]))
        opt=torch.optim.Adam(list(net.parameters())+list(emb.parameters()),1e-3)
        for _ in range(300):
            idx=tr[torch.randperm(len(tr))[:1024]]; opt.zero_grad()
            xb=torch.cat([Ht[idx],emb(A[idx])],-1)
            nn.functional.mse_loss(net(xb),Hn[idx]).backward(); opt.step()
        with torch.no_grad():
            xte=torch.cat([Ht[te],emb(A[te])],-1); pred=net(xte)
            sse=((pred-Hn[te])**2).sum(); sst=((Hn[te]-Hn[te].mean(0))**2).sum(); return 1-(sse/sst).item()
    res={}
    for level,name in [(0,"L1"),(1,"L2")]:
        O,A,NO=collect(level,6000,seed=level); perm=torch.randperm(len(A)); cut=int(0.8*len(A)); tr,te=perm[:cut],perm[cut:]
        res[name]={}
        for kind in ["random","jepa","ppo_l1"]:
            enc=encoder(kind); Ht=std(feats(enc,O)); Hn=std(feats(enc,NO))
            res[name][kind]=dict(eff_rank=round(eff_rank(Ht),2),idm_acc=round(idm(Ht,Hn,A,tr,te),3),fwd_r2=round(fwd(Ht,Hn,A,tr,te),3))
    return ("probe","transfer",res)

if __name__=="__main__":
    mp.set_start_method("spawn")
    od=Path("/tmp/exp010_sweep"); od.mkdir(exist_ok=True)
    tasks=[("ppo",("resc",True,1,70)),("ppo",("resc",False,1,70)),("ppo",("random",True,1,70)),
           ("ppo",("resc",True,2,70)),("ppo",("resc",False,2,70)),("ppo",("random",True,2,70)),
           ("probe",None)]
    funcs={"ppo":ppo_task,"probe":probe_task}
    with mp.Pool(7) as pool:
        async_res=[pool.apply_async(funcs[t],(arg,)) for t,arg in tasks]
        for r in async_res:
            kind,tag,data=r.get()
            line=f"[{kind}] {tag}: {json.dumps(data)}"
            print(line,flush=True)
            with open(od/"results.txt","a") as f: f.write(line+"\n")
    print("=== SWEEP COMPLETE ===",flush=True)
