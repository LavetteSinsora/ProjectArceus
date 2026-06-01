"""DIRECT evidence that the phantom (normalized) advantage shapes the policy in
early training for the JEPA encoder but not the random CNN.

Terminal-only reward, NO success occurs in this window, so the only driver of
the policy update is the critic-derived advantage. For each encoder/seed we run
25 PPO updates and, on a FIXED probe set of states, log:
  H        : mean policy entropy  (max ln4=1.386)
  KLunif   : KL( mean-policy || uniform )  -> how far the marginal policy moved
  pbar[a]  : marginal action prob (mean over probe states)
And from each rollout (the thing that drives the update) we log the
WHOLE-BATCH-NORMALIZED advantage broken down by action:
  advA[a]  : mean normalized advantage of transitions whose action == a
A consistent (state/action-structured) advantage => some action's advA stays
systematically +/-, the policy drifts that way (KLunif grows, H falls).
Pure noise => advA ~ 0, policy stays uniform.

We also report the causal link corr( Σ_t advA[a], Δlogit[a] ): did the actions
that accumulated positive advantage get their logits pushed up?

Run: uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/10_phantom_policy_shaping.py
"""
import os, sys, json
from pathlib import Path
import multiprocessing as mp
ROOT = Path(__file__).resolve().parents[5]
DBG  = Path(__file__).resolve().parents[1]

def worker(spec):
    kind, seed = spec
    import torch; torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"]="1"
    sys.path.insert(0,str(ROOT)); import numpy as np
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
    ENC=ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
    DEV=torch.device("cpu"); NUPD=25
    torch.manual_seed(seed); np.random.seed(seed); m=ActorCritic()
    if kind!="random": m.encoder.load_state_dict(torch.load(ENC,map_location="cpu",weights_only=False)["encoder"])
    if kind=="resc":
        s=2.3/152.0; orig=m.encoder.forward; m.encoder.forward=lambda x,_o=orig,_s=s:_o(x)*_s
    # fixed probe set of states
    penv=VecLS20Env("ls20",8,200,seed=1000+seed); pobs=[penv.current_obs()]
    rng=np.random.default_rng(seed)
    for _ in range(64): o,*_=penv.step(rng.integers(0,4,8)); pobs.append(o)
    P=torch.from_numpy(np.concatenate(pobs,0))
    def probe():
        with torch.no_grad():
            lg,_,_=m.forward(P); pi=torch.softmax(lg,-1)
            H=torch.distributions.Categorical(probs=pi).entropy().mean().item()
            pbar=pi.mean(0)                       # marginal action dist
            kl=(pbar*torch.log(pbar/0.25)).sum().item()
            logit_mean=lg.mean(0)                 # mean logits over probe states
        return H,kl,pbar.numpy(),logit_mean.numpy()
    H0,kl0,pbar0,logit0=probe()
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],3e-4)
    envs=VecLS20Env("ls20",8,200,seed=seed)
    Hs=[H0]; KLs=[kl0]; PBs=[pbar0.tolist()]; advA_cum=np.zeros(4)
    for u in range(NUPD):
        roll=collect_rollout(envs,m,DEV,128); compute_gae(roll,0.99,0.95)
        adv=roll.advantages.reshape(-1); act=roll.actions.reshape(-1)
        advn=((adv-adv.mean())/(adv.std()+1e-8)).numpy(); a=act.numpy()
        advA=np.array([advn[a==k].mean() if (a==k).any() else 0.0 for k in range(4)])
        advA_cum+=advA
        ppo_update(m,opt,roll,PPOConfig(),DEV)
        H,kl,pbar,logit=probe(); Hs.append(H); KLs.append(kl); PBs.append(pbar.tolist())
    _,_,_,logitF=probe(); dlogit=logitF-logit0
    # causal link: did actions with more accumulated advantage get higher logits?
    corr=float(np.corrcoef(advA_cum,dlogit)[0,1]) if np.std(advA_cum)>1e-9 and np.std(dlogit)>1e-9 else float("nan")
    return f"{kind}_s{seed}", dict(kind=kind,seed=seed,H=Hs,KL=KLs,PB=PBs,
                                   advA_cum=advA_cum.tolist(),dlogit=dlogit.tolist(),adv_dlogit_corr=corr)

if __name__=="__main__":
    mp.set_start_method("spawn")
    specs=[(k,s) for k in ["random","raw","resc"] for s in [0,1,2]]
    with mp.Pool(min(9,len(specs))) as pool:
        res=dict(pool.map(worker,specs))
    json.dump(res, open(DBG/"data/policy_shaping.json","w"), indent=1)
    # quick console summary
    import statistics as st
    for k in ["random","raw","resc"]:
        ks=[v for v in res.values() if v["kind"]==k]
        Hend=st.mean(v["H"][-1] for v in ks); KLend=st.mean(v["KL"][-1] for v in ks)
        corr=st.mean(v["adv_dlogit_corr"] for v in ks if v["adv_dlogit_corr"]==v["adv_dlogit_corr"])
        maxadv=st.mean(max(abs(x) for x in v["advA_cum"]) for v in ks)
        print(f"{k:>7}: H {ks[0]['H'][0]:.3f}->{Hend:.3f}  KL(pi||unif)_end={KLend:.4f}  "
              f"max|cum adv per action|={maxadv:.2f}  corr(cumAdv,Δlogit)={corr:.2f}", flush=True)
    print("saved data/policy_shaping.json")
