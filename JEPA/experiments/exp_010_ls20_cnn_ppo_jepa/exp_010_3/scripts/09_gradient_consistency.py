"""Operationalize 'state-consistent phantom advantage' and disentangle it from
feature norm. Zero-reward rollout, untrained policy, per encoder.

Metrics (mean over seeds):
  Vstd        : std of V(s) across rollout states (raw, pre-norm)
  adv_std_raw : std of GAE advantages BEFORE normalization
  |g_raw|     : L2 norm of the actor-loss gradient using RAW advantages
  |g_norm|    : L2 norm of the actor-loss gradient using NORMALIZED advantages
  splithalf_cos_RAW  : cosine between actor-grads on two disjoint halves, RAW adv
  splithalf_cos_NORM : same, with per-half NORMALIZED adv   <-- 'consistency'
  dH_1upd     : entropy change after ONE real PPO update (zero reward)

Reasoning: a CONSISTENT (state/action-correlated) advantage gives a reproducible
gradient direction -> high split-half cosine. Pure noise -> cosine ~0.
Advantage normalization divides out magnitude, so |g_norm| is ~equal across
encoders even when Vstd differs 70x -> 'amplification' of a tiny-but-consistent
signal. The norm only sets RAW magnitude; consistency is norm-invariant.
"""
import sys, numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
ROOT=Path("/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo")
sys.path.insert(0,str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
ENC=ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
DEV=torch.device("cpu")

def build(kind,seed):
    torch.manual_seed(seed); np.random.seed(seed); m=ActorCritic()
    if kind!="random": m.encoder.load_state_dict(torch.load(ENC,map_location="cpu",weights_only=False)["encoder"])
    if kind=="resc":
        s=2.3/152.0; orig=m.encoder.forward; m.encoder.forward=lambda x,_o=orig,_s=s:_o(x)*_s  # fixed scale (avoid extra env build)
    return m

def actor_grad(m, obs, act, adv):
    m.zero_grad(set_to_none=True)
    logp,ent,val,feat = m.evaluate(obs, act)
    loss = -(logp*adv).mean()
    loss.backward()
    return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None and p.requires_grad])

def cos(a,b): return float((a@b)/(a.norm()*b.norm()+1e-12))

def run(kind, seed):
    m=build(kind,seed)
    envs=VecLS20Env("ls20",8,200,seed=seed)
    roll=collect_rollout(envs,m,DEV,128); compute_gae(roll,0.99,0.95)
    assert float(roll.rewards.sum())==0.0
    obs=roll.obs.reshape(-1,64,64); act=roll.actions.reshape(-1); adv=roll.advantages.reshape(-1)
    n=len(adv); idx=torch.randperm(n); A=idx[:n//2]; B=idx[n//2:]
    Vstd=roll.values.std().item(); adv_std=adv.std().item()
    g_raw =actor_grad(m,obs,act,adv)
    g_norm=actor_grad(m,obs,act,(adv-adv.mean())/(adv.std()+1e-8))
    # split-half, RAW
    gA=actor_grad(m,obs[A],act[A],adv[A]); gB=actor_grad(m,obs[B],act[B],adv[B]); ch_raw=cos(gA,gB)
    # split-half, per-half NORMALIZED (mirrors PPO minibatch normalization)
    aA=(adv[A]-adv[A].mean())/(adv[A].std()+1e-8); aB=(adv[B]-adv[B].mean())/(adv[B].std()+1e-8)
    gAn=actor_grad(m,obs[A],act[A],aA); gBn=actor_grad(m,obs[B],act[B],aB); ch_norm=cos(gAn,gBn)
    # one real PPO update, entropy change
    with torch.no_grad():
        lg,_,_=m.forward(obs); H0=torch.distributions.Categorical(logits=lg).entropy().mean().item()
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],3e-4)
    ppo_update(m,opt,roll,PPOConfig(),DEV)
    with torch.no_grad():
        lg,_,_=m.forward(obs); H1=torch.distributions.Categorical(logits=lg).entropy().mean().item()
    return dict(Vstd=Vstd,adv_std=adv_std,g_raw=g_raw.norm().item(),g_norm=g_norm.norm().item(),
                ch_raw=ch_raw,ch_norm=ch_norm,dH=H1-H0)

if __name__=="__main__":
    import statistics as st
    print(f"{'enc':>8} {'Vstd':>7} {'adv_std_raw':>11} {'|g_raw|':>9} {'|g_norm|':>9} {'cos_RAW':>8} {'cos_NORM':>9} {'dH_1upd':>8}")
    for kind in ["random","raw","resc"]:
        rs=[run(kind,s) for s in [0,1,2]]
        ag=lambda k: st.mean(r[k] for r in rs)
        print(f"{kind:>8} {ag('Vstd'):>7.3f} {ag('adv_std'):>11.4f} {ag('g_raw'):>9.3f} {ag('g_norm'):>9.3f} "
              f"{ag('ch_raw'):>8.3f} {ag('ch_norm'):>9.3f} {ag('dH'):>8.4f}", flush=True)
