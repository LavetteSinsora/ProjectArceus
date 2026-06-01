"""Full per-update curves for the four HERO conditions (seed 0, 90 updates),
saved to exp_010_3/data/curves.json for the report figures.

  random_base   : random-init encoder + GAE value baseline   (solves)
  resc_base     : norm-matched JEPA encoder + value baseline  (fails, ent collapses w/o reward)
  froz_base     : frozen JEPA encoder + value baseline        (fails)
  resc_nobase   : norm-matched JEPA encoder, NO value baseline (RESCUED)

Logs per update: policy_entropy, value_loss, grad_norm, n_success.
Run from repo root:  uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/06_figure_curves.py
"""
import os, sys, json
from pathlib import Path
import multiprocessing as mp
ROOT = Path(__file__).resolve().parents[5]  # scripts/debug/exp_010../experiments/JEPA/Code Repo
DATA = Path(__file__).resolve().parents[1] / "data"

def worker(spec):
    kind, baseline = spec
    import torch; torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"]="1"
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update
    ENC = ROOT/"JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
    DEV=torch.device("cpu"); SEED=0; NUPD=90
    torch.manual_seed(SEED); np.random.seed(SEED)
    m=ActorCritic(); frozen=False
    if kind!="random":
        m.encoder.load_state_dict(torch.load(ENC,map_location="cpu",weights_only=False)["encoder"])
    if kind=="resc":
        with torch.no_grad():
            env=VecLS20Env("ls20",n_envs=8,max_episode_steps=200,seed=999); o=env.current_obs()
            for _ in range(20): o,*_=env.step(np.random.randint(0,4,size=8))
            cur=m.features(torch.from_numpy(o)).norm(dim=-1).mean().item()
        s=2.3/cur; orig=m.encoder.forward; m.encoder.forward=lambda x,_o=orig,_s=s:_o(x)*_s
    if kind=="froz":
        for p in m.encoder.parameters(): p.requires_grad_(False)
        m.encoder.eval(); frozen=True
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],3e-4)
    envs=VecLS20Env("ls20",n_envs=8,max_episode_steps=200,seed=SEED)
    def mc(roll,g):
        T,N=roll.rewards.shape; ret=torch.zeros(T,N); nxt=torch.zeros(N)
        for t in reversed(range(T)):
            nt=(~roll.dones[t]).float(); nxt=roll.rewards[t]+g*nxt*nt; ret[t]=nxt
        return ret
    ent=[]; vl=[]; gn=[]; cs=[]; cum=0
    for u in range(1,NUPD+1):
        roll=collect_rollout(envs,m,DEV,128); compute_gae(roll,0.99,0.95)
        if not baseline:
            r=mc(roll,0.99); roll.advantages=r; roll.returns=r
        st=ppo_update(m,opt,roll,PPOConfig(),DEV)
        cum+=int((roll.rewards>0).sum().item())
        ent.append(round(st.entropy,4)); vl.append(round(st.value_loss,4))
        gn.append(round(st.grad_norm_total,4)); cs.append(cum)
    tag=f"{kind}_{'base' if baseline else 'nobase'}"
    return tag, dict(entropy=ent, value_loss=vl, grad_norm=gn, cum_success=cs)

if __name__=="__main__":
    mp.set_start_method("spawn")
    DATA.mkdir(exist_ok=True)
    specs={"random_base":("random",True),"resc_base":("resc",True),
           "froz_base":("froz",True),"resc_nobase":("resc",False)}
    with mp.Pool(4) as pool:
        out=dict(pool.map(worker, list(specs.values())))
    json.dump(out, open(DATA/"curves.json","w"), indent=1)
    print("saved", DATA/"curves.json", "keys:", list(out))
