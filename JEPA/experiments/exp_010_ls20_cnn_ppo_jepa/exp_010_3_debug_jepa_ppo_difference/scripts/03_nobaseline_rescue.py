"""CAUSAL test of the phantom-advantage mechanism.

If a structured encoder fails because the *value baseline* manufactures
state-consistent phantom advantages from zero reward, then REMOVING the
baseline (Monte-Carlo advantages = discounted reward-to-go; with zero reward
the advantages are exactly 0) should RESCUE the structured encoder: it should
keep exploring at high entropy and stumble on the sparse reward like the
random-encoder baseline does.

Compare, over 90 updates each (seed 0):
  resc  + GAE-baseline   (already known: never stumbles, entropy collapses)
  resc  + NO baseline    (prediction: explores stably, stumbles, learns)
  random+ NO baseline    (control: still works)
"""
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path("/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo")
sys.path.insert(0, str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update

ENC = ROOT / "JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
DEV = torch.device("cpu")
N_UPD, SEED = 90, 0

def build(kind):
    torch.manual_seed(SEED); np.random.seed(SEED)
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
    return m.to(DEV)

def mc_returns(roll, gamma):
    """Discounted reward-to-go within episodes (bootstrap 0 at truncation)."""
    T, N = roll.rewards.shape
    ret = torch.zeros(T, N)
    nxt = roll.bootstrap_value.float() * 0.0   # no bootstrap (pure MC)
    for t in reversed(range(T)):
        nonterm = (~roll.dones[t]).float()
        nxt = roll.rewards[t] + gamma * nxt * nonterm
        ret[t] = nxt
    return ret

def run(kind, baseline):
    m = build(kind)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-4)
    envs = VecLS20Env("ls20", n_envs=8, max_episode_steps=200, seed=SEED)
    first=None; tot=0; log=[]
    for u in range(1, N_UPD+1):
        roll = collect_rollout(envs, m, DEV, 128)
        compute_gae(roll, 0.99, 0.95)
        if not baseline:                       # replace GAE advantages with MC returns
            mc = mc_returns(roll, 0.99)
            roll.advantages = mc               # no value baseline subtracted
            roll.returns = mc                  # value head still regresses MC return
        st = ppo_update(m, opt, roll, PPOConfig(), DEV)
        n = int((roll.rewards>0).sum().item()); tot+=n
        if n>0 and first is None: first=u
        log.append((u, st.entropy, n))
    return kind, baseline, first, tot, log

if __name__ == "__main__":
    for kind, base in [("resc", False), ("random", False)]:
        tag = f"{kind}_{'GAE' if base else 'NObaseline'}"
        print(f"\n=== {tag} ===", flush=True)
        k,b,fs,tot,log = run(kind, base)
        for u,e,n in log:
            if u in (1,5,10,20,30,40,50,70,90):
                print(f"  u{u:>3} ent={e:.3f} succ_this_upd={n}", flush=True)
        print(f"  >> first success @update {fs}, total successes = {tot}", flush=True)
