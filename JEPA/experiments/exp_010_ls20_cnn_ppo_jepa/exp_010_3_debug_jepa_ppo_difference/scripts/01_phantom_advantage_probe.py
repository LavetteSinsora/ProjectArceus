"""Test the PHANTOM-ADVANTAGE mechanism (no real training, zero reward).

Claim: with terminal-only reward and NO success yet, the only source of a
policy gradient is the value baseline. A *structured* encoder lets the critic
fit approximation noise into a state-varying V(s); GAE turns that into
state-consistent advantages; PPO chases them and collapses exploration. A
*collapsed/random* encoder has V(s)~const -> no phantom advantages -> the
policy stays a stable uniform explorer.

For each encoder we: collect ONE rollout with the untrained policy (rewards are
all 0), compute GAE, and report
  feat_cos  : mean pairwise cosine of features (1.0 = collapsed)
  V_std     : std of V(s) across rollout states  (phantom value structure)
  adv_std   : std of GAE advantages
  dent_1upd : entropy AFTER one PPO update minus before (how much exploration
              the value-noise destroys in a single update, with zero reward)
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

def build(kind, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ActorCritic()
    if kind != "random":
        sd = torch.load(ENC, map_location="cpu", weights_only=False)["encoder"]
        m.encoder.load_state_dict(sd)
    if kind == "resc":
        with torch.no_grad():
            env = VecLS20Env("ls20", n_envs=8, max_episode_steps=200, seed=999)
            o = env.current_obs()
            for _ in range(20): o,*_ = env.step(np.random.randint(0,4,size=8))
            cur = m.features(torch.from_numpy(o)).norm(dim=-1).mean().item()
        s = 2.3/cur; orig = m.encoder.forward
        m.encoder.forward = lambda x,_o=orig,_s=s: _o(x)*_s
    return m.to(DEV)

def probe(kind, seed):
    m = build(kind, seed)
    envs = VecLS20Env("ls20", n_envs=8, max_episode_steps=200, seed=seed)
    roll = collect_rollout(envs, m, DEV, 128)
    compute_gae(roll, 0.99, 0.95)
    assert float(roll.rewards.sum())==0.0, "expected zero reward in first rollout"
    feats = roll.features.reshape(-1, roll.features.shape[-1])
    fn = torch.nn.functional.normalize(feats, dim=-1)
    cos = (fn @ fn.T)[torch.triu(torch.ones(len(fn),len(fn)),1)>0].mean().item()
    V_std = roll.values.std().item()
    adv_std = roll.advantages.std().item()
    # one PPO update, measure entropy change
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-4)
    with torch.no_grad():
        logits,_,_ = m.forward(roll.obs.reshape(-1,64,64))
        ent0 = torch.distributions.Categorical(logits=logits).entropy().mean().item()
    st = ppo_update(m, opt, roll, PPOConfig(), DEV)
    with torch.no_grad():
        logits,_,_ = m.forward(roll.obs.reshape(-1,64,64))
        ent1 = torch.distributions.Categorical(logits=logits).entropy().mean().item()
    return dict(feat_cos=cos, V_std=V_std, adv_std=adv_std, ent0=ent0, ent1=ent1, dent=ent1-ent0)

if __name__ == "__main__":
    print(f"{'kind':>8} {'seed':>4} {'feat_cos':>8} {'V_std':>8} {'adv_std':>8} {'ent0':>6} {'ent1':>6} {'dent_1upd':>9}")
    for seed in [0,1,2]:
        for kind in ["random","raw","resc"]:
            r = probe(kind, seed)
            print(f"{kind:>8} {seed:>4} {r['feat_cos']:>8.3f} {r['V_std']:>8.3f} "
                  f"{r['adv_std']:>8.4f} {r['ent0']:>6.3f} {r['ent1']:>6.3f} {r['dent']:>9.4f}", flush=True)
