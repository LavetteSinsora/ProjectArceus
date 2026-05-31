"""Fig 5: distribution of V(s) across states visited in a random rollout, under
a RANDOM (untrained) value head, for three frozen encoders.

Same visited states for all encoders (actions are uniform-random, encoder-
independent), and the value head is identically seeded — so any difference in
the V(s) distribution is due purely to the encoder. This visualizes the
'phantom value structure': random CNN -> tight spike (V approx const -> no
phantom advantages); JEPA -> broad spread (state-structured V). The zoomed panel
shows the NORM-MATCHED JEPA encoder still spreads far more than random, so the
spread comes from feature DIRECTION structure, not just feature norm.

Run from repo root:
  uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/08_value_histogram.py
"""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
EXP = Path(__file__).resolve().parents[1].parent  # exp_010 dir
DBG = Path(__file__).resolve().parents[1]
ENC = EXP/"exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"

def make(kind):
    torch.manual_seed(0); np.random.seed(0)
    m = ActorCritic()                       # value_head identically seeded across kinds
    if kind != "random":
        m.encoder.load_state_dict(torch.load(ENC, map_location="cpu", weights_only=False)["encoder"])
    if kind == "resc":
        s = 2.3/152.0; orig = m.encoder.forward
        m.encoder.forward = lambda x,_o=orig,_s=s: _o(x)*_s
    return m.eval()

# collect a fixed set of visited states with uniform-random actions (encoder-independent)
env = VecLS20Env("ls20", n_envs=8, max_episode_steps=200, seed=0)
rng = np.random.default_rng(0)
states = [env.current_obs()]
for _ in range(700):
    obs,*_ = env.step(rng.integers(0,4,size=8)); states.append(obs)
S = torch.from_numpy(np.concatenate(states,0))   # (~5600, 64,64)

V = {}
for kind in ["random","jepa","resc"]:
    m = make(kind)
    with torch.no_grad():
        _, val, feat = m.forward(S)
    V[kind] = val.numpy()
    print(f"{kind:>7}: V mean={val.mean():.4f} std={val.std():.4f} min={val.min():.3f} max={val.max():.3f} feat|h|={feat.norm(dim=-1).mean():.2f}", flush=True)

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
# (a) raw scale: random spike vs JEPA broad
lo,hi = float(min(V['jepa'].min(),-1)), float(max(V['jepa'].max(),1))
bins = np.linspace(lo, hi, 80)
ax[0].hist(V['jepa'], bins=bins, color="#d95f02", alpha=0.6, label=f"JEPA  (V_std={V['jepa'].std():.2f})", density=True)
ax[0].hist(V['random'], bins=bins, color="#1b9e77", alpha=0.85, label=f"random CNN  (V_std={V['random'].std():.3f})", density=True)
ax[0].set(title="(a) V(s) over visited states — raw scale\n(untrained value head, same states)", xlabel="V(s)", ylabel="density")
ax[0].legend(fontsize=8)
# (b) zoomed: random vs norm-matched JEPA (both small norm) -> still broader
m = 4*V['random'].std()
b2 = np.linspace(-max(m, 0.25), max(m, 0.25), 70)
ax[1].hist(V['resc'], bins=b2, color="#7570b3", alpha=0.6, label=f"JEPA norm-matched  (V_std={V['resc'].std():.3f})", density=True)
ax[1].hist(V['random'], bins=b2, color="#1b9e77", alpha=0.85, label=f"random CNN  (V_std={V['random'].std():.3f})", density=True)
ax[1].set(title="(b) zoomed: random vs NORM-MATCHED JEPA\n(spread is from feature directions, not norm)", xlabel="V(s)", ylabel="density")
ax[1].legend(fontsize=8)
fig.suptitle("Fig 5 — V(s) distribution across a random rollout (random value head): random CNN is a spike (V≈const); JEPA spreads",
             y=1.06, fontsize=10.5, weight="bold")
fig.savefig(DBG/"figures/fig5_value_hist.png", bbox_inches="tight", dpi=120)
json.dump({k:{"mean":float(v.mean()),"std":float(v.std()),"min":float(v.min()),"max":float(v.max())} for k,v in V.items()},
          open(DBG/"data/value_hist.json","w"), indent=1)
print("saved", DBG/"figures/fig5_value_hist.png")
