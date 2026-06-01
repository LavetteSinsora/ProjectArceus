"""Decisive control for exp_010: is 10_2's failure caused by the pretrained
encoder's LARGE FEATURE NORM (-> value-head recalibration -> large pre-reward
gradients into the shared encoder), or by the representation CONTENT?

Four matched PPO runs (same env seed, same seeded policy/value heads):
  A random   : random-init encoder            (= 10_0 baseline)
  B raw-pre  : pretrained encoder, as-is       (= 10_2, "loud")
  C resc-pre : pretrained encoder, output rescaled to ~random-init norm
               (identical feature DIRECTIONS, only the scale changed -> "quiet")
  D froz-pre : pretrained encoder, FROZEN (heads adapt to fixed good features)

If C (and/or D) solves like A while B never does, the failure is the norm-
induced pre-reward gradient, not the representation content.
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo")
sys.path.insert(0, str(ROOT))
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import ActorCritic
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.rollout import collect_rollout, compute_gae
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ppo import PPOConfig, ppo_update

ENC = ROOT / "JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt"
DEVICE = torch.device("cpu")   # small model; CPU avoids MPS overhead for fairness
N_UPDATES = 140
N_ENVS, ROLLOUT, MAXEP, SEED = 8, 128, 200, 0


def build(kind):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = ActorCritic()                       # heads identically seeded across kinds
    if kind != "random":
        sd = torch.load(ENC, map_location="cpu", weights_only=False)["encoder"]
        m.encoder.load_state_dict(sd)
    if kind == "resc":
        # measure current mean feature norm, scale so it ~matches random-init (~2.3)
        with torch.no_grad():
            env = VecLS20Env("ls20", n_envs=8, max_episode_steps=MAXEP, seed=99)
            obs = env.current_obs()
            for _ in range(20):
                obs, *_ = env.step(np.random.randint(0, 4, size=8))
            f = m.features(torch.from_numpy(obs))
            cur = f.norm(dim=-1).mean().item()
        scale = 2.3 / cur
        orig = m.encoder.forward
        m.encoder.forward = lambda x, _o=orig, _s=scale: _o(x) * _s
        print(f"  [resc] cur_norm={cur:.1f} scale={scale:.4f}", flush=True)
    if kind == "froz":
        for p in m.encoder.parameters():
            p.requires_grad_(False)
        m.encoder.eval()
    return m.to(DEVICE)


def run(kind):
    m = build(kind)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-4)
    pcfg = PPOConfig()
    envs = VecLS20Env("ls20", n_envs=N_ENVS, max_episode_steps=MAXEP, seed=SEED)
    first_succ = None
    total_succ = 0
    log = []
    for u in range(1, N_UPDATES + 1):
        roll = collect_rollout(envs, m, DEVICE, ROLLOUT)
        compute_gae(roll, 0.99, 0.95)
        st = ppo_update(m, opt, roll, pcfg, DEVICE, clip_params=params)
        nsucc = int((roll.rewards > 0).sum().item())
        total_succ += nsucc
        if nsucc > 0 and first_succ is None:
            first_succ = u
        log.append((u, st.entropy, st.grad_norm_total, st.value_loss, nsucc))
    # quick deterministic-ish eval: greedy rollout success rate over 16 eps
    return kind, first_succ, total_succ, log


if __name__ == "__main__":
    results = {}
    for kind in (sys.argv[1:] or ["random", "raw", "resc", "froz"]):
        print(f"\n=== {kind} ===", flush=True)
        k, fs, ts, log = run(kind)
        results[k] = dict(first_succ=fs, total_succ=ts)
        # print a few checkpoints
        for u, e, g, v, n in log:
            if u in (1, 2, 3, 5, 10, 20, 40, 70, 100, 140):
                print(f"  u{u:>3} ent={e:.3f} grad={g:6.2f} vloss={v:8.3f} succ_this_upd={n}", flush=True)
        print(f"  >> first observed success at update {fs}, total successes in {N_UPDATES} updates = {ts}", flush=True)
    print("\nSUMMARY:", json.dumps(results), flush=True)
