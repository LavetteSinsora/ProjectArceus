"""exp_010_4 — collapse-prevention recipe study for the LS20-L1 JEPA encoder.

Goal: find the recipe that yields the best *frozen state representation* (prior)
for LS20 Level 1, using the FIXED exp_010 CNN trunk (trunk_dim=256, post-ReLU).

We compare action-conditioned forward-JEPA trained with different anti-collapse
mechanisms, on identical real-LS20 random transitions, judged by a label-light
eval harness (representation health + forward predictivity + action-decodability
+ a frame-diff agent-localization probe). No IDM-as-loss (per the modern recipe);
IDM is used only as a frozen probe.

Recipes (all: same fixed trunk -> 128-d projector -> predict next z given action):
  baseline_sg : MSE(pred, sg(z_next)), no regulariser           (the exp_010_2 trap)
  vicreg      : MSE(pred, z_next) + VICReg(var+cov) on z          (delays collapse)
  sigreg      : MSE(pred, z_next) + lambda*SIGReg(z)  [LeJEPA]     (isotropic-Gaussian)
  ema         : MSE(pred, sg(z_next^EMA)), EMA target             (BYOL/I-JEPA)
  sigreg_ema  : ema target + SIGReg(z)

Modern grounding: LeJEPA (Balestriero & LeCun, 2511.08544) — regularise embeddings
toward an isotropic Gaussian via SIGReg (random 1-D slices + Epps-Pulley CF test);
the isotropic Gaussian provably minimises downstream linear-probe risk = "best prior".

Run:  PYTHONPATH=<repo> python -m ...exp_010_4_jepa_recipe_research.study
Outputs go to /tmp/jepa4 (fast local disk); the report step copies the keepers.
"""
from __future__ import annotations
import json, os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
import sys; sys.path.insert(0, REPO)
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import (
    CNNEncoder, one_hot_frame, TRUNK_DIM, N_ACTIONS)
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.ls20_vec_env import VecLS20Env

OUT = "/tmp/jepa4"; os.makedirs(OUT, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ZDIM = 128
torch.manual_seed(0); np.random.seed(0)


# ----------------------------------------------------------------- data
def collect(n, seed, n_envs=16, max_steps=200):
    env = VecLS20Env("ls20", n_envs=n_envs, max_episode_steps=max_steps, seed=seed)
    rng = np.random.default_rng(seed)
    O, A, NO = [], [], []
    obs = env.current_obs()
    while len(O) < n:
        a = rng.integers(0, env.n_actions, size=n_envs)
        no, _, dones, _ = env.step(a)
        for i in range(n_envs):
            if not dones[i]:                       # skip reset frames
                O.append(obs[i]); A.append(int(a[i])); NO.append(no[i])
        obs = no
    return (np.asarray(O[:n], np.uint8), np.asarray(A[:n], np.int64),
            np.asarray(NO[:n], np.uint8))


def get_data():
    p = f"{OUT}/buf.npz"
    if os.path.exists(p):
        d = np.load(p); return {k: d[k] for k in d.files}
    t0 = time.time()
    o, a, no = collect(60000, 0)
    vo, va, vno = collect(8000, 999)
    d = dict(o=o, a=a, no=no, vo=vo, va=va, vno=vno)
    np.savez_compressed(p, **d)
    print(f"collected in {time.time()-t0:.0f}s  train={len(o)} val={len(vo)}", flush=True)
    return d


# ----------------------------------------------------------------- model
class JEPA(nn.Module):
    def __init__(self, ema=False):
        super().__init__()
        self.trunk = CNNEncoder(trunk_dim=TRUNK_DIM)          # FIXED arch
        self.proj = nn.Linear(TRUNK_DIM, ZDIM)
        self.act_emb = nn.Embedding(N_ACTIONS, 32)
        self.pred = nn.Sequential(nn.Linear(ZDIM + 32, 256), nn.ReLU(),
                                  nn.Linear(256, ZDIM))
        if ema:
            self.t_trunk = CNNEncoder(trunk_dim=TRUNK_DIM)
            self.t_proj = nn.Linear(TRUNK_DIM, ZDIM)
            self._sync(); [p.requires_grad_(False) for p in self.t_params()]
    def t_params(self):
        return list(self.t_trunk.parameters()) + list(self.t_proj.parameters())
    def o_params(self):
        return list(self.trunk.parameters()) + list(self.proj.parameters())
    @torch.no_grad()
    def _sync(self):
        for s, t in zip(self.o_params(), self.t_params()): t.copy_(s.data)
    @torch.no_grad()
    def ema_update(self, m=0.996):
        for s, t in zip(self.o_params(), self.t_params()): t.mul_(m).add_(s.data, alpha=1-m)
    def h(self, obs_u8):
        return self.trunk(one_hot_frame(obs_u8))
    def z(self, obs_u8): return self.proj(self.h(obs_u8))
    def z_tgt(self, obs_u8): return self.t_proj(self.t_trunk(one_hot_frame(obs_u8)))
    def predict(self, z_t, a): return self.pred(torch.cat([z_t, self.act_emb(a)], -1))


# ----------------------------------------------------------------- losses
def vicreg(z, zp):
    def vc(x):
        x = x - x.mean(0)
        std = torch.sqrt(x.var(0) + 1e-4)
        v = F.relu(1 - std).mean()
        c = (x.T @ x) / (x.shape[0] - 1)
        cov = (c - torch.diag(torch.diag(c))).pow(2).sum() / x.shape[1]
        return v, cov
    v1, c1 = vc(z); v2, c2 = vc(zp)
    return 25*(v1+v2) + 1.0*(c1+c2)

def sigreg(z, n_slices=256, n_points=17, dev="cpu"):
    """SIGReg: project onto random unit dirs, Epps-Pulley CF test vs N(0,1).
    Penalises deviation of each 1-D marginal from a standard normal -> drives the
    embedding to an isotropic Gaussian (no collapse, fixed scale)."""
    D = z.shape[1]
    dirs = F.normalize(torch.randn(D, n_slices, device=z.device), dim=0)
    p = z @ dirs                                    # (B, S) raw projections
    t = torch.linspace(-5, 5, n_points, device=z.device)
    ang = p[..., None] * t                          # (B, S, T)
    re = torch.cos(ang).mean(0); im = torch.sin(ang).mean(0)   # (S, T)
    phi0 = torch.exp(-0.5 * t**2)                   # CF of N(0,1)
    w = torch.exp(-0.5 * t**2)                      # Epps-Pulley weight
    ep = (((re - phi0)**2 + im**2) * w).mean(-1)    # (S,)
    return ep.mean()


# ----------------------------------------------------------------- train
def train(recipe, data, steps=1000, bs=128, lr=3e-4, log_every=200):
    o = torch.tensor(data["o"]); a = torch.tensor(data["a"]); no = torch.tensor(data["no"])
    n = o.shape[0]; idx = np.arange(n)
    ema = recipe in ("ema", "sigreg_ema")
    torch.manual_seed(1234); np.random.seed(1234)   # same init + data order for all recipes
    m = JEPA(ema=ema).to(DEV)
    opt = torch.optim.Adam(m.o_params() + list(m.act_emb.parameters())
                           + list(m.pred.parameters()), lr=lr)
    curve = []
    for step in range(steps + 1):
        if step % log_every == 0:
            curve.append({"step": step, **evaluate(m, data)})
            r = curve[-1]
            print(f"[{recipe}] {step:>4} rank={r['rank']:.1f} std={r['std']:.3f} "
                  f"cos={r['cos']:.3f} |h|={r['norm']:.1f} fwdR2={r['fwd_r2']:.3f} "
                  f"idm={r['idm_acc']:.3f} loc={r['loc_r2']:.3f}", flush=True)
        b = np.random.choice(idx, bs, replace=False)
        ob, ab, nob = o[b].to(DEV), a[b].to(DEV), no[b].to(DEV)
        z = m.z(ob); zp = m.predict(z, ab)
        if ema:
            with torch.no_grad(): zt = m.z_tgt(nob)
            loss = F.mse_loss(zp, zt)
            if recipe == "sigreg_ema": loss = loss + 1.0 * sigreg(z)
        else:
            zn = m.z(nob)
            if recipe == "baseline_sg":
                loss = F.mse_loss(zp, zn.detach())
            elif recipe == "vicreg":
                loss = F.mse_loss(zp, zn) + vicreg(z, zn)
            elif recipe == "sigreg":
                loss = F.mse_loss(zp, zn) + 1.0 * sigreg(z)
            else:
                raise ValueError(recipe)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ema: m.ema_update()
    return m, curve


# ----------------------------------------------------------------- eval harness
@torch.no_grad()
def feats(m, obs_u8, bs=1024):
    out = []
    for i in range(0, obs_u8.shape[0], bs):
        out.append(m.h(torch.tensor(obs_u8[i:i+bs]).to(DEV)).cpu())
    return torch.cat(out)

def eff_rank(H):
    Hc = H - H.mean(0); C = (Hc.T @ Hc) / (Hc.shape[0]-1)
    ev = torch.linalg.eigvalsh(C).clamp(min=0)
    return ((ev.sum()**2) / (ev.pow(2).sum()+1e-12)).item()

def ridge_r2(Xtr, Ytr, Xva, Yva, lam=1e-2):
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr),1))],1)
    Xva = np.concatenate([Xva, np.ones((len(Xva),1))],1)
    W = np.linalg.solve(Xtr.T@Xtr + lam*np.eye(Xtr.shape[1]), Xtr.T@Ytr)
    pred = Xva@W; ss = ((pred-Yva)**2).sum(); tot = ((Yva-Yva.mean(0))**2).sum()
    return float(1 - ss/tot)

def changed_centroid(o, no):
    """Frame-diff proxy for agent location: centroid (row,col) of changed cells."""
    d = (o.astype(np.int16) != no.astype(np.int16))
    out = np.full((len(o), 2), np.nan)
    for i in range(len(o)):
        ys, xs = np.nonzero(d[i])
        if len(ys) > 0: out[i] = [ys.mean(), xs.mean()]
    return out

@torch.no_grad()
def evaluate(m, data, n=3000):
    m.eval()
    Hv = feats(m, data["vo"][:n]).float()
    Hn = feats(m, data["vno"][:n]).float()
    av = torch.tensor(data["va"][:n])
    # representation health (on trunk h = downstream prior)
    rank = eff_rank(Hv); std = Hv.std(0).mean().item(); norm = Hv.norm(dim=1).mean().item()
    Hn1 = F.normalize(Hv, dim=1); cos = (Hn1 @ Hn1.mean(0, keepdim=True).T).mean().item()
    # forward predictivity in z-space (normalized R^2 vs predict-mean)
    zt = m.proj(Hv.to(DEV)); zn = m.proj(Hn.to(DEV)); zp = m.predict(zt, av.to(DEV))
    resid = (zp-zn).pow(2).sum(1).mean(); totz = (zn-zn.mean(0)).pow(2).sum(1).mean()
    fwd_r2 = float((1 - resid/ (totz+1e-9)).item())
    # action-decodability (frozen linear IDM on [h_t,h_next], labels=actions)
    half = n//2
    Xtr = torch.cat([Hv[:half], Hn[:half]],1).numpy(); ytr = av[:half].numpy()
    Xva = torch.cat([Hv[half:], Hn[half:]],1).numpy(); yva = av[half:].numpy()
    idm = ridge_classify(Xtr, ytr, Xva, yva, N_ACTIONS)
    # agent-localization probe (frame-diff proxy), only valid (moved) rows
    loc = changed_centroid(data["vo"][:n], data["vno"][:n])
    ok = ~np.isnan(loc[:,0]);
    if ok.sum() > 200:
        Hh = Hv.numpy()[ok]; Y = loc[ok]; k = len(Hh)//2
        loc_r2 = ridge_r2(Hh[:k], Y[:k], Hh[k:], Y[k:])
    else:
        loc_r2 = float("nan")
    m.train()
    return dict(rank=rank, std=std, norm=norm, cos=cos, fwd_r2=fwd_r2,
                idm_acc=idm, loc_r2=loc_r2)

def ridge_classify(Xtr, ytr, Xva, yva, k, lam=1e-1):
    Y = np.eye(k)[ytr]
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr),1))],1)
    Xva = np.concatenate([Xva, np.ones((len(Xva),1))],1)
    W = np.linalg.solve(Xtr.T@Xtr + lam*np.eye(Xtr.shape[1]), Xtr.T@Y)
    return float((np.argmax(Xva@W,1) == yva).mean())


# ----------------------------------------------------------------- references
@torch.no_grad()
def eval_trunk_only(trunk, data, n=3000):
    """Eval a bare trunk (random-init or loaded exp_010_2) — no projector/predictor."""
    vo = data["vo"][:n]; vno = data["vno"][:n]
    def h(obs): return trunk(one_hot_frame(obs))
    Hv = torch.cat([h(torch.tensor(vo[i:i+1024]).to(DEV)).cpu()
                    for i in range(0, len(vo), 1024)]).float()
    Hn = torch.cat([h(torch.tensor(vno[i:i+1024]).to(DEV)).cpu()
                    for i in range(0, len(vno), 1024)]).float()
    av = torch.tensor(data["va"][:n])
    rank = eff_rank(Hv); std = Hv.std(0).mean().item(); norm = Hv.norm(dim=1).mean().item()
    Hn1 = F.normalize(Hv, dim=1); cos = (Hn1 @ Hn1.mean(0, keepdim=True).T).mean().item()
    half = n//2
    idm = ridge_classify(torch.cat([Hv[:half],Hn[:half]],1).numpy(), av[:half].numpy(),
                         torch.cat([Hv[half:],Hn[half:]],1).numpy(), av[half:].numpy(), N_ACTIONS)
    loc = changed_centroid(data["vo"][:n], data["vno"][:n]); ok = ~np.isnan(loc[:,0])
    Hh = Hv.numpy()[ok]; Y = loc[ok]; k = len(Hh)//2
    loc_r2 = ridge_r2(Hh[:k], Y[:k], Hh[k:], Y[k:]) if ok.sum()>200 else float('nan')
    return dict(rank=rank, std=std, norm=norm, cos=cos, fwd_r2=float('nan'),
                idm_acc=idm, loc_r2=loc_r2)


def main():
    data = get_data()
    results = {}
    # references
    torch.manual_seed(0); ri = CNNEncoder(trunk_dim=TRUNK_DIM).to(DEV).eval()
    results["random_init"] = {"final": eval_trunk_only(ri, data), "curve": []}
    ckpt = os.path.join(REPO, "JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/"
                        "exp_010_2_jepa_random_pretrain/jepa_pretrained/encoder_final.pt")
    if os.path.exists(ckpt):
        e2 = CNNEncoder(trunk_dim=TRUNK_DIM).to(DEV)
        e2.load_state_dict(torch.load(ckpt, map_location=DEV)["encoder"]); e2.eval()
        results["exp_010_2"] = {"final": eval_trunk_only(e2, data), "curve": []}
    # recipes
    saved = {}
    for rc in ["baseline_sg", "vicreg", "sigreg", "ema"]:
        t0 = time.time()
        m, curve = train(rc, data)
        results[rc] = {"final": curve[-1], "curve": curve, "secs": time.time()-t0}
        saved[rc] = m
        torch.save({"trunk": m.trunk.state_dict()}, f"{OUT}/encoder_{rc}.pt")
        json.dump(results, open(f"{OUT}/metrics.json","w"), indent=1)  # incremental
    json.dump(results, open(f"{OUT}/metrics.json","w"), indent=1)
    print("DONE -> /tmp/jepa4/metrics.json", flush=True)


if __name__ == "__main__":
    main()
