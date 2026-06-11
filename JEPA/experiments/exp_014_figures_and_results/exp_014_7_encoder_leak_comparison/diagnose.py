"""exp_014_7 — Encoder comparison: in which representation does RND counting work,
and how badly does counting one state LEAK into counting another?

The encoder-axis sibling of exp_014_6. exp_014_6 fixed the encoder (a frozen random
projection) and swept the leak μ. Here we fix the leak (μ=0 = standard RND by
default) and sweep the ENCODER, running one independent RND loop per encoder over
the SAME real LS20-L2 rollout stream:

    pixel   — raw masked board → RND directly (normalised indices, D=4096; or
              one-hot D=65536 with --pixel-onehot). No structure.
    linproj — a FROZEN random LINEAR projection of the one-hot board (D=256). The
              exp_014_1/_5/_6 RND input; included to tie back to the lineage.
    random  — a FROZEN random-init CNN encoder (D=256). A fixed nonlinear map.
    idm     — OUR encoder: ICM inverse-dynamics φ, trained ONLINE (D=256). The only
              encoder that must separate states (it names the action between them).

LEAK ISOLATION — the driver/probe split
---------------------------------------
The monitored set is partitioned into:
    * DRIVERS — heavily-visited near-reset states. These get distilled.
    * PROBES  — sibling near-reset states that are ACTION-BLOCKED from the start,
                so they are (almost) never visited and never distilled.
Leak is then the DIRECT signal: a probe's novelty should stay high (it is never
visited). If its novelty decays anyway — purely because nearby drivers are being
distilled — that is the count leaking across states. Pixel/linproj/random should
leak (probes collapse with the drivers); idm should not (probes stay novel).

INFLUENCE (cross-talk) MATRIX
-----------------------------
At chosen snapshots we also measure, per encoder, an N×N influence matrix: clone
the predictor, distill on state i ALONE for a few steps, record the fractional
novelty drop on every state j, then restore. infl[i,j] = 1 − nov_j_after/nov_j_before.
Diagonal-dominant = clean per-state counting (idm); dense off-diagonal = leak.

PROTOCOL
--------
  0. HARVEST: short uniform-random roam → n_drivers + n_probes distinct near-reset
     states; the most-visited become drivers, the rest probes. Probes are blocked
     from the start (their incoming (s,a) transitions are forbidden).
  1. PHASE FREE (--updates-free): random policy, probes blocked. Each update, BEFORE
     the distill, measure every monitored state's novelty (pairs with its cumulative
     visit count), the monitored-set geometry, and (at snapshots) the influence
     matrix; then roll out, count visits, train the online IDM φ, and distill each
     encoder's RND predictor on its features of the rollout's masked next-states.
  2. PHASE MASKED (--updates-masked, optional): ALSO block the drivers → everything
     abandoned (the organic-forget view, now across encoders).

Saved to results/encoder_leak_series_<tag>.npz for future analysis (see README).

CPU-only. Figures via plot.py; multi-seed aggregation via aggregate.py.

Run:
    uv run python -m JEPA.experiments.exp_014_figures_and_results.\
exp_014_7_encoder_leak_comparison.diagnose --updates-free 30
    uv run ... --n-drivers 3 --n-probes 2 --influence-every 5
    uv run ... --pixel-onehot --suffix _onehot
    uv run ... --seed 1 --suffix _seed1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import (
    CNNEncoder, one_hot_frame,
)
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi
from JEPA.experiments.exp_014_figures_and_results.exp_014_1_rnd_saturation.diagnose import (
    mask_board, state_key, FRAME, N_COLORS,
)
from JEPA.experiments.exp_014_figures_and_results.exp_014_6_organic_forget.diagnose import (
    choose_masked_actions,
)

DEVICE = torch.device("cpu")
ENCODER_NAMES = ["pixel", "linproj", "random", "idm"]


# ── encoders: masked board (B,64,64) uint8 → feature tensor (B, D) ─────────────

def _onehot_flat(masked: np.ndarray) -> np.ndarray:
    """(B,64,64) palette indices → (B, 64*64*16) float32 one-hot."""
    B = masked.shape[0]
    flat = masked.reshape(B, FRAME * FRAME).astype(np.int64)
    oh = np.zeros((B, FRAME * FRAME, N_COLORS), dtype=np.float32)
    r = np.arange(B)[:, None]; c = np.arange(FRAME * FRAME)[None, :]
    oh[r, c, flat] = 1.0
    return oh.reshape(B, -1)


class PixelEncoder:
    """Raw pixel space: normalised palette indices (D=4096, default) or flattened
    one-hot (D=65536, principled but heavy)."""

    def __init__(self, onehot: bool):
        self.onehot = onehot
        self.dim = FRAME * FRAME * (N_COLORS if onehot else 1)
        self.trainable = False

    def __call__(self, masked: np.ndarray) -> torch.Tensor:
        if self.onehot:
            return torch.from_numpy(_onehot_flat(masked))
        flat = masked.reshape(masked.shape[0], FRAME * FRAME).astype(np.float32)
        return torch.from_numpy(flat / (N_COLORS - 1))


class LinProjEncoder:
    """Frozen random LINEAR projection of the one-hot board (the exp_014 RND input)."""

    def __init__(self, seed: int, dim: int):
        self.dim = dim
        self.trainable = False
        g = torch.Generator().manual_seed(seed)
        in_dim = FRAME * FRAME * N_COLORS
        self.W = torch.randn(in_dim, dim, generator=g) / (in_dim ** 0.5)

    def __call__(self, masked: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(_onehot_flat(masked)) @ self.W


class FrozenRandomCNN:
    """A random-init CNNEncoder, frozen. A fixed nonlinear feature map (D=256)."""

    def __init__(self, seed: int, trunk_dim: int = 256):
        self.dim = trunk_dim
        self.trainable = False
        torch.manual_seed(seed)
        self.enc = CNNEncoder(n_colors=N_COLORS, frame_size=FRAME,
                              trunk_dim=trunk_dim).to(DEVICE)
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.eval()

    @torch.no_grad()
    def __call__(self, masked: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(masked.astype(np.int64))
        return self.enc(one_hot_frame(x, N_COLORS).to(DEVICE)).cpu()


class IDMEncoder:
    """OUR encoder: ICM inverse-dynamics φ, trained ONLINE (D=256)."""

    def __init__(self, seed: int, lr: float, beta: float, trunk_dim: int = 256):
        self.dim = trunk_dim
        self.trainable = True
        self.beta = beta
        torch.manual_seed(seed)
        self.icm = ICMModule(n_actions=4, n_colors=N_COLORS, frame_size=FRAME,
                             trunk_dim=trunk_dim).to(DEVICE)
        self.opt = torch.optim.Adam(self.icm.parameters(), lr=lr)

    @torch.no_grad()
    def __call__(self, masked: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(masked.astype(np.int64)).to(DEVICE)
        return self.icm.encode(x).cpu()

    def freeze(self) -> None:
        """Stop training φ — make it a stationary ruler for the probe."""
        for p in self.icm.parameters():
            p.requires_grad_(False)
        self.icm.eval()
        self.trainable = False

    def train_rollout(self, prev_masked, actions, next_masked, epochs, minibatches,
                      grad_clip, rng) -> dict:
        n = prev_masked.shape[0]
        if n == 0:
            return {"inv_acc": float("nan"), "fwd_err": float("nan")}
        o = torch.from_numpy(prev_masked.astype(np.int64)).to(DEVICE)
        no = torch.from_numpy(next_masked.astype(np.int64)).to(DEVICE)
        a = torch.from_numpy(actions.astype(np.int64)).to(DEVICE)
        mb = max(1, n // minibatches); idx = np.arange(n)
        ia = me = 0.0; steps = 0
        for _ in range(epochs):
            rng.shuffle(idx)
            for s in range(0, n, mb):
                sel = idx[s:s + mb]
                l_inv, l_fwd, acc, err = self.icm.losses_on_batch(o[sel], no[sel], a[sel])
                loss = (1.0 - self.beta) * l_inv + self.beta * l_fwd
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.icm.parameters(), grad_clip)
                self.opt.step()
                ia += acc; me += err; steps += 1
        return {"inv_acc": ia / steps, "fwd_err": me / steps}


def build_encoders(cfg) -> dict:
    return {
        "pixel":   PixelEncoder(onehot=cfg.pixel_onehot),
        "linproj": LinProjEncoder(seed=cfg.seed + 3, dim=cfg.linproj_dim),
        "random":  FrozenRandomCNN(seed=cfg.seed + 7),
        "idm":     IDMEncoder(seed=cfg.seed + 13, lr=cfg.icm_lr, beta=cfg.beta),
    }


def warmup_idm(envs, idm_encoder: IDMEncoder, rng, cfg) -> dict:
    """Pre-train the online IDM φ on `idm_warmup_episodes` episodes of random-policy
    data BEFORE the probe, so idm enters the measurement as a TRAINED encoder rather
    than a random one. Collects transitions until that many episodes terminate (with a
    safety cap), then trains inverse+forward for `idm_warmup_epochs` passes. Frozen
    encoders need no warm-up; only φ is touched here."""
    N = envs.n_envs
    envs.reset_all()
    cur_m = mask_board(envs.current_obs())
    prev_l, act_l, next_l = [], [], []
    episodes = 0; steps = 0
    step_cap = max(1, cfg.idm_warmup_episodes) * cfg.max_episode_steps * 4
    while episodes < cfg.idm_warmup_episodes and steps < step_cap:
        a = rng.integers(0, envs.n_actions, size=N)
        nobs, _r, dones, _i = envs.step(a); nm = mask_board(nobs)
        for i in range(N):
            if dones[i]:
                episodes += 1; continue          # skip the terminal→reset transition
            prev_l.append(cur_m[i]); act_l.append(int(a[i])); next_l.append(nm[i])
        cur_m = nm; steps += 1
    if not prev_l:
        return {"episodes": episodes, "transitions": 0, "inv_acc": float("nan"),
                "fwd_err": float("nan")}
    prev = np.stack(prev_l); acts = np.array(act_l, dtype=np.int64); nxt = np.stack(next_l)
    stats = idm_encoder.train_rollout(prev, acts, nxt, cfg.idm_warmup_epochs,
                                      cfg.minibatches, cfg.grad_clip, rng)
    stats.update({"episodes": episodes, "transitions": len(prev_l)})
    return stats


# ── geometry / influence ──────────────────────────────────────────────────────

def geometry(feats: torch.Tensor) -> dict:
    """Pairwise geometry of the monitored states under one encoder. Cross-encoder
    comparison needs scale-free metrics, so alongside raw L2/cosine we report the
    centered cosine and the unit-norm (chord) L2 (∈[0,2]). Means over off-diagonal
    pairs; raw 5×5 matrices returned too."""
    X = feats.cpu().numpy().astype(np.float64)
    n = X.shape[0]; off = ~np.eye(n, dtype=bool)
    l2 = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = Xn @ Xn.T
    l2u = np.sqrt(np.clip(2.0 - 2.0 * cos, 0.0, None))
    Xc = X - X.mean(axis=0, keepdims=True)
    Xcn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    cos_c = Xcn @ Xcn.T
    return {"mean_l2": float(l2[off].mean()), "mean_cos": float(cos[off].mean()),
            "mean_l2_unit": float(l2u[off].mean()), "mean_cos_centered": float(cos_c[off].mean()),
            "l2_mat": l2, "cos_mat": cos}


def influence_matrix(rnd: RNDPhi, mon_feats: torch.Tensor, k: int, lr: float):
    """N×N generalization-leak / cross-talk matrix — a property of the ENCODER's
    representation, not of how far the online predictor has trained. For each source
    i: RESET the predictor to its random init, fit ONLY state i for k steps, and
    record the fractional novelty drop on every state j relative to the init novelty:
        infl[i,j] = 1 − nov_j(fit only i) / nov_j(init).
    Diagonal ≈ 1 (fitting i kills i's novelty); off-diagonal = leak (fitting i also
    suppresses j because the encoder couldn't separate them). Non-destructive: the
    live predictor state is snapshotted and restored. Returns (infl (N,N), base (N,)).
    Mirrors the resolution test in exp_016 frozen_encoder_resolution.py."""
    N = mon_feats.shape[0]
    infl = np.zeros((N, N))
    live = [p.detach().clone() for p in rnd.predictor.parameters()]   # restore at end

    def _to_init():
        with torch.no_grad():
            for p, p0 in zip(rnd.predictor.parameters(), rnd._pred_init):
                p.copy_(p0.to(p.device))

    _to_init()
    with torch.no_grad():
        base = rnd.novelty(mon_feats).cpu().numpy()                    # init novelty
    for i in range(N):
        _to_init()
        topt = torch.optim.Adam(rnd.predictor.parameters(), lr=lr)
        fi = mon_feats[i:i + 1]
        for _ in range(k):
            topt.zero_grad(); rnd.distill_loss(fi).backward(); topt.step()
        with torch.no_grad():
            after = rnd.novelty(mon_feats).cpu().numpy()
        infl[i] = 1.0 - after / (base + 1e-12)
    with torch.no_grad():
        for p, p0 in zip(rnd.predictor.parameters(), live):
            p.copy_(p0)
    return infl, base


def distill_predictor(rnd, opt, feats, idx_shuffle, epochs, minibatches,
                      grad_clip) -> None:
    """Distill one RND predictor on PRECOMPUTED features. The encoder is frozen within
    an update (φ is trained once, before this), so its features are constant across the
    inner steps → we encode ONCE per update (in the caller) and reuse here, instead of
    re-encoding every minibatch. ~minibatches×epochs fewer encodes — the difference
    between seconds and minutes per update for the CNN encoders. Leak after each step."""
    if feats.shape[0] == 0:
        return
    for _ep in range(epochs):
        for batch_idx in np.array_split(idx_shuffle, minibatches):
            if len(batch_idx) == 0:
                rnd.apply_leak(); continue
            opt.zero_grad()
            rnd.distill_loss(feats[batch_idx]).backward()
            torch.nn.utils.clip_grad_norm_(rnd.predictor.parameters(), grad_clip)
            opt.step(); rnd.apply_leak()


def _encode_chunked(encoder, masked, chunk):
    if masked.shape[0] <= chunk:
        return encoder(masked).to(DEVICE)
    return torch.cat([encoder(masked[s:s + chunk]).to(DEVICE)
                      for s in range(0, masked.shape[0], chunk)], dim=0)


# ── harvest: drivers + probes ─────────────────────────────────────────────────

def harvest_driver_probe(envs, rng, n_drivers, n_probes, max_dist, pre_roam):
    """Short uniform-random roam → the n_drivers+n_probes most-visited distinct
    near-reset states. The most-visited are DRIVERS; the rest PROBES. Returns the
    monitored boards/keys (drivers first, then probes), the is_probe mask, the set
    of (s,a) transitions that land on a PROBE (blocked from the start), and stats."""
    N = envs.n_envs
    visit: dict[bytes, int] = defaultdict(int)
    first_step: dict[bytes, int] = {}
    exemplar: dict[bytes, np.ndarray] = {}
    trans: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))

    obs = envs.current_obs(); om = mask_board(obs)
    prev_keys = [state_key(om[i]) for i in range(N)]
    ep_step = np.zeros(N, dtype=np.int64)
    for i in range(N):
        first_step.setdefault(prev_keys[i], 0); exemplar.setdefault(prev_keys[i], om[i])
    for _ in range(pre_roam):
        a = rng.integers(0, envs.n_actions, size=N)
        nobs, _r, dones, _i = envs.step(a); nm = mask_board(nobs)
        for i in range(N):
            if dones[i]:
                ep_step[i] = 0; prev_keys[i] = state_key(nm[i])
                first_step.setdefault(prev_keys[i], 0); exemplar.setdefault(prev_keys[i], nm[i])
                continue
            ep_step[i] += 1; nk = state_key(nm[i])
            visit[nk] += 1; exemplar.setdefault(nk, nm[i])
            fs = first_step.get(nk)
            if fs is None or ep_step[i] < fs:
                first_step[nk] = int(ep_step[i])
            trans[(prev_keys[i], int(a[i]))][nk] += 1
            prev_keys[i] = nk

    n_total = n_drivers + n_probes
    cand = [(k, visit[k]) for k, fs in first_step.items()
            if 1 <= fs <= max_dist and visit[k] > 0]
    cand.sort(key=lambda kv: kv[1], reverse=True)
    chosen = [k for k, _ in cand[:n_total]]
    if len(chosen) < n_total:                       # relax distance cap if too few
        # never fall back to a RESET state (first_step==0): it is re-entered every
        # episode and cannot be action-blocked, so it makes a useless probe.
        extra = [k for k, _ in sorted(visit.items(), key=lambda kv: kv[1], reverse=True)
                 if k not in chosen and first_step.get(k, 0) != 0]
        chosen += extra[: n_total - len(chosen)]
    chosen = chosen[:n_total]

    driver_keys = chosen[:n_drivers]
    probe_keys = chosen[n_drivers:n_total]
    probe_set = set(probe_keys)
    blocked_probe = {sa for sa, nexts in trans.items()
                     if any(nk in probe_set for nk in nexts)}

    monitored_keys = driver_keys + probe_keys
    monitored_masked = np.stack([exemplar[k] for k in monitored_keys])
    is_probe = np.array([0] * len(driver_keys) + [1] * len(probe_keys), dtype=np.int64)
    stats = {"first_step": [int(first_step.get(k, -1)) for k in monitored_keys],
             "harvest_visits": [int(visit[k]) for k in monitored_keys],
             "n_blocked_probe_sa": len(blocked_probe), "n_states_seen": len(visit)}
    return monitored_masked, monitored_keys, is_probe, blocked_probe, stats


def collect_rollout(envs, rng, monitored_keys, monitored_set, probe_set, block_keys,
                    blocked_sa, cfg):
    """One rollout under the always-masked policy (masks against blocked_sa, which
    always blocks probes and — in the masked phase — drivers too). Counts visits to
    every monitored state; learns blocks online for any state in block_keys that is
    entered anyway. Returns prev/actions/next (valid transitions), a per-transition
    `next_is_probe` mask (so the RND distill can exclude probe states → their novelty
    moves only via LEAK), the per-state visits, and the mask-failure count."""
    N = envs.n_envs
    mon_index = {k: j for j, k in enumerate(monitored_keys)}
    visits = np.zeros(len(monitored_keys), dtype=np.int64); fails = 0
    prev_l, act_l, next_l, nprobe_l = [], [], [], []
    obs = envs.current_obs(); cur_m = mask_board(obs)
    cur_keys = [state_key(cur_m[i]) for i in range(N)]
    for _t in range(cfg.rollout_steps):
        a = choose_masked_actions(cur_keys, blocked_sa, envs.n_actions, rng)
        nobs, _r, dones, _i = envs.step(a); nm = mask_board(nobs)
        new_keys = cur_keys[:]
        for i in range(N):
            if dones[i]:
                new_keys[i] = state_key(nm[i]); continue
            nk = state_key(nm[i])
            if nk in monitored_set:
                visits[mon_index[nk]] += 1
                if nk in block_keys:
                    blocked_sa.add((cur_keys[i], int(a[i]))); fails += 1
            prev_l.append(cur_m[i]); act_l.append(int(a[i])); next_l.append(nm[i])
            nprobe_l.append(nk in probe_set)
            new_keys[i] = nk
        cur_m = nm; cur_keys = new_keys
    if prev_l:
        return (np.stack(prev_l), np.array(act_l, dtype=np.int64), np.stack(next_l),
                np.array(nprobe_l, dtype=bool), visits, fails)
    z = np.zeros((0, FRAME, FRAME), dtype=np.uint8)
    return z, np.zeros(0, dtype=np.int64), z, np.zeros(0, dtype=bool), visits, fails


# ── main ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20")
    p.add_argument("--level", type=int, default=1, help="0-indexed (1 = LS20 L2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-drivers", type=int, default=3, help="visited (distilled) states")
    p.add_argument("--n-probes", type=int, default=2, help="held-out (blocked) states "
                   "whose novelty decay = leak")
    p.add_argument("--monitor-max-dist", type=int, default=6)
    p.add_argument("--updates-free", type=int, default=30)
    p.add_argument("--updates-masked", type=int, default=0,
                   help=">0 also abandons the drivers in a second phase")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--rnd-epochs", type=int, default=4, help="RND distill epochs/update")
    p.add_argument("--rnd-lr", type=float, default=1e-3)
    p.add_argument("--mu", type=float, default=0.0, help="leak for ALL predictors")
    p.add_argument("--icm-lr", type=float, default=1e-3)
    p.add_argument("--icm-epochs", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--idm-warmup-episodes", type=int, default=20,
                   help="pre-train φ on this many random-policy episodes BEFORE the "
                        "probe so idm starts trained, not random (0 = no warm-up)")
    p.add_argument("--idm-warmup-epochs", type=int, default=30,
                   help="training passes over the warm-up buffer")
    p.add_argument("--idm-online", action="store_true",
                   help="keep training φ online during the probe (non-stationary). "
                        "Default after warm-up is to FREEZE φ (stationary ruler, clean "
                        "leak attribution, fair vs the frozen encoders).")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--linproj-dim", type=int, default=256)
    p.add_argument("--pixel-onehot", action="store_true",
                   help="feed pixel-RND the one-hot board (D=65536, slow)")
    p.add_argument("--influence-every", type=int, default=0,
                   help="compute the N×N influence matrix every N updates (0 = only "
                        "at the final update; influence is the heaviest extra)")
    p.add_argument("--influence-steps", type=int, default=20, help="distill steps/source")
    p.add_argument("--influence-lr", type=float, default=None, help="defaults to rnd-lr")
    p.add_argument("--pre-roam", type=int, default=800)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--suffix", default="")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    cfg = _parse()
    if cfg.influence_lr is None:
        cfg.influence_lr = cfg.rnd_lr
    HERE = Path(__file__).resolve().parent
    RES_DIR = HERE / "results"; RES_DIR.mkdir(parents=True, exist_ok=True)
    TAG = f"{cfg.game}_L{cfg.level + 1}{cfg.suffix}"
    n_enc = len(ENCODER_NAMES)

    rng = np.random.default_rng(cfg.seed)
    envs = VecLS20EnvLevel(env_name=cfg.game, n_envs=cfg.n_envs,
                           max_episode_steps=cfg.max_episode_steps,
                           seed=cfg.seed, level_index=cfg.level)

    mon_masked, monitored_keys, is_probe, blocked_probe, hstats = harvest_driver_probe(
        envs, rng, cfg.n_drivers, cfg.n_probes, cfg.monitor_max_dist, cfg.pre_roam)
    n_mon = len(monitored_keys)
    monitored_set = set(monitored_keys)
    probe_keys = {k for k, p in zip(monitored_keys, is_probe) if p}
    driver_keys = {k for k, p in zip(monitored_keys, is_probe) if not p}
    print(f"[exp_014_7/{TAG}] {cfg.n_drivers} drivers + {cfg.n_probes} probes  "
          f"is_probe={is_probe.tolist()}  first_step={hstats['first_step']}  "
          f"harvest_visits={hstats['harvest_visits']}  blocked_probe(s,a)="
          f"{hstats['n_blocked_probe_sa']}  μ={cfg.mu}  "
          f"pixel={'onehot' if cfg.pixel_onehot else 'index'}", flush=True)

    encoders = build_encoders(cfg)

    # ── pre-train the online IDM φ on random-policy data BEFORE the probe ─────────
    warm = {"episodes": 0, "transitions": 0, "inv_acc": float("nan"), "fwd_err": float("nan")}
    if cfg.idm_warmup_episodes > 0:
        warm = warmup_idm(envs, encoders["idm"], np.random.default_rng(cfg.seed + 5), cfg)
        if not cfg.idm_online:
            encoders["idm"].freeze()
    idm_frozen = (cfg.idm_warmup_episodes > 0) and (not cfg.idm_online)
    print(f"[exp_014_7/{TAG}] IDM warm-up: {warm['episodes']} episodes / "
          f"{warm['transitions']} transitions → inv_acc={warm['inv_acc']:.3f} "
          f"fwd_err={warm['fwd_err']:.3f}  φ during probe={'FROZEN' if idm_frozen else 'online'}",
          flush=True)

    rnds, opts = {}, {}
    for name in ENCODER_NAMES:
        torch.manual_seed(cfg.seed + 100)
        r = RNDPhi(dim=encoders[name].dim, hidden=256, out=256, leak=cfg.mu).to(DEVICE)
        rnds[name] = r
        opts[name] = torch.optim.Adam(r.predictor.parameters(), lr=cfg.rnd_lr)

    total = cfg.updates_free + cfg.updates_masked
    nov = np.zeros((n_enc, n_mon, total + 1))
    mean_l2 = np.zeros((n_enc, total + 1)); mean_cos = np.zeros((n_enc, total + 1))
    mean_l2_unit = np.zeros((n_enc, total + 1)); mean_cos_centered = np.zeros((n_enc, total + 1))
    pair_l2 = np.zeros((n_enc, total + 1, n_mon, n_mon))
    pair_cos = np.zeros((n_enc, total + 1, n_mon, n_mon))
    cum_visits = np.zeros((n_mon, total + 1), dtype=np.int64)
    visits_per_update = np.zeros((total, n_mon), dtype=np.int64)
    fails_per_update = np.zeros(total, dtype=np.int64)
    phase = np.zeros(total, dtype=np.int64)
    idm_inv_acc = np.full(total, np.nan); idm_fwd_err = np.full(total, np.nan)
    infl_list = []          # (n_enc, n_mon, n_mon) per snapshot
    infl_updates = []

    def measure(u_slot, do_influence):
        infl_snap = np.zeros((n_enc, n_mon, n_mon)) if do_influence else None
        for e, name in enumerate(ENCODER_NAMES):
            feats = encoders[name](mon_masked).to(DEVICE)
            with torch.no_grad():
                nov[e, :, u_slot] = rnds[name].novelty(feats).cpu().numpy()
            g = geometry(feats)
            mean_l2[e, u_slot] = g["mean_l2"]; mean_cos[e, u_slot] = g["mean_cos"]
            mean_l2_unit[e, u_slot] = g["mean_l2_unit"]
            mean_cos_centered[e, u_slot] = g["mean_cos_centered"]
            pair_l2[e, u_slot] = g["l2_mat"]; pair_cos[e, u_slot] = g["cos_mat"]
            if do_influence:
                infl_snap[e], _ = influence_matrix(
                    rnds[name], feats, cfg.influence_steps, cfg.influence_lr)
        if do_influence:
            infl_list.append(infl_snap); infl_updates.append(u_slot)

    measure(0, do_influence=(cfg.influence_every > 0))
    envs.reset_all()
    loop_rng = np.random.default_rng(cfg.seed + 1)
    blocked_sa = set(blocked_probe)
    run_visits = np.zeros(n_mon, dtype=np.int64)

    for u in range(total):
        cum_visits[:, u] = run_visits
        masking = u >= cfg.updates_free
        phase[u] = int(masking)
        block_keys = set(probe_keys) | (set(driver_keys) if masking else set())
        prev, acts, nxt, next_is_probe, visits, fails = collect_rollout(
            envs, loop_rng, monitored_keys, monitored_set, set(probe_keys),
            block_keys, blocked_sa, cfg)
        visits_per_update[u] = visits; fails_per_update[u] = fails
        run_visits = run_visits + visits

        # IDM φ trains on ALL transitions (it is the encoder, not the count) UNLESS it
        # was frozen after warm-up — then it is a stationary ruler for the probe.
        if not idm_frozen:
            idm_stats = encoders["idm"].train_rollout(
                prev, acts, nxt, cfg.icm_epochs, cfg.minibatches, cfg.grad_clip, loop_rng)
            idm_inv_acc[u] = idm_stats["inv_acc"]; idm_fwd_err[u] = idm_stats["fwd_err"]
        else:
            idm_inv_acc[u] = warm["inv_acc"]; idm_fwd_err[u] = warm["fwd_err"]

        # RND predictors are distilled on NON-probe next-states only, so a probe's
        # novelty can move only via leak from the (driver) states being counted.
        # Encode each encoder's features ONCE per update (frozen within the update),
        # then reuse across the inner distill steps.
        nxt_rnd = nxt[~next_is_probe]
        idx_shuffle = np.arange(nxt_rnd.shape[0]); loop_rng.shuffle(idx_shuffle)
        for name in ENCODER_NAMES:
            feats = (_encode_chunked(encoders[name], nxt_rnd, 256) if nxt_rnd.shape[0]
                     else torch.zeros((0, encoders[name].dim), device=DEVICE))
            distill_predictor(rnds[name], opts[name], feats, idx_shuffle,
                              cfg.rnd_epochs, cfg.minibatches, cfg.grad_clip)

        last = (u == total - 1)
        do_infl = (cfg.influence_every > 0 and (u + 1) % cfg.influence_every == 0) or last
        measure(u + 1, do_influence=do_infl)
        ph = "MASK" if masking else "free"
        dn = nov[:, is_probe == 0, u + 1].mean(); pn = nov[:, is_probe == 1, u + 1].mean()
        print(f"  update {u + 1:>2}/{total} [{ph}]  drv_visits="
              f"{int(visits[is_probe == 0].sum())}  probe_visits={int(visits[is_probe == 1].sum())}"
              f"  inv_acc={idm_inv_acc[u]:.2f}  driver_nov={dn:.2e}  probe_nov={pn:.2e}",
              flush=True)
    cum_visits[:, total] = run_visits

    influence = (np.stack(infl_list, axis=1) if infl_list
                 else np.zeros((n_enc, 0, n_mon, n_mon)))   # (n_enc, n_snap, N, N)

    npz_path = RES_DIR / f"encoder_leak_series_{TAG}.npz"
    np.savez_compressed(
        npz_path,
        nov=nov, cum_visits=cum_visits, visits_per_update=visits_per_update,
        is_probe=is_probe, fails_per_update=fails_per_update, phase=phase,
        mean_l2=mean_l2, mean_cos=mean_cos,
        mean_l2_unit=mean_l2_unit, mean_cos_centered=mean_cos_centered,
        pair_l2=pair_l2, pair_cos=pair_cos,
        influence=influence, influence_updates=np.array(infl_updates, dtype=np.int64),
        idm_inv_acc=idm_inv_acc, idm_fwd_err=idm_fwd_err,
        mon_masked=mon_masked, encoder_names=np.array(ENCODER_NAMES),
        encoder_dims=np.array([encoders[n].dim for n in ENCODER_NAMES]),
        updates_free=cfg.updates_free, updates_masked=cfg.updates_masked,
        rollout_env_steps=cfg.rollout_steps * cfg.n_envs,
        first_step=np.array(hstats["first_step"], dtype=np.int64),
        mu=cfg.mu, pixel_onehot=int(cfg.pixel_onehot), seed=cfg.seed,
        idm_warmup_episodes=cfg.idm_warmup_episodes, idm_frozen=int(idm_frozen),
        idm_warmup_inv_acc=float(warm["inv_acc"]), idm_warmup_transitions=warm["transitions"],
    )
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "game": cfg.game, "level_index": cfg.level, "level_tag": TAG,
        "encoder_names": ENCODER_NAMES,
        "encoder_dims": {n: encoders[n].dim for n in ENCODER_NAMES},
        "n_drivers": cfg.n_drivers, "n_probes": cfg.n_probes, "is_probe": is_probe.tolist(),
        "monitor_first_step": hstats["first_step"], "harvest_visits": hstats["harvest_visits"],
        "updates_free": cfg.updates_free, "updates_masked": cfg.updates_masked,
        "rollout_env_steps_per_update": cfg.rollout_steps * cfg.n_envs,
        "minibatches": cfg.minibatches, "rnd_epochs": cfg.rnd_epochs, "rnd_lr": cfg.rnd_lr,
        "mu": cfg.mu, "icm_lr": cfg.icm_lr, "icm_epochs": cfg.icm_epochs, "beta": cfg.beta,
        "linproj_dim": cfg.linproj_dim, "pixel_onehot": cfg.pixel_onehot,
        "influence_every": cfg.influence_every, "influence_steps": cfg.influence_steps,
        "idm_warmup_episodes": cfg.idm_warmup_episodes, "idm_warmup_epochs": cfg.idm_warmup_epochs,
        "idm_frozen_during_probe": idm_frozen, "idm_warmup_inv_acc": float(warm["inv_acc"]),
        "idm_warmup_transitions": warm["transitions"],
        "seed": cfg.seed, "final_cum_visits": run_visits.tolist(),
        "final_mean_l2_unit": {n: float(mean_l2_unit[e, -1]) for e, n in enumerate(ENCODER_NAMES)},
        "final_mean_cos": {n: float(mean_cos[e, -1]) for e, n in enumerate(ENCODER_NAMES)},
        "final_novelty": {n: float(nov[e, :, -1].mean()) for e, n in enumerate(ENCODER_NAMES)},
        "npz": str(npz_path),
    }
    if influence.shape[1] > 0:
        off = ~np.eye(n_mon, dtype=bool)
        d_idx = np.where(is_probe == 0)[0]; p_idx = np.where(is_probe == 1)[0]
        summary["final_leak_offdiag_mean"] = {
            n: float(influence[e, -1][off].mean()) for e, n in enumerate(ENCODER_NAMES)}
        summary["final_leak_diag_mean"] = {
            n: float(np.diag(influence[e, -1]).mean()) for e, n in enumerate(ENCODER_NAMES)}
        # the cleanest leak number: distilling a DRIVER, how much a never-distilled
        # PROBE drops (0 = no leak, →1 = the probe is fully counted by proxy).
        if len(d_idx) and len(p_idx):
            summary["final_leak_driver_to_probe"] = {
                n: float(influence[e, -1][np.ix_(d_idx, p_idx)].mean())
                for e, n in enumerate(ENCODER_NAMES)}
    json_path = RES_DIR / f"encoder_leak_summary_{TAG}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[exp_014_7] wrote series  -> {npz_path}")
    print(f"[exp_014_7] wrote summary -> {json_path}")
    for e, n in enumerate(ENCODER_NAMES):
        msg = (f"[exp_014_7] {n:8s}: l2u={mean_l2_unit[e,-1]:.3f} cos={mean_cos[e,-1]:+.3f} "
               f"probe_nov={nov[e, is_probe==1, -1].mean():.2e} "
               f"driver_nov={nov[e, is_probe==0, -1].mean():.2e}")
        if influence.shape[1] > 0:
            d_idx = np.where(is_probe == 0)[0]; p_idx = np.where(is_probe == 1)[0]
            if len(d_idx) and len(p_idx):
                msg += f"  leak(drv→probe)={influence[e,-1][np.ix_(d_idx,p_idx)].mean():+.3f}"
        print(msg)

    if not cfg.no_plot:
        from JEPA.experiments.exp_014_figures_and_results.exp_014_7_encoder_leak_comparison.plot import (
            make_figures,
        )
        make_figures(npz_path, HERE / "figures", TAG)


if __name__ == "__main__":
    main()
