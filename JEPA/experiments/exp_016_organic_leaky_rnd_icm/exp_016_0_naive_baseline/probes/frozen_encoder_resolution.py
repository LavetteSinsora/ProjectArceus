"""exp_016_0 probe — does a FROZEN-RANDOM encoder separate LS20-L1 states?

The worry (the reason we'd even consider an IDM-trained encoder): LS20 frames are
near-identical pixel-wise, so a frozen random encoder might map distinct board
states to near-identical representations — giving RND no resolution to tell states
apart (1 visit ≈ N visits, the "no count resolution" failure from the lineage).

This probe MEASURES it directly:
  1. Enumerate the distinct board states reachable on LS20 Level 1 by a long
     uniform-random roam. State identity = the UI/timer-masked 64x64 color board
     (rows 60-63 zeroed — the marching step-timer makes every raw frame unique).
  2. Push every distinct state through a frozen-random CNNEncoder (the project's
     real exp_010 encoder, random init).
  3. Report the AVERAGE PAIRWISE COSINE SIMILARITY of the representations, plus
     the controls needed to read that number:
       - input pixel cosine (how similar the states are to begin with),
       - centered cosine (removes the ReLU positive-orthant bias),
       - a "max-distinct" control: random-noise frames through the SAME encoder
         (the separation ceiling — what genuinely-different inputs look like),
       - the frozen random LINEAR projection (exp_014's RND input) for reference.

Higher cosine = states look MORE alike in rep space = WORSE resolution for RND.

CPU-only. Run from repo root:
    uv run python -m JEPA.experiments.exp_016_organic_leaky_rnd_icm.\
exp_016_0_naive_baseline.probes.frozen_encoder_resolution
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel
from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import CNNEncoder, one_hot_frame
from JEPA.experiments.exp_013_headline_experiment.exp_013_1b_leaky_rnd_on_icm_phi.rnd_phi import RNDPhi

FRAME = 64
N_COLORS = 16
TIMER_ROW0 = 60          # rows 60-63 = step-timer/energy UI (mask for true identity)
DEVICE = torch.device("cpu")


# ── state identity ────────────────────────────────────────────────────────────

def mask_board(frame: np.ndarray) -> np.ndarray:
    m = frame.copy()
    m[..., TIMER_ROW0:, :] = 0
    return m


def state_key(masked: np.ndarray) -> bytes:
    return masked.tobytes()


def harvest(game: str, level: int, seed: int, roam_steps: int, n_envs: int):
    """Long uniform-random roam → distinct states. Returns (masked_states (S,64,64),
    raw_states (S,64,64), n_raw_distinct)."""
    envs = VecLS20EnvLevel(env_name=game, n_envs=n_envs, max_episode_steps=200,
                           seed=seed, level_index=level)
    rng = np.random.default_rng(seed)
    masked_ex: dict[bytes, np.ndarray] = {}
    raw_ex: dict[bytes, np.ndarray] = {}
    raw_keys: set[bytes] = set()
    for _ in range(roam_steps):
        a = rng.integers(0, envs.n_actions, size=envs.n_envs)
        nobs, _r, dones, _i = envs.step(a)
        m = mask_board(nobs)
        for i in range(envs.n_envs):
            if dones[i]:
                continue
            k = state_key(m[i])
            masked_ex.setdefault(k, m[i].copy())
            raw_ex.setdefault(k, nobs[i].copy())
            raw_keys.add(nobs[i].tobytes())
    masked = np.stack(list(masked_ex.values()))
    raw = np.stack(list(raw_ex.values()))
    return masked, raw, len(raw_keys)


# ── similarity ───────────────────────────────────────────────────────────────

def pairwise_cosine_stats(X: np.ndarray) -> dict:
    """X: (S, D). Mean/std/min/max/percentiles of off-diagonal pairwise cosine."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    C = Xn @ Xn.T
    S = C.shape[0]
    off = C[~np.eye(S, dtype=bool)]
    return {
        "mean": float(off.mean()), "std": float(off.std()),
        "min": float(off.min()), "p50": float(np.percentile(off, 50)),
        "max": float(off.max()),
    }


def encode_cnn(states_uint8: np.ndarray, seed: int) -> np.ndarray:
    """Frozen-random CNNEncoder → (S, trunk_dim). states_uint8: (S,64,64)."""
    torch.manual_seed(seed)
    enc = CNNEncoder(n_colors=N_COLORS, frame_size=FRAME, trunk_dim=256).to(DEVICE)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    with torch.no_grad():
        x = torch.from_numpy(states_uint8.astype(np.int64))
        feats = enc(one_hot_frame(x, N_COLORS).to(DEVICE))
    return feats.cpu().numpy()


def input_pixel_features(masked: np.ndarray) -> np.ndarray:
    """Flatten the one-hot board to a raw input vector (S, 64*64*16)."""
    S = masked.shape[0]
    flat = masked.reshape(S, FRAME * FRAME).astype(np.int64)
    oh = np.zeros((S, FRAME * FRAME, N_COLORS), dtype=np.float32)
    r = np.arange(S)[:, None]; c = np.arange(FRAME * FRAME)[None, :]
    oh[r, c, flat] = 1.0
    return oh.reshape(S, -1)


def linear_projection_features(masked: np.ndarray, seed: int, dim: int = 256) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    in_dim = FRAME * FRAME * N_COLORS
    W = (torch.randn(in_dim, dim, generator=g) / (in_dim ** 0.5)).numpy()
    return input_pixel_features(masked) @ W


def near_dup_fraction(X: np.ndarray, thresh: float) -> float:
    """Fraction of off-diagonal pairs with raw cosine > thresh."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    C = Xn @ Xn.T
    S = C.shape[0]
    off = C[~np.eye(S, dtype=bool)]
    return float((off > thresh).mean())


def rnd_leak_test(feats: np.ndarray, seed: int, fit_frac: float = 0.5,
                  steps: int = 600, lr: float = 1e-3) -> dict:
    """THE resolution test (mirrors the lineage's '99.9% leak').

    Split states into a FIT half and a HELD-OUT half. Train a standard RND
    predictor (leak=0) to memorise the FIT half, then measure how much the
    held-out states' novelty DROPPED purely from training on the fit half.
      leak% = 100·(1 − nov_heldout_after / nov_heldout_before).
    leak ≈ 0%  → held-out stays novel → encoder gives RND real resolution.
    leak ≈100% → fitting some states kills novelty everywhere → no resolution.
    """
    D = feats.shape[1]
    X = torch.from_numpy(feats.astype(np.float32))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_fit = max(1, int(len(idx) * fit_frac))
    fit, held = idx[:n_fit], idx[n_fit:]
    Xf, Xh = X[fit], X[held]

    torch.manual_seed(seed)
    rnd = RNDPhi(dim=D, hidden=256, out=256, leak=0.0).to(DEVICE)
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=lr)

    nov_fit0 = float(rnd.novelty(Xf).mean())
    nov_held0 = float(rnd.novelty(Xh).mean())
    for _ in range(steps):
        opt.zero_grad()
        rnd.distill_loss(Xf).backward()
        opt.step()
    nov_fit1 = float(rnd.novelty(Xf).mean())
    nov_held1 = float(rnd.novelty(Xh).mean())
    leak = 100.0 * (1.0 - nov_held1 / max(nov_held0, 1e-12))
    fitdrop = 100.0 * (1.0 - nov_fit1 / max(nov_fit0, 1e-12))
    return {"leak_pct": leak, "fit_drop_pct": fitdrop,
            "held_before": nov_held0, "held_after": nov_held1,
            "fit_after": nov_fit1}


def _fmt(name: str, s: dict) -> str:
    return (f"  {name:<34} mean={s['mean']:+.4f}  std={s['std']:.4f}  "
            f"[min {s['min']:+.3f} | p50 {s['p50']:+.3f} | max {s['max']:+.3f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20")
    p.add_argument("--level", type=int, default=0, help="0-indexed (0 = Level 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roam-steps", type=int, default=4000, help="per-env roam steps")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--enc-seeds", type=int, default=5, help="frozen encoders to average")
    p.add_argument("--leak-lr", type=float, default=1e-4, help="predictor lr in leak test")
    p.add_argument("--leak-steps", type=int, default=300, help="distill steps in leak test")
    args = p.parse_args()

    print(f"[probe] harvesting {args.game} L{args.level+1} "
          f"({args.n_envs} envs × {args.roam_steps} steps)…", flush=True)
    masked, raw, n_raw = harvest(args.game, args.level, args.seed,
                                 args.roam_steps, args.n_envs)
    S = masked.shape[0]
    print(f"[probe] distinct MASKED board states: {S}   "
          f"distinct RAW frames (timer-confounded): {n_raw}\n")

    pix = input_pixel_features(masked)
    lin = linear_projection_features(masked, args.seed)
    cnn = encode_cnn(masked, seed=args.seed * 1000)
    rng = np.random.default_rng(args.seed + 99)
    noise = rng.integers(0, N_COLORS, size=(S, FRAME, FRAME)).astype(np.uint8)
    cnn_noise = encode_cnn(noise, seed=args.seed * 1000)

    # ── 1) average pairwise cosine (what you asked for) ──
    print("AVERAGE PAIRWISE COSINE SIMILARITY (raw; higher = more alike):")
    print(_fmt("one-hot pixels (the states)", pairwise_cosine_stats(pix)))
    print(_fmt("frozen random LINEAR proj", pairwise_cosine_stats(lin)))
    print(_fmt("frozen random CNN encoder", pairwise_cosine_stats(cnn)))
    print(_fmt("CONTROL: random-noise → CNN", pairwise_cosine_stats(cnn_noise)))
    print("  (raw cosine is positively biased by shared background + ReLU; the leak"
          "\n   test below is the resolution metric that actually matters for RND.)")

    # ── 2) near-duplicate pairs the encoder cannot separate ──
    print("\nNEAR-DUPLICATE PAIR FRACTION (raw cosine above threshold):")
    for name, X in [("pixels", pix), ("linear", lin), ("CNN", cnn),
                    ("noise→CNN", cnn_noise)]:
        print(f"  {name:<12} >0.99: {near_dup_fraction(X,0.99):6.1%}   "
              f">0.999: {near_dup_fraction(X,0.999):6.1%}")

    # ── 3) RND generalization-leak (THE resolution metric) ──
    print(f"\nRND GENERALIZATION-LEAK  (fit half the {S} states, "
          f"measure novelty drop on the unseen half):")
    print("  encoder         leak%   fit_drop%   held(before→after)")
    for name, X in [("pixels", pix), ("linear-proj", lin), ("CNN-random", cnn),
                    ("noise→CNN", cnn_noise)]:
        r = rnd_leak_test(X, seed=args.seed, steps=args.leak_steps, lr=args.leak_lr)
        print(f"  {name:<14} {r['leak_pct']:5.1f}    {r['fit_drop_pct']:5.1f}     "
              f"{r['held_before']:.3e} → {r['held_after']:.3e}")
    print("  leak≈0%  → unseen states stay novel → RND has real resolution here.")
    print("  leak→100% → fitting some states kills novelty everywhere → no resolution.")


if __name__ == "__main__":
    main()
