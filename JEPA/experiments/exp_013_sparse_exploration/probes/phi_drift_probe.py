"""READ-ONLY probe: how fast does the ICM encoder phi stabilize during training?

Motivation
----------
RND / pseudo-counts need a STATIONARY embedding to count against. ICM's phi is
trained by inverse dynamics, so it MOVES during training: the same state s maps
to a different phi(s) over time, which smears any count computed in phi-space.
We measure that motion empirically on the exp_011_0 ICM checkpoints (LS20 L1).

Method
------
1. Build ONE fixed probe set of distinct LS20 L1 frames (states visited by a
   uniform-random policy, deduped by raw frame bytes). Held fixed across all
   checkpoints and all seeds.
2. For each checkpoint (per seed, ordered by training step) load `icm.phi`,
   encode the fixed probe set -> Phi_t  (N, D).
3. Drift on the SAME states between consecutive checkpoints:
      - cosine: mean cos(phi_t(s), phi_{t+1}(s))   (raw features)
      - L2    : mean || phi_t(s)/||.|| - phi_{t+1}(s)/||.|| ||   (unit-normed)
4. Resolution: at each t, the typical INTER-STATE distance = mean pairwise L2
   among distinct states (on unit-normed features). The key diagnostic ratio is
      drift_L2(t->t+1) / inter_state_dist(t+1)
   When << 1, phi has stopped moving relative to the gap it must keep between
   states -> RND-on-phi is trustworthy.

This is READ-ONLY: it loads checkpoints and rolls out a random policy to gather
probe frames. It does not modify environment_files/, the shared env wrappers, or
any exp_011 files.

Caveat: checkpoints start at step 102400, so the first ~100k env steps of phi
drift (likely the largest) are NOT captured. Conclusions are about drift AFTER
100k steps.

Run:
    uv run python JEPA/experiments/exp_013_sparse_exploration/probes/phi_drift_probe.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from JEPA.experiments.exp_011_ls20_icm.shared.icm import ICMModule
from JEPA.experiments.exp_011_ls20_icm.shared.ls20_vec_env_level import VecLS20EnvLevel

HERE = Path(__file__).resolve().parent
CKPT_ROOT = (
    HERE.parents[1]
    / "exp_011_ls20_icm"
    / "exp_011_0_icm_baseline"
    / "checkpoints"
)
OUT_JSON = HERE / "phi_drift_results.json"
OUT_FIG = HERE / "phi_drift.png"

N_PROBE = 1500          # target distinct frames in the probe set
N_ENVS = 8
MAX_EP_STEPS = 200
PROBE_SEED = 12345      # rng seed for the random-policy frame collection
LEVEL_INDEX = 0         # LS20 Level 1


def device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# 1. fixed probe set: distinct random-policy frames
# ---------------------------------------------------------------------------

def build_probe_set(n_target: int = N_PROBE) -> np.ndarray:
    """Roll a uniform-random policy on LS20 L1, dedupe frames by bytes."""
    rng = np.random.default_rng(PROBE_SEED)
    env = VecLS20EnvLevel(
        env_name="ls20", n_envs=N_ENVS, max_episode_steps=MAX_EP_STEPS,
        seed=PROBE_SEED, level_index=LEVEL_INDEX,
    )
    n_actions = env.n_actions
    seen: dict[bytes, np.ndarray] = {}

    def add(frames: np.ndarray):
        for f in frames:
            key = f.tobytes()
            if key not in seen:
                seen[key] = f.copy()

    add(env.current_obs())
    max_iters = 20000
    it = 0
    while len(seen) < n_target and it < max_iters:
        actions = rng.integers(0, n_actions, size=N_ENVS)
        next_obs, _, _, _ = env.step(actions)
        add(next_obs)
        it += 1
    frames = np.stack(list(seen.values()), axis=0).astype(np.uint8)
    print(f"[probe] collected {frames.shape[0]} distinct frames "
          f"in {it} random steps (target {n_target})")
    return frames


# ---------------------------------------------------------------------------
# 2. load checkpoints / encode
# ---------------------------------------------------------------------------

def list_runs() -> list[Path]:
    return sorted(p for p in CKPT_ROOT.iterdir() if p.is_dir())


def list_steps(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in glob.glob(str(run_dir / "step_*.pt")):
        m = re.search(r"step_(\d+)\.pt", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), Path(p)))
    return sorted(out, key=lambda x: x[0])


def load_phi_module(ckpt_path: Path, device: str) -> ICMModule:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    icm = ICMModule()
    icm.load_state_dict(ck["icm"])
    icm.eval()
    return icm.to(device)


def load_policy_encoder_state(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ck.get("encoder", None)


@torch.no_grad()
def encode_probe(icm: ICMModule, frames_u8: torch.Tensor, device: str,
                 chunk: int = 256) -> torch.Tensor:
    feats = []
    for s in range(0, frames_u8.shape[0], chunk):
        o = frames_u8[s:s + chunk].to(device)
        feats.append(icm.encode(o).cpu())
    return torch.cat(feats, dim=0)


# ---------------------------------------------------------------------------
# 3/4. drift + resolution metrics
# ---------------------------------------------------------------------------

def unit_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def mean_pairwise_l2(xn: torch.Tensor, max_pts: int = 800,
                     seed: int = 0) -> float:
    """Mean pairwise L2 among (a subsample of) unit-normed features."""
    n = xn.shape[0]
    if n > max_pts:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_pts]
        xn = xn[idx]
    d = torch.cdist(xn, xn)               # (m, m)
    m = xn.shape[0]
    triu = torch.triu_indices(m, m, offset=1)
    return float(d[triu[0], triu[1]].mean().item())


def consecutive_drift(phi_a: torch.Tensor, phi_b: torch.Tensor):
    """Drift between same states across two checkpoints."""
    cos = torch.nn.functional.cosine_similarity(phi_a, phi_b, dim=-1)
    an, bn = unit_norm(phi_a), unit_norm(phi_b)
    l2 = (an - bn).norm(dim=-1)
    return float(cos.mean()), float(cos.std()), float(l2.mean()), float(l2.std())


def main():
    dev = device_str()
    print(f"[device] {dev}")
    print(f"[ckpt root] {CKPT_ROOT}")

    frames_np = build_probe_set()
    frames_u8 = torch.from_numpy(frames_np)   # (N,64,64) uint8
    N = frames_u8.shape[0]

    runs = list_runs()
    print(f"[runs] {[r.name for r in runs]}")

    results = {
        "n_probe_states": N,
        "level_index": LEVEL_INDEX,
        "probe_seed": PROBE_SEED,
        "device": dev,
        "per_seed": {},
        "caveat": "Checkpoints start at step 102400; the first ~100k env "
                  "steps of phi drift are NOT captured.",
    }

    for run in runs:
        steps = list_steps(run)
        if not steps:
            continue
        seed_name = run.name
        print(f"\n=== {seed_name} : steps {[s for s, _ in steps]} ===")

        # phi (ICM encoder)
        phis = []
        for step, path in steps:
            icm = load_phi_module(path, dev)
            phis.append((step, encode_probe(icm, frames_u8, dev)))
            del icm
        # inter-state resolution per checkpoint (phi)
        resolution = [mean_pairwise_l2(unit_norm(phi)) for _, phi in phis]

        intervals = []
        for i in range(len(phis) - 1):
            s0, pa = phis[i]
            s1, pb = phis[i + 1]
            cos_m, cos_s, l2_m, l2_s = consecutive_drift(pa, pb)
            res_next = resolution[i + 1]
            intervals.append({
                "step_from": s0, "step_to": s1,
                "cos_mean": cos_m, "cos_std": cos_s,
                "l2_mean": l2_m, "l2_std": l2_s,
                "inter_state_l2_at_to": res_next,
                "drift_over_resolution": l2_m / res_next if res_next > 0 else float("nan"),
            })

        # policy encoder ("encoder") contrast: build a CNNEncoder via ICM phi
        # slot is different; the policy encoder is the exp_010 model encoder.
        # We reuse ICMModule.phi architecture only for phi; for the policy
        # encoder we load the exp_010 CNNEncoder directly.
        enc_intervals = _policy_encoder_drift(steps, frames_u8, dev)

        results["per_seed"][seed_name] = {
            "steps": [s for s, _ in steps],
            "phi_inter_state_l2": resolution,
            "phi_intervals": intervals,
            "policy_encoder_intervals": enc_intervals,
        }
        for iv in intervals:
            print(f"  phi {iv['step_from']}->{iv['step_to']}: "
                  f"cos={iv['cos_mean']:.4f} l2={iv['l2_mean']:.4f} "
                  f"res={iv['inter_state_l2_at_to']:.4f} "
                  f"ratio={iv['drift_over_resolution']:.3f}")

    # averages across seeds (phi)
    results["averaged_phi"] = _average_intervals(results["per_seed"], "phi_intervals")
    results["averaged_policy_encoder"] = _average_intervals(
        results["per_seed"], "policy_encoder_intervals")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {OUT_JSON}")

    try:
        _make_figure(results)
        print(f"[saved] {OUT_FIG}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")

    return results


def _policy_encoder_drift(steps, frames_u8, dev):
    """Encode the probe set with the POLICY encoder for contrast.

    The policy network in exp_010/011 wraps a CNNEncoder; the checkpoint key
    'encoder' is that CNNEncoder's state_dict. We instantiate a bare CNNEncoder
    and apply the same one-hot front-end ICM.encode uses.
    """
    from JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.shared.model import (
        CNNEncoder, one_hot_frame, N_COLORS, FRAME_SIZE, TRUNK_DIM,
    )
    feats = []
    ok = True
    for step, path in steps:
        sd = load_policy_encoder_state(path)
        if sd is None:
            ok = False
            break
        enc = CNNEncoder(n_colors=N_COLORS, frame_size=FRAME_SIZE, trunk_dim=TRUNK_DIM)
        try:
            enc.load_state_dict(sd)
        except Exception:
            ok = False
            break
        enc.eval().to(dev)
        with torch.no_grad():
            chunk = 256
            out = []
            for s in range(0, frames_u8.shape[0], chunk):
                o = frames_u8[s:s + chunk].to(dev)
                out.append(enc(one_hot_frame(o, N_COLORS)).cpu())
            feats.append((step, torch.cat(out, 0)))
        del enc
    if not ok:
        return []
    res = [mean_pairwise_l2(unit_norm(f)) for _, f in feats]
    out_iv = []
    for i in range(len(feats) - 1):
        s0, pa = feats[i]
        s1, pb = feats[i + 1]
        cos_m, cos_s, l2_m, l2_s = consecutive_drift(pa, pb)
        rn = res[i + 1]
        out_iv.append({
            "step_from": s0, "step_to": s1,
            "cos_mean": cos_m, "l2_mean": l2_m,
            "inter_state_l2_at_to": rn,
            "drift_over_resolution": l2_m / rn if rn > 0 else float("nan"),
        })
    return out_iv


def _average_intervals(per_seed: dict, key: str):
    buckets: dict[tuple[int, int], list[dict]] = {}
    for sd in per_seed.values():
        for iv in sd.get(key, []):
            buckets.setdefault((iv["step_from"], iv["step_to"]), []).append(iv)
    out = []
    for (s0, s1), ivs in sorted(buckets.items()):
        out.append({
            "step_from": s0, "step_to": s1,
            "n_seeds": len(ivs),
            "cos_mean": float(np.mean([x["cos_mean"] for x in ivs])),
            "l2_mean": float(np.mean([x["l2_mean"] for x in ivs])),
            "inter_state_l2_at_to": float(np.mean([x["inter_state_l2_at_to"] for x in ivs])),
            "drift_over_resolution": float(np.mean([x["drift_over_resolution"] for x in ivs])),
        })
    return out


def _make_figure(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # left: phi drift L2 + cos per seed
    ax = axes[0]
    for seed_name, sd in results["per_seed"].items():
        xs = [iv["step_to"] for iv in sd["phi_intervals"]]
        l2 = [iv["l2_mean"] for iv in sd["phi_intervals"]]
        ax.plot(xs, l2, marker="o", label=f"{seed_name[-5:]} L2")
    avg = results["averaged_phi"]
    ax.plot([iv["step_to"] for iv in avg], [iv["l2_mean"] for iv in avg],
            "k--", lw=2.5, marker="s", label="avg L2")
    ax.set_xlabel("training step (interval end)")
    ax.set_ylabel("phi drift: unit-norm L2 (same states)")
    ax.set_title("ICM phi drift between consecutive checkpoints")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # right: drift / inter-state resolution ratio
    ax = axes[1]
    for seed_name, sd in results["per_seed"].items():
        xs = [iv["step_to"] for iv in sd["phi_intervals"]]
        r = [iv["drift_over_resolution"] for iv in sd["phi_intervals"]]
        ax.plot(xs, r, marker="o", label=f"{seed_name[-5:]}")
    ax.plot([iv["step_to"] for iv in avg],
            [iv["drift_over_resolution"] for iv in avg],
            "k--", lw=2.5, marker="s", label="avg")
    ax.axhline(1.0, color="r", ls=":", label="drift = resolution")
    ax.set_xlabel("training step (interval end)")
    ax.set_ylabel("drift L2 / inter-state L2")
    ax.set_title("phi drift relative to inter-state resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)


if __name__ == "__main__":
    main()
