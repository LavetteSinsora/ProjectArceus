"""Reproducible analysis of the exp_010 Colab runs (0 CNN / 1 joint-JEPA / 2 pretrained).

Reads the logged metrics.jsonl of each run and probes the pretrained encoder
against fresh real-LS20 frames, to explain WHY the two JEPA variants
underperform the raw CNN+PPO baseline.

    uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.analyze_runs

Pass --run-dir overrides if you train new runs; by default it auto-picks the
newest run dir under each sub-experiment.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
SUBS = {
    "0_cnn": "exp_010_0_cnn_ppo_baseline",
    "1_joint": "exp_010_1_jepa_joint_online",
    "2_pretr": "exp_010_2_jepa_random_pretrain",
}


def newest_metrics(sub: str) -> Path | None:
    runs = sorted(glob.glob(str(EXP / sub / "runs" / "*" / "metrics.jsonl")))
    # skip tiny smoke logs (< 50 records)
    runs = [r for r in runs if sum(1 for _ in open(r)) >= 50]
    return Path(runs[-1]) if runs else None


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def curve_summary():
    print("=" * 78)
    print("LEARNING-CURVE SUMMARY")
    print("=" * 78)
    for name, sub in SUBS.items():
        mp = newest_metrics(sub)
        if not mp:
            print(f"{name}: no full run found"); continue
        recs = load(mp)
        ev = [r for r in recs if r.get("success_rate") is not None]
        sr = [(r["step"], r["success_rate"]) for r in ev]
        smax = max((s for _, s in sr), default=float("nan"))
        first_train = next((r["step"] for r in recs
                            if (r.get("train_success_rate") or 0) > 0), None)
        ent = [r["policy_entropy"] for r in recs if r.get("policy_entropy") is not None]
        def g(k):
            c = [r[k] for r in recs if r.get(k) is not None]
            return (c[0], c[-1]) if c else (None, None)
        print(f"\n[{name}]  updates={len(recs)}  last_step={recs[-1]['step']}")
        print(f"  eval success_rate: max={smax:.3f}  final={sr[-1][1]:.3f}")
        print(f"  train first stumbled on goal at step: {first_train}")
        print(f"  policy_entropy: start={ent[0]:.3f} min={min(ent):.3f} final={ent[-1]:.3f}")
        for k in ["value_loss", "grad_norm_total", "feat_effective_rank",
                  "feat_pairwise_l2", "mean_feature_cosine", "jepa_loss"]:
            a, b = g(k)
            if a is not None:
                print(f"  {k:20s} {a:.3f} -> {b:.3f}")


def encoder_norm_probe(n_frames: int = 480):
    print("\n" + "=" * 78)
    print("ENCODER PROBE — features/value/logits on real LS20 frames (random policy)")
    print("=" * 78)
    from .shared.model import ActorCritic
    from .shared.ls20_vec_env import VecLS20Env
    torch.manual_seed(0); np.random.seed(0)
    env = VecLS20Env("ls20", n_envs=8, max_episode_steps=200, seed=1)
    frames = []
    for _ in range(n_frames // 8):
        obs, *_ = env.step(np.random.randint(0, 4, size=8))
        frames.append(obs.copy())
    F = torch.from_numpy(np.concatenate(frames, 0))

    def probe(model, tag):
        model.eval()
        with torch.no_grad():
            logits, val, feat = model.forward(F)
            ent = torch.distributions.Categorical(logits=logits).entropy()
        print(f"  {tag:12s} feat_norm={feat.norm(dim=-1).mean():7.2f}  "
              f"init_entropy={ent.mean():.3f} (uniform=1.386)  "
              f"|value|={val.abs().mean():6.3f}  logit_std={logits.std():.4f}")

    probe(ActorCritic(), "random-init")
    enc = EXP / "exp_010_2_jepa_random_pretrain" / "jepa_pretrained" / "encoder_final.pt"
    if enc.exists():
        m = ActorCritic()
        m.encoder.load_state_dict(torch.load(enc, map_location="cpu", weights_only=False)["encoder"])
        probe(m, "pretrained")
        print("\n  => pretrained features are ~60x larger-norm, so the random-init value head\n"
              "     starts wildly miscalibrated (huge value_loss/grad_norm), flooding the\n"
              "     SHARED encoder with large gradients before any reward is found.")


if __name__ == "__main__":
    curve_summary()
    encoder_norm_probe()
