"""
Gradient-flow diagnostic probe for exp_003_1.

For each requested checkpoint we:
  1. Load online + target encoder, predictor, action_embed, policy.
  2. Roll out short episodes to populate a LatentBuffer with realistic transitions
     (mirroring the training rollout, including is_initial flag).
  3. Replicate one training-step forward/backward exactly as train.py does:
       - split batch by is_initial
       - online encode (live placeholders for initial; stored h_query for recurrent)
       - target encode -> h_targets (no grad)
       - flow_loss.backward()
  4. Walk every named parameter, log:
       - param L2 norm
       - grad L2 norm
       - grad-to-param ratio
  5. Aggregate by sub-module: SA stack, Perceiver(placeholders / round_i / output_norm),
     predictor (per-MLP / time_embed), action_embed.
  6. Forward-pass activation diagnostics: norms / std / pairwise cossim of the
     four output latents, sa_out, perceiver-internal queries each round,
     flow ODE step cossim (predictor near-identity check).
  7. Decompose flow loss by initial vs recurrent sub-batch: which one drives
     the placeholder gradient?

Outputs human-readable text + a compact JSON to results/.

Run:
  uv run python -m JEPA.experiments.exp_003_1_ema_target.grad_probe \
      --checkpoints step_005000.pt step_040000.pt step_080000.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_1_ema_target.config import Config
from JEPA.experiments.exp_003_1_ema_target.models import load_models_with_target
from JEPA.experiments.exp_003_1_ema_target.train import (
    LatentBuffer, load_checkpoint,
)
from JEPA.experiments.exp_003_1_ema_target.reward_shaping import is_end_of_life
from JEPA.shared.env_wrapper import LS20Env

# --------------------------------------------------------------------------- #
# small numeric helpers

def l2(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().item())

def stat_dict(t: torch.Tensor) -> dict:
    t = t.detach().float()
    return {
        "shape": list(t.shape),
        "mean":  float(t.mean().item()),
        "std":   float(t.std().item()),
        "norm":  float(t.norm().item()),
        "rms":   float((t.pow(2).mean().sqrt()).item()),
    }

def effective_rank(M: torch.Tensor) -> float:
    """Entropy-based effective rank of the (rows, cols) matrix M."""
    if M.dim() == 1:
        return 1.0
    s = torch.linalg.svdvals(M.detach().float().cpu())
    s = s[s > 1e-12]
    p = s / s.sum()
    H = -(p * (p + 1e-30).log()).sum()
    return float(torch.exp(H).item())


# --------------------------------------------------------------------------- #
# rollout to populate a buffer (mirrors train.py)

def collect_buffer(encoder, target_encoder, action_embed, predictor,
                   cfg: Config, n_transitions: int, device) -> LatentBuffer:
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(_repo_root / "environment_files"))
    raw = arc.make(cfg.game_id)
    env = LS20Env(raw)

    buf = LatentBuffer(
        n_latents=cfg.n_latents, d_model=cfg.d_model,
        capacity=max(n_transitions * 2, 1024),
        recency_fraction=cfg.recency_fraction,
        recent_window=cfg.recent_buffer_size,
    )

    frame_np = env.reset()
    h_t = None
    ep_buf: list = []
    n_added = 0

    while n_added < n_transitions:
        frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
        with torch.no_grad():
            if h_t is None:
                queries = encoder.perceiver.get_initial_queries(1, device)
                init_flag = True
            else:
                queries = h_t.detach()
                init_flag = False
            h_current, _, _ = encoder(frame_t, queries)

        action_idx = int(np.random.randint(0, cfg.n_actions))
        next_np, is_terminal = env.step(action_idx)
        life_end = is_end_of_life(frame_np, next_np, is_terminal)

        with torch.no_grad():
            next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)
            h_next, _, _ = target_encoder(next_t, h_current.detach())

        ep_buf.append((
            frame_np.copy(),
            queries.squeeze(0).detach().cpu().numpy(),
            action_idx,
            h_next.squeeze(0).cpu().numpy(),
            init_flag,
        ))
        h_t = h_current

        if life_end or len(ep_buf) >= 130:
            for f, q, a, ht, ini in ep_buf[:-1]:
                buf.add(f, q, a, ht, ini)
                n_added += 1
                if n_added >= n_transitions:
                    break
            ep_buf = []
            h_t = None
            frame_np = env.reset() if (life_end and is_terminal) else next_np
        else:
            frame_np = next_np
    return buf


# --------------------------------------------------------------------------- #
# the actual probe

def probe_checkpoint(ckpt_path: Path, cfg: Config, device, batch_size: int = 64,
                    n_buffer: int = 1024) -> dict:
    enc, tenc, pred, aemb, pol, _ = load_models_with_target(cfg, device)
    step = load_checkpoint(ckpt_path, enc, tenc, pred, aemb, pol, device)
    enc.train(); pred.train(); aemb.train(); tenc.eval()

    print(f"\n[probe] === {ckpt_path.name} (step {step}) ===")

    print(f"[probe] collecting {n_buffer} transitions ...")
    buf = collect_buffer(enc, tenc, aemb, pred, cfg, n_buffer, device)
    print(f"[probe] buffer size: {len(buf)}, "
          f"initials in buffer ≈ "
          f"{int(buf._is_initial[:len(buf)].sum())}/{len(buf)}")

    batch = buf.sample(min(batch_size, len(buf)), device)

    # ── activation pass ───────────────────────────────────────────────────────
    with torch.no_grad():
        # encode patches and SA
        sa_out = enc.encode_patches(batch.frames)
        # placeholder vs recurrent paths separately, then run perceiver
        q_init = enc.perceiver.get_initial_queries(batch.frames.shape[0], device)
        h_from_placeholder, _ = enc.perceiver(q_init, sa_out)
        h_from_recurrent, _ = enc.perceiver(batch.h_queries, sa_out)
        # full forward (mixed) so we can compare against forward used in training
        # walk through perceiver rounds manually for activation tracking
        round_outs = []
        h = q_init.clone()
        for r, rb in enumerate(enc.perceiver.rounds):
            h, _ = rb(h, sa_out)
            round_outs.append(h.detach().clone())
        h_pre_norm = h
        h_post_norm = enc.perceiver.output_norm(h)

    # cross-state std and effective rank (from the placeholder-path latents — this
    # matches the diagnostic train.py uses)
    bsz = h_from_placeholder.shape[0]
    h_flat = h_from_placeholder.reshape(bsz, -1)
    cross_state_std = float(h_from_placeholder.std(dim=0).mean().item())
    eff_rank_batch = effective_rank(h_flat)
    eff_rank_within = effective_rank(h_from_placeholder[0])  # 4 latents x 128
    pair_cs = []
    n_lat = cfg.n_latents
    for i in range(n_lat):
        for j in range(i + 1, n_lat):
            pair_cs.append(F.cosine_similarity(
                h_from_placeholder[:, i, :], h_from_placeholder[:, j, :], dim=-1
            ).mean().item())
    avg_pair_cossim_within = float(np.mean(pair_cs))
    same_lat_cs = []
    for i in range(n_lat):
        v = h_from_placeholder[:, i, :]
        v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        cs = (v_norm @ v_norm.T)
        # mean off-diagonal
        mask = ~torch.eye(cs.shape[0], dtype=torch.bool, device=cs.device)
        same_lat_cs.append(float(cs[mask].mean().item()))

    # placeholder vs recurrent path divergence
    ph_rec_cs = F.cosine_similarity(
        h_from_placeholder.reshape(bsz, -1),
        h_from_recurrent.reshape(bsz, -1), dim=-1).mean().item()
    ph_rec_l2 = (h_from_placeholder - h_from_recurrent).norm(dim=-1).mean().item()

    # predictor near-identity probe
    with torch.no_grad():
        a_emb = aemb(batch.actions)
        # one ODE step from current latents, action 0
        h_pred = pred.predict(h_from_placeholder, a_emb)
        ode_cossim = F.cosine_similarity(
            h_from_placeholder.reshape(bsz, -1),
            h_pred.reshape(bsz, -1), dim=-1).mean().item()
        ode_l2 = (h_from_placeholder - h_pred).norm(dim=-1).mean().item()

    activations = {
        "sa_out_norm":         l2(sa_out) / sa_out.numel() ** 0.5,
        "sa_out_std":          float(sa_out.std().item()),
        "h_round0_norm_mean":  float(round_outs[0].norm(dim=-1).mean().item()),
        "h_round1_norm_mean": (float(round_outs[1].norm(dim=-1).mean().item())
                               if len(round_outs) > 1 else None),
        "h_pre_outnorm_mean": float(h_pre_norm.norm(dim=-1).mean().item()),
        "h_post_outnorm_mean": float(h_post_norm.norm(dim=-1).mean().item()),
        "latent_cross_state_std": cross_state_std,
        "latent_within_eff_rank": eff_rank_within,
        "latent_batch_eff_rank":  eff_rank_batch,
        "latent_pairwise_cossim_within": avg_pair_cossim_within,
        "latent_same_idx_cross_state_cossim": same_lat_cs,
        "placeholder_vs_recurrent_path_cossim": ph_rec_cs,
        "placeholder_vs_recurrent_path_l2":     ph_rec_l2,
        "predictor_ode_cossim_to_input": ode_cossim,
        "predictor_ode_l2_step":         ode_l2,
    }

    # ── gradient pass: replicate one training step ────────────────────────────
    enc.zero_grad(set_to_none=True)
    pred.zero_grad(set_to_none=True)
    aemb.zero_grad(set_to_none=True)

    init_mask = batch.is_initial
    rec_mask = ~init_mask
    B = batch.frames.shape[0]

    h_t_fresh = torch.empty(B, cfg.n_latents, cfg.d_model, device=device)

    if init_mask.any():
        n_init = int(init_mask.sum().item())
        q_init = enc.perceiver.get_initial_queries(n_init, device)
        h_init, _, _ = enc(batch.frames[init_mask], q_init)
        h_t_fresh[init_mask] = h_init
    if rec_mask.any():
        h_rec, _, _ = enc(batch.frames[rec_mask], batch.h_queries[rec_mask].detach())
        h_t_fresh[rec_mask] = h_rec

    a_emb = aemb(batch.actions)
    flow_loss, per_lat = pred.compute_loss(h_t_fresh, batch.h_targets.detach(), a_emb)
    flow_loss.backward()

    # walk all parameters
    by_param = []
    for name, p in enc.named_parameters():
        g = p.grad
        by_param.append({
            "module":  "encoder",
            "name":    name,
            "n":       p.numel(),
            "p_norm":  l2(p),
            "p_std":   float(p.detach().float().std().item()) if p.numel() > 1 else 0.0,
            "g_norm":  l2(g) if g is not None else 0.0,
            "g_rms":   (float(g.detach().float().pow(2).mean().sqrt().item())
                        if g is not None else 0.0),
        })
    for name, p in pred.named_parameters():
        g = p.grad
        by_param.append({"module": "predictor", "name": name, "n": p.numel(),
                         "p_norm": l2(p), "p_std": float(p.detach().float().std().item()) if p.numel() > 1 else 0.0,
                         "g_norm": l2(g) if g is not None else 0.0,
                         "g_rms": (float(g.detach().float().pow(2).mean().sqrt().item()) if g is not None else 0.0)})
    for name, p in aemb.named_parameters():
        g = p.grad
        by_param.append({"module": "action_embed", "name": name, "n": p.numel(),
                         "p_norm": l2(p), "p_std": float(p.detach().float().std().item()) if p.numel() > 1 else 0.0,
                         "g_norm": l2(g) if g is not None else 0.0,
                         "g_rms": (float(g.detach().float().pow(2).mean().sqrt().item()) if g is not None else 0.0)})

    # group sums
    def group_g_sq(filter_fn) -> float:
        return sum((row["g_norm"] ** 2) for row in by_param if filter_fn(row))

    groups = {
        "encoder.color_embed":  ("encoder", lambda n: n.startswith("color_embed")),
        "encoder.patch_proj":   ("encoder", lambda n: n.startswith("patch_proj")),
        "encoder.sa_blocks":    ("encoder", lambda n: n.startswith("sa_blocks")),
        "encoder.sa_norm":      ("encoder", lambda n: n.startswith("sa_norm")),
        "perceiver.placeholders":   ("encoder", lambda n: n == "perceiver.placeholders"),
        "perceiver.round0.cross":   ("encoder", lambda n: n.startswith("perceiver.rounds.0.cross_attn")),
        "perceiver.round0.self":    ("encoder", lambda n: n.startswith("perceiver.rounds.0.self_attn")),
        "perceiver.round1.cross":   ("encoder", lambda n: n.startswith("perceiver.rounds.1.cross_attn")),
        "perceiver.round1.self":    ("encoder", lambda n: n.startswith("perceiver.rounds.1.self_attn")),
        "perceiver.output_norm":    ("encoder", lambda n: n.startswith("perceiver.output_norm")),
        "predictor.time_embed":     ("predictor", lambda n: n.startswith("time_embed")),
        "predictor.mlps":           ("predictor", lambda n: n.startswith("mlps")),
        "action_embed":             ("action_embed", lambda n: True),
    }
    group_grad = {}
    for label, (mod, pred_fn) in groups.items():
        sq = sum((row["g_norm"] ** 2)
                 for row in by_param
                 if row["module"] == mod and pred_fn(row["name"]))
        n = sum(row["n"]
                for row in by_param
                if row["module"] == mod and pred_fn(row["name"]))
        group_grad[label] = {
            "g_norm":   sq ** 0.5,
            "n_params": n,
            "g_rms":    (sq / max(n, 1)) ** 0.5,
        }

    # global norms (matching train.py reporting buckets)
    global_grads = {
        "flow_loss":   float(flow_loss.item()),
        "per_lat":     [float(x) for x in per_lat.tolist()],
        "enc_total":   sum(group_grad[k]["g_norm"] ** 2 for k in group_grad
                           if k.startswith("encoder.") or k.startswith("perceiver.")) ** 0.5,
        "enc_sa":      sum(group_grad[k]["g_norm"] ** 2
                           for k in ("encoder.color_embed", "encoder.patch_proj",
                                     "encoder.sa_blocks", "encoder.sa_norm")) ** 0.5,
        "enc_perceiver": sum(group_grad[k]["g_norm"] ** 2 for k in group_grad
                             if k.startswith("perceiver.")) ** 0.5,
        "predictor_total": sum(group_grad[k]["g_norm"] ** 2
                               for k in ("predictor.time_embed", "predictor.mlps")) ** 0.5,
    }

    # ── decompose loss by sub-batch (initial vs recurrent) ────────────────────
    # repeat backward separately on each subset
    init_loss_val = rec_loss_val = None
    init_grad_perceiver = rec_grad_perceiver = None
    init_grad_placeholder = rec_grad_placeholder = None

    def _grad_norm_of(group_pred, mod_pred):
        sq = 0.0
        if mod_pred == "encoder":
            for n, p in enc.named_parameters():
                if group_pred(n) and p.grad is not None:
                    sq += p.grad.detach().float().pow(2).sum().item()
        return sq ** 0.5

    if init_mask.any():
        enc.zero_grad(set_to_none=True); pred.zero_grad(set_to_none=True); aemb.zero_grad(set_to_none=True)
        n_init = int(init_mask.sum().item())
        q_i = enc.perceiver.get_initial_queries(n_init, device)
        h_i, _, _ = enc(batch.frames[init_mask], q_i)
        a_i = aemb(batch.actions[init_mask])
        loss_i, _ = pred.compute_loss(h_i, batch.h_targets[init_mask].detach(), a_i)
        loss_i.backward()
        init_loss_val = float(loss_i.item())
        init_grad_perceiver = _grad_norm_of(lambda n: n.startswith("perceiver."), "encoder")
        init_grad_placeholder = _grad_norm_of(lambda n: n == "perceiver.placeholders", "encoder")
        init_grad_sa = _grad_norm_of(lambda n: not n.startswith("perceiver."), "encoder")
    else:
        init_grad_sa = None

    if rec_mask.any():
        enc.zero_grad(set_to_none=True); pred.zero_grad(set_to_none=True); aemb.zero_grad(set_to_none=True)
        h_r, _, _ = enc(batch.frames[rec_mask], batch.h_queries[rec_mask].detach())
        a_r = aemb(batch.actions[rec_mask])
        loss_r, _ = pred.compute_loss(h_r, batch.h_targets[rec_mask].detach(), a_r)
        loss_r.backward()
        rec_loss_val = float(loss_r.item())
        rec_grad_perceiver = _grad_norm_of(lambda n: n.startswith("perceiver."), "encoder")
        rec_grad_placeholder = _grad_norm_of(lambda n: n == "perceiver.placeholders", "encoder")
        rec_grad_sa = _grad_norm_of(lambda n: not n.startswith("perceiver."), "encoder")
    else:
        rec_grad_sa = None

    decomp = {
        "n_initial": int(init_mask.sum().item()),
        "n_recurrent": int(rec_mask.sum().item()),
        "init_loss": init_loss_val,
        "rec_loss":  rec_loss_val,
        "init_grad_perceiver": init_grad_perceiver,
        "rec_grad_perceiver":  rec_grad_perceiver,
        "init_grad_placeholder": init_grad_placeholder,
        "rec_grad_placeholder":  rec_grad_placeholder,
        "init_grad_sa_etc": init_grad_sa,
        "rec_grad_sa_etc":  rec_grad_sa,
    }

    # ── singular value decomposition of stored target encoder embeddings ──────
    # (across the buffer, gives the *actual* representation manifold)
    with torch.no_grad():
        H = torch.from_numpy(buf._h_targets[:len(buf)]).reshape(len(buf), -1).float()
        s = torch.linalg.svdvals(H.cpu())
        s_top = s[:10].tolist()
        eff_rank_buffer = effective_rank(H)

    # ── EMA divergence between online and target ──────────────────────────────
    ema_diff = 0.0
    ema_norm = 0.0
    for (n_o, p_o), (n_t, p_t) in zip(enc.named_parameters(), tenc.named_parameters()):
        ema_diff += (p_o.detach() - p_t.detach()).pow(2).sum().item()
        ema_norm += p_o.detach().pow(2).sum().item()
    ema_relative = (ema_diff / max(ema_norm, 1e-12)) ** 0.5

    summary = {
        "step":           step,
        "ckpt":           ckpt_path.name,
        "activations":    activations,
        "global_grads":   global_grads,
        "group_grads":    group_grad,
        "loss_decomp":    decomp,
        "ema_relative_param_distance": ema_relative,
        "buffer_target_singular_top10": s_top,
        "buffer_target_eff_rank": eff_rank_buffer,
    }

    print_summary(summary)
    return summary


def print_summary(s: dict) -> None:
    a = s["activations"]
    g = s["global_grads"]
    gg = s["group_grads"]
    d = s["loss_decomp"]
    print(f"\n--- {s['ckpt']} step {s['step']} ---")
    print(f"flow_loss = {g['flow_loss']:.5f}   per_lat = "
          f"[{','.join(f'{x:.5f}' for x in g['per_lat'])}]")
    print(f"\n[activations]")
    print(f"  sa_out: norm/√n={a['sa_out_norm']:.3f}  std={a['sa_out_std']:.3f}")
    print(f"  perceiver h round0 norm-mean = {a['h_round0_norm_mean']:.3f}")
    if a['h_round1_norm_mean'] is not None:
        print(f"  perceiver h round1 norm-mean = {a['h_round1_norm_mean']:.3f}")
    print(f"  pre-output_norm  norm-mean   = {a['h_pre_outnorm_mean']:.3f}")
    print(f"  post-output_norm norm-mean   = {a['h_post_outnorm_mean']:.3f}  (≈ √d=11.31)")
    print(f"  latent cross_state_std = {a['latent_cross_state_std']:.4f}")
    print(f"  latent within-state effective rank = {a['latent_within_eff_rank']:.3f} / {len(a['latent_same_idx_cross_state_cossim'])}")
    print(f"  latent batch effective rank        = {a['latent_batch_eff_rank']:.3f}")
    print(f"  pairwise cossim across the 4 latents (within state) = {a['latent_pairwise_cossim_within']:.4f}")
    print(f"  same-latent-idx cross-state cossim = "
          f"[{','.join(f'{x:.4f}' for x in a['latent_same_idx_cross_state_cossim'])}]")
    print(f"  placeholder-path vs recurrent-path: cossim={a['placeholder_vs_recurrent_path_cossim']:.4f}  l2={a['placeholder_vs_recurrent_path_l2']:.4f}")
    print(f"  predictor ODE one-step:  cossim={a['predictor_ode_cossim_to_input']:.4f}  l2-step={a['predictor_ode_l2_step']:.4f}")

    print(f"\n[global grad norms]")
    print(f"  total encoder    : {g['enc_total']:.4f}")
    print(f"  encoder.SA stack : {g['enc_sa']:.4f}")
    print(f"  encoder.perceiver: {g['enc_perceiver']:.4f}")
    print(f"  predictor        : {g['predictor_total']:.4f}")

    print(f"\n[per-group grad norms]                  g_norm     #params  g_rms")
    for k, v in gg.items():
        print(f"  {k:36s}  {v['g_norm']:9.4f}  {v['n_params']:8d}  {v['g_rms']:.6f}")

    print(f"\n[loss decomposition init vs recurrent]")
    print(f"  init  ({d['n_initial']:3d} samples)  loss={d['init_loss']}  "
          f"perceiver_grad={d['init_grad_perceiver']}  placeholder_grad={d['init_grad_placeholder']}  sa_etc={d['init_grad_sa_etc']}")
    print(f"  recur ({d['n_recurrent']:3d} samples)  loss={d['rec_loss']}  "
          f"perceiver_grad={d['rec_grad_perceiver']}  placeholder_grad={d['rec_grad_placeholder']}  sa_etc={d['rec_grad_sa_etc']}")
    print(f"\n[EMA online↔target relative param distance] = {s['ema_relative_param_distance']:.4f}")
    print(f"[buffer target embeddings] eff_rank = {s['buffer_target_eff_rank']:.3f}, "
          f"top-10 singular values = "
          f"[{','.join(f'{x:.2f}' for x in s['buffer_target_singular_top10'])}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=[
        "step_005000.pt", "step_040000.pt", "step_080000.pt",
    ])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-buffer",   type=int, default=1024)
    ap.add_argument("--out", default="results/grad_probe.json")
    args = ap.parse_args()

    cfg = Config()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device = {device}")

    ckpt_root = Path(__file__).parent / "checkpoints"
    out_path = Path(__file__).parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in args.checkpoints:
        p = ckpt_root / name
        if not p.exists():
            print(f"[probe] skipping missing {p}")
            continue
        summaries.append(probe_checkpoint(p, cfg, device,
                                          batch_size=args.batch_size,
                                          n_buffer=args.n_buffer))

    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"\n[probe] wrote {out_path} ({len(summaries)} checkpoints)")


if __name__ == "__main__":
    main()
