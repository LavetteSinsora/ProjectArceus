"""exp_008_1 — JEPA overfit test.

For each checkpoint in the exp_007_4 sweep:
  1. load encoder, JEPA predictor, IDM head
  2. score the loaded modules on
       (a) transitions collected by the loaded stochastic policy
       (b) transitions collected by a uniform-random agent (collected once,
           cached, reused across all checkpoints)
  3. write per-checkpoint, per-source, per-action breakdown of:
       jepa_mse, idm_ce, idm_acc, n_samples
     plus an overall (action="all") row, plus an action_histogram row.

Outputs land in cfg.output_dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.device import pick_device
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.model import (
    ActorCritic,
    one_hot_frame,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.vec_env import VecMiniEnv
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_4_jepa_sg_idm.models import (
    ActionConditionedPredictor,
    InverseDynamicsModel,
)

from JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_1_JEPA_overfit_test.config import Config
from JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_1_JEPA_overfit_test.collect import (
    collect_transitions,
    random_action_fn,
    trained_action_fn,
)


# ───────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ───────────────────────────────────────────────────────────────────────────

def _list_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Sorted by update number; `final.pt` (if present) appended last."""
    update_files = sorted(ckpt_dir.glob("update_*.pt"))
    final = ckpt_dir / "final.pt"
    out = list(update_files)
    if final.exists():
        out.append(final)
    return out


def _parse_update(p: Path) -> int | None:
    """Returns int update number, or None for `final.pt`."""
    if p.stem == "final":
        return None
    return int(p.stem.split("_")[-1])


def _load_modules(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]  # dict (saved via asdict)
    model = ActorCritic().to(device)
    predictor = ActionConditionedPredictor(
        d_feat=model.encoder.trunk_dim,
        n_actions=model.n_actions,
        d_action=int(cfg["d_action"]),
        hidden=int(cfg["predictor_hidden"]),
    ).to(device)
    idm = InverseDynamicsModel(
        d_feat=model.encoder.trunk_dim,
        n_actions=model.n_actions,
        hidden=int(cfg["idm_hidden"]),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    predictor.load_state_dict(ck["predictor_state_dict"])
    idm.load_state_dict(ck["idm_state_dict"])
    model.eval(); predictor.eval(); idm.eval()
    return model, predictor, idm, ck.get("update", _parse_update(ckpt_path)), ck.get("env_step")


# ───────────────────────────────────────────────────────────────────────────
# Scoring
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _score(
    model: ActorCritic,
    predictor: ActionConditionedPredictor,
    idm: InverseDynamicsModel,
    transitions: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Compute per-transition (jepa_se, idm_ce, idm_correct) over all
    transitions (no done-filtering yet — caller will mask)."""
    obs = transitions["obs"]
    actions = transitions["actions"]
    next_obs = transitions["next_obs"]
    M = obs.shape[0]

    jepa_se = np.empty(M, dtype=np.float32)
    idm_ce = np.empty(M, dtype=np.float32)
    idm_correct = np.empty(M, dtype=np.bool_)

    for i in range(0, M, batch_size):
        j = min(M, i + batch_size)
        ob = obs[i:j].to(device)
        nb = next_obs[i:j].to(device)
        ac = actions[i:j].to(device)

        h_t = model.encoder(one_hot_frame(ob))
        h_n = model.encoder(one_hot_frame(nb))
        h_pred = predictor(h_t, ac)

        se = ((h_pred - h_n.detach()) ** 2).mean(dim=1)
        logits = idm(h_t, h_n)
        ce = F.cross_entropy(logits, ac, reduction="none")
        correct = (logits.argmax(dim=-1) == ac)

        jepa_se[i:j] = se.detach().cpu().numpy()
        idm_ce[i:j] = ce.detach().cpu().numpy()
        idm_correct[i:j] = correct.detach().cpu().numpy()

    return {"jepa_se": jepa_se, "idm_ce": idm_ce, "idm_correct": idm_correct}


def _aggregate_rows(
    update_label: int | str,
    source: str,
    transitions: dict[str, torch.Tensor],
    scored: dict[str, np.ndarray],
) -> list[dict]:
    """Build the per-action + overall rows for one (checkpoint, source)."""
    actions = transitions["actions"].numpy()
    dones = transitions["dones"].numpy()
    valid = ~dones

    rows: list[dict] = []
    # Overall.
    sel = valid
    rows.append({
        "update": update_label,
        "source": source,
        "action": "all",
        "n_samples": int(sel.sum()),
        "jepa_mse": float(scored["jepa_se"][sel].mean()) if sel.any() else float("nan"),
        "idm_ce": float(scored["idm_ce"][sel].mean()) if sel.any() else float("nan"),
        "idm_acc": float(scored["idm_correct"][sel].mean()) if sel.any() else float("nan"),
    })
    # Per-action.
    for a in range(4):
        sel = valid & (actions == a)
        rows.append({
            "update": update_label,
            "source": source,
            "action": str(a),
            "n_samples": int(sel.sum()),
            "jepa_mse": float(scored["jepa_se"][sel].mean()) if sel.any() else float("nan"),
            "idm_ce": float(scored["idm_ce"][sel].mean()) if sel.any() else float("nan"),
            "idm_acc": float(scored["idm_correct"][sel].mean()) if sel.any() else float("nan"),
        })
    return rows


def _action_histogram(transitions: dict[str, torch.Tensor]) -> dict[int, int]:
    actions = transitions["actions"].numpy()
    dones = transitions["dones"].numpy()
    valid = ~dones
    return {int(a): int(((actions == a) & valid).sum()) for a in range(4)}


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="final.pt only, 1024 transitions per source")
    p.add_argument("--n-transitions", type=int, default=None,
                   help="override Config.n_transitions_per_source")
    p.add_argument("--ckpt-glob", type=str, default=None,
                   help="only run on checkpoints whose basename matches this glob "
                        "(e.g. 'final.pt' or 'update_00095*.pt')")
    args = p.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.n_transitions_per_source = 1024
    if args.n_transitions is not None:
        cfg.n_transitions_per_source = args.n_transitions

    device = pick_device()
    print(f"[exp_008_1] device={device}")
    print(f"[exp_008_1] n_transitions_per_source={cfg.n_transitions_per_source}")

    ckpt_dir = Path(cfg.ckpt_sweep_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")
    ckpts = _list_checkpoints(ckpt_dir)
    if args.smoke:
        ckpts = [c for c in ckpts if c.stem == "final"]
    elif args.ckpt_glob is not None:
        ckpts = [c for c in ckpts if c.match(args.ckpt_glob)]
    if not ckpts:
        raise RuntimeError("no checkpoints selected")
    print(f"[exp_008_1] {len(ckpts)} checkpoint(s) to process")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / ("per_checkpoint_smoke.csv" if args.smoke else "per_checkpoint.csv")
    hist_path = out_dir / ("action_histogram_smoke.json" if args.smoke else "action_histogram.json")
    summary_path = out_dir / ("summary_smoke.json" if args.smoke else "summary.json")
    meta_path = out_dir / ("run_meta_smoke.json" if args.smoke else "run_meta.json")

    # Resolve level path to absolute once; defends against any cwd surprises.
    level_path_abs = str(Path(cfg.level_path).resolve())
    if not Path(level_path_abs).exists():
        raise FileNotFoundError(f"level config not found: {level_path_abs}")

    # ── 1. Collect the random transitions ONCE (policy-independent). ───────
    t0 = time.time()
    rng_env = VecMiniEnv(level_path_abs, n_envs=cfg.n_envs, seed=cfg.seed_random)
    rng_env.reset_all()
    # Build the trained-collection env once and just reset_all() each ckpt;
    # this both avoids the per-iter construction cost and sidesteps a
    # FileNotFoundError seen sporadically on repeated MiniLS20Env construction.
    tr_env = VecMiniEnv(level_path_abs, n_envs=cfg.n_envs, seed=cfg.seed_trained)
    print(f"[exp_008_1] collecting random source ({cfg.n_transitions_per_source} target valid)...")
    random_transitions = collect_transitions(
        rng_env,
        random_action_fn(n_actions=4, n_envs=cfg.n_envs, seed=cfg.seed_random),
        cfg.n_transitions_per_source,
    )
    random_valid = int((~random_transitions["dones"]).sum())
    print(f"[exp_008_1]   random collected: {random_transitions['obs'].shape[0]} total, "
          f"{random_valid} valid  ({time.time() - t0:.1f}s)")

    # ── 2. Per-checkpoint loop. ────────────────────────────────────────────
    all_rows: list[dict] = []
    histograms: dict[str, dict] = {}
    histograms["random"] = _action_histogram(random_transitions)

    csv_path.unlink(missing_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["update", "source", "action", "n_samples",
                           "jepa_mse", "idm_ce", "idm_acc"])
        writer.writeheader()

    for idx, ckpt_path in enumerate(ckpts):
        update_label = _parse_update(ckpt_path)
        update_str = "final" if update_label is None else str(update_label)
        t_ck = time.time()

        model, predictor, idm, ckpt_update, ckpt_env_step = _load_modules(ckpt_path, device)
        # Prefer the stored 'update' field over the filename for the label.
        if ckpt_update is not None:
            label_for_row = ckpt_update if ckpt_path.stem != "final" else f"final({ckpt_update})"
        else:
            label_for_row = update_str

        # 2a. Trained-policy transitions (depends on this checkpoint's weights).
        tr_env.reset_all()
        trained_transitions = collect_transitions(
            tr_env,
            trained_action_fn(model, device),
            cfg.n_transitions_per_source,
        )
        tr_valid = int((~trained_transitions["dones"]).sum())

        # 2b. Score.
        scored_tr = _score(model, predictor, idm, trained_transitions, device, cfg.score_batch_size)
        scored_rn = _score(model, predictor, idm, random_transitions, device, cfg.score_batch_size)

        # 2c. Aggregate.
        rows = []
        rows += _aggregate_rows(label_for_row, "trained", trained_transitions, scored_tr)
        rows += _aggregate_rows(label_for_row, "random", random_transitions, scored_rn)

        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["update", "source", "action", "n_samples",
                               "jepa_mse", "idm_ce", "idm_acc"])
            for r in rows:
                writer.writerow(r)

        all_rows.extend(rows)
        histograms[f"trained_{update_str}"] = _action_histogram(trained_transitions)

        # Pretty per-ckpt log line.
        tr_all = next(r for r in rows if r["source"] == "trained" and r["action"] == "all")
        rn_all = next(r for r in rows if r["source"] == "random"  and r["action"] == "all")
        gap = rn_all["jepa_mse"] / tr_all["jepa_mse"] if tr_all["jepa_mse"] > 0 else float("nan")
        print(f"[exp_008_1] [{idx+1:2d}/{len(ckpts)}] {ckpt_path.name}  "
              f"upd={update_str:>6}  tr_valid={tr_valid}  "
              f"jepa(tr)={tr_all['jepa_mse']:.4f}  jepa(rn)={rn_all['jepa_mse']:.4f}  "
              f"gap={gap:5.2f}x  idm_acc(tr)={tr_all['idm_acc']:.3f}  "
              f"idm_acc(rn)={rn_all['idm_acc']:.3f}  ({time.time()-t_ck:.1f}s)")

        # Free the per-ckpt state.
        del model, predictor, idm, trained_transitions, scored_tr, scored_rn
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    # ── 3. Write summary, histograms, run meta. ────────────────────────────
    hist_path.write_text(json.dumps(histograms, indent=2))

    # Headline summary: last checkpoint's "all" rows.
    last_label = all_rows[-1]["update"] if all_rows else None
    final_tr = next((r for r in reversed(all_rows)
                     if r["source"] == "trained" and r["action"] == "all"), None)
    final_rn = next((r for r in reversed(all_rows)
                     if r["source"] == "random" and r["action"] == "all"), None)
    summary = {
        "n_checkpoints": len(ckpts),
        "final_checkpoint_update": last_label,
        "final": {
            "trained": final_tr,
            "random": final_rn,
            "jepa_mse_ratio_random_over_trained": (
                final_rn["jepa_mse"] / final_tr["jepa_mse"]
                if final_tr and final_tr["jepa_mse"] > 0 else None
            ),
            "idm_acc_drop_trained_minus_random": (
                final_tr["idm_acc"] - final_rn["idm_acc"]
                if final_tr and final_rn else None
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    meta = {
        "config": asdict(cfg),
        "n_checkpoints": len(ckpts),
        "checkpoints": [str(c) for c in ckpts],
        "random_collected_total": int(random_transitions["obs"].shape[0]),
        "random_collected_valid": random_valid,
        "wall_clock_s": time.time() - t0,
        "device": str(device),
        "smoke": bool(args.smoke),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[exp_008_1] wrote:\n  {csv_path}\n  {hist_path}\n  {summary_path}\n  {meta_path}")
    print(f"[exp_008_1] total wall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
