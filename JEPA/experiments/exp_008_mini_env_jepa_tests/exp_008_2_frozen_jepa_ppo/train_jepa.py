"""Offline JEPA pretraining on uniform-random-agent data.

Loss:
    L = L_jepa + lambda_idm * L_idm
    L_jepa = MSE(predictor(encoder(o_t), a_t), sg(encoder(o_{t+1})))
    L_idm  = CE(idm(encoder(o_t), encoder(o_{t+1})), a_t)
                                               # grads flow into both endpoints

This is the exp_007_4 loss minus the PPO half. The encoder and predictor
classes are imported verbatim from the exp_007 codebase — no new model code.

Usage:
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_jepa --env 1rot
    uv run python -m JEPA.experiments.exp_008_mini_env_jepa_tests.exp_008_2_frozen_jepa_ppo.train_jepa --env 2rot --smoke

Outputs (per env_tag):
    jepa_runs/<env_tag>_<timestamp>/
        config.json
        metrics.jsonl
        encoder_final.pt          # state dicts; used by train_ppo.py
        encoder_epoch_<NN>.pt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.device import pick_device
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.shared.model import (
    CNNEncoder,
    one_hot_frame,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_4_jepa_sg_idm.models import (
    ActionConditionedPredictor,
    InverseDynamicsModel,
)
from JEPA.experiments.exp_007_mini_env_cnn_ppo_baseline.exp_007_3_jepa_sg.diagnostics import (
    all_diagnostics,
)

from .collect import buffer_path
from .config import ENV_TAG_TO_LEVEL, JEPA_RUNS_DIR, JEPATrainConfig, level_path_for


def _make_run_dir(env_tag: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run = JEPA_RUNS_DIR / f"{env_tag}_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def _log_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _split_indices(n: int, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_split))
    return idx[n_val:], idx[:n_val]      # (train, val)


def train_jepa(cfg: JEPATrainConfig, smoke: bool = False) -> Path:
    device = pick_device()
    print(f"[jepa] device={device}  env={cfg.env_tag}  smoke={smoke}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Load random buffer (env-specific) ───────────────────────────────
    buf_path = buffer_path(cfg.env_tag, cfg.seed)
    if not buf_path.exists():
        raise FileNotFoundError(
            f"random buffer not found at {buf_path}.  "
            f"Run collect.py --env {cfg.env_tag} first."
        )
    print(f"[jepa] loading buffer: {buf_path}")
    buf = torch.load(buf_path, weights_only=False)

    obs_all: torch.Tensor = buf["obs"]            # (M, 32, 32) uint8
    actions_all: torch.Tensor = buf["actions"]    # (M,)        int64
    next_obs_all: torch.Tensor = buf["next_obs"]  # (M, 32, 32) uint8
    dones_all: torch.Tensor = buf["dones"]        # (M,)        bool

    # Drop done-crossing transitions entirely (the predictor / IDM should
    # never see (s_terminal, a, s_reset) pairs).
    valid_mask = ~dones_all
    obs = obs_all[valid_mask]
    actions = actions_all[valid_mask]
    next_obs = next_obs_all[valid_mask]
    n_valid = obs.shape[0]
    print(f"[jepa] buffer: {obs_all.shape[0]} total, {n_valid} valid transitions")

    train_idx, val_idx = _split_indices(n_valid, cfg.val_split, cfg.seed)
    print(f"[jepa] split: train={len(train_idx)} val={len(val_idx)}")

    # ── Build modules ───────────────────────────────────────────────────
    encoder = CNNEncoder().to(device)
    predictor = ActionConditionedPredictor(
        d_feat=encoder.trunk_dim,
        n_actions=4,
        d_action=cfg.d_action,
        hidden=cfg.predictor_hidden,
    ).to(device)
    idm = InverseDynamicsModel(
        d_feat=encoder.trunk_dim,
        n_actions=4,
        hidden=cfg.idm_hidden,
    ).to(device)

    all_params = (
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(idm.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=cfg.learning_rate)

    # ── Run dir + config dump ───────────────────────────────────────────
    run_dir = _make_run_dir(cfg.env_tag)
    (run_dir / "config.json").write_text(json.dumps({
        **asdict(cfg),
        "level_path": level_path_for(cfg.env_tag),
        "buffer_path": str(buf_path),
        "buffer_meta": buf.get("meta", {}),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "smoke": smoke,
    }, indent=2))
    log_path = run_dir / "metrics.jsonl"

    n_epochs = 1 if smoke else cfg.n_epochs
    print(f"[jepa] training {n_epochs} epoch(s), batch_size={cfg.batch_size}")

    # ── Train ───────────────────────────────────────────────────────────
    t0 = time.time()
    global_step = 0
    for epoch in range(1, n_epochs + 1):
        rng = np.random.default_rng(cfg.seed + epoch)
        order = train_idx.copy()
        rng.shuffle(order)

        encoder.train(); predictor.train(); idm.train()

        epoch_jepa = epoch_idm = epoch_acc = 0.0
        n_batches = 0
        for start in range(0, len(order), cfg.batch_size):
            sel = order[start:start + cfg.batch_size]
            if len(sel) == 0:
                continue

            mb_obs = obs[sel].to(device)
            mb_actions = actions[sel].to(device)
            mb_next_obs = next_obs[sel].to(device)

            h_t = encoder(one_hot_frame(mb_obs))
            h_tp1 = encoder(one_hot_frame(mb_next_obs))

            pred = predictor(h_t, mb_actions)
            loss_jepa = (pred - h_tp1.detach()).pow(2).sum(dim=-1).mean()

            idm_logits = idm(h_t, h_tp1)
            loss_idm = F.cross_entropy(idm_logits, mb_actions)

            with torch.no_grad():
                idm_acc = (idm_logits.argmax(dim=-1) == mb_actions).float().mean().item()

            loss = loss_jepa + cfg.idm_loss_weight * loss_idm

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            optimizer.step()

            epoch_jepa += float(loss_jepa.item())
            epoch_idm += float(loss_idm.item())
            epoch_acc += float(idm_acc)
            n_batches += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                _log_jsonl(log_path, {
                    "step": global_step,
                    "epoch": epoch,
                    "split": "train",
                    "jepa_loss": float(loss_jepa.item()),
                    "idm_loss": float(loss_idm.item()),
                    "idm_acc": float(idm_acc),
                    "grad_norm": float(grad_norm),
                    "wall_clock_s": time.time() - t0,
                })
                if smoke:
                    break  # one batch per epoch is plenty for smoke

        # ── End-of-epoch validation ─────────────────────────────────────
        val_jepa, val_idm, val_acc, collapse = _evaluate(
            encoder, predictor, idm, obs, actions, next_obs, val_idx,
            cfg.batch_size, device,
        )
        msg = (
            f"[jepa] epoch={epoch:3d}/{n_epochs} "
            f"train_jepa={epoch_jepa / max(1, n_batches):.4f} "
            f"train_idm={epoch_idm / max(1, n_batches):.4f} "
            f"train_idm_acc={epoch_acc / max(1, n_batches):.3f} "
            f"val_jepa={val_jepa:.4f} val_idm={val_idm:.4f} val_acc={val_acc:.3f} "
            f"cos={collapse['feat_cosine_consecutive']:.3f} "
            f"std={collapse['feat_std']:.3f} "
            f"rank={collapse['feat_effective_rank']:.1f}"
        )
        print(msg)
        _log_jsonl(log_path, {
            "step": global_step,
            "epoch": epoch,
            "split": "epoch_summary",
            "train_jepa_loss": epoch_jepa / max(1, n_batches),
            "train_idm_loss": epoch_idm / max(1, n_batches),
            "train_idm_acc": epoch_acc / max(1, n_batches),
            "val_jepa_loss": val_jepa,
            "val_idm_loss": val_idm,
            "val_idm_acc": val_acc,
            **collapse,
            "wall_clock_s": time.time() - t0,
        })

        if epoch % cfg.save_every_epochs == 0 or epoch == n_epochs:
            _save_ckpt(
                run_dir / f"encoder_epoch_{epoch:03d}.pt",
                encoder, predictor, idm, cfg, epoch, global_step, val_jepa, val_idm, val_acc,
            )

    final_path = run_dir / "encoder_final.pt"
    _save_ckpt(
        final_path, encoder, predictor, idm, cfg, n_epochs, global_step, val_jepa, val_idm, val_acc,
    )
    print(f"[jepa] done. final encoder -> {final_path}")
    return run_dir


@torch.no_grad()
def _evaluate(
    encoder, predictor, idm,
    obs, actions, next_obs,
    val_idx, batch_size, device,
) -> tuple[float, float, float, dict[str, float]]:
    """Returns (val_jepa_loss, val_idm_loss, val_idm_acc, collapse_metrics).

    Collapse metrics computed on h_t over the whole val split:
        feat_cosine_consecutive — mean cos(h_t, h_{t+1}); 1 ⇒ collapse
        feat_std                — mean std of feature dims; 0 ⇒ collapse
        feat_pairwise_l2        — mean pairwise distance; 0 ⇒ collapse
        feat_effective_rank     — entropy-based rank; 1 ⇒ rank-1 collapse
    """
    nan_collapse = {
        "feat_cosine_consecutive": float("nan"),
        "feat_std": float("nan"),
        "feat_pairwise_l2": float("nan"),
        "feat_effective_rank": float("nan"),
    }
    if len(val_idx) == 0:
        return float("nan"), float("nan"), float("nan"), nan_collapse

    encoder.eval(); predictor.eval(); idm.eval()
    j_sum = i_sum = a_sum = 0.0
    n = 0
    cos_sum = 0.0
    cos_n = 0
    h_t_chunks: list[torch.Tensor] = []   # for global collapse diagnostics

    for start in range(0, len(val_idx), batch_size):
        sel = val_idx[start:start + batch_size]
        if len(sel) == 0:
            continue
        mb_obs = obs[sel].to(device)
        mb_actions = actions[sel].to(device)
        mb_next_obs = next_obs[sel].to(device)
        h_t = encoder(one_hot_frame(mb_obs))
        h_tp1 = encoder(one_hot_frame(mb_next_obs))
        pred = predictor(h_t, mb_actions)
        j = (pred - h_tp1).pow(2).sum(dim=-1).mean().item()
        logits = idm(h_t, h_tp1)
        i = F.cross_entropy(logits, mb_actions).item()
        a = (logits.argmax(dim=-1) == mb_actions).float().mean().item()
        bs = mb_obs.shape[0]
        j_sum += j * bs
        i_sum += i * bs
        a_sum += a * bs
        n += bs

        h_t_n = F.normalize(h_t, dim=-1, eps=1e-8)
        h_tp1_n = F.normalize(h_tp1, dim=-1, eps=1e-8)
        cos_sum += (h_t_n * h_tp1_n).sum(-1).sum().item()
        cos_n += bs

        # Cap stored features at ~4096 rows so the SVD stays cheap.
        if sum(c.shape[0] for c in h_t_chunks) < 4096:
            h_t_chunks.append(h_t.detach())

    collapse: dict[str, float] = {
        "feat_cosine_consecutive": cos_sum / max(1, cos_n),
    }
    if h_t_chunks:
        h_all = torch.cat(h_t_chunks, dim=0)
        collapse.update(all_diagnostics(h_all))
    else:
        collapse.update({
            "feat_std": float("nan"),
            "feat_pairwise_l2": float("nan"),
            "feat_effective_rank": float("nan"),
        })

    return j_sum / max(1, n), i_sum / max(1, n), a_sum / max(1, n), collapse


def _save_ckpt(path: Path, encoder, predictor, idm, cfg, epoch, step,
               val_jepa, val_idm, val_acc) -> None:
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "idm_state_dict": idm.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "step": step,
        "val_jepa_loss": float(val_jepa),
        "val_idm_loss": float(val_idm),
        "val_idm_acc": float(val_acc),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=sorted(ENV_TAG_TO_LEVEL), required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 1 train batch, tiny val pass")
    args = p.parse_args()

    cfg = JEPATrainConfig(env_tag=args.env, seed=args.seed)
    if args.epochs is not None:
        cfg.n_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr

    train_jepa(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
