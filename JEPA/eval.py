"""
Evaluation and visualisation for the JEPA world model.

Loads a checkpoint and reports:
  1. Sanity checks (forward pass shapes, embedding variance)
  2. World model prediction quality (JEPA loss on a fresh episode)
  3. Policy rollout statistics (mean reward, episode length, level completions)
  4. Embedding drift: how much latent representations change per step on average

Run:
    cd JEPA && uv run python eval.py --checkpoint jepa_checkpoint.pt --episodes 10
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from encoder import Encoder
from predictor import Predictor
from action_embed import ActionEmbedding
from policy import PolicyNetwork
from env_wrapper import LS20Env
from train import compute_patch_weights, jepa_loss


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg: Config = ckpt["config"]

    encoder = Encoder(cfg.d_model, cfg.d_color, cfg.n_heads, cfg.n_blocks, cfg.ffn_dim, cfg.patch_size).to(device)
    target_encoder = copy.deepcopy(encoder)
    predictor = Predictor(cfg.d_model, cfg.d_action).to(device)
    action_embed = ActionEmbedding(cfg.n_actions, cfg.d_action).to(device)
    policy = PolicyNetwork(cfg.d_model, cfg.n_actions).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    target_encoder.load_state_dict(ckpt["target_encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    action_embed.load_state_dict(ckpt["action_embed"])
    policy.load_state_dict(ckpt["policy"])

    encoder.eval(); target_encoder.eval(); predictor.eval()
    action_embed.eval(); policy.eval()

    print(f"[eval] Loaded checkpoint from step {ckpt.get('step', '?')}")
    return cfg, encoder, target_encoder, predictor, action_embed, policy


def sanity_checks(cfg: Config, encoder, predictor, action_embed, policy, device):
    print("\n── Sanity Checks ──────────────────────────────────────────────")
    dummy = torch.zeros(1, 64, 64, dtype=torch.uint8, device=device)
    z = encoder(dummy)
    assert z.shape == (1, 16, cfg.d_model), f"encoder output shape {z.shape}"
    print(f"  Encoder output shape: {tuple(z.shape)} ✓")

    a = action_embed(torch.zeros(1, dtype=torch.long, device=device))
    pred = predictor(z, a)
    assert pred.shape == (1, 16, cfg.d_model), f"predictor output shape {pred.shape}"
    print(f"  Predictor output shape: {tuple(pred.shape)} ✓")

    h = policy.initial_state().to(device)
    idx, lp, h2 = policy.act(h, z.squeeze(0))
    assert 0 <= idx < cfg.n_actions, f"action out of range: {idx}"
    print(f"  Policy step: action={idx}, log_prob={lp.item():.4f}, h shape={tuple(h2.shape)} ✓")


def run_episodes(cfg, encoder, target_encoder, predictor, action_embed, policy, device, n_episodes):
    from arc_agi import Arcade, OperationMode
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).parent.parent)
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_Path(_repo_root) / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)

    all_rewards, all_lengths, level_completions = [], [], 0
    all_jepa_losses = []
    all_emb_drifts = []

    for ep in range(n_episodes):
        frame_np = env.reset()
        h = policy.initial_state().to(device)
        ep_rewards, ep_drift = [], []
        step = 0

        while True:
            frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

            with torch.no_grad():
                z_t = encoder(frame_t)
                z_curr = target_encoder(frame_t).squeeze(0)

            action_idx, _, h = policy.act(h.detach(), z_t.squeeze(0), env.available_actions)
            next_frame_np, is_terminal = env.step(action_idx)
            next_frame_t = torch.from_numpy(next_frame_np).unsqueeze(0).to(device)

            with torch.no_grad():
                z_next = target_encoder(next_frame_t).squeeze(0)
                reward = (z_next - z_curr).pow(2).mean().sqrt().item()

                # JEPA prediction quality on this transition
                a_emb = action_embed(torch.tensor([action_idx], device=device))
                pred = predictor(z_t, a_emb)
                w = compute_patch_weights(frame_t, next_frame_t, cfg.change_weight_max)
                jloss = jepa_loss(pred, target_encoder(next_frame_t), w, z_t, cfg.variance_reg_lambda)
                all_jepa_losses.append(jloss.item())

                drift = (z_next - z_curr).pow(2).mean().sqrt().item()

            ep_rewards.append(reward)
            ep_drift.append(drift)
            step += 1

            if is_terminal or step > 500:
                break
            frame_np = next_frame_np

        if env.level_completed:
            level_completions += 1

        all_rewards.append(np.mean(ep_rewards))
        all_lengths.append(step)
        all_emb_drifts.extend(ep_drift)

    return {
        "mean_reward": np.mean(all_rewards),
        "mean_ep_len": np.mean(all_lengths),
        "level_completions": level_completions,
        "mean_jepa_loss": np.mean(all_jepa_losses),
        "mean_emb_drift": np.mean(all_emb_drifts),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="jepa_checkpoint.pt")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    ckpt_path = Path(__file__).parent / args.checkpoint
    cfg, encoder, target_encoder, predictor, action_embed, policy, = load_checkpoint(str(ckpt_path), device)

    sanity_checks(cfg, encoder, predictor, action_embed, policy, device)

    print(f"\n── Policy Rollout ({args.episodes} episodes) ─────────────────────────────")
    stats = run_episodes(cfg, encoder, target_encoder, predictor, action_embed, policy, device, args.episodes)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    print("\n── Embedding Health ────────────────────────────────────────────")
    from arc_agi import Arcade, OperationMode
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).parent.parent)
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_Path(_repo_root) / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)
    frames = []
    frame_np = env.reset()
    for _ in range(64):
        action_idx = np.random.randint(0, cfg.n_actions)
        next_np, done = env.step(action_idx)
        frames.append(frame_np)
        frame_np = next_np if not done else env.reset()
    frames_t = torch.from_numpy(np.stack(frames)).to(device)
    with torch.no_grad():
        z = encoder(frames_t)
    print(f"  Embedding std (across batch): {z.std(dim=0).mean().item():.4f}")
    print(f"  Embedding mean norm: {z.norm(dim=-1).mean().item():.4f}")


if __name__ == "__main__":
    main()
