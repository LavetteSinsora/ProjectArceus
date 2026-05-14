"""
At t=0 only: dump the FULL 4×4 pairwise JSD matrix and per-row diagnostics for
Round 0's cross-attention. Identifies which latent query is the outlier when
the mean pairwise JSD would otherwise hide a "3-similar + 1-different" pattern.

For each Perceiver round and each head, prints:
  - 4×4 pairwise JSD matrix on softmax-attention rows
  - 4×4 pairwise cosine sim on RAW logit rows
  - per-row "outlier score":  mean JSD of row i against the other 3 rows
  - which queries are anomalously close to each other (q_proj cosine)

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_3_state_only_reward.analyze_cross_attn_t0_full \\
        --checkpoint JEPA/experiments/exp_003_3_state_only_reward/checkpoints/step_015000.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_3_state_only_reward.config import Config
from JEPA.experiments.exp_003_3_state_only_reward.models import load_models
from JEPA.shared.env_wrapper import LS20Env


def _pairwise_jsd_matrix(P: torch.Tensor) -> torch.Tensor:
    """Full K×K pairwise JSD matrix over rows of P (K × V), each a distribution."""
    P = P.clamp_min(1e-12)
    M = 0.5 * (P.unsqueeze(0) + P.unsqueeze(1))
    kl = (P.unsqueeze(1) * (P.unsqueeze(1).log() - M.log())).sum(-1)
    return 0.5 * (kl + kl.transpose(0, 1))


def _pairwise_cos_matrix(v: torch.Tensor) -> torch.Tensor:
    """Full K×K pairwise cosine matrix over rows of v."""
    vn = F.normalize(v, dim=-1)
    return vn @ vn.T


def _print_matrix(M: torch.Tensor, fmt: str = "{:+.4f}"):
    K = M.shape[0]
    print("         " + "".join(f"{j:>11}" for j in range(K)))
    for i in range(K):
        row = " ".join(fmt.format(M[i, j].item()).rjust(10) for j in range(K))
        print(f"   row{i}  {row}")


@torch.no_grad()
def analyze_t0(ckpt_path: str):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"[t0-full] device={device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, dict):
        valid = {f.name for f in dataclasses.fields(Config)}
        cfg = Config(**{k: v for k, v in cfg_raw.items() if k in valid})
    else:
        cfg = cfg_raw
    encoder, sp, ap, ae, pol, _ = load_models(cfg, device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    print(f"[t0-full] loaded step={ckpt.get('step', 0)}")

    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(_repo_root / "environment_files"))
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)
    frame_np = env.reset()
    frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

    queries = encoder.perceiver.get_initial_queries(1, device)   # (1, 4, d_model)
    sa_out = encoder.encode_patches(frame_t)                     # (1, 16, d_model)

    h = queries
    for r_idx, round_block in enumerate(encoder.perceiver.rounds):
        ca = round_block.cross_attn
        q_in = ca.norm_q(h)
        k_in = ca.norm_kv(sa_out)
        Q = ca.q_proj(q_in).view(1, h.shape[1], ca.n_heads, ca.d_head).permute(0, 2, 1, 3).squeeze(0)   # (heads, 4, d_head)
        K = ca.k_proj(k_in).view(1, 16, ca.n_heads, ca.d_head).permute(0, 2, 1, 3).squeeze(0)            # (heads, 16, d_head)
        scale = math.sqrt(ca.d_head)
        logits = (Q @ K.transpose(-1, -2)) / scale       # (heads, 4, 16)
        attn   = F.softmax(logits, dim=-1)               # (heads, 4, 16)

        print(f"\n══════════════════════════════════════════════════════════════════")
        print(f"  Round {r_idx} — Cross-attention at t=0 (placeholder queries)")
        print(f"══════════════════════════════════════════════════════════════════")
        print(f"  h_input cossim pairwise mean: {_pairwise_cos_matrix(h.squeeze(0))[torch.triu_indices(4,4,1).unbind()].mean().item():+.4f}")

        for h_idx in range(ca.n_heads):
            print(f"\n  ── head {h_idx} " + "─" * 50)

            jsd_M = _pairwise_jsd_matrix(attn[h_idx])             # 4×4
            logit_cos_M = _pairwise_cos_matrix(logits[h_idx])     # 4×4 on raw logits
            q_cos_M = _pairwise_cos_matrix(Q[h_idx])              # 4×4 on Q post-q_proj

            print(f"  4×4 softmax-row JSD matrix:")
            _print_matrix(jsd_M, "{:+.4f}")

            iu = torch.triu_indices(4, 4, 1)
            mean_jsd = jsd_M[iu[0], iu[1]].mean().item()
            max_jsd = jsd_M[iu[0], iu[1]].max().item()
            min_jsd = jsd_M[iu[0], iu[1]].min().item()
            print(f"    pairwise: mean={mean_jsd:.4f}  min={min_jsd:.4f}  max={max_jsd:.4f}  ratio max/min={max_jsd/(min_jsd+1e-12):.1f}×")

            # Per-row outlier score: mean JSD of row i vs the other 3 rows
            outlier_score = torch.tensor([
                (jsd_M[i, :].sum() - jsd_M[i, i]) / 3.0 for i in range(4)
            ])
            print(f"    per-row outlier score (mean JSD vs others):")
            for i in range(4):
                bar = "█" * int(outlier_score[i].item() * 100)
                print(f"      L{i}: {outlier_score[i].item():.4f}  {bar}")
            outlier_idx = int(outlier_score.argmax().item())
            print(f"    → most-different row: L{outlier_idx} "
                  f"(outlier_score = {outlier_score[outlier_idx].item():.4f})")

            print(f"  4×4 raw-logit cosine matrix:")
            _print_matrix(logit_cos_M, "{:+.4f}")
            print(f"  4×4 post-q_proj Q cosine matrix:")
            _print_matrix(q_cos_M, "{:+.4f}")

        # Advance the recurrent stream through Round r_idx so Round r_idx+1 sees the right input
        attn_full = F.softmax(
            (ca.q_proj(q_in).view(1, h.shape[1], ca.n_heads, ca.d_head).transpose(1, 2) @
             ca.k_proj(k_in).view(1, 16, ca.n_heads, ca.d_head).transpose(1, 2).transpose(-1, -2)) / scale,
            dim=-1,
        )
        V = ca.v_proj(sa_out).view(1, 16, ca.n_heads, ca.d_head).transpose(1, 2)
        out_attn = (attn_full @ V).transpose(1, 2).contiguous().view(1, h.shape[1], -1)
        h = h + ca.out_proj(out_attn)
        h = ca.ffn(h)
        h = round_block.self_attn(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    analyze_t0(args.checkpoint)


if __name__ == "__main__":
    main()
