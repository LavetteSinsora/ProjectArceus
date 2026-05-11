"""
Debug: Why do the 4 Perceiver latent tokens converge to near-identical representations?

Hooks into every intermediate stage of the Perceiver to track:
  1. Cosine similarity between the 4 latent vectors at each stage
  2. Cross-attention weight patterns (which patches does each latent attend to?)
  3. Self-attention weight patterns among latents (is it acting like mean-pooling?)

Run:
    cd "Code Repo"
    uv run python -m JEPA.experiments.exp_003_0_normalized_latent_jepa.debug_latent_collapse
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.experiments.exp_003_0_normalized_latent_jepa.config import Config
from JEPA.experiments.exp_003_0_normalized_latent_jepa.models import load_models
from JEPA.experiments.exp_003_0_normalized_latent_jepa.reward_shaping import is_end_of_life
from JEPA.shared.env_wrapper import LS20Env

CKPT = Path(__file__).parent / "checkpoints" / "step_395000.pt"
OUT_DIR = Path(__file__).parent / "results" / "latent_collapse_debug"
N_EPISODES = 3
MAX_STEPS_PER_EP = 60


# ── Cosine similarity helpers ─────────────────────────────────────────────────

def pairwise_cossim(h: torch.Tensor) -> float:
    """h: (n, d) → mean of all unique pairwise cosine similarities."""
    n = h.shape[0]
    normed = F.normalize(h.float(), dim=-1)
    sims = [normed[i].dot(normed[j]).item() for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(sims))


def all_pairwise_cossim(h: torch.Tensor) -> list:
    """h: (n, d) → list of all (i,j) cosine similarities."""
    n = h.shape[0]
    normed = F.normalize(h.float(), dim=-1)
    return [(i, j, normed[i].dot(normed[j]).item()) for i in range(n) for j in range(i + 1, n)]


# ── Instrumented forward pass ─────────────────────────────────────────────────

class InstrumentedEncoder:
    """Wraps encoder to capture intermediate Perceiver representations."""

    def __init__(self, encoder):
        self.enc = encoder

    @torch.no_grad()
    def forward_with_probes(self, frame_t: torch.Tensor, queries: torch.Tensor) -> dict:
        """
        Returns a dict with:
          - 'input_queries':         (n_latents, d)  — queries entering the Perceiver
          - 'after_cross_r{i}':      (n_latents, d)  — after cross-attn in round i
          - 'after_self_r{i}':       (n_latents, d)  — after self-attn in round i
          - 'output':                (n_latents, d)  — final output-normed latents
          - 'cross_attn_w_r{i}':     (n_heads, n_latents, 16) — cross-attn weights round i
          - 'self_attn_w_r{i}':      (n_heads, n_latents, n_latents) — self-attn weights round i
          - 'sa_out':                (16, d)         — SA-normed patch embeddings
        """
        probes = {}
        enc = self.enc
        B = frame_t.shape[0]

        # ── Stage 1: patch embedding + SA blocks ──────────────────────────────
        sa_out = enc.encode_patches(frame_t)  # (B, 16, d)
        probes["sa_out"] = sa_out[0].cpu()

        # ── Perceiver: manual round-by-round forward ──────────────────────────
        h = queries  # (B, n_latents, d)
        probes["input_queries"] = h[0].cpu()

        for r, round_block in enumerate(enc.perceiver.rounds):
            # --- Cross-attention ---
            cross = round_block.cross_attn
            B_, Lq, D = h.shape
            Lc = sa_out.shape[1]
            Q = cross.q_proj(cross.norm_q(h)).view(B_, Lq, cross.n_heads, cross.d_head).transpose(1, 2)
            K = cross.k_proj(cross.norm_kv(sa_out)).view(B_, Lc, cross.n_heads, cross.d_head).transpose(1, 2)
            V = cross.v_proj(sa_out).view(B_, Lc, cross.n_heads, cross.d_head).transpose(1, 2)
            attn_w = F.softmax((Q @ K.transpose(-2, -1)) / cross.scale, dim=-1)
            probes[f"cross_attn_w_r{r}"] = attn_w[0].cpu()  # (n_heads, n_latents, 16)
            out = (attn_w @ V).transpose(1, 2).contiguous().view(B_, Lq, D)
            h_after_cross = h + cross.out_proj(out)
            h_after_cross = cross.ffn(h_after_cross)
            probes[f"after_cross_r{r}"] = h_after_cross[0].cpu()  # (n_latents, d)

            # --- Self-attention among latents ---
            sa = round_block.self_attn
            B_, L, D = h_after_cross.shape
            hh = sa.norm(h_after_cross)
            Q2 = sa.q_proj(hh).view(B_, L, sa.n_heads, sa.d_head).transpose(1, 2)
            K2 = sa.k_proj(hh).view(B_, L, sa.n_heads, sa.d_head).transpose(1, 2)
            V2 = sa.v_proj(hh).view(B_, L, sa.n_heads, sa.d_head).transpose(1, 2)
            attn_w2 = F.softmax((Q2 @ K2.transpose(-2, -1)) / sa.scale, dim=-1)
            probes[f"self_attn_w_r{r}"] = attn_w2[0].cpu()  # (n_heads, n_latents, n_latents)
            out2 = (attn_w2 @ V2).transpose(1, 2).contiguous().view(B_, L, D)
            h_after_self = h_after_cross + sa.out_proj(out2)
            h_after_self = sa.ffn(h_after_self)
            probes[f"after_self_r{r}"] = h_after_self[0].cpu()  # (n_latents, d)
            h = h_after_self

        h_final = enc.perceiver.output_norm(h)
        probes["output"] = h_final[0].cpu()

        return probes


def probe_stages_cossim(probes: dict, n_latents: int, n_rounds: int) -> dict:
    """Extract avg pairwise cosine similarity at every stage."""
    stages = {}
    stages["0_input_queries"] = pairwise_cossim(probes["input_queries"])
    for r in range(n_rounds):
        stages[f"{2*r+1}_after_cross_r{r}"] = pairwise_cossim(probes[f"after_cross_r{r}"])
        stages[f"{2*r+2}_after_self_r{r}"] = pairwise_cossim(probes[f"after_self_r{r}"])
    stages[f"{2*n_rounds+1}_output_normed"] = pairwise_cossim(probes["output"])
    return stages


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    encoder, _, _, _, _ = load_models(cfg, device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    instr = InstrumentedEncoder(encoder)

    from arc_agi import Arcade, OperationMode
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_repo_root / "environment_files"),
    )
    raw_env = arc.make(cfg.game_id)
    env = LS20Env(raw_env)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Accumulate per-stage cosine similarities across all timesteps
    # Key: stage label → list of float
    stage_cossims: dict[str, list] = defaultdict(list)

    # Also track: timestep-0 probes (first frame, placeholder queries)
    t0_probes_list = []

    # Track per-episode, per-step data for a detailed plot
    all_ep_data = []  # list of ep_data; ep_data = list of (step, stage_dict)

    for ep in range(N_EPISODES):
        frame_np = env.reset()
        h_t = None
        ep_data = []

        for t in range(MAX_STEPS_PER_EP):
            frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)

            if h_t is None:
                queries = encoder.perceiver.get_initial_queries(1, device)
            else:
                queries = h_t.detach()

            probes = instr.forward_with_probes(frame_t, queries)
            stages = probe_stages_cossim(probes, cfg.n_latents, cfg.n_perceiver_rounds)

            for k, v in stages.items():
                stage_cossims[k].append(v)
            ep_data.append((t, stages, probes))

            if t == 0:
                t0_probes_list.append(probes)

            # Advance
            with torch.no_grad():
                h_next, _, _ = encoder(frame_t, queries)
            h_t = h_next

            action_idx = np.random.randint(0, cfg.n_actions)
            next_np, is_terminal = env.step(action_idx)
            life_end = is_end_of_life(frame_np, next_np, is_terminal)

            if life_end:
                break
            frame_np = next_np

        all_ep_data.append(ep_data)
        print(f"[ep {ep+1}] {t+1} steps collected")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("AVERAGE PAIRWISE COSINE SIMILARITY AT EACH PERCEIVER STAGE")
    print("(averaged over all timesteps across all episodes)")
    print("="*70)
    stage_order = sorted(stage_cossims.keys())
    for k in stage_order:
        vals = stage_cossims[k]
        label = k.split("_", 1)[1]  # strip numeric prefix
        print(f"  {label:35s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # ── Print t=0 detail ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TIMESTEP 0 (placeholder queries) — DETAILED PAIRWISE")
    print("="*70)
    p0 = t0_probes_list[0]
    for r in range(cfg.n_perceiver_rounds):
        for tag in [f"after_cross_r{r}", f"after_self_r{r}"]:
            h = p0[tag]
            pairs = all_pairwise_cossim(h)
            print(f"\n  Stage: {tag}")
            for i, j, sim in pairs:
                print(f"    cos(L{i}, L{j}) = {sim:.4f}")

    print("\n  Stage: output")
    for i, j, sim in all_pairwise_cossim(p0["output"]):
        print(f"    cos(L{i}, L{j}) = {sim:.4f}")

    # ── Print cross-attention patterns at t=0 ────────────────────────────────
    print("\n" + "="*70)
    print("CROSS-ATTENTION WEIGHTS AT T=0 (which patches does each latent attend to?)")
    print("="*70)
    for r in range(cfg.n_perceiver_rounds):
        w = p0[f"cross_attn_w_r{r}"]  # (n_heads, n_latents, 16)
        w_avg = w.mean(0)              # (n_latents, 16) — averaged over heads
        print(f"\n  Round {r} — head-averaged cross-attn (rows=latents, cols=patches):")
        print(f"  {'':12s}" + "".join(f"P{p:2d} " for p in range(16)))
        for i in range(cfg.n_latents):
            row = "  ".join(f"{v:.2f}" for v in w_avg[i].tolist())
            print(f"    Latent {i}: {row}")
        # Entropy of each latent's attention distribution
        print(f"\n  Cross-attn entropy per latent (max = log(16) = {np.log(16):.2f}):")
        for i in range(cfg.n_latents):
            p_avg = w_avg[i]
            ent = -(p_avg * (p_avg + 1e-9).log()).sum().item()
            print(f"    Latent {i}: H={ent:.3f}")
        # How similar are the attention maps between different latents?
        print(f"\n  Cosine similarity of cross-attn distributions (w_avg rows):")
        for i in range(cfg.n_latents):
            for j in range(i + 1, cfg.n_latents):
                sim = F.cosine_similarity(w_avg[i].unsqueeze(0), w_avg[j].unsqueeze(0)).item()
                print(f"    cos(L{i}_attn, L{j}_attn) = {sim:.4f}")

    # ── Print self-attention weights at t=0 ──────────────────────────────────
    print("\n" + "="*70)
    print("SELF-ATTENTION WEIGHTS AMONG LATENTS AT T=0")
    print("(rows=query latent, cols=key latent; uniform → mean-pooling)")
    print("="*70)
    for r in range(cfg.n_perceiver_rounds):
        w = p0[f"self_attn_w_r{r}"]   # (n_heads, n_latents, n_latents)
        w_avg = w.mean(0)              # (n_latents, n_latents)
        print(f"\n  Round {r} — head-averaged self-attn (4×4 matrix):")
        for i in range(cfg.n_latents):
            row = "  ".join(f"{v:.3f}" for v in w_avg[i].tolist())
            print(f"    Latent {i}: [{row}]")
        # Entropy: uniform = log(4) ≈ 1.386; peaked = 0
        print(f"  Self-attn entropy (max=log(4)={np.log(4):.3f}, uniform → mean-pool):")
        for i in range(cfg.n_latents):
            p_i = w_avg[i]
            ent = -(p_i * (p_i + 1e-9).log()).sum().item()
            print(f"    Latent {i} query: H={ent:.3f}")

    # ── FIGURES ───────────────────────────────────────────────────────────────
    stage_labels = [k.split("_", 1)[1] for k in stage_order]
    stage_means  = [float(np.mean(stage_cossims[k])) for k in stage_order]

    # Figure 1: bar chart of cosine similarity across Perceiver pipeline stages
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(stage_labels))
    ax.bar(x, stage_means, color="steelblue", alpha=0.8)
    ax.axhline(1.0, color="red",  linewidth=1, linestyle="--", label="collapse (=1)")
    ax.axhline(0.0, color="gray", linewidth=1, linestyle="--", label="orthogonal (=0)")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=25, ha="right")
    ax.set_ylabel("Avg pairwise cosine similarity")
    ax.set_title("Exp-003: Where do the 4 latent tokens converge?\n"
                 "Avg pairwise cos-sim at each Perceiver stage (all timesteps, all episodes)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "stage_cossim_bar.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'stage_cossim_bar.png'}")

    # Figure 2: cross-attention heatmap per round at t=0
    fig, axes = plt.subplots(1, cfg.n_perceiver_rounds, figsize=(14, 4))
    if cfg.n_perceiver_rounds == 1:
        axes = [axes]
    for r, ax in enumerate(axes):
        w = p0[f"cross_attn_w_r{r}"].mean(0).numpy()  # (n_latents, 16)
        im = ax.imshow(w, vmin=0, vmax=w.max(), cmap="Blues", aspect="auto")
        ax.set_xlabel("Patch index (0-15)")
        ax.set_ylabel("Latent index")
        ax.set_title(f"Cross-attn weights (round {r})\nT=0, head-averaged")
        ax.set_yticks(range(cfg.n_latents))
        ax.set_yticklabels([f"L{i}" for i in range(cfg.n_latents)])
        ax.set_xticks(range(16))
        plt.colorbar(im, ax=ax, fraction=0.03)
    plt.suptitle("Cross-attention: which patches does each latent attend to?\n"
                 "Uniform rows → all latents read the same information → collapse",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cross_attn_heatmap_t0.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'cross_attn_heatmap_t0.png'}")

    # Figure 3: self-attention heatmap per round at t=0
    fig, axes = plt.subplots(1, cfg.n_perceiver_rounds, figsize=(10, 4))
    if cfg.n_perceiver_rounds == 1:
        axes = [axes]
    for r, ax in enumerate(axes):
        w = p0[f"self_attn_w_r{r}"].mean(0).numpy()  # (n_latents, n_latents)
        im = ax.imshow(w, vmin=0, vmax=1, cmap="Oranges", aspect="auto")
        ax.set_xlabel("Key latent")
        ax.set_ylabel("Query latent")
        ax.set_title(f"Self-attn weights (round {r})\nT=0, head-averaged")
        ax.set_xticks(range(cfg.n_latents))
        ax.set_xticklabels([f"L{i}" for i in range(cfg.n_latents)])
        ax.set_yticks(range(cfg.n_latents))
        ax.set_yticklabels([f"L{i}" for i in range(cfg.n_latents)])
        plt.colorbar(im, ax=ax, fraction=0.06)
        # Annotate values
        for ii in range(cfg.n_latents):
            for jj in range(cfg.n_latents):
                ax.text(jj, ii, f"{w[ii,jj]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
        # Entropy annotation
        for ii in range(cfg.n_latents):
            p_i = torch.tensor(w[ii])
            ent = -(p_i * (p_i + 1e-9).log()).sum().item()
            ax.text(cfg.n_latents - 0.45, ii, f"H={ent:.2f}", ha="left", va="center",
                    fontsize=7, color="dimgray")
    plt.suptitle(f"Self-attention among latents (T=0)\nUniform matrix (all 0.25) = pure mean-pooling → kills diversity\n"
                 f"max entropy = log(4)={np.log(4):.2f}", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "self_attn_heatmap_t0.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'self_attn_heatmap_t0.png'}")

    # Figure 4: cosine similarity over time (step index) for episode 0 — one curve per stage
    ep0 = all_ep_data[0]
    t_vals = [t for t, _, _ in ep0]
    fig, ax = plt.subplots(figsize=(12, 4))
    for k in stage_order:
        label = k.split("_", 1)[1]
        vals = [ep_stage[k] for _, ep_stage, _ in ep0]
        ax.plot(t_vals, vals, marker=".", markersize=4, label=label, alpha=0.85)
    ax.axhline(1.0, color="red",  linewidth=0.8, linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Timestep within episode")
    ax.set_ylabel("Avg pairwise cosine similarity")
    ax.set_title("Exp-003 episode 0: cos-sim between the 4 latents at each Perceiver stage\n"
                 "(t=0 uses placeholder queries; t>0 uses h_{t-1})")
    ax.legend(loc="right", fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cossim_over_time_ep0.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'cossim_over_time_ep0.png'}")

    print("\nDone. All figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
