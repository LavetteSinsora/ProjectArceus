"""
Eval pass — N-episode sampling rollouts collecting metrics from
metrics.md §1.2, 1.3, 1.4@eval, 1.5, 2.1, 2.2, 3.1, 4.3@eval, 6.2, 6.3.

Cadence: every cfg.eval_freq env steps. Episodes run until is_end_of_life
fires. The policy is sampled (not greedy) so the N trajectories are
decorrelated despite LS20 being deterministic.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from JEPA.experiments.exp_003_2_action_pred_no_ema.reward_shaping import is_end_of_life

from .attention import (
    pairwise_jsd,
    patch_sa_row_jsd,
    patch_sa_temporal_jsd,
    latent_self_attn_row_jsd_per_round,
    read_sa_attn,
    read_latent_self_attn,
)
from .exploration import ExplorationTracker
from .representation import latent_pairwise_cossim, ht_htp1_cossim


def _set_eval_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_or_nan(xs):
    return float(np.mean(xs)) if xs else float("nan")


@torch.no_grad()
def run_eval(
    encoder,
    state_predictor,
    action_predictor,
    action_embed,
    policy,
    env,
    cfg,
    step: int,
    device: torch.device,
) -> dict:
    """
    Run cfg.n_eval_episodes episodes under eval-mode encoder/policy/predictors;
    return a dict of section-tagged scalars.
    """
    encoder.eval(); state_predictor.eval(); action_predictor.eval()
    action_embed.eval(); policy.eval()

    # Register a forward hook that captures the round-0 latent SA *output*
    # (post-SA latents) — distinct from the cached attention matrix.
    post_sa_cap = {"out": None}

    def _hook(_mod, _inp, out):
        post_sa_cap["out"] = out.detach()
    h_handle = encoder.perceiver.rounds[0].self_attn.register_forward_hook(_hook)

    # Buckets across all episodes
    pair_t1, pair_t10, pair_t20 = [], [], []
    pair_postSA_t1 = []
    ht_htp1_eval = []
    H1_HT = []
    T_episode_list = []
    sa_row_jsd_vals = []
    sa_attn_per_step_all_eps = []   # for temporal JSD; per-step list of per-block tensors
    perc_self_jsd_per_round = [[] for _ in range(cfg.n_perceiver_rounds)]
    action_entropy_eval_vals = []
    coverage_pct_vals = []
    cross_hits_vals = []

    exploration = ExplorationTracker(cfg.game_id)

    try:
        for ep in range(cfg.n_eval_episodes):
            _set_eval_seeds(cfg.seed + step + ep)
            frame_np = env.reset()
            exploration.reset()
            h_t = None
            H_1 = None
            H_T = None
            H_t_list = []
            sa_attn_per_step = []
            T = 0

            while True:
                frame_t = torch.from_numpy(frame_np).unsqueeze(0).to(device)
                queries = (encoder.perceiver.get_initial_queries(1, device)
                           if h_t is None else h_t)
                h_current, _, _ = encoder(frame_t, queries)
                H_t_list.append(h_current.squeeze(0))    # (L, D)

                # Capture sec1.3 (post-SA latents) at t=1 only
                if T == 0:
                    post_sa = post_sa_cap["out"]
                    if post_sa is not None:
                        pair_postSA_t1.append(latent_pairwise_cossim(post_sa[0]))

                # Capture SA-block attention per step for sec2.1 / sec2.2
                step_sa = [read_sa_attn(b) for b in encoder.sa_blocks]
                # Clone tensors so they survive the next forward overwrite
                step_sa = [a.clone() if a is not None else None for a in step_sa]
                sa_attn_per_step.append(step_sa)

                # Capture perceiver latent SA at each step → sec3.1
                perc_self = [read_latent_self_attn(r) for r in encoder.perceiver.rounds]
                jsds = latent_self_attn_row_jsd_per_round(perc_self)
                for r_idx, v in enumerate(jsds):
                    if r_idx < len(perc_self_jsd_per_round) and np.isfinite(v):
                        perc_self_jsd_per_round[r_idx].append(v)

                # Sample action from policy (mask available)
                avail = env.available_actions
                action_idx, _, _ = policy.act(h_current.squeeze(0), avail)
                exploration.step(frame_np)

                # Step env
                next_np, is_terminal = env.step(action_idx)
                eol = is_end_of_life(frame_np, next_np, is_terminal)

                # Compute h_next for action-predictor entropy + ht/htp1 cossim
                next_t = torch.from_numpy(next_np).unsqueeze(0).to(device)
                h_next, _, _ = encoder(next_t, h_current)
                ht_htp1_eval.append(
                    ht_htp1_cossim(h_current.squeeze(0), h_next.squeeze(0))
                )
                logits_a = action_predictor(h_current, h_next)
                # Mask to available actions before computing entropy
                mask = torch.full_like(logits_a, float("-inf"))
                for a in avail:
                    idx = int(a) - 1
                    if 0 <= idx < cfg.n_actions:
                        mask[0, idx] = 0.0
                log_p = F.log_softmax(logits_a + mask, dim=-1)
                p = log_p.exp()
                action_entropy_eval_vals.append(
                    float(-(p * log_p).nan_to_num(0.0).sum(-1).item())
                )

                T += 1
                if H_1 is None:
                    H_1 = h_current.squeeze(0).clone()
                H_T = h_current.squeeze(0).clone()

                h_t = h_current
                frame_np = next_np
                if eol:
                    break

            T_episode_list.append(T)
            exploration.step(frame_np)  # final frame

            # sec1.2 — pairwise cossim of H_t at fixed t indices
            if len(H_t_list) >= 1:
                pair_t1.append(latent_pairwise_cossim(H_t_list[0]))
            if len(H_t_list) >= 10:
                pair_t10.append(latent_pairwise_cossim(H_t_list[9]))
            if len(H_t_list) >= 20:
                pair_t20.append(latent_pairwise_cossim(H_t_list[19]))

            # sec1.5 — H_1 vs H_T
            if H_1 is not None and H_T is not None:
                H1_HT.append(
                    float(F.cosine_similarity(H_1, H_T, dim=-1).mean().item())
                )

            # sec2.1 — patch SA row JSD averaged over time per episode
            for step_sa in sa_attn_per_step:
                v = patch_sa_row_jsd(step_sa)
                if np.isfinite(v):
                    sa_row_jsd_vals.append(v)
            sa_attn_per_step_all_eps.append(sa_attn_per_step)

            # Exploration
            cov = exploration.coverage_pct()
            ch = exploration.cross_hits()
            if isinstance(cov, float) and np.isfinite(cov):
                coverage_pct_vals.append(cov)
            if isinstance(ch, (int, float)) and np.isfinite(float(ch)):
                cross_hits_vals.append(float(ch))

    finally:
        h_handle.remove()
        encoder.train(); state_predictor.train(); action_predictor.train()
        action_embed.train(); policy.train()

    # sec2.2 — temporal JSD per episode, then averaged
    sa_temporal = []
    for per_step in sa_attn_per_step_all_eps:
        v = patch_sa_temporal_jsd(per_step)
        if np.isfinite(v):
            sa_temporal.append(v)

    return {
        "sec1/latent_pairwise_cossim_t1":   _mean_or_nan(pair_t1),
        "sec1/latent_pairwise_cossim_t10":  _mean_or_nan(pair_t10),
        "sec1/latent_pairwise_cossim_t20":  _mean_or_nan(pair_t20),
        "sec1/round0_postSA_pairwise_cossim_t1": _mean_or_nan(pair_postSA_t1),
        "sec1/ht_htp1_cossim_eval":         _mean_or_nan(ht_htp1_eval),
        "sec1/H1_HT_cossim":                _mean_or_nan(H1_HT),
        "sec1/T_episode_eval":              _mean_or_nan(T_episode_list),
        "sec2/patch_sa_row_jsd":            _mean_or_nan(sa_row_jsd_vals),
        "sec2/patch_sa_temporal_jsd":       _mean_or_nan(sa_temporal),
        **{f"sec3/latent_self_attn_row_jsd_r{r}": _mean_or_nan(perc_self_jsd_per_round[r])
           for r in range(cfg.n_perceiver_rounds)},
        "sec4/action_pred_entropy_eval":    _mean_or_nan(action_entropy_eval_vals),
        "sec6/reachable_tile_coverage_pct": _mean_or_nan(coverage_pct_vals),
        "sec6/cross_hits_per_episode":      _mean_or_nan(cross_hits_vals),
    }
