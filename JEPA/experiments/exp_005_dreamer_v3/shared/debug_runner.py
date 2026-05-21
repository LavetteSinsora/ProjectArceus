"""Smoke test for Dreamer V3.

Builds every module on synthetic data, prints shapes, runs a single
forward+backward, and measures per-step latency so we can extrapolate
training time before committing to a long run.

Usage:
    uv run python -m JEPA.experiments.exp_005_dreamer_v3.shared.debug_runner
"""

from __future__ import annotations

import time

import torch

from .config_base import ConfigBase
from .device import resolve_device
from .models import load_models
from .models.functional import (
    PercentileReturnScale,
    lambda_returns,
    make_twohot_bins,
    symexp,
    symlog,
    twohot_decode,
    twohot_encode,
)


def main():
    cfg = ConfigBase(device="auto")
    device = resolve_device(cfg.device)
    print(f"[debug] device={device}")
    print(f"[debug] cfg.deter={cfg.deter} stoch=({cfg.n_groups}x{cfg.n_classes}) actions={cfg.n_actions}")

    # ── 1. functional unit tests ────────────────────────────────────────────
    x = torch.linspace(-30, 30, 61, device=device)
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4), "symlog/symexp roundtrip failed"
    bins = make_twohot_bins(-20.0, 20.0, 255, device=device)
    soft = twohot_encode(symlog(x.clamp(-20, 20)), bins)
    assert soft.shape == (61, 255)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(61, device=device), atol=1e-5), "twohot does not sum to 1"
    recon = symexp(twohot_decode(soft, bins))
    err = (recon - x.clamp(-20, 20)).abs().max().item()
    assert err < 0.5, f"twohot roundtrip error too high: {err}"
    print(f"[debug] symlog/twohot roundtrip OK (max err {err:.4f})")

    # ── 2. Build models ─────────────────────────────────────────────────────
    wm, actor, critic, critic_ema, actor_p2e, critic_p2e, critic_p2e_ema = load_models(cfg, device)
    critic_ema.to(device); critic_p2e_ema.to(device)
    n_params_wm = sum(p.numel() for p in wm.parameters())
    n_params_actor = sum(p.numel() for p in actor.parameters())
    n_params_critic = sum(p.numel() for p in critic.parameters())
    print(f"[debug] params: wm={n_params_wm/1e6:.2f}M actor={n_params_actor/1e6:.2f}M critic={n_params_critic/1e6:.2f}M")

    # ── 3. Forward observe() ───────────────────────────────────────────────
    B, T = cfg.batch_size, cfg.batch_length
    obs = torch.rand(B, T, cfg.obs_channels, cfg.obs_size, cfg.obs_size, device=device) - 0.5
    actions_idx = torch.randint(0, cfg.n_actions, (B, T), device=device)
    actions_oh = torch.nn.functional.one_hot(actions_idx, num_classes=cfg.n_actions).float()
    rewards = torch.zeros(B, T, device=device)
    rewards[0, -1] = 1.0
    conts = torch.ones(B, T, device=device); conts[0, -1] = 0.0

    out = wm.observe(obs, actions_oh)
    print(f"[debug] post.h {tuple(out.post.h.shape)}  post.z {tuple(out.post.z.shape)}  post.logits {tuple(out.post.logits.shape)}")
    print(f"[debug] features {tuple(out.features.shape)}")
    assert out.post.h.shape == (B, T, cfg.deter)
    assert out.post.z.shape == (B, T, cfg.n_groups, cfg.n_classes)

    # ── 4. WM losses ───────────────────────────────────────────────────────
    obs_flat = obs.reshape(B * T, *obs.shape[2:])
    L_pred = -(out.recon_dist.log_prob(obs_flat).mean()
               + out.reward_dist.log_prob(rewards.reshape(-1)).mean()
               + out.continue_dist.log_prob(conts.reshape(-1)).mean())
    L_dyn, L_rep = wm.rssm.kl_loss(out.post, out.prior, free_nats=cfg.free_nats)
    L_wm = cfg.beta_pred * L_pred + cfg.beta_dyn * L_dyn + cfg.beta_rep * L_rep
    print(f"[debug] L_pred {L_pred.item():.3f}  L_dyn {L_dyn.item():.3f}  L_rep {L_rep.item():.3f}  L_wm {L_wm.item():.3f}")
    L_wm.backward()
    # Ensure gradients flow into the GRU and the encoder
    enc_grad = wm.encoder.proj.weight.grad
    gru_grad = wm.rssm.gru.weight_ih.grad
    assert enc_grad is not None and enc_grad.abs().sum().item() > 0, "no grad in encoder"
    assert gru_grad is not None and gru_grad.abs().sum().item() > 0, "no grad in RSSM GRU"
    print(f"[debug] encoder/gru gradients OK")

    # ── 5. Imagine() ───────────────────────────────────────────────────────
    start_h = out.post.h.reshape(B * T, -1).detach()
    start_z = out.post.z.reshape(B * T, cfg.n_groups, cfg.n_classes).detach()
    H = cfg.imag_horizon
    imag = wm.imagine(start_h, start_z, actor, horizon=H)
    print(f"[debug] imag.features {tuple(imag.features.shape)}  imag.actions {tuple(imag.actions.shape)}")
    assert imag.features.shape == (H, B * T, cfg.deter + cfg.n_groups * cfg.n_classes)
    assert imag.actions.shape == (H, B * T, cfg.n_actions)

    # Lambda returns
    rew = imag.reward_dist.mean().reshape(H, -1)
    cont = imag.continue_dist.mean().reshape(H, -1)
    v_dist = critic(imag.features.reshape(H * B * T, -1))
    v = v_dist.mean().reshape(H, -1)
    v_ext = torch.cat([v.detach(), v.detach()[-1:]], dim=0)
    R = lambda_returns(rew.detach(), v_ext, cont.detach(), gamma=cfg.gamma, lam=cfg.lam)
    print(f"[debug] λ-returns shape {tuple(R.shape)}, mean {R.mean().item():.4f}")

    # Percentile scale
    ps = PercentileReturnScale()
    s = ps.update(R)
    print(f"[debug] percentile scale {s:.4f} (will be ~0 on random init, max(1, S) used downstream)")

    # ── 6. Latency timing ───────────────────────────────────────────────────
    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    _sync()
    t0 = time.time()
    N_TIMING = 5
    for _ in range(N_TIMING):
        out = wm.observe(obs, actions_oh)
        L_pred = -(out.recon_dist.log_prob(obs_flat).mean()
                   + out.reward_dist.log_prob(rewards.reshape(-1)).mean()
                   + out.continue_dist.log_prob(conts.reshape(-1)).mean())
        L_dyn, L_rep = wm.rssm.kl_loss(out.post, out.prior, free_nats=cfg.free_nats)
        L_wm = cfg.beta_pred * L_pred + cfg.beta_dyn * L_dyn + cfg.beta_rep * L_rep
        for p in wm.parameters():
            if p.grad is not None: p.grad = None
        L_wm.backward()

        start_h = out.post.h.reshape(B * T, -1).detach()
        start_z = out.post.z.reshape(B * T, cfg.n_groups, cfg.n_classes).detach()
        imag = wm.imagine(start_h, start_z, actor, horizon=H)
    _sync()
    per_step = (time.time() - t0) / N_TIMING
    print(f"[timing] per gradient-step (WM forward+backward + imagine): {per_step*1000:.1f} ms")
    print(f"[timing] extrapolated 500K env-steps @ train_ratio={cfg.train_ratio}: "
          f"{(cfg.batch_size * cfg.batch_length / cfg.train_ratio):.1f} env steps/update → "
          f"{cfg.max_env_steps * cfg.train_ratio / (cfg.batch_size * cfg.batch_length):.0f} updates → "
          f"{per_step * cfg.max_env_steps * cfg.train_ratio / (cfg.batch_size * cfg.batch_length) / 3600:.2f} h")


if __name__ == "__main__":
    main()
