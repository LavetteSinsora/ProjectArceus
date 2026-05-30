# exp_010_0 — CNN + PPO baseline on the real LS20 game

> The "7_0 recipe, applied to LS20." A deliberately naive vanilla CNN + PPO
> agent with **terminal-only** reward, run on the real 64×64 ARC-AGI-3 LS20
> game instead of the 32×32 mini-env. This is the zero against which the two
> JEPA-encoder variants (10_1, 10_2) are measured.
>
> Parent: [../SYSTEM_CARD.md](../SYSTEM_CARD.md).

---

## 1. What this is

Same model family, optimiser, and PPO hyperparameters as
[exp_007_0_naive](../../exp_007_mini_env_cnn_ppo_baseline/exp_007_0_naive/),
with two forced changes for the real env (see parent §1):

1. **64×64 input** → one extra stride-2 conv so the flattened map stays 8×8.
2. **Terminal-only reward** is the only option — the real LS20 game exposes no
   player-rotation accessor, so exp_007_2's rotation-match shaping cannot be
   computed. `r_t = +1` iff the step cleared the level, else `0` (including
   running out of the game's step budget and our `max_episode_steps`
   truncation).

No JEPA, no auxiliary loss, no frame stacking, no exploration bonus. Just a CNN
encoder shared by a policy head and a value head, trained by clipped-surrogate
PPO.

## 2. Hypothesis

Real LS20 Level 1 under *sparse* reward is plausibly much harder than the
mini-env was: a 64×64 grid roughly quadruples the random-walk hitting time
versus 32×32, and there is no shaping to convert the puzzle into a two-phase
reward landscape. Expectation: `success_rate` is slow to depart from chance,
and may not within a few million env steps. If it solves quickly, real LS20
Level 1 is easier under sparse reward than the mini-env analysis suggested —
itself an interesting result.

## 3. Config

`config.py`: `jepa_mode="none"`, `total_env_steps=3_000_000`, 8 envs ×
128 rollout steps (1024 transitions/update), `max_episode_steps=200`. All PPO
knobs inherited from `shared/config_base.py` (γ 0.99, λ 0.95, clip 0.2,
value clip 0.2, c_v 0.5, c_ent 0.01, Adam 3e-4, 4 epochs × 4 minibatches).

## 4. Metrics

Headline: `success_rate` (periodic stochastic eval). Health: `policy_entropy`,
`approx_kl`, `clipfrac`, `grad_norm_total`, `mean_feature_cosine`. See parent §6.

## 5. How to run

```bash
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_0_cnn_ppo_baseline.train
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_0_cnn_ppo_baseline.train --smoke
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_0_cnn_ppo_baseline.eval \
    --checkpoint <path-to>/checkpoints/step_XXXXXXXX.pt
```

Checkpoints → `checkpoints/step_*.pt`; metrics → `runs/<run>/metrics.jsonl`.
Both are picked up by the main dashboard (port 8787).
