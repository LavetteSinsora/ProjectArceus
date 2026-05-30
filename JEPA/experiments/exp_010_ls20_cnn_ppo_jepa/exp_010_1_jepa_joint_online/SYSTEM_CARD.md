# exp_010_1 — Joint online JEPA + PPO on real LS20

> The **on-policy** JEPA variant. Random-init encoder *and* policy. Run PPO
> from scratch, and on every update *also* train the shared encoder with a
> JEPA(+IDM) loss on the data the current policy just collected. The encoder
> is shaped by the policy gradient and the world-model objective at the same
> time — no separate pretraining phase.
>
> Parent: [../SYSTEM_CARD.md](../SYSTEM_CARD.md).
> Sibling: [exp_010_2_jepa_random_pretrain](../exp_010_2_jepa_random_pretrain/)
> (the off-policy / random-data counterpart).

---

## 1. What "on-policy JEPA" means here

This is the spec's first 10_1 variant, clarified: **we do not collect a
separate dataset.** Encoder and policy both start from random init. Each PPO
iteration:

1. Collect a rollout with the current policy (stochastic actions).
2. Run the standard PPO update — the encoder gets the policy/value gradient.
3. Form `(s_t, a_t, s_{t+1})` pairs **from that same rollout** (dropping
   episode-ending steps, whose `s_{t+1}` is a reset) and take a JEPA(+IDM)
   gradient step on encoder + predictor + IDM:
   `L = jepa_coef · MSE(predictor(h_t,a), sg(h_{t+1})) + idm_coef · CE(idm(h_t,h_{t+1}), a)`.

So the encoder's world-model training distribution is exactly **whatever the
PPO agent visits** — it co-evolves with the policy. This is the exp_007_3 /
exp_007_4 idea (CNN+PPO with a stop-gradient JEPA + IDM aux loss) carried to
the real 64×64 LS20 game.

## 2. Hypothesis

The JEPA + IDM auxiliary loss gives the encoder a denser learning signal than
the sparse terminal reward alone, which should (a) keep the representation from
collapsing early and (b) *possibly* speed up policy learning vs the 10_0
baseline. The honest open question is whether (b) materialises on real LS20 or
whether the aux loss is merely inert (or, in the failure case, fights the
policy gradient for encoder capacity). The collapse diagnostics
(`mean_feature_cosine`, `feat_effective_rank`) and the `jepa_loss` /`idm_acc`
traces are there to tell which.

## 3. Config

`config.py`: `jepa_mode="online"`, `jepa_coef=1.0`, `idm_coef=1.0`,
`jepa_epochs=1` (one JEPA pass over each rollout), `total_env_steps=3_000_000`.
Everything else inherited (parent §4–5). Encoder, predictor, and IDM share one
Adam optimiser so the encoder receives both gradients.

## 4. Metrics

Baseline metrics (parent §6) **plus** per-update `jepa_loss`, `idm_loss`,
`idm_acc`. The comparison of interest is `success_rate`-vs-`env_step` against
exp_010_0, with the collapse diagnostics as the mechanism explanation.

## 5. How to run

```bash
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_1_jepa_joint_online.train
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_1_jepa_joint_online.train --smoke
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_1_jepa_joint_online.eval --checkpoint <ckpt>
```

## 6. Caveats

- The JEPA target is the encoder's *own* next-state encoding with a stop-
  gradient (no EMA target). On a collapsing encoder this can be self-reinforcing;
  the IDM term and the collapse diagnostics are the guard against reading a
  collapsed encoder as "healthy."
- On-policy data inherits the policy's coverage: once PPO becomes directed, the
  encoder only ever world-models a narrow trajectory tube (the exact effect
  exp_008_1 measured on mini-env). That is a feature of this variant, not a bug
  — it is precisely what distinguishes it from the random-data sibling 10_2.
