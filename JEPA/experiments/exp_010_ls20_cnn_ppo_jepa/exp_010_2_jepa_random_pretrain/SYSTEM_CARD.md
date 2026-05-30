# exp_010_2 — Random-policy JEPA pretraining → unfrozen PPO on real LS20

> The **random-policy** JEPA variant. Collect `(s,a,s')` transitions with a
> uniform-random agent, train a JEPA encoder (+predictor+IDM) on that fixed
> buffer until the JEPA loss plateaus, then use the encoder as the initial
> weight for PPO and fine-tune it (**unfrozen**). Mirrors exp_008_2/008_4 on
> mini-env, now on the real 64×64 LS20 game.
>
> Parent: [../SYSTEM_CARD.md](../SYSTEM_CARD.md).
> Sibling: [exp_010_1_jepa_joint_online](../exp_010_1_jepa_joint_online/)
> (the on-policy counterpart).

---

## 1. Pipeline

Three stages, each its own entry point:

1. **`collect.py`** — `n_envs` uniform-random agents in the real LS20 env drain
   `(s_t, a_t, s_{t+1})` tuples into `data/random_buffer.npz` until
   `n_random_transitions` (default 500k) are stored. Episode-ending steps are
   skipped (their `s'` is a reset, not a real transition — same rule as
   exp_008). The collector records `env_steps_used` in `data/random_buffer.meta.json`
   — **the number of environment steps the data cost, which the spec asks us to
   report for this variant.**

2. **`train_jepa.py`** — train encoder + `ActionConditionedPredictor` + `IDM`
   on the buffer (90/10 train/val split) with loss
   `jepa_coef·MSE(predictor(h_t,a), sg(h_{t+1})) + idm_coef·CE(idm(h_t,h_{t+1}),a)`.
   **Plateau rule:** stop when held-out `val_jepa_loss` fails to improve by
   more than `pretrain_min_delta` (relative) for `pretrain_patience` epochs, or
   at `pretrain_max_epochs`. The best-val encoder bundle is saved to
   `jepa_pretrained/encoder_final.pt`. Loss curves → `runs/jepa_pretrain_*/metrics.jsonl`.

3. **`train_ppo.py`** — build a fresh `ActorCritic`, load `encoder_final.pt`
   into its encoder, leave `requires_grad=True` (unfrozen; `--freeze` flips it
   for a frozen ablation), and run the **identical** PPO recipe as exp_010_0.
   The encoder and heads share one Adam optimiser; only the *initial encoder
   weights* differ from the baseline.

## 2. Hypothesis

Random-data JEPA encodes the env's generic geometry (edges, the player sprite,
walls) without any reward bias and — unlike the on-policy sibling — without
collapsing onto a narrow trajectory tube. On mini-env, exp_008_2 found the
*frozen* random-data encoder was a *worse* warm start than learning the CNN
from scratch, while exp_008_4 asked whether *unfreezing* recovers it. exp_010_2
runs the unfrozen test on the real game: does a random-data encoder init beat,
tie, or lose to the from-scratch 10_0 baseline on `success_rate`-vs-`env_step`?

## 3. Config

`config.py`: `jepa_mode="none"` (PPO phase has no online JEPA — the encoder is
only *initialised* from the pretrained weights), `freeze_encoder=False`,
`n_random_transitions=500_000`, `pretrain_max_epochs=40`, `pretrain_patience=4`,
`pretrain_min_delta=0.002`, `total_env_steps=3_000_000`. `init_encoder_ckpt` is
resolved at runtime by `train_ppo.py` to the latest `encoder_final.pt`.

## 4. Metrics

- **Pretraining:** `train/val_jepa_loss`, `train/val_idm_loss`, `idm_acc`, and
  the reported `env_steps_used` for the random data.
- **PPO:** the baseline metric set (parent §6). A startup parameter signature
  is loaded; the encoder is expected to move under PPO (unfrozen) — if it
  doesn't, the "unfrozen" claim is hollow (cf. exp_008_4 §4.3).

## 5. How to run

```bash
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_2_jepa_random_pretrain.collect
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_2_jepa_random_pretrain.train_jepa
uv run python -m JEPA.experiments.exp_010_ls20_cnn_ppo_jepa.exp_010_2_jepa_random_pretrain.train_ppo
# frozen ablation:
uv run python -m ....exp_010_2_jepa_random_pretrain.train_ppo --freeze
# everything tiny, end-to-end:
uv run python -m ....exp_010_2_jepa_random_pretrain.collect --smoke
uv run python -m ....exp_010_2_jepa_random_pretrain.train_jepa --smoke
uv run python -m ....exp_010_2_jepa_random_pretrain.train_ppo --smoke
```

## 6. Caveats

- **Uniform-random under-samples deep states.** A random agent rarely reaches
  states a good policy would; the pretrained encoder may be uninformative
  exactly where it matters most. This is the standard limitation of the
  random-data baseline (exp_008_2 §8) — a curiosity-driven collector is the
  natural follow-up.
- **Plateau ≠ good.** A flat `val_jepa_loss` only means training converged on
  *this* buffer; it does not certify the features are useful for control. The
  downstream PPO curve is the actual test.
