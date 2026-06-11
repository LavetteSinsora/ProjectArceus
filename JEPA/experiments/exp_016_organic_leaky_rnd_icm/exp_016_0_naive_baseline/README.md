# exp_016_0 — naive leaky-RND on inverse-dynamics features

The deliberately-minimal exploration baseline: a REINFORCE policy (no value baseline)
driven by a Random-Network-Distillation curiosity bonus computed on features from a
small CNN encoder trained by inverse dynamics. Built to *expose* failure modes, not
hide them. Full design rationale: [SYSTEM_CARD.md](SYSTEM_CARD.md).

## Layout

**Core (the agent + training loop)**
- `config.py` — every hyperparameter (the spec's §4 table).
- `actor.py` — policy-only actor (no value head), timer-masked input.
- `tracker.py` — IDM encoder + inverse head + replay buffer; leaky-RND update (+ the `idm_layernorm` ablation).
- `diagnostics.py` — state registry, probe-state harvest, drift metrics.
- `trainer.py` — the update loop + all per-update logging.

**Entry points**
- `run.py` — train one run (`--game --level --seed --idm-layernorm --no-reward-zscore ...`).
- `train.py` — alias of `run.py` so the shared dashboard discovers the experiment.
- `debug_runner.py` — replays a checkpoint's policy for the dashboard behavior viewer.

**Analysis tools**
- `dashboard.py` — render a per-run multi-panel `dashboard.png` from `metrics.jsonl`.
- `build_synthesis.py` — build `findings/ULTIMATE_CAUSE.html` (the plain-language writeup).
- `probes/` — standalone diagnostics (`frozen_encoder_resolution.py`, `feature_scale_evidence.py`).

**Outputs**
- `runs/<run>/` — `metrics.jsonl`, `state_novelty.jsonl`, `config.json`, `dashboard.png`.
- `checkpoints/<run>/step_*.pt` — policy + tracker snapshots (for the behavior viewer).
- `findings/` — the analysis writeups: `ULTIMATE_CAUSE.html` (start here), the four
  `INVESTIGATION_*.md` reports, and supporting figures.

## Run
```
# train (LS20 Level 1, with checkpoints)
uv run python -m JEPA.experiments.exp_016_organic_leaky_rnd_icm.exp_016_0_naive_baseline.run --game ls20 --level 0 --seed 0

# per-run dashboard figure
uv run python -m JEPA.experiments.exp_016_organic_leaky_rnd_icm.exp_016_0_naive_baseline.dashboard

# rebuild the synthesis HTML
uv run python -m JEPA.experiments.exp_016_organic_leaky_rnd_icm.exp_016_0_naive_baseline.build_synthesis

# interactive behavior viewer + metrics page (shared dashboard server)
uv run python -m JEPA.dashboard.server          # → http://127.0.0.1:8787
```

## Headline findings (see `findings/ULTIMATE_CAUSE.html`)
Three surprising behaviors, three separate causes: the encoder becomes
action-predictive only because of the **timer mask**; the novelty numbers balloon
because the encoder output is **not normalized**; and the policy **freezes on the
small game** because it runs out of new states and the self-referential curiosity
reward turns negative. First reward on LS20-L1 was reached at ~33k env-steps
(random ≈ 50k).
