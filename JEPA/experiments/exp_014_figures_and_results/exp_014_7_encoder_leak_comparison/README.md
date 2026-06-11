# exp_014_7 — Encoder comparison: where does RND counting work, and where does it leak?

The **encoder-axis** sibling of exp_014_6. exp_014_6 fixed the encoder and swept the
leak μ; here we fix the leak (μ=0 = standard RND by default) and sweep the
**encoder**, running one independent RND loop per encoder over the *same* real
LS20-L2 rollout stream:

| name | representation | trained | dim |
|---|---|---|---|
| `pixel`   | masked board → RND directly (normalised indices, or one-hot via `--pixel-onehot`) | no | 4096 / 65536 |
| `linproj` | frozen random **linear** projection of the one-hot board (the exp_014_1/5/6 RND input) | no | 256 |
| `random`  | frozen random-init **CNN** encoder | no | 256 |
| `idm`     | **our** ICM inverse-dynamics φ, trained **online** | **yes** | 256 |

## The claim

For ARC/LS20, distinct board states are near-identical, so in pixel/linproj/random
space their representations sit almost on top of each other and a single RND
predictor **interpolates**: distilling one state drives down the novelty of others —
*counting one state leaks into counting its neighbours*. The IDM encoder must
separate states (it names the action between them), so each state's novelty tracks
its **own** visit count.

## Leak isolation — the driver/probe split

The monitored set is split into:
- **drivers** — heavily-visited near-reset states; these are distilled, and
- **probes** — sibling near-reset states that are **action-blocked from the start**
  *and excluded from the RND distillation set entirely**.

So a probe is **never directly counted**. Any decay in its novelty is therefore
**pure leak** from the drivers being counted nearby. Pixel/linproj/random → probes
collapse with the drivers; idm → probes stay novel. (Reset states, `first_step==0`,
are never chosen as probes — they can't be blocked.)

## IDM warm-up (so idm enters the probe as a *trained* encoder)

Without warm-up, idm starts from random init and is effectively just another random
encoder for the first many updates — an unfair comparison. So before the probe we
**pre-train φ on `--idm-warmup-episodes` (default 20) episodes of random-policy
data** (inverse+forward, `--idm-warmup-epochs` passes). After warm-up, φ is **frozen
during the probe by default** (`--idm-online` to keep training it). Freezing:
- removes the φ-drift confound (`finding_phi_drift`) so a probe's novelty change is
  attributable to leak, not the ruler moving under it, and
- makes idm a stationary ruler, fair vs the frozen pixel/linproj/random encoders.

The warm-up touches only φ (the other encoders are already frozen); the RND
predictors still start fresh at probe begin. `idm_inv_acc` then reports the (frozen)
warm-up accuracy each update — watch it reach well above chance (0.25 for 4 actions)
for the separation/leak contrast to appear.

## Influence / cross-talk matrix

A per-encoder N×N generalization-leak matrix (a property of the *representation*,
snapshot-independent): for each source i, reset the predictor to its random init, fit
**only** state i, and record the fractional novelty drop on every state j:
`infl[i,j] = 1 − nov_j(fit only i) / nov_j(init)`. Diagonal ≈ 1; off-diagonal = leak.
The headline scalar is **driver→probe leak** = mean `infl[driver, probe]` (0 = no
leak, →1 = the probe is fully counted by proxy). This mirrors the resolution test in
exp_016 `frozen_encoder_resolution.py`. Computed at `--influence-every N` updates (0
= only the final update; it's the heaviest extra). Tune `--influence-steps` /
`--influence-lr` so the source is fit without the single-sample-Adam interference
that masks the leak (cf. `exp_014_5_rnd_forget`).

## Run

```bash
# single run (experiment 1 = scatter; experiment 2 = add --updates-masked)
uv run python -m JEPA.experiments.exp_014_figures_and_results.\
exp_014_7_encoder_leak_comparison.diagnose --updates-free 30 --n-drivers 3 --n-probes 2

# multi-seed sweep + auto-aggregate (RND target + IDM init are seed-dependent)
uv run python -m JEPA.experiments.exp_014_figures_and_results.\
exp_014_7_encoder_leak_comparison.run_sweep --seeds 0 1 2 3 4 --jobs 3 -- \
  --updates-free 30 --influence-every 0

# re-aggregate existing seeds, or re-plot one run
uv run python -m ...exp_014_7_encoder_leak_comparison.aggregate --seeds 0 1 2 3 4
uv run python -m ...exp_014_7_encoder_leak_comparison.plot results/encoder_leak_series_<tag>.npz
```

CPU-only. Heaviest knobs: `--pixel-onehot` (D=65536) and frequent `--influence-every`.
For a publishable result, raise `--updates-free` so the drivers span ~1→10³ visits and
the online φ separates them (watch `inv_acc` climb toward ~1.0), and use ≥5 seeds.

## Figures

Per run (`plot.py`): `encoder_scatter` (novelty vs visit count, log–log, per encoder
+ pooled; o=driver, X=probe), **`encoder_probe_leak`** (probe novelty over updates —
the leak headline), `encoder_geometry` (scale-free chord-L2 + cosine),
**`encoder_influence`** (N×N cross-talk heatmaps), `encoder_novelty_over_updates`.

Aggregated (`aggregate.py`): `agg_geometry`, `agg_probe_leak`, **`agg_leak_bar`**
(driver→probe leak per encoder, median ± IQR across seeds — the one-number leak
comparison), and `encoder_leak_aggregate_<tag>.json`.

## Saved data (`results/encoder_leak_series_<tag>.npz`)

`nov[enc,state,update]`, `cum_visits`, `visits_per_update`, `is_probe`,
`fails_per_update`, `mean_l2`/`mean_cos` + scale-free `mean_l2_unit`/`mean_cos_centered`,
the full 5×5 `pair_l2`/`pair_cos`, `influence[enc,snapshot,i,j]` + `influence_updates`,
`idm_inv_acc`/`idm_fwd_err`, `mon_masked` (the monitored boards), `encoder_names`,
`encoder_dims`, and config. Human-readable `encoder_leak_summary_<tag>.json` alongside.

### Caveats
- **Cross-encoder raw L2 is not comparable** (different dim/norm) — use `mean_l2_unit`
  (chord) for separation. Raw `mean_l2`/`mean_cos` + matrices are saved as requested.
- **Centered cosine is unreliable at small N_MONITOR** (pinned near −1/(n−1)); saved,
  not plotted. Raise `--n-drivers`/`--n-probes` to use it.
- **idm is non-stationary**: its novelty/geometry reflect visitation *and* φ-drift
  (see `finding_phi_drift`); read alongside `idm_inv_acc`.
- **Fully isolating a probe on LS20-L2 is imperfect** via action-masking alone (states
  are highly connected); `fails_per_update` reports any probe entries. The
  exclude-from-distill mechanism makes the leak test robust regardless.
- **Influence depends on `--influence-steps`/`--influence-lr`**: too few/weak → source
  not fit; too many → even a good encoder memorises and "leaks". Tune as in exp_014_5.

## Suggested follow-ups (not yet built)
Count-fidelity scalar (Spearman novelty vs −log visits); leak-vs-distance curve;
encoder × leak-rate grid (does the leak only pay off in a separating encoder?);
a contrasting cell (re86-L1 uncontrollable φ, tu93-L1 solvable).
