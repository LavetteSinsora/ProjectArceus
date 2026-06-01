# exp_012_2 — multi-step curiosity probe (cheap pre-build check)

**Decision being made:** the proposed exp_012 modification is to replace ICM's
1-step forward-prediction-error reward with a **k-step discounted open-loop
rollout error** (k=5, γ=0.7), and optionally to train the forward model on
multi-step rollouts too. Before building any of that, this probe answers the one
question that decides whether the idea *can* help on a small deterministic maze:

> As the ICM forward model learns the dynamics and 1-step curiosity dies, does
> the **k-step error decay slower** and keep enough **magnitude** *and*
> **cross-state spread** to still drive exploration — or does it collapse too?

If 5-step error also goes flat, multi-step buys nothing here → pivot to
disagreement (Plan2Explore-style ensemble) or RND. If it stays meaningfully
larger *and* keeps state-to-state variance, the multi-step idea is worth
implementing as exp_012_2's training variant.

This runs entirely on the **unmodified `ICMModule`** (exp_011/shared/icm.py).
No policy retraining, no new model.

## What it does

1. **Replay (the decisive part).** Collect one pool of uniform-random
   transitions on LS20 L1, train a *fresh* `ICMModule` with the existing 1-step
   ICM loss (`losses_on_batch`, β=0.2, lr 1e-3), and every update read out, on a
   held-out window set (envs disjoint from training), the per-horizon h=1..K
   squared error in two modes:
   - **open-loop** (imagination): `phî_{t+h} = f(…f(f(phi_t,a_t),a_{t+1})…)`
   - **teacher-forced**: one `f`-step from the *true* `phi_{t+h-1}`
   plus the discounted-sum reward `Σ_h γ^{h-1}·err_ol[h]`, each horizon's
   cross-state std, and `‖phi‖²`. This reproduces the collapse and shows each
   horizon's *relative* decay.

2. **Snapshot.** Same read-out on the real trained exp_011_0 ICM checkpoints
   (4 steps × 3 seeds) to confirm the replay matches reality at the stages that
   were actually saved.

## Why a replay at all (limitation, stated up front)

exp_011_0 only saved checkpoints every 102.4k steps — *all* of them are well
**after** the ~20-update curiosity collapse. So the trained checkpoints cannot
show the *decay during* collapse; the from-scratch replay is how we see it
cheaply. The replay trains on a fixed random-policy pool (a proxy for the
near-uniform early-training regime where the real collapse happens). On a small
deterministic maze the held-out envs traverse the same state space, so
generalisation error collapses too — which is itself part of the finding.
Definitive per-horizon decay timing would need frequent *early* ICM checkpoints
(a one-line `save_every` knob on a short re-run) — that is the **next** step, not
this one.

## Run

```bash
uv run python -m JEPA.experiments.exp_012_ls20_intrinsic_exploration.exp_012_2_multistep_probe.probe
# knobs: --n_envs --steps --K --gamma --replay_updates --max_windows --skip_snapshot
```

Outputs: `results.json`, `figures/fig1_replay_decay.png` (per-horizon decay +
current-vs-proposed reward), `figures/fig2_replay_spread.png` (cross-state spread
+ magnitude gain), `figures/fig3_snapshot.png` (real checkpoints).

## How to read the verdict

- **magnitude gain** `mean(k-disc)/mean(1-step)`: how much bigger the proposed
  reward is. >1 always (errors compound); not sufficient on its own — a bigger
  but flatter signal is no better for exploration.
- **cross-state CV** `std/mean`: a reward that is large but nearly constant
  across states gives **no exploration gradient**. The multi-step idea only wins
  if k-step CV ≥ 1-step CV *and* stays alive longer.
- **decay slope** (fig1/fig3): does k-step error decay slower than 1-step as the
  model learns?

## Findings (full run: n_envs=16, steps=400, 80 replay updates, 309 held-out windows)

**Verdict: the k-step discounted-error reward, by itself, will NOT fix ICM
exploration on LS20 L1.** Three reasons, all visible in the figures:

1. **k-step is a near-constant ×5 copy of 1-step.** On a log axis the k-step
   discounted reward tracks the 1-step reward with a fixed ~4–5× vertical offset
   throughout training (fig1 right; fig2 right settles at ~4–5×, flat). They rise
   and fall *together*. Since ICM **auto-calibrates η to fix the mean reward**,
   that ×5 magnitude gain is entirely absorbed by recalibration — it changes
   nothing the policy sees.
2. **No gain in cross-state discrimination.** What survives η-renormalisation is
   the *shape across states* (CV = std/mean). Final CV: 1-step **0.773**,
   k-disc **0.698** — k-step is if anything slightly *less* discriminative. A
   reward that doesn't differentiate states better gives no better exploration
   gradient. (per-horizon std, fig2 left, just tracks the mean.)
3. **It collapses in lockstep at the critical moment.** Around update ~13 the
   1-step error nearly dies (mean ~0.09, std ~0.006) — the analog of the real
   agent's ~20-update curiosity death. The k-step error dips *right along with
   it* (fig1), 5× higher in raw terms but renormalised away. Multi-step offers no
   rescue exactly when it's needed.

**Bonus finding (reframes the disease).** Raw forward error does **not** stay
collapsed: after the update-13 trough it rebounds to O(5–18) and stabilises
(fig1) — because the encoder φ is a *non-stationary target* driven by the inverse
loss (echoes the JEPA forward-target collapse note). The real trained checkpoints
confirm it independently: raw 1-step error is ~1.3–15 at every saved step, never
near zero. So the real "r^i → 1e-5 collapse" was **η-scaling + on-policy
distribution narrowing into a learned region**, *not* the forward model becoming
globally accurate. The horizon is not the disease; **prediction-error magnitude
is roughly state-independent and is the wrong novelty signal here.**

**Robustness.** The replay's mid-run rebound is partly an artifact of the fixed
pool + joint inverse/forward training (moving φ); the update-13 trough is the
cleanest analog of the real collapse. But the two load-bearing conclusions — the
constant-multiple relationship and the unchanged CV — hold in *both* the replay
*and* the independent real checkpoints, so they don't depend on replay dynamics.

**Recommendation.** Skip the multi-step reward variant. The signal must track
*novelty/visitation*, which prediction-error magnitude does not. Pivot to a
quantity that → 0 only where the model is *confident* (learned) and stays high in
genuinely novel regions: **ensemble disagreement over rollouts** (Plan2Explore,
Sekar 2020) or the **RND pseudo-count** (already prototyped in
`exp_012_1_rnd_baseline`). Multi-step *training* (Idea B) was not directly tested,
but is unlikely to change the discrimination property for the same reason.
