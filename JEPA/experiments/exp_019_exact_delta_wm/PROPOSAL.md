# exp_019 — Exact sparse-delta world model + in-model Go-Explore

*Autonomous research experiment. Proposal written before code, per repo convention.*

## Where the repo stands (motivation)

Eighteen experiments + the `claude_automate` campaign leave a clear triangle of failure:

1. **Curiosity RL (exp_010–018, leaky RND/ICM):** mechanistically interesting, but 8/12
   game×level cells remain unsolved by *any* intrinsic-reward method. Exploration does not
   align with the reward sequence; entropy collapse governs outcomes.
2. **Go-Explore + recurrent BC (claude_automate):** solves LS20 L1–L4 + TU93 L1 at 100%,
   but it is per-level brute force (up to 321k real env steps for L4) and the distilled
   policy transfers nothing. Every new level pays full price.
3. **Frame-level world model (claude_automate exp 006):** trained on L1–L3, it transfers to
   unseen L4–L6 at ~99% **pixel** accuracy — but **0.0% exact-frame accuracy**. Go-Explore
   needs exact frames (cell hashing + exact replay), so model-based planning was declared
   "not viable" and abandoned.

Failure (3) is the pivotal one. The dynamics *do* transfer; the model is just parameterized
so that it can never get a frame exactly right (~34 wrong pixels/frame, concentrated on the
moving content — precisely what planning needs). If exactness were fixed, (2)'s search could
run inside the model and (2)'s real-step cost would collapse to a verification budget —
giving the few-shot level transfer that nothing in the repo has achieved.

## Hypothesis

**H1 (exactness):** An ARC transition changes a tiny, local set of cells (avatar moves,
timer ticks). A model that predicts the *sparse delta* — per-cell P(change) and
P(new colour | change) — with the unchanged cells copied by construction, can reach
near-100% **exact** next-frame accuracy where the full-frame U-Net got 0%, with no more
data and a smaller network.

**H2 (transfer planning):** With an exact-enough delta model trained only on LS20 L1–L3,
Go-Explore run *inside the model* (1 real reset for the seed frame) plus real-env
verification of candidate trajectories solves held-out L4 in **≪ 321k real env steps** —
the from-scratch Go-Explore cost. Verification failures are not wasted: each divergence
yields a true transition that fine-tunes the model (a deterministic MBRL correction loop).

## Method

### Model — `DeltaWorldModel`
- Input: one-hot frame (16,64,64) + action embedding broadcast.
- Backbone: small conv net (CPU-sized; dilated convs for receptive field, no/light pooling
  to preserve spatial precision).
- Heads (per cell): `change` logit (does this cell change?) and 16-way `colour` logits.
- Loss: BCE on change (positively weighted ~sparse classes) + CE on colour **only at
  changed cells**. Aux heads: terminal.
- Inference: `next = frame.copy(); next[p(change)>τ] = argmax colour`. Copying is exact by
  construction — the model only has to get the (few) changed cells right.

### Evaluation gates (make-or-break, in order)
- **Gate A:** exact next-frame accuracy (UI-masked and unmasked) on held-out L4
  transitions. exp_006 baseline: 0.0%. Need ≳95% to proceed.
- **Gate B:** n-step open-loop rollout exactness on L4 (replay the known 100-action
  solution in the model; report first-divergence step distribution).
- **Gate C:** in-model Go-Explore on L4 → real env steps to first completion, counting
  every real reset/step (seed + verifications + correction data). Compare: 321.6k
  (from-scratch Go-Explore), and report the verification/correction breakdown.

### Data
- Train: L1–L3 transitions — replayed cached solutions (gives the rare completion +
  modifier-tile transitions) + random rollouts + random-burst (Go-Explore-style) coverage.
- Held-out: L4 (its cached solution is used for *evaluation only* — never trained on).
- Strictly count and log all real env steps used at L4 time.

### Ablations (rigor)
1. Delta vs full-frame parameterization (same backbone, same data) — isolates H1's cause.
2. Data-size scaling (exactness vs #transitions).
3. Loss weighting on/off.
4. Cross-game probe: same recipe on tu93 (different game, different UI row).

## Why this is the right experiment
- It attacks the *documented* blocker with the *smallest* change (output parameterization),
  not a bigger model. Favors simplicity/elegance per the brief.
- The core artifact is a trained neural dynamics model (learning-based requirement).
- Success = first weight-level transfer result in the repo (model trained on some levels
  makes an unseen level cheap). Failure is also informative: it would show exactness, not
  pixel accuracy, is the irreducible obstacle, sharpening exp_006's verdict.

## Compute constraints (sandbox reality, logged for honesty)
4-core aarch64 CPU, 3 GB RAM, 45 s per shell call (no persistent background processes) —
training runs are chunked with checkpoint-resume; model batch inference is used inside
Go-Explore for throughput. Offline env runs at ~2k steps/s, so "imagined steps are cheaper
than real steps" holds for the *sample-efficiency* metric (real ARC-AGI-3 deployment steps
are server-bound and scarce), not for local wall clock.
