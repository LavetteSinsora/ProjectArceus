# How we diagnosed representation collapse in a JEPA world model

*A narrative walkthrough of a real debugging session. No prior context required.*

---

## What we were building

Our project is a reinforcement-learning agent for a small grid-world puzzle game (called LS20). The agent sees a 64×64 image of colored tiles every step, picks one of four actions (move N/S/E/W, roughly), and gets a small reward signal.

Rather than feeding pixels straight into a policy network, we are training a **world model** alongside the policy. A world model is exactly what it sounds like: a learned simulator. Given the current state and a chosen action, it tries to predict what the *next* state will look like — but not in pixel space. Instead, it predicts in a compressed latent space. This style of architecture is called **JEPA** (Joint Embedding Predictive Architecture), introduced by Yann LeCun's group. The pitch is that learning to predict in an abstract embedding space is much easier than learning to predict pixels, and the learned embeddings make a good substrate for a policy to act on.

So we have three things being trained together:

1. **An encoder**: takes a 64×64 frame and produces a small set of latent vectors (we use 4 latent vectors, each 128-dimensional).
2. **A predictor**: takes the current latent and the chosen action, predicts what the next latent should be.
3. **A policy**: looks at the current latent and decides which action to take.

The encoder and predictor train together via a self-supervised loss: encode frame `t`, encode frame `t+1`, ask the predictor to map the first to the second, measure the error. The policy trains separately via REINFORCE, where the *reward* is the predictor's prediction error itself — a "curiosity" reward that encourages the agent to seek surprising transitions.

## The architecture in slightly more detail

The encoder has two stages. **Stage one** is a small Vision Transformer: it chops the 64×64 frame into 16 patches of 16×16, embeds each patch into 128 dimensions, and runs two self-attention blocks so each patch can incorporate context from the others. This produces 16 patch embeddings, one per location.

**Stage two** is a Perceiver Resampler. Think of it as a learned mechanism that pools the 16 patch embeddings down into just 4 summary vectors. Specifically, it maintains 4 "query" vectors and runs two rounds of cross-attention: in each round, the queries read information from the 16 patches and update themselves. At the end, those 4 updated queries are our final latent representation.

A crucial detail: **the latents are recurrent**. After encoding frame `t` and producing 4 latents, those 4 latents become the *queries* for encoding frame `t+1`. This means each new encoding builds on the last one — the latents carry information forward through time, like the hidden state of an RNN. (On the very first frame of an episode, we don't have prior latents to use as queries, so we use 4 learned "placeholder" vectors instead.)

The predictor is conceptually simpler: it takes the 4 latents for the current state plus an action embedding, and outputs 4 latents for the predicted next state. It's a flow-matching predictor — we'll explain what that means in the next section.

The policy is a small MLP that reads the 4 latents and outputs a probability distribution over the 4 actions.

## The training loss, explained

Two losses train simultaneously: the **world-model loss** and the **policy loss**. We focus on the world-model loss because that's where the trouble was.

### Flow matching, intuitively

A typical world-model loss would be: encode `s_t`, predict the next latent, compute the squared error against the encoded `s_{t+1}`. Done.

We use something slightly more elaborate called **flow matching**, borrowed from generative modeling. The idea is to teach the predictor to be a continuous map between distributions. In our case, between "the encoding of the current state" and "the encoding of the next state". Here's the mechanic:

Let `h_t` be the current encoding and `h_{t+1}` the next-state encoding. We pick a random number `τ` between 0 and 1, and form a linear interpolation:

```
x_τ = (1 − τ) · h_t  +  τ · h_{t+1}
```

This `x_τ` is a point somewhere on the straight line between `h_t` and `h_{t+1}`. We feed it (along with `τ` itself and the action embedding) into the predictor, and ask the predictor to recover the *endpoint* `h_{t+1}`. The loss is the squared error between the prediction and the true `h_{t+1}`.

```
loss = || h_{t+1} − predictor(x_τ, τ, action) ||²
```

That's the entire loss for one sample. We average it across the batch.

The reason for this construction (rather than a plain "predict `h_{t+1}` from `h_t`" loss) is that it teaches the predictor a smooth, continuous mapping: given any partial-mix between the two states, it should know how to get to the endpoint. At inference time, we can use this to take small steps and produce intermediate predictions if we want.

**Key point for this story**: each training sample uses **one** random `τ`, not several. We don't sum or average across multiple `τ` values per sample. So when we look at gradient norms later, those numbers reflect one prediction per sample, full stop.

### The curiosity reward

The policy is trained with the prediction error itself as a reward. But the *exact* form of "prediction error" used as reward is different from the training loss above. For the reward, we run the predictor as a generative model: starting from `h_t`, take three small Euler steps using the learned flow field, and arrive at a predicted `h_{t+1}`. Then compute squared error against the actual `h_{t+1}`.

The training loss uses linear interpolation at a random `τ` and asks the predictor for the endpoint. The reward uses three iterative integration steps starting from `h_t` and measures end-to-end prediction quality. **These are different functions of the same predictor weights**. This becomes important later.

## How gradients flow through this thing

To diagnose where training is going wrong, you need a clear picture of which parameters get gradient signal from which loss term. Here's the map.

The world-model loss `|| h_{t+1} − predictor(x_τ, τ, a) ||²` produces a gradient with respect to the prediction. From there, gradient flows backward in two directions:

**Backward through the predictor.** Every parameter of the predictor MLP, the time embedding (which encodes `τ`), and the action embedding receives gradient. Standard stuff.

**Backward through `x_τ` back to `h_t`.** Because `x_τ` is a linear combination involving `h_t`, gradient flows into `h_t`. From there it flows back through the entire encoder: through the Perceiver's cross-attention and self-attention layers, through the patch self-attention blocks, and into the patch embedding. So the encoder learns to produce latents that are "predictable."

**Crucially: gradient does *not* flow through `h_{t+1}`.** We deliberately stop the gradient there. If we let gradient flow through `h_{t+1}` too, the encoder would find a trivial solution: just emit the same constant vector regardless of input — then `h_t = h_{t+1}` and the loss is zero. This is called **representation collapse**, and stopping the gradient through `h_{t+1}` is the standard JEPA trick to discourage it.

In our setup, we go one step further: `h_{t+1}` is produced by an **EMA target encoder** — a second copy of the encoder whose weights are an exponential moving average of the online encoder's weights, and which never receives gradient. This is the BYOL/MoCo-style stabilization: by computing the target with a slowly-changing copy of the network, the loss landscape is smoothed and the trivial fixed point becomes (in theory) harder to reach.

So at a high level: every world-model gradient step is trying to make the online encoder produce latents from which a predictor can recover the EMA-encoder's latents for the next frame, without ever changing the EMA-encoder via gradient. Only the online encoder, predictor, and action embedding receive gradient from this loss.

The policy is trained separately by REINFORCE, with the curiosity reward described above. Its gradients are isolated from the world-model loss; only the policy network gets policy-loss gradient.

## The symptoms: when something seems off

We had been running training for around 85,000 steps with this setup, watching a dashboard of metrics. A few numbers caught our eye:

- **The loss was dropping nicely** from around 0.16 to 0.007 — a 20× reduction. By itself, this looks great.
- **But the gradient norm on one specific part of the encoder was huge and flat**: the Perceiver portion of the encoder consistently had gradient norm around 11.5, while the rest of the encoder sat at 0.005 to 0.01 — three orders of magnitude smaller.
- **Effective rank of the latent representations was suspiciously low**: only around 1.4 out of a maximum of 4.
- **The "ODE cosine similarity" metric** — which measures how much the predictor changes its input — was 0.99+, meaning the predictor was barely doing anything.

A loss curve that drops while other diagnostics deteriorate is a classic warning sign. We needed to look more carefully.

## Probe 1: where is the gradient actually going?

The gradient norm reported in the dashboard was a single number per major component — "encoder.SA blocks", "encoder.perceiver", "predictor MLPs". Useful for trends, but too coarse to tell us *where* the gradient was concentrated within the Perceiver.

So we wrote a probe that loads a saved checkpoint, runs a forward and backward pass identical to a training step, then walks every single named parameter and records the L2 norm of its gradient and weight. Then we aggregated by sub-module.

The result, at step 80,000:

```
encoder.SA stack          grad norm:   0.009
perceiver.placeholders                  0.011
perceiver.round0.cross   grad norm:    20.44    <-- 99.99% of everything
perceiver.round0.self                   0.072
perceiver.round1.cross                  0.053
perceiver.round1.self                   0.049
perceiver.output_norm                   0.006
predictor.time_embed                    0.008
predictor.mlps                          0.099
```

The entire encoder's reported gradient norm of ~20 came from **a single cross-attention block** — the very first one in the Perceiver, which is where the queries first read from the patch embeddings.

This is a strange and very specific failure mode. It says: of the millions of parameters being trained, the gradient is overwhelmingly trying to update one sub-block, and only that one.

To localize further, we split the batch into two parts: transitions where the queries were placeholders (the start of an episode) versus transitions where the queries were prior latents (mid-episode). We computed the loss and gradient separately for each subset.

The result was even more striking. The mid-episode subset (60 of 64 samples) had **a loss of 0.002 but a Perceiver gradient of 42.04**. The start-of-episode subset (4 samples) had loss 0.11 but gradient only 0.81. The contrast is exactly backwards from what you'd expect: tiny loss producing huge gradient.

When you see "tiny loss, huge gradient," you are sitting near a degenerate point in the loss landscape. The function value is small because outputs are nearly constant, but the local geometry has narrow ravines or singularities, so small parameter changes cause large output changes. The optimizer is spending all its energy maintaining the near-constant state rather than learning anything.

## Probe 2: what does the representation actually look like?

This led us to look directly at the latent representations themselves. We gathered 1024 transitions through the EMA target encoder and asked: are the 4 latent tokens different from each other? Do the latents vary across game states? How much of the 128-dimensional space is actually being used?

The key tool here is **effective rank**, computed from singular values. Plain rank counts how many directions of a matrix are non-zero. Effective rank is a softer version: if your singular values are `[100, 100, 100, 100]`, effective rank is 4. If they are `[100, 0.001, 0.001, 0.001]`, effective rank is approximately 1, because one direction dominates. The formula is `exp(entropy of normalized singular values)`.

Effective rank has natural upper bounds. For a matrix of shape `(N, D)`, effective rank is at most `min(N, D)`. So if we stack 1024 latent samples each of size 128, the maximum is 128. If we flatten all 4 tokens into a 512-dim row, the maximum is 512.

We measured effective rank at several checkpoints, at three granularities:

| Step | All 4 tokens flattened (max 512) | Each token individually (max 128) | Pairwise cosine between tokens |
|------|----------------------------------|-----------------------------------|--------------------------------|
| 5K   | 6.4                              | 6.4                               | +0.9996 (essentially identical) |
| 20K  | 17.8                             | 16.9                              | +0.91 to +0.97 (still very similar) |
| 40K  | 27.2                             | 22                                | mostly near 0 (decorrelated) |
| 60K  | 2.7                              | 2.7                               | +0.99 (re-collapsed) |
| 80K  | 1.9                              | 1.9                               | +0.97 (deep collapse) |

This is the entire story in one table.

At step 5K, the four latent tokens are functionally **the same vector**, occupying a 6-dimensional subspace of the 128-dimensional latent space. The encoder is essentially producing one repeated vector, four times.

By step 20K, that subspace has expanded to about 17 dimensions, but the four tokens are still essentially copies of each other.

By step 40K, something remarkable happens: the four tokens **decorrelate**. Their pairwise cosine similarities drop near zero, meaning each token now reads different information from the patches. Each token individually spans about 22 dimensions. This is the only window in training where the representation is doing something legitimately useful.

Then it falls apart. By step 60K the four tokens have **collapsed back together** (cosine 0.99+) and the effective rank has crashed to 2.7. By step 80K it's down to 1.9 — meaning all the variance across 1024 different game states lives in essentially **two directions** of a 128-dimensional space. The norm standard deviation is 0.001 — every latent has exactly the same magnitude, because the output LayerNorm forces it.

So the model briefly worked, then fell off a cliff.

## Probe 3: was the loss function even what we thought it was?

While diagnosing the above, a separate worry came up. Flow matching uses a random `τ` per sample — could the gradient norms be inflated by some kind of accumulation we hadn't noticed?

We re-read the loss code carefully. The answer is no: each sample contributes **one** squared-error term for **one** randomly drawn `τ`. There is no inner loop over multiple `τ` values, no summation. The training-time gradient is an unbiased Monte Carlo estimate of the τ-integrated loss, evaluated at one point per sample. So the large gradient norms we observed are not a measurement artifact — they are real.

But this same investigation surfaced something more interesting: **the curiosity reward and the training loss are not the same function**. The training loss is `|| h_{t+1} − predictor(linear-interp at random τ) ||²`. The reward is `|| h_{t+1} − three-step-Euler-rollout(h_t) ||²`. They measure related but different things.

The implication: when the encoder collapses to a near-constant function, `h_t ≈ h_{t+1}` for every transition, so the rollout-based reward also goes to zero — but it goes to zero *uniformly*, regardless of which action was taken. The policy receives a flat, uninformative reward landscape, so it cannot push back against the collapse via the policy gradient. The two halves of the system have decoupled.

## Putting it all together: what actually broke

Here's the unified picture across both probes:

The encoder started in a partial-collapse state and trained itself out of it for the first 40,000 steps. During this phase, the four latent tokens were almost identical copies, occupying a small subspace, and the predictor easily satisfied the loss by being near-identity (predicting `h_{t+1} ≈ h_t`). The gradient was modest because both sides of the loss were slack.

Around step 40,000, the four tokens decorrelated. Each one was now spanning ~22 dimensions, with low cross-correlation. The gradient on the first cross-attention block was high (gradient norm ~13), reflecting genuine learning pressure. This was the moment when the system was *actually working*.

But this state was not stable. Without an explicit penalty against collapse, and with the predictor still able to satisfy the loss by being near-identity, the easier solution — collapse — was always available downhill. Between step 40K and 60K, the encoder slid back into a collapsed state, deeper than the original one. By step 80K, all the encoder's outputs lived along a single direction (one singular value of 367, the next 23, then 17, then progressively negligible). The four latent tokens were once again essentially identical to each other.

In the collapsed regime, the loss kept decreasing because the predictor had refined its "be near-identity" behavior. The gradient norm on the first cross-attention block kept rising not because anything was being learned, but because the loss landscape had become a narrow, steep ravine around the trivial solution — small parameter perturbations caused large output movements but in directions that the LayerNorm at the output then immediately squashed back to the same constant vector. The system was burning gradient energy maintaining the collapse.

The EMA target encoder, which was supposed to prevent this kind of collapse, did not help here. Its purpose is to provide a stable target — and indeed, the EMA tracked the online encoder faithfully, with relative parameter distance dropping from 1.7% at step 5K to 0.5% at step 80K. But "stable target" cannot prevent collapse when the online encoder is monotonically *drifting toward* the collapsed solution. The EMA is downstream of the online encoder; if the online encoder collapses slowly, the EMA collapses slowly too. We saw the EMA target's effective rank drop from 24.5 (at step 40K) to 1.9 (at step 80K) in lockstep.

## Why the collapse happens, mechanistically

Several factors compound:

**The trivial solution is always reachable.** The loss `|| h_{t+1} − predictor(...) ||²` is zero when both `h_t` and `h_{t+1}` are the same constant vector and the predictor outputs that vector. The stop-gradient on `h_{t+1}` and the EMA target make this slightly harder to reach, but do not eliminate it as a basin of attraction.

**The output LayerNorm hides one symptom of collapse.** Without it, a collapsing encoder would also produce vectors of shrinking magnitude, which is easy to detect. With the LayerNorm always forcing magnitude to √128 ≈ 11.3, the only signal of collapse is in the *direction* of the latent vectors, which requires more careful instrumentation to see.

**The predictor's near-identity bias.** Because consecutive game frames are very similar (the agent only moves one tile per action, most of the frame is unchanged), `h_t` and `h_{t+1}` should genuinely be close. A predictor that learns "output approximately the input" gets most of the loss right. Once it learns this, the encoder is free to collapse without paying a loss penalty.

**The reward signal doesn't fight the collapse.** When the encoder collapses, the curiosity reward also goes to zero, so the policy can't apply pressure to keep the representation diverse via behavior. Both halves of the system collapse together.

**Within-state token redundancy is not penalized.** Nothing in our loss says "the four latent tokens should be different from each other." So the cheapest way for the Perceiver to produce 4 latents is to emit four copies of the same vector. The effective rank within a state never exceeded 1.6 out of 4 — the four-token capacity is wasted throughout.

## Mitigations to try

Given the diagnosis, several interventions are worth trying. In rough order of expected leverage:

**1. Add a variance regularizer on the latents.** This is the simplest fix and was pioneered by VICReg. For each latent dimension, penalize whenever its standard deviation across the batch drops below 1. This directly fights the per-state collapse and gives a clear signal whenever the model tries to slide into the trivial solution. Apply it to the *pre-LayerNorm* latents so it actually constrains the manifold's spread, not just the LayerNorm's outputs (which are forced to a constant norm by definition).

**2. Add a covariance off-diagonal penalty.** Within a batch of latents, compute the covariance matrix across the 128 dimensions. Penalize the off-diagonal entries (squared sum). This pushes the dimensions to be statistically uncorrelated, raising the effective rank. Combined with the variance penalty above, this is the full VICReg recipe.

**3. Penalize within-state token redundancy.** Specifically penalize the cosine similarity between the 4 latent tokens within a single state. This addresses the 1.4/4 within-state rank directly — the four tokens should be different *summaries* of the patch information, not four copies of one summary.

**4. Restructure the curiosity reward.** Right now the reward goes to zero whenever the encoder collapses, removing any policy-side pressure against collapse. A simple fix: divide the reward by some measure of latent movement, so a flat predictor on a flat encoder no longer produces zero reward — it produces undefined or negative reward, which is bad for the policy and gives it a reason to seek non-degenerate transitions.

**5. Investigate why the first cross-attention specifically.** The fact that all the gradient pressure concentrates on `perceiver.round0.cross` suggests something specific about how the placeholders meet the patch context in the first round. Two checks worth running: the singular spectrum of the Q-projection weights at step 40K vs step 80K, and the attention-weight distributions over the 16 patches. If queries collapse to producing identical attention patterns over a single patch, that would explain the redundancy and the gradient concentration simultaneously.

**6. Use the eff-rank metric as a hard training signal.** Currently we monitor loss and gradient norm. The diagnostic that actually predicted the failure is per-token effective rank — it rose from 6 to 22 (good) then dropped to 2 (bad), and this happened well before the loss numbers showed anything alarming. Adding a hard stop or a logged warning when effective rank drops below a threshold would catch this much earlier in future runs.

## What we learned, beyond this specific bug

Loss curves alone are not sufficient diagnostics for representation learning. A loss that decreases monotonically can be hiding a system that is degenerating along axes the loss does not measure. In our case, the loss function — viewed in isolation — was completely satisfied by the collapsed solution, because it never directly measured whether the representation had any structure.

Effective rank, standard deviation across samples, and within-state diversity are far better signals of whether a self-supervised encoder is learning anything. They should be primary metrics, not auxiliary ones.

Gradient norms are diagnostic, but only when broken down by sub-module. The single number "encoder gradient norm = 11" is essentially uninterpretable — it could mean healthy learning or it could mean, as in our case, that a single block has captured all the gradient pressure while the rest of the network goes nowhere.

And finally: when training a coupled system (encoder, predictor, policy, EMA target all influencing each other), it is worth tracing exactly which losses send gradient to which parameters. The decoupling of curiosity reward from training loss in our setup was invisible until we drew the gradient graph and noticed that the two loss functions were not the same function, even though they shared every parameter. That kind of slow architectural drift — where two pieces of code were obviously the same when they were written, and have quietly diverged — is exactly the sort of thing only careful re-reading catches.
