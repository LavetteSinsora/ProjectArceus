# How to do AI/RL research — lessons from the exp_010 "phantom advantage" investigation

A reflection on the **process** (not the result) of figuring out why warm-starting PPO from a
JEPA-pretrained encoder fails on sparse-reward LS20 while a random encoder succeeds. The
lessons below are distilled from the actual path — the wrong turns, the corrections forced by
skeptical questions, and an adversarial self-review. They are meant as durable, transferable
guidance, with the concrete misstep that taught each one.

The single habit that matters most, stated up front:

> **After every claim, ask "what is the cheapest experiment that could prove this wrong?" — and
> run it before believing yourself.** The explanation that survives is the one that survived
> several genuine attempts to kill it, not the one that sounded most elegant.

---

## I. Hypothesis discipline

1. **The first/obvious explanation is usually wrong or incomplete — and it may be baked into
   your code.** The repo's own `analyze_runs.py` asserted the failure was "large feature norm →
   miscalibrated value head → entropy collapse." It was wrong. Treat *every* existing
   explanation — your own, the codebase's, a paper's — as a hypothesis to falsify, not a fact.

2. **Aim to falsify, not to confirm.** For each hypothesis, design a control that can *kill* it.
   We killed three in a row — "the features are bad" (the frozen rep still reaches the goal),
   "it's the norm" (norm-matching still fails), "it's encoder drift" (freezing still fails) —
   before the survivor. A hypothesis that survives a real kill-attempt is worth something; one
   you only tried to confirm is worth nothing.

3. **The decisive experiment is a single-variable intervention.** "Remove the value critic,
   change nothing else" → rescue. Cross-condition comparisons (random vs JEPA encoder) confound
   the encoder's content with its norm, its rank, its training, everything. Find the one knob
   that, flipped alone, turns the effect on and off.

4. **Replace a dying hypothesis; don't patch it.** Each modification ("loud encoder" →
   "structured V → phantom advantage") was *forced* by a control that contradicted the prior
   story. When the data says no, change the theory — and say in the writeup that you did.

## II. Measurement — look at the right thing

5. **Timing/causality, not just endpoint correlation.** The "entropy collapse" story died the
   moment we plotted *when* entropy fell relative to the first reward: it fell at *zero reward*,
   and was still near-max in the window that actually mattered. Plot the time course before
   trusting an aggregate.

6. **Decompose marginal vs conditional / aggregate vs structured.** `KL(marginal policy ‖
   uniform)` looked identical for random and JEPA; the real difference lived entirely in
   `I(S;A)` (state-conditional structure). A marginal statistic can completely hide the effect
   you're chasing.

7. **Know what your pipeline already equalized.** PPO normalizes advantages to mean-0/unit-var,
   so comparing the *marginal advantage distribution* across encoders is uninformative *by
   construction*. Before comparing quantity X, ask: "has a normalization step already
   standardized X?" (It had — the real signal was in the structure normalization can't touch.)

8. **Correlation ≠ causation in your *own* plots.** "Reward drives encoder differentiation"
   (feature-cosine 0.99→0.73) is confounded: a sharpening policy visits a *different* set of
   states, which changes the measured cosine for reasons unrelated to the claim. Strip causal
   verbs unless you have an intervention or a fixed measurement set.

9. **Capacity ≠ optimization.** "Frozen JEPA reproduces the L2 win; frozen random can't" is a
   representation-*capacity* result — and it vanishes the instant the encoder is allowed to
   train. Never let "the representation *can* express X" leak into "the algorithm *will* learn X."

## III. Statistics & honesty

10. **Be ruthless about small n.** Our binary solve/fail table (n=3) had *no* significant
    pairwise difference, and the random encoder itself failed on 1 of 3 seeds. Lean on
    continuous, mechanistic, controlled measurements; treat binary outcomes at tiny n as
    anecdotes.

11. **Reproduce before believing.** A skew/kurtosis difference looked real at 3 seeds and
    *vanished* at 8. Noise masquerades as signal at small n — re-derive any "clean" effect with
    more seeds before it enters the story.

12. **Every number in the writeup needs a script that regenerates it.** A "41× reward
    dominance" number was an isolated-trajectory artifact (it dilutes under advantage
    normalization + multi-env rollouts) and wasn't even saved anywhere — so it was cut. If you
    can't regenerate it on demand, it does not go in the paper.

13. **Hedge to the evidence; state the scope.** Lead on what's controlled, fence what's brittle,
    name the limits (one game, two levels, ≤3 seeds, short runs). Overselling is a bug, and a
    reviewer (or a future you) will find it.

## IV. Language & reasoning precision

14. **Words smuggle in claims — name things exactly.** "Collapsed" implied low-rank; the random
    features were full-rank, merely *dominated by a state-invariant component*. "Cancellation"
    was loose (advantages are zero-mean by construction). A sloppy word silently becomes a wrong
    mental model that derails the next three experiments.

15. **Find the right level of abstraction.** The question "how can the *same* advantage reinforce
    differently?" forced the real mechanism into view: it is **credit generalization through the
    shared representation** (`Δπ` at state s′ ∝ `h(s)·h(s′)`), not the advantage value at s. The
    right abstraction dissolves the apparent paradox.

16. **Don't jump reasoning steps — each "therefore" is a place to insert a measurement.** When a
    step felt too obvious to check ("entropy is high, so it's exploring well"), it was exactly
    where the error hid (entropy ≠ coverage).

## V. Process

17. **Red-team yourself before you publish.** Two adversarial critics (a theory critic + an
    evidence auditor) found real errors: the imprecise cancellation framing, the unsupported
    41×, the Fig-10 causal confound, and the statistical emptiness of the binary table. Spawn the
    skeptic you would *fear* as a reviewer, and do it *before* you're attached to the story.

18. **Actively hunt for the boundary conditions — where does the result *not* hold?** A finding
    without its scope is half-finished. (See the appendix: the random encoder's "safety" depends
    on LS20's near-identical frames; with distinct representations it would also drift. That is a
    clean, named, *untested* boundary — exactly the kind to flag.)

19. **A probing question is a free experiment design — welcome it.** Nearly every correction here
    was triggered by a skeptical question ("is the cosine really 0.99, and what does it *mean*?",
    "doesn't normalization wash it out?", "should it explore *truly* randomly?"). Adversarial
    engagement, from anyone, is the cheapest source of better experiments.

20. **Fast iteration is a research multiplier.** Parallelizing tiny-model runs (many
    single-threaded processes, ~10/12 cores instead of ~4) turned a serial slog into quick
    loops; more cycles → more falsification attempts → a better-tested answer. Tooling is part of
    the method.

## VI. RL-specific gotchas that bit us (worth internalizing)

21. **Under terminal-only reward, before the first success the actor gradient is a *pure
    function of the critic*.** Reward enters the policy only through the advantage; with r=0,
    `Â` is a functional of V alone. So a randomly-wrong value function is the *only* thing
    shaping the policy in the entire pre-reward phase — that is where "phantom" guidance comes
    from.

22. **Entropy is not exploration coverage.** Max per-step action entropy ≠ good *state* coverage
    (a uniform walk is diffusive and slow). And uniform is the entropy *maximum*, so *any* drift
    — even unbiased noise — lowers entropy. A falling entropy curve does not by itself mean
    "the agent is learning something real."

23. **"Explore uniformly" is a fallback, not the optimum.** Provably-good pre-reward exploration
    is *directed* by epistemic novelty (counts/curiosity). The pathology is biasing on
    *value-function noise* (zero information about the world), which is strictly worse than
    uniform; biasing on *novelty* is better than uniform. The lesson is "don't synthesize a
    direction from noise," not "never be directed."

24. **On-policy (PPO) vs off-policy (DQN + ε-greedy) face the signal-less phase differently —
    know which you're in.** On-policy actor-critic *acts from* and *updates* the same policy, so
    the critic shapes exploration directly (the phantom's door). Value-based methods *decouple*
    behavior (forced ε-random) from the value estimates, so a random Q cannot corrupt
    exploration — they update Q off-policy from every transition regardless of the action's
    source.

25. **Know what your defaults are *for*.** The entropy bonus, the small/zero value-head
    initialization, and training the encoder *from scratch* are partly there to keep this exact
    phantom small. Warm-starting from an informative, large-norm representation quietly removes
    all three protections at once — which is why exp_010 hit the failure so cleanly.

---

## Appendix — a worked example of lesson #18: stress-testing this result's generality

A reader's question (the right one): *"If this is a genuine mechanism, shouldn't all PPO suffer?
And isn't the random encoder safe only because LS20 frames barely differ?"* The honest answers,
and how confident we are in each:

- **The "random encoder is safe" result does NOT generalize** — it is a property of LS20's
  near-identical frames, which make a random CNN map states to near-identical representations
  (V≈const → no phantom). With inputs/architectures that give a random encoder *distinct*
  representations, its phantom advantages would localize and *not* cancel, and it would drift
  too. *(Reasoned, and cleanly testable: make the random encoder's features state-varying — input
  noise, a deeper random net, an appended one-hot position — and check whether its `I(S;A)` rises
  and the advantage disappears. Not yet run.)*
- **Severity is a product, and standard practice shrinks each factor:**
  `≈ (representation state-variation) × (value-head output scale) × (length of signal-less
  phase) ÷ (entropy pressure)`. The entropy bonus, small value-head init, and from-scratch
  encoder training each keep one factor small; warm-starting from an informative large-norm
  encoder removes all three.
- **The field's real mitigations** for "the value function has no signal for a long time"
  (the heart of hard-exploration RL): **intrinsic motivation** (RND, ICM, count-based bonuses)
  gives V a real reward-independent signal so it isn't random and exploration is *directed*;
  **max-entropy / strong entropy regularization** resists premature collapse; **Go-Explore /
  episodic archives** decouple reaching a state from the policy. *(These are established field
  knowledge, stated from principle, not measured here.)*
- **ε-greedy is a value-based mechanism**, not PPO's; it sidesteps the phantom by *decoupling*
  the behavior policy (forced uniform during high ε) from the value estimates. PPO is on-policy
  and has no ε-greedy, which is precisely why the critic can shape its exploration.

Flagging which claims are **measured** (the LS20 mechanism, the critic-removal rescue, the
norm-controlled value spread), which are **reasoned but testable** (the representation-similarity
boundary), and which are **field knowledge cited from principle** (the mitigations) — that
labeling *is* the discipline this document is about.
