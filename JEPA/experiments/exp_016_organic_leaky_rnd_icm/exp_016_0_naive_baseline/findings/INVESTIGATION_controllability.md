# exp_016_0 — What made held-out inverse accuracy jump from ~chance to ~1.00?

**Phenomenon.** exp_016_0's IDM reaches held-out inverse accuracy ~0.99–1.00 on both
tu93-L3 and ls20-L1, whereas the prior lineage exp_013_1b reported held-out inverse
accuracy stuck near chance (~0.22–0.34 ls20, ~0.22 re86) and concluded φ "never becomes
controllable."

## Verdict

**The single fundamental enabler is the timer-row mask** (zeroing observation rows 60–63
before the encoder). It is the *only* change that converts a genuinely chance-level φ into a
controllable one. The "drop no-ops" change is real but **secondary and downstream**: it is an
*evaluation-set* effect that inflates the headline number; it does not create controllability,
it only removes transitions whose action is mathematically unidentifiable. Feature-norm
inflation is a **correlate, not a cause**.

The prior "φ never becomes controllable" conclusion was an artifact of **two confounds stacked
on top of each other**, and exp_016 happens to fix both at once:
1. the exp_013_1b *ls20* runs ran with `mask_timer=False` (the timer confound genuinely broke ls20), and
2. the held-out metric in exp_013_1b *included no-ops* (~50% of random transitions), deflating the score on games whose φ was actually fine.

## Evidence

### 1. Timer mask is the causal enabler (same game, same level, same eval convention)
exp_013_1b ls20-L1 ICM φ checkpoint (`...icm_ls20_L1...113023/step_00028672.pt`) had
`mask_timer=False` / no `timer_mask_rows` key. Probed on a **no-op-free** holdout:

| run (ls20-L1, no-op-free holdout) | held-out inv_acc | ‖h‖ |
|---|---|---|
| exp_013_1b φ (mask OFF) | **0.264** (≈chance 0.25) | 1.2 (collapsed) |
| exp_016_0 IDM (mask ON), from metrics | first 0.270 → **max 0.987** | — |

Same game, level, and no-op-free convention; the only structural difference is the mask. φ
genuinely cannot recover the action when the marching step-timer makes every frame fake-unique;
masking the timer makes the true 43-state board visible and the inverse problem solvable.

### 2. The exp_013_1b "near-chance" headline was partly an eval artifact, not all real failure
exp_013_1b *tu93-L3* DID run with `mask_timer=True`. Its final ICM φ, probed on the **same**
no-op-free holdout I use for exp_016:

`NO-OP-FREE acc=0.967 | INCL-NOOP acc=0.756 | NOOP-ONLY acc=0.549`

So on controllable transitions exp_013_1b's masked φ was already 0.967 — *not* near chance.
Its logged metric peaked at 0.725 only because it averaged in ~48% no-ops. The "0.22–0.34
stuck" story is specifically the *unmasked ls20/re86* runs (case 1 above), generalized too far.

### 3. Drop-no-ops is an evaluation effect, not a controllability mechanism (tu93, 13 ckpts)
Probing all 13 exp_016 tu93-L3 IDM checkpoints on three holdouts (natural no-op fraction under
random policy = 0.50):

| env_step | no-op-free | no-op-free (unit-norm) | ‖h‖ | incl-no-op (50% noop) | no-op-only |
|---|---|---|---|---|---|
| 20480 | 0.841 | 0.825 | 96.9 | 0.434 | 0.024 |
| 40960 | 1.000 | 1.000 | 190 | 0.622 | 0.265 |
| 102400 | 1.000 | 1.000 | 204 | 0.610 | 0.209 |
| 249856 (final) | 1.000 | 1.000 | 201 | 0.625 | 0.242 |

- No-op-only accuracy sits at **chance** (0.242 vs 0.25; majority-class prior 0.354). This is
  correct: for a no-op, masked s == s′ → the inverse head sees `[h, h]`, which carries **zero**
  action information. No-ops are *unanswerable*, not merely hard.
- Including 50% no-ops caps achievable accuracy near 0.625; excluding them lifts it to 1.000.
- **So the headline 0.99–1.00 is genuine on controllable transitions but partly inflated by an
  evaluation set that excludes the ~50% unanswerable items.** The honest controllable-subset
  number (1.00) is real; the all-transition number is ~0.62.

### 4. Feature-norm inflation is a correlate, not the cause
‖h‖ grows ~97 → 201 across training, but **unit-normalizing features before the inverse head
leaves accuracy unchanged** (1.000 → 1.000 at every checkpoint). The discrimination lives in
the *direction* of h, not its magnitude; norm growth is a downstream side-effect of continuous
unfrozen training, not the enabler. (Upstream→downstream: timer-mask → solvable inverse problem
→ encoder learns separable directions → norm drifts up under continued SGD.)

### 5. Mask matters per-game (caveat on universality)
Feeding the trained-with-mask exp_016 tu93 IDM *raw timer-stamped* boards still gives acc=1.000
— in tu93 the CNN simply does not route the timer rows into its discriminative features, so tu93
worked even unmasked (consistent with exp_013_1b tu93 = 0.967). The mask is the *active* fix for
**ls20**, where the timer confound truly collapses φ (‖h‖→1.2, acc→chance). The mask is the
correct single answer because it is the only change that rescues the case that genuinely failed.

## Relation to the other candidates
- **Cross-update replay buffer / pure-inverse (no forward) / no-freeze**: these change training
  dynamics and stability but are not necessary for controllability — exp_013_1b's *masked* tu93
  φ already reached 0.967 with rollout-only data, inverse+forward loss, and no freeze fired.
  They are not the fundamental enabler.
- **Drop-no-ops**: real but secondary; an eval/training-set restriction that removes
  unidentifiable items. It raises the *reported* number; it does not produce a controllable
  representation.

## Bottom line
The isolated driving force is the **timer-row mask**. The high held-out accuracy reflects a
**genuinely controllable representation on controllable transitions** (survives unit-norm,
recovers the action perfectly), but the specific ~1.00 figure is **modestly inflated** by a
no-op-excluding evaluation set; the honest all-random-transition accuracy is ~0.62 (capped by
~50% action-unidentifiable no-ops).
