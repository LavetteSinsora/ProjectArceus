# L1→L2 φ-transfer — does it help? (ls20)

**Verdict: NO measurable benefit on ls20 L2; a slight early disadvantage. n=2 seeds, cap 120k.**

## Setup
- Source φ: an ls20-L1 ICM run trained to 28,672 steps (holdout inv_acc ~0.7 on L1),
  `exp013_1_rndicm_icm_ls20_L1_seed0_20260601_113023/checkpoints/step_00028672.pt`.
- B (`exp_013_1_rnd_icm`) on **ls20 L2**, `--init-phi-ckpt <source>` (XFER) vs random init (RAND),
  seeds 0/1, 120k cap, stop-on-first-reward. φ-load CONFIRMED in logs (`φ INIT-FROM-CKPT …`).

## Results (env_steps to first reward; holdout inv_acc)
| run | holdout init | holdout final | solved |
|---|---|---|---|
| RAND s0 | 0.256 | 0.735 | ✗ (censored 119k) |
| RAND s1 | 0.261 | 0.743 | ✗ |
| XFER s0 | 0.256 | 0.749 | ✗ |
| XFER s1 | 0.261 | 0.704 | ✗ |

Holdout trajectory (seed 0): XFER 0.256→0.38(u20)→0.41(u29); **RAND 0.256→0.53(u20)→0.67(u29)** — random-init
climbs *faster*. Neither solves ls20 L2.

## Why transfer doesn't help (as implemented)
1. **Only the φ ENCODER transfers, not the inverse-dynamics HEAD.** So holdout inv_acc starts at
   chance (~0.26) for XFER too — the controllability signal must relearn from scratch regardless,
   erasing the head-start the transfer was supposed to give.
2. **The L1-specialised φ may need to UN-learn before adapting to L2's layout** — consistent with
   XFER climbing *slower* than RAND early.
3. ls20 L2 is a frontier cell (random E=∞, both baselines 0/8); the bottleneck is finding the
   reward sequence, which a slightly-better φ does not address.

## Recommendation
Drop encoder-only transfer as a lever, OR test transferring the inverse head too (so controllability
isn't reset). Don't spend frontier-sweep budget on B-xfer on the strength of this result.
