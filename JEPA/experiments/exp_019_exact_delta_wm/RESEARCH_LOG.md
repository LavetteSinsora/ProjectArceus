# exp_019 RESEARCH LOG (append-only)

## 2026-06-10 — session start

- Reviewed all prior experiments (exp_001–018, claude_automate). Key blocker chosen as
  target: exp_006's world model transfers at 99% pixel acc but 0% exact-frame acc → killed
  model-based planning. See PROPOSAL.md.
- Sandbox setup: aarch64 CPU-only. torch 2.7.1+cpu installed (chunked PyPI download —
  pytorch.org index blocked by proxy; first downloaded x86_64 wheel by mistake, sandbox is
  aarch64; 2.12 aarch64 wheel is CUDA-SBSA, 2.7.1 aarch64 is CPU). arc-agi 0.9.8 /
  arcengine 0.9.3 require Python≥3.12, sandbox has 3.10 → vendored both pure-Python
  packages from the repo's macOS .venv; they compile clean under 3.10. Offline env verified:
  ls20 reset + 100 steps in 0.05 s (~2k steps/s).
- Constraint discovered: background processes do NOT survive between shell calls; max 45 s
  per call. All training/search will be chunked with checkpoint-resume.
- Assets found: cached Go-Explore solutions for ls20 L1 (33 acts), L2 (60), L3 (60),
  L4 (100), tu93 L1 (43) in `claude_automate/experiments/solve_*/solution.json` — replaying
  them gives the rare completion/modifier transitions for the train set (L1–L3) and an
  evaluation rollout for held-out L4.
- env API: `step(a) -> (frame, terminal)`, `level_completed` property, `_MASKED_ROWS` =
  rows 61–62 for ls20 (row 63 for tu93/re86/g50t).

## 2026-06-10 — data + first training results

- Data collected (solution replay ×2 + 40 random eps + ~20 solution-prefix random bursts
  per level): L1 7 592 / L2 4 460 / L3 4 514 transitions (train pool = 16 566), L4 6 548
  (eval-only). Completion transitions in train: 11. Sparse-delta structure confirmed:
  ~1% of cells change per transition (≈38 masked cells = avatar block + bars); 26% of
  transitions are no-ops outside the UI timer; board occupies rows ~5–58 in 5-px bands;
  rows 61–62 = timer (100% change rate).
- Sandbox training speed: ch=64×6 layers = 1.3 s/batch → too slow. Settled on ch=32,
  5 dilated layers [1,2,4,8,1], ~0.2M params, ~90 batches (32) per 36 s chunk.
- **v1** (lr 3e-4, pos_weight 30, raw frames): exact-masked on held-out L4 = 0% @ step 272,
  2.7% @ 539, 17.1% @ 1015. Diagnosis @633: change-head precision 0.505 / recall 0.818 —
  81% of wrong cells are FALSE changes; τ sweep non-monotonic (τ=.99 → 25%). Timer rows
  never exactly predicted (unmasked exact stuck at 0%).
- **Design fix (v2): canonical-masked state.** UI rows 61–62 zeroed in input AND target —
  the model state is the UI-masked frame, the same object Go-Explore hashes. Removes the
  always-changing rows from the change head's burden and keeps rollouts clean. Cost:
  model can no longer see energy level → cannot predict energy-death timing (acceptable —
  verifier catches it; documented). Also lr 3e-4→1e-3, pos_weight 30→10.
- **v2 trajectory:** 8.3% @ 270, 23.9% @ 928 (vs v1 17% @ 1015) — better but still rising.
- **Key diagnostic:** exactness on *train* levels is only 38% (L1) / 46% (L3) — the model
  is UNDERFITTING, not transfer-limited: held-out L4 (24%) is close to held-in. Action:
  train much longer (loss still falling); capacity increase only if train exactness
  plateaus below ~95%.
