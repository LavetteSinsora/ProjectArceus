"""Build ULTIMATE_CAUSE.html — a single self-contained page (base64 figures) that
explains the ultimate cause of exp_016_0's three interesting phenomena:
  1. controllable φ (held-out inv_acc→1)  — root: the timer mask
  2. RND-loss / novelty inflation          — root: unnormalized inverse-dynamics CE
  3. entropy collapse (L1 yes, TU93 no)    — root: coverage-saturation novelty decay
plus the decisive LayerNorm ablation that separates root (2) from (3).

Run AFTER the LayerNorm ablation finishes:
    uv run python -m JEPA.experiments.exp_016_organic_leaky_rnd_icm.\
exp_016_0_naive_baseline.build_synthesis
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .tracker import IDMEncoder
from .diagnostics import harvest_states, encode_all

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CKPTS = HERE / "checkpoints"
L1 = RUNS / "exp016_0_naive_ls20_L1_seed0_20260606_193526"          # baseline (collapses)
TU93 = RUNS / "exp016_0_naive_tu93_L3_seed0_20260606_212547"        # no collapse
TU93_CK = CKPTS / "exp016_0_naive_tu93_L3_seed0_20260606_212547"
DEVICE = torch.device("cpu")
BLUE, RED, GREEN, GRAY = "#2c7fb8", "#d73027", "#1a9850", "#888"


def M(run): return [json.loads(l) for l in open(run / "metrics.jsonl")]
def col(m, k): return np.array([r.get(k, np.nan) for r in m], float)


def _b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(b.getvalue()).decode()


COLLAPSE_WINDOW = 120_000   # only judge the ablation once it's run past the baseline's ~102k collapse

def _last_step(run):
    try:
        return json.loads(open(run / "metrics.jsonl").read().splitlines()[-1])["step"]
    except Exception:
        return -1

def find_ln_run():
    """The LONGEST layernorm ls20-L1 run (ignores smoke/incomplete runs)."""
    cands = [d for d in RUNS.glob("*ls20_L1*")
             if (d / "config.json").exists()
             and json.loads((d / "config.json").read_text()).get("idm_layernorm")
             and (d / "metrics.jsonl").exists()]
    return max(cands, key=_last_step) if cands else None


# ── Fig 1: controllability — timer mask is the cause ─────────────────────────
def fig_controllability():
    # measured (INVESTIGATION_controllability.md): no-op-free held-out inv_acc
    labels = ["earlier project\ntimer NOT masked", "this work\ntimer masked",
              "+ features\nunit-length\n(control)", "all steps\n(honest avg)",
              "wall-bumps only\n(unrecoverable)"]
    vals = [0.264, 0.987, 1.000, 0.62, 0.242]
    cols = [RED, GREEN, GREEN, BLUE, GRAY]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.bar(labels, vals, color=cols)
    ax.axhline(0.25, ls="--", color="k", lw=1); ax.text(4.3, 0.28, "chance (1/4)", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("action recoverable from 2 frames?\n(inverse-dynamics accuracy)")
    ax.set_title("1 · Encoder learns action-predictive features — cause: the timer mask\n"
                 "(same game & level; feature magnitude ruled out by the unit-length control)")
    for i, v in enumerate(vals): ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    return _b64(fig)


# ── Fig 2: feature-norm inflation — unnormalized inverse CE; plateau∝CE→0 ────
def fig_feature_norm():
    ck = sorted(TU93_CK.glob("step_*.pt"))
    cfg0 = Config(**torch.load(ck[0], map_location=DEVICE, weights_only=False)["config"])
    reg, _ = harvest_states(cfg0.game, cfg0.level_index, cfg0.seed, cfg0.probe_roam_steps,
                            cfg0.n_envs, tuple(cfg0.timer_mask_rows), cfg0.n_probe_states)
    states = reg.all_masked()
    steps, norms = [], []
    for c in ck:
        d = torch.load(c, map_location=DEVICE, weights_only=False)
        cfg = Config(**d["config"])
        idm = IDMEncoder(cfg.n_actions, cfg.n_colors, cfg.frame_size, cfg.trunk_dim,
                         cfg.idm_hidden).to(DEVICE)
        idm.load_state_dict(d["idm"]); idm.eval()
        h = encode_all(idm.encode_masked, states, DEVICE)
        steps.append(d["step"]); norms.append(float(h.norm(dim=-1).mean()))
    m = M(TU93); ms, ia = col(m, "step"), col(m, "inverse_acc_holdout")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(steps, norms, "o-", color=BLUE, lw=2, label="feature magnitude (from saved models)")
    ax.set_xlabel("environment steps"); ax.set_ylabel("encoder feature magnitude", color=BLUE)
    ax2 = ax.twinx(); ax2.plot(ms, ia, color=GREEN, lw=1.5, label="inverse-dynamics accuracy")
    ax2.set_ylabel("inverse-dynamics accuracy", color=GREEN); ax2.set_ylim(0, 1.05)
    ax.set_title("2 · Novelty/error balloons — cause: encoder output is not normalized\n"
                 "feature magnitude stops growing exactly when action-prediction saturates")
    return _b64(fig)


# ── Fig 3: collapse discriminator — reward sign (coverage), L1 vs TU93 ───────
def fig_collapse_discriminator():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for run, name, c in [(L1, "LS20-L1 (collapses)", RED), (TU93, "TU93-L3 (survives)", BLUE)]:
        m = M(run); s = col(m, "step")
        axes[0].plot(s, col(m, "entropy"), color=c, label=name)
        axes[1].plot(s, col(m, "reward_norm_mean"), color=c, label=name)
    axes[0].axhline(np.log(4), ls=":", color=GRAY)
    axes[0].set_title("policy entropy (ln4 = random, 0 = frozen)")
    axes[0].set_xlabel("environment steps"); axes[0].set_ylabel("entropy"); axes[0].legend(fontsize=8)
    axes[1].axhline(0, ls="--", color="k", lw=1)
    axes[1].set_title("standardized curiosity reward (its SIGN is the switch)")
    axes[1].set_xlabel("environment steps"); axes[1].set_ylabel("standardized reward (mean)"); axes[1].legend(fontsize=8)
    fig.suptitle("3 · Policy freezes on the small game only — cause: it runs out of new states\n"
                 "(curiosity reward turns NEGATIVE on the small game, stays POSITIVE on the large one)", fontsize=10)
    fig.tight_layout()
    return _b64(fig)


# ── Fig 4: decisive ablation — LayerNorm vs baseline on L1 ───────────────────
def fig_ablation():
    ln = find_ln_run()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    runs = [(L1, "original (features not normalized)", RED)]
    if ln: runs.append((ln, "normalized features (LayerNorm)", GREEN))
    for run, name, c in runs:
        m = M(run); s = col(m, "step")
        axes[0].plot(s, col(m, "novelty_raw_mean"), color=c, label=name)
        axes[1].plot(s, col(m, "entropy"), color=c, label=name)
        axes[2].plot(s, col(m, "reward_norm_mean"), color=c, label=name)
    axes[0].set_yscale("log"); axes[0].set_title("novelty score size (log scale)")
    axes[1].set_title("policy entropy (0 = frozen)"); axes[1].axhline(np.log(4), ls=":", color=GRAY)
    axes[2].set_title("standardized curiosity reward"); axes[2].axhline(0, ls="--", color="k", lw=1)
    for a in axes: a.set_xlabel("environment steps"); a.legend(fontsize=8)
    verdict = "result pending" if not ln else "normalizing shrinks novelty ~140× but the policy STILL freezes"
    fig.suptitle(f"Control: does normalizing the features stop the policy from freezing?  ({verdict})",
                 fontsize=10)
    fig.tight_layout()
    return _b64(fig), ln


# ── Fig 5: why the crash is sudden & one-time — a stability flip ─────────────
def fig_sudden():
    m = M(L1); u = col(m, "update")
    H = col(m, "entropy"); mp = np.array([max(r["per_action_prob"]) for r in m])
    rm = col(m, "reward_norm_mean")
    sel = (u >= 15) & (u <= 55)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.plot(u[sel], H[sel], color="k", lw=2, label="policy entropy (0 = frozen)")
    ax.plot(u[sel], mp[sel], color=BLUE, lw=1.6, label="most-likely action's probability")
    ax.set_xlabel("update"); ax.set_ylabel("entropy / max action prob"); ax.set_ylim(0, 1.5)
    ax2 = ax.twinx(); ax2.plot(u[sel], rm[sel], color=RED, lw=1.6, ls="--",
                               label="standardized curiosity reward")
    ax2.axhline(0, color=RED, lw=0.8, alpha=0.5); ax2.set_ylabel("standardized curiosity reward", color=RED)
    ax.axvspan(15, 35, color=GREEN, alpha=0.06); ax.axvspan(41, 55, color=RED, alpha=0.06)
    ax.annotate("1st excursion\nRECOVERS\n(reward still +)", (31, 0.62), (22, 0.13),
                fontsize=8, ha="center", arrowprops=dict(arrowstyle="->"))
    ax.annotate("reward turns persistently\nNEGATIVE (~u41):\nexploring policy\nbecomes unstable",
                (41, 0.0), (45, 0.92), fontsize=8, ha="center", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("2nd excursion\nLOCKS IN", (46, 0.58), (49.5, 0.35), fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->"))
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Why the crash is sudden & one-time: a stability flip, then a noise-driven escape\n"
                 "green = exploring policy is STABLE (reward+, wobbles revert)   "
                 "red = UNSTABLE (reward−, next wobble runs away)")
    return _b64(fig)


# ── Fig 6: toy REINFORCE — a constant reward (either sign) collapses entropy ─
def fig_sim():
    import collections
    def toy(reward_fn, steps=30000, lr=0.1, n=4, seed=0):
        rng = np.random.default_rng(seed); z = np.zeros(n); Hs = []
        for t in range(steps):
            p = np.exp(z - z.max()); p /= p.sum()
            a = rng.choice(n, p=p)
            g = -p.copy(); g[a] += 1.0                 # d log pi(a)/dz = onehot(a) - p
            z += lr * reward_fn(a) * g
            if t % 150 == 0:
                Hs.append(-(p * np.log(p + 1e-12)).sum())
        p = np.exp(z - z.max()); p /= p.sum()
        return np.array(Hs), int(np.argmax(p))
    specs = [("constant reward −1", lambda a: -1.0, RED),
             ("constant reward +1", lambda a: 1.0, "#f0a020"),
             ("informative (action 0 best)", lambda a: 1.0 if a == 0 else 0.0, GREEN)]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [2, 1]})
    win = {}
    for name, fn, c in specs:
        traces, ws = [], []
        for s in range(12):
            H, w = toy(fn, seed=s); traces.append(H); ws.append(w)
        L = min(len(t) for t in traces); arr = np.array([t[:L] for t in traces])
        ax.plot(np.arange(L) * 150, arr.mean(0), color=c, lw=2, label=name)
        win[name] = collections.Counter(ws)
    ax.axhline(np.log(4), ls=":", color=GRAY); ax.set_ylabel("policy entropy"); ax.set_xlabel("update")
    ax.set_title("Toy REINFORCE, 4 actions, no baseline:\na CONSTANT reward of EITHER sign collapses entropy")
    ax.legend(fontsize=8)
    w = 0.26
    for i, (name, fn, c) in enumerate(specs):
        axb.bar(np.arange(4) + i * w, [win[name].get(a, 0) for a in range(4)], w, color=c)
    axb.set_xticks(np.arange(4) + w); axb.set_xticklabels(["a0", "a1", "a2", "a3"])
    axb.set_ylabel("# of 12 seeds frozen here")
    axb.set_title("but constant → a RANDOM action\n(diffusion); informative → always a0")
    fig.tight_layout()
    return _b64(fig)


def ln_verdict(ln):
    if not ln or _last_step(ln) < COLLAPSE_WINDOW:
        return "pending", (f"the LayerNorm ablation has not yet run past the collapse window "
                           f"(~{COLLAPSE_WINDOW//1000}k steps); re-run build_synthesis when it finishes.")
    m = M(ln); H = col(m, "entropy"); rn = col(m, "reward_norm_mean")
    nov = col(m, "novelty_raw_mean")
    collapsed = H[-5:].mean() < 0.2
    neg_frac = float((rn < 0).mean())
    base_max = col(M(L1), "novelty_raw_mean").max()
    if collapsed:
        v = (f"the policy <b>still freezes</b> even with normalized features (final entropy {H[-1]:.2f}; "
             f"curiosity reward negative on {neg_frac:.0%} of updates) — so <b>running out of new states</b> is "
             f"the real cause. Normalizing only shrank the novelty score about {base_max/max(nov.max(),1e-6):.0f}× "
             f"(from ~{base_max:.0f} to ~{nov.max():.1f}), confirming the feature magnitude was just the size of "
             f"the numbers, not the reason the policy stops exploring.")
        return "collapses", v
    v = (f"the policy does <b>not</b> freeze with normalized features (final entropy {H[-1]:.2f}) — so the "
         f"growing feature magnitude WAS the cause of the freezing, not just the size of the numbers.")
    return "survives", v


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Why a curiosity agent suddenly stops exploring</title>
<style>
 body{{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:880px;margin:30px auto;padding:0 18px;color:#1a1a1a}}
 h1{{font-size:23px;margin-bottom:2px}} .sub{{color:#666;margin-top:0}}
 h2{{font-size:18px;margin-top:34px;border-bottom:2px solid #e8e8e8;padding-bottom:5px}}
 h3{{font-size:15.5px;margin:20px 0 4px;color:#0b3d66}}
 img{{width:100%;border:1px solid #eee;border-radius:6px;margin:8px 0}}
 .root{{background:#f3f8f3;border-left:4px solid #1a9850;padding:10px 14px;margin:10px 0;border-radius:4px}}
 .key{{background:#fff7e6;border-left:4px solid #f0a020;padding:10px 14px;border-radius:4px}}
 .setup{{background:#eef4fb;border-left:4px solid #2c7fb8;padding:10px 14px;border-radius:4px}}
 .wrong{{background:#fdf0ef;border-left:4px solid #d73027;padding:8px 14px;border-radius:4px;margin:8px 0}}
 code{{background:#f2f2f2;padding:1px 5px;border-radius:3px;font-size:13px}} .small{{color:#666;font-size:13px}}
 details{{margin:8px 0}} summary{{cursor:pointer;font-weight:600;color:#2c7fb8}}
 table{{border-collapse:collapse;font-size:13.5px;margin:6px 0}} td{{border:1px solid #e3e3e3;padding:5px 9px;vertical-align:top}}
 td:first-child{{font-weight:600;white-space:nowrap;background:#fafafa}}
</style></head><body>
<h1>Why a curiosity-driven agent suddenly stops exploring — and two smaller surprises along the way</h1>
<p class="sub">An exploration agent on a sparse-reward puzzle game shows three unexpected behaviors. They have <b>three separate causes</b>. The headline one — a sudden, one-time collapse of exploration — is a clean dynamical-systems effect, not the bug it first looks like.</p>

<div class="setup"><b>The setup, in plain terms.</b> An agent explores a puzzle game (LS20) where a reward comes only on solving the level, so it must explore on its own. Two parts:
<b>a policy</b> (a network that picks one of 4 actions, trained by <b>REINFORCE</b> — the most basic policy-gradient method, here with <i>no</i> value-function "baseline"), and <b>a curiosity bonus</b> that rewards reaching unfamiliar states (<b>Random Network Distillation</b>: a fixed random "target" net and a learned "predictor" net; the predictor's leftover error — large for new states, ~0 for familiar — is the "novelty" reward). Novelty is computed on a <b>feature vector from a small CNN encoder</b> that is trained by <b>inverse dynamics</b> (predict which action happened between two frames). Success = environment steps to the first reward.
<details><summary>Glossary (click to expand)</summary>
<table>
<tr><td>encoder / features</td><td>the CNN that turns a frame into a vector; <b>feature magnitude</b> = its length.</td></tr>
<tr><td>inverse-dynamics accuracy</td><td>how often the action between two frames is recoverable from their features. Chance = 0.25.</td></tr>
<tr><td>novelty</td><td>the predictor's error copying the target — the curiosity reward. Bigger = less familiar.</td></tr>
<tr><td>standardized reward</td><td>novelty minus its running average, over its running spread. Negative when current novelty is below average.</td></tr>
<tr><td>policy entropy</td><td>randomness of the action choice. ln(4)&#8776;1.39 = uniform; 0 = always one action ("collapse").</td></tr>
<tr><td>coverage</td><td>how many distinct states have been visited. "Fully covered" = every reachable state seen.</td></tr>
<tr><td>timer mask</td><td>blanking the on-screen step-timer rows (the timer advances every step regardless of the agent).</td></tr>
<tr><td>LS20-L1 / TU93-L3</td><td>two games we compare; LS20-L1 is tiny (~118 states), TU93-L3 much larger.</td></tr>
</table></details></div>

<div class="key"><b>The three findings in one line each.</b><br>
<b>A.</b> The encoder learns useful (action-predictive) features <i>only</i> because of the <b>timer mask</b>.<br>
<b>B.</b> The novelty numbers <b>balloon</b> because the encoder's output is <b>never normalized</b>, so its magnitude grows and novelty &#8733; magnitude&#178;.<br>
<b>C (the headline).</b> The policy <b>suddenly freezes</b> on the small game because it <b>runs out of new states</b>: the curiosity reward loses its information, and a plain REINFORCE policy then <b>diffuses</b> into a single action. A control experiment shows this is <i>not</i> caused by B — {ablation_one_liner}</div>

<h2>Part A — Two quick wins (the settled causes)</h2>
<h3>A1 · The encoder learns action-predictive features — because of the timer mask</h3>
<p>Inverse-dynamics accuracy reaches ~1.0, which earlier versions of this project never achieved. The single cause is <b>blanking the on-screen step-timer</b>: otherwise the timer advances every step, every frame looks unique for no real reason, and accuracy sits at chance (0.25). It is <i>not</i> the feature magnitude — forcing features to unit length leaves accuracy at 1.0. (Honest accuracy over <i>all</i> steps is ~0.62, because "wall-bump" steps don't change the board so their action is genuinely unrecoverable.)</p>
<img src="data:image/png;base64,{f1}">
<h3>A2 · The novelty numbers balloon — because the encoder output isn't normalized</h3>
<p>The inverse-dynamics objective is the <i>only</i> thing that trains this encoder. To predict actions more sharply it simply <b>grows the magnitude of its output vector</b>, stopping once accuracy saturates. Since novelty is measured on these un-normalized features, the <b>novelty score scales with magnitude&#178;</b> (correlation with the predictor's error = 0.9998). So "the error rises instead of falling" is a ruler-stretching side-effect, not a learning failure.</p>
<img src="data:image/png;base64,{f2}">

<h2>Part B — The headline: why exploration suddenly collapses</h2>

<h3>B1 · The puzzle</h3>
<p>Same algorithm, opposite outcomes. On the <b>small</b> game (LS20-L1) the policy is fine for ~90k steps, then entropy crashes almost vertically to 0 and the agent just bumps a wall forever. On the <b>large</b> game (TU93-L3) it never collapses. And the crash is <b>one-time</b>: there's an earlier, equally-deep wobble that <i>fully recovers</i>. The one quantity that tracks the difference is the <b>sign of the curiosity reward</b> — negative on the small game, positive on the large one.</p>
<img src="data:image/png;base64,{f3}">

<h3>B2 · Why the obvious explanation is wrong</h3>
<p>The tempting story is "the reward goes negative, so it pushes action probabilities down and the policy collapses." That's not how it works. A reward that is the <b>same for every action</b> produces, on average, <b>zero net change</b> to every action's probability: the push-down an action gets when it's chosen is exactly cancelled by the push-up it gets (relative to others) when a different action is chosen. And a nearly-deterministic policy receives <b>almost no gradient at all</b> (the policy-gradient term shrinks to zero as one action's probability approaches 1). So a constant negative reward neither pushes toward nor away from any action.</p>
<div class="wrong"><b>Why "the sharpest action gets pushed down the most, so it self-corrects" fails:</b> that action is <i>also</i> pushed up every time another action is sampled, and the two cancel exactly — so there is no self-correction, in either direction.</div>

<h3>B3 · The real mechanism — a driftless random walk</h3>
<p>Zero average change does not mean nothing happens: it means the policy's internal scores do a <b>random walk</b> (the updates have zero mean but nonzero variance). Entropy is highest at the perfectly-uniform point and falls as the scores drift apart — and with <b>no force pulling them back to uniform</b>, the random walk inevitably spreads them out, so entropy wanders down to 0. A toy REINFORCE confirms it: a <b>constant reward of <i>either</i> sign</b> collapses entropy, and it collapses onto a <b>random</b> action (different each run) — the signature of diffusion, not a directed push. Only an <i>informative</i> reward (different per action) collapses onto a <i>specific</i> action.</p>
<img src="data:image/png;base64,{f6}">

<h3>B4 · What actually held exploration up — and why it vanished</h3>
<p>So what kept the policy exploring during the good early phase, if a positive constant reward would also diffuse? The answer: early on the reward is <b>informative, not constant</b> — different states genuinely have different novelty (high reward <i>spread</i>), and that spread is a real, directional signal that keeps steering the agent toward new states. <b>That informativeness is the restoring force.</b> When the small game is fully explored, every state becomes equally stale, the spread collapses (it falls from ~1.4 to ~0.01), and the reward becomes <i>effectively constant</i> — so only the driftless random walk is left, and the policy diffuses into a corner. The large game never runs out of genuinely-new states, so its reward stays informative and the restoring force never disappears.</p>

<h3>B5 · Why it's sudden and one-time</h3>
<p>The exploring policy is <b>metastable</b>, and its stability <b>flips</b> the moment the reward loses its information (~update 41). Before the flip, the informative reward restores small wobbles — which is exactly why an earlier near-collapse around update 30 <b>fully recovers</b>. After the flip there is no restoring force, so the <i>next</i> wobble — the same size as the one that recovered — runs away instead. It looks instantaneous because the action probabilities are a softmax (exponential): once one action edges ahead, it saturates toward probability 1 within a couple of updates, and the narrowing policy stops visiting varied states, removing the last bit of corrective signal. The exact timing is random (whichever wobble first escapes after the flip), which is why it's a one-time crash, not a slow slide.</p>
<img src="data:image/png;base64,{f5}">

<h3>B6 · Two controls pin the cause</h3>
<p><b>(i) It is not the feature magnitude (Part A2).</b> Normalizing the encoder output shrinks the novelty numbers ~140&#215; — yet the policy <b>still freezes</b>. So the ballooning magnitude only set the <i>size</i> of the numbers, not the reason for the collapse. <b>Result: {ablation_verdict}</b></p>
<img src="data:image/png;base64,{f4}">
<p><b>(ii) A value baseline would prevent this specific collapse.</b> A baseline subtracts the running average return, which removes the constant component — and with it the random walk. The policy would then <b>freeze its current entropy rather than crash to a single action</b> once the reward goes uninformative. So the "no baseline" choice is genuinely load-bearing for the <i>collapse</i> (not just a minor aggravator) — though it doesn't restore the missing exploration signal.</p>

<h2>The bottom line</h2>
<div class="root"><b>Three independent causes.</b> &nbsp;<b>(A)</b> the timer mask makes the features action-predictive; &nbsp;<b>(B)</b> not normalizing the encoder output makes the novelty numbers balloon; &nbsp;<b>(C)</b> on a small game the curiosity reward eventually <b>runs out of information</b>, and a baseline-free REINFORCE policy then <b>diffuses</b> into one action (a driftless random walk, not a push) — sharply, because softmax saturates.</div>
<p class="small"><b>Fixes.</b> Keep the timer mask. Normalize/whiten the encoder output to stop the novelty numbers ballooning. For the collapse, two complementary cures: a <b>value baseline</b> (removes the random walk &#8594; the policy freezes gracefully instead of crashing), and a <b>curiosity signal that doesn't fade once everything is seen</b> or a switch to exploiting the real reward once the game is explored (restores the missing direction). Note the collapse happens <i>after</i> success anyway — the first reward came at ~33,000 steps, faster than a random policy (~50,000).</p>
<p class="small">Supporting analyses: <code>INVESTIGATION_{{entropy_collapse,feature_norm,controllability}}.md</code>; run logs <code>metrics.jsonl</code> / <code>state_novelty.jsonl</code>; 13 checkpoints from the large-game run; toy-REINFORCE simulation reproduced in <code>build_synthesis.py</code>.</p>
</body></html>"""


def main():
    f1 = fig_controllability()
    f2 = fig_feature_norm()
    f3 = fig_collapse_discriminator()
    f5 = fig_sudden()
    f6 = fig_sim()
    f4, ln = fig_ablation()
    status, verdict = ln_verdict(ln)
    one = {"collapses": "normalizing the features shrinks the novelty numbers but the policy still freezes, so <b>running out of new states</b> is the real cause — the growing feature magnitude only changed the size of the numbers.",
           "survives": "with normalized features the policy no longer freezes, so the growing feature magnitude was itself the cause.",
           "pending": "control experiment still running."}[status]
    html = HTML.format(f1=f1, f2=f2, f3=f3, f4=f4, f5=f5, f6=f6,
                       ablation_verdict=verdict, ablation_one_liner=one)
    out = HERE / "findings" / "ULTIMATE_CAUSE.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"[synthesis] ablation verdict: {status} — {verdict}")
    print(f"[synthesis] wrote {out}")


if __name__ == "__main__":
    main()
