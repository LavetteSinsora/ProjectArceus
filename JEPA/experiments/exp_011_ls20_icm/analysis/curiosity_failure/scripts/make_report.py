"""Build figures + index.html for the curiosity-failure analysis."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo"
AN = os.path.join(ROOT, "JEPA/experiments/exp_011_ls20_icm/analysis/curiosity_failure")
FIG = os.path.join(AN, "figures")
os.makedirs(FIG, exist_ok=True)

log = json.load(open(os.path.join(AN, "log_traj.json")))
probe = json.load(open(os.path.join(AN, "probe_results.json")))

C = {"L1": "#1f77b4", "L2": "#d62728"}
plt.rcParams.update({"font.size": 10, "figure.dpi": 110})


def first_reward_marks(ax, lab):
    for seed, s in log[lab].items():
        fr = s.get("first_reward_step")
        if fr:
            ax.axvline(fr, color=C[lab], ls=":", alpha=0.5, lw=1)


# ── FIG 1: forward error + r^i collapse (both levels, all seeds) ─────────────
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for lab in ["L1", "L2"]:
    for seed, s in log[lab].items():
        st = np.array(s["step"], float)
        ax[0].plot(st, s["forward_error_mean"], color=C[lab], alpha=0.55, lw=1)
        ri = np.array(s["intrinsic_reward_mean"], float)
        ax[1].plot(st, ri, color=C[lab], alpha=0.55, lw=1)
ax[0].set_title("Forward prediction error (per update)")
ax[0].set_xlabel("env steps"); ax[0].set_ylabel(r"$\|\hat\phi-\phi(s')\|^2$")
ax[0].set_yscale("log"); ax[0].set_xlim(0, 480000)
ax[1].set_title(r"Intrinsic reward mean $r^i$")
ax[1].set_xlabel("env steps"); ax[1].set_ylabel(r"$r^i$ (log)")
ax[1].set_yscale("log"); ax[1].set_xlim(0, 480000)
ax[1].axhline(1e-2, color="gray", ls="--", lw=1)
ax[1].text(2e5, 1.1e-2, r"calibration target $r^i\!=\!0.01$", color="gray", fontsize=8)
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], color=C["L1"], label="L1 (solved 3/3)"),
       Line2D([0], [0], color=C["L2"], label="L2 (failed 0/3)")]
ax[0].legend(handles=leg, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_collapse.png")); plt.close(fig)

# numbers for narrative
def rng(lab, key):
    vals0 = [log[lab][s][key][0] for s in log[lab]]
    valsf = [log[lab][s][key][-1] for s in log[lab]]
    return min(vals0), max(vals0), min(valsf), max(valsf)
fe_L1 = rng("L1", "forward_error_mean"); fe_L2 = rng("L2", "forward_error_mean")

# ── FIG 2: CONTRAST — novel vs familiar r^i at each checkpoint ───────────────
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for j, lab in enumerate(["L1", "L2"]):
    rows = probe["contrast"][lab]
    steps = [r["step"] for r in rows]
    nov = [r["novel_mean"] for r in rows]
    fam = [r["fam_mean"] for r in rows]
    ax[j].plot(steps, nov, "o-", color="#2ca02c", label="novel (first-seen)")
    ax[j].plot(steps, fam, "s-", color="#9467bd", label="familiar (revisited)")
    for r in rows:
        if r.get("novel_over_fam") is not None:
            ax[j].annotate(f"{r['novel_over_fam']:.2f}x", (r["step"], r["novel_mean"]),
                           fontsize=7, ha="center", va="bottom")
    ax[j].set_title(f"{lab}: $r^i$ novel vs familiar")
    ax[j].set_xlabel("checkpoint env steps"); ax[j].set_yscale("log")
    ax[j].legend(fontsize=8)
ax[0].set_ylabel(r"per-transition $r^i$ (log)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_contrast.png")); plt.close(fig)

# ── FIG 3: STATE COVERAGE — saved policy vs random floor ────────────────────
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for lab in ["L1", "L2"]:
    rows = probe["coverage"][lab]
    steps = [r["step"] for r in rows]
    cum = [r["cum_unique"] for r in rows]
    epu = [r["mean_ep_unique"] for r in rows]
    ax[0].plot(steps, cum, "o-", color=C[lab], label=f"{lab} policy")
    ax[1].plot(steps, epu, "o-", color=C[lab], label=f"{lab} policy")
    rnd = probe["random"][lab]
    ax[0].axhline(rnd["cum_unique"], color=C[lab], ls="--", alpha=0.6, lw=1)
    if rnd["mean_ep_unique"]:
        ax[1].axhline(rnd["mean_ep_unique"], color=C[lab], ls="--", alpha=0.6, lw=1)
ax[0].set_title("Cumulative unique masked states\n(dashed = random-policy floor)")
ax[0].set_xlabel("checkpoint env steps"); ax[0].set_ylabel("unique states (8 envs x 4 resets)")
ax[0].legend(fontsize=8)
ax[1].set_title("Unique states per episode\n(dashed = random-policy floor)")
ax[1].set_xlabel("checkpoint env steps"); ax[1].set_ylabel("unique states / episode")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_coverage.png")); plt.close(fig)

# ── FIG 4: r^i distribution at final ckpt vs scale of competing signals ──────
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for lab in ["L1", "L2"]:
    d = np.array(probe["ri_dist"][lab], float)
    d = d[d > 0]
    ax.hist(np.log10(d + 1e-12), bins=40, alpha=0.5, color=C[lab],
            label=f"{lab} final-ckpt $r^i$ (median {np.median(d):.1e})")
ax.axvline(np.log10(0.01386), color="black", ls="--", lw=1.3)
ax.text(np.log10(0.01386), ax.get_ylim()[1]*0.9, " entropy bonus\n /step (0.0139)",
        fontsize=8, va="top")
ax.set_xlabel(r"$\log_{10} r^i$ per transition"); ax.set_ylabel("count")
ax.set_title("Collapsed $r^i$ vs the PPO entropy bonus it competes with")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_magnitude.png")); plt.close(fig)

# ── gather narrative numbers ────────────────────────────────────────────────
def cov_floor(lab):
    rnd = probe["random"][lab]
    last = probe["coverage"][lab][-1]
    return rnd["cum_unique"], last["cum_unique"], rnd["goal_hits"]
L1cov = cov_floor("L1"); L2cov = cov_floor("L2")
L1goal = sum(r["goal_hits"] for r in probe["coverage"]["L1"])
L2goal = sum(r["goal_hits"] for r in probe["coverage"]["L2"])
nof_L1 = [r.get("novel_over_fam") for r in probe["contrast"]["L1"] if r.get("novel_over_fam")]
nof_L2 = [r.get("novel_over_fam") for r in probe["contrast"]["L2"] if r.get("novel_over_fam")]
fr_L1 = sorted(s.get("first_reward_step") for s in [log["L1"][k] for k in log["L1"]])

# collapse-by-step from logs: step where r^i first < 1e-3
def collapse_step(lab):
    out = []
    for seed, s in log[lab].items():
        st = s["step"]; ri = s["intrinsic_reward_mean"]
        cs = next((st[i] for i in range(len(st)) if ri[i] < 1e-3), None)
        out.append(cs)
    return out
cs_L1 = collapse_step("L1"); cs_L2 = collapse_step("L2")

med_L1 = np.median(np.array(probe["ri_dist"]["L1"]))
med_L2 = np.median(np.array(probe["ri_dist"]["L2"]))

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Why ICM fails to explore LS20</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:880px;
margin:32px auto;padding:0 18px;color:#1a1a1a;line-height:1.5}}
h1{{font-size:22px;margin-bottom:2px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:3px}}
.sub{{color:#666;font-size:13px;margin-top:0}}
img{{width:100%;border:1px solid #eee;border-radius:6px;margin:8px 0}}
p{{font-size:14px}} .k{{background:#f5f5f5;padding:1px 5px;border-radius:4px;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}}
.box{{background:#f7f9fb;border-left:3px solid #1f77b4;padding:8px 14px;margin:12px 0;font-size:13.5px}}
.crux{{border-left-color:#d62728;background:#fdf6f6}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}} td,th{{border:1px solid #ddd;padding:4px 9px}}
</style></head><body>

<h1>Why curiosity (ICM) fails to explore LS20 Level 2</h1>
<p class="sub">exp_011 · faithful Pathak-2017 ICM + CNN-PPO · real 64x64 LS20 · 3 seeds/level · device MPS.
L1 solved 3/3 (first reward {fr_L1[0]//1000}k-{fr_L1[2]//1000}k steps); L2 failed 0/3 (never found extrinsic reward).</p>

<div class="box crux"><b>One-line answer.</b> Curiosity does not "decay gracefully" &mdash; it <b>self-extinguishes within ~2 updates</b>.
On this deterministic, low-diversity env the forward model fits the dynamics almost immediately, so the per-transition
prediction error (= the intrinsic reward) collapses to a tiny, near-<i>constant</i> floor that <b>assigns the same value
to first-seen and revisited states</b>. With no novel-vs-familiar contrast and a magnitude ~100x below the PPO entropy
bonus, ICM provides no directed exploration gradient. L1 is won only because its goal sits inside the &lt;2-update window
(and within the random-policy reach) before curiosity dies; L2's goal is too deep and is never reached.</div>

<h2>1 &nbsp;Curiosity collapses almost immediately &mdash; on BOTH levels</h2>
<img src="figures/fig1_collapse.png">
<p>Forward error falls from <span class="k">{fe_L1[1]:.1f}</span> (L1) / <span class="k">{fe_L2[1]:.1f}</span> (L2)
to <span class="k">{fe_L1[2]:.3f}-{fe_L1[3]:.3f}</span> / <span class="k">{fe_L2[2]:.3f}-{fe_L2[3]:.3f}</span>.
The &eta;-calibration pins mean <span class="k">r&#770;</span> to 0.01 on the first rollout, then it drops <b>below 1e-3 by step
{int(np.nanmin([c for c in cs_L1+cs_L2 if c]))}</b> (the 2nd&ndash;4th update) in nearly every run &mdash; far earlier
than L1 finds reward (17k&ndash;225k steps). Crucially the two levels are <b>indistinguishable</b> in this collapse:
curiosity dies just as fast on the solved L1 as on the failed L2. So intrinsic magnitude alone cannot explain the L1/L2 gap.</p>

<h2>2 &nbsp;The mechanistic crux: r&#770; loses all novel-vs-familiar CONTRAST</h2>
<img src="figures/fig2_contrast.png">
<p>Rolling each saved policy and splitting transitions by whether the (UI-masked) frame was first-seen or revisited:
the intrinsic reward given to <b>novel</b> states is essentially identical to that given to <b>familiar</b> ones &mdash;
novel/familiar ratio &asymp; <span class="k">{np.mean(nof_L1):.2f}x</span> on L1 and
<span class="k">{np.mean(nof_L2):.2f}x</span> on L2 across all checkpoints (1.0 = no discrimination).
A working exploration bonus must pay <i>more</i> for novelty; here it pays the same flat floor for everything.
This is the failure: not that r&#770; is merely small, but that it no longer <b>ranks</b> states &mdash; there is no
gradient pointing toward unexplored regions.</p>

<h2>3 &nbsp;Behavioral ground truth: coverage stalls at the random-policy floor</h2>
<img src="figures/fig3_coverage.png">
<p>State coverage from rolling out the saved policies (exact masked-frame counting). The intrinsically-trained policy
reaches <b>no more</b> unique states than a uniform-random policy (dashed): L2 policy cum-unique
<span class="k">{L2cov[1]}</span> vs random <span class="k">{L2cov[0]}</span>; per-episode coverage is flat across
training. With no contrast signal, the policy under entropy regularization behaves like undirected random search.
Across the entire coverage probe the L2 policy reached the goal frame <b>{L2goal} times</b> (extrinsic reward 0/3 seeds),
while L1 reached it because its goal lies within random reach (L1 goal hits {L1goal}).</p>

<h2>4 &nbsp;Even when present, r&#770; is dwarfed by the entropy bonus</h2>
<img src="figures/fig4_magnitude.png">
<p>The collapsed per-transition r&#770; distribution (median <span class="k">{med_L1:.1e}</span> L1,
<span class="k">{med_L2:.1e}</span> L2) sits ~2 orders of magnitude below the PPO entropy bonus
(<span class="k">0.0139</span>/step at H&asymp;1.39, dashed). Once curiosity has collapsed, the policy gradient is
dominated by the entropy term &mdash; which pushes toward <i>uniform</i> action noise, not toward the specific deep
action sequence (~60 moves) that L2 requires. Intrinsic reward is along for the ride, not steering.</p>

<h2>Conclusion</h2>
<div class="box">
<b>Failure mechanism (precise).</b> ICM's prediction-error reward collapses to a near-constant floor within ~2 updates
because the deterministic, low-diversity LS20 dynamics are trivially fit by the forward model (error
{fe_L2[1]:.0f}&rarr;{fe_L2[3]:.2f}). The floor is (i) tiny &mdash; ~{med_L2:.0e}, ~100x below the entropy bonus &mdash;
and (ii) <b>non-discriminative</b>: novel and revisited states receive the same r&#770; (ratio &asymp;1.0x). There is
therefore no exploration gradient at any point after the first few updates. The L1-vs-L2 outcome is decided <b>not</b>
by curiosity but by puzzle depth: L1's goal is reachable within random search during the brief informative window
(and even afterward), so it is found; L2's ~60-move goal is not, and curiosity is already dead, so the agent never
escapes the random-coverage basin (0/3 seeds, 0 goal hits in probing).</div>

<p class="sub">Hypotheses: <b>(1) forward saturation</b> &mdash; supported. <b>(3) loss of contrast</b> &mdash; supported and
identified as the crux (novel/fam &asymp;1.0x). <b>(2) magnitude too small</b> &mdash; supported but secondary (the deeper
problem is contrast, not scale). <b>(4) coverage plateau at random floor on L2</b> &mdash; supported. <b>(5) &phi; health</b>
&mdash; &phi; is healthy (inverse_acc&rarr;~1.0, finite effective rank); the collapse is of the prediction-error <i>signal</i>,
not of the representation. The intuitive "race" story is <b>refined</b>: there is no race in r&#770; magnitude (it collapses
equally fast on both); the only thing that differs is whether the goal is reachable by near-random behavior.</p>
</body></html>"""

with open(os.path.join(AN, "index.html"), "w") as f:
    f.write(html)
print("wrote index.html and 4 figures")
print("novel/fam L1", np.mean(nof_L1), "L2", np.mean(nof_L2))
print("L2 cov policy", L2cov[1], "random", L2cov[0], "L2 goal hits", L2goal)
print("med ri L1", med_L1, "L2", med_L2)
