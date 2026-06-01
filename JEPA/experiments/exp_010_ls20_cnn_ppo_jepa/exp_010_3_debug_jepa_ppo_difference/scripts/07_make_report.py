"""Build figures + the single self-contained report.html (PNGs as base64).

Run from repo root after 04 (sweep) and 06 (curves) have produced:
  data/curves.json, data/phantom.json, data/sweep_results.txt,
  data/control140_seed0.json, data/rescue_seed0.json
and the logged metrics.jsonl of exp_010_0/1/2.
"""
import json, glob, base64, io, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP=Path(__file__).resolve().parents[2]; DBG=Path(__file__).resolve().parents[1]; DATA=DBG/"data"; FIG=DBG/"figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi":120,"font.size":10,"axes.grid":True,"grid.alpha":0.3,"axes.axisbelow":True})

def metrics(sub):
    rs=sorted(glob.glob(str(EXP/sub/"runs"/"*"/"metrics.jsonl"))); rs=[r for r in rs if sum(1 for _ in open(r))>=50]
    return [json.loads(l) for l in open(rs[-1])] if rs else []
def b64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
def load(name):
    p=DATA/name; return json.load(open(p)) if p.exists() else None

figs={}
# ---------- Fig 1 : the phenomenon ----------
runs={"10_0  CNN+PPO (random enc)":("exp_010_0_cnn_ppo_baseline","#1b9e77"),
      "10_1  joint-online JEPA":("exp_010_1_jepa_joint_online","#d95f02"),
      "10_2  JEPA-pretrained enc":("exp_010_2_jepa_random_pretrain","#7570b3")}
fig,ax=plt.subplots(1,3,figsize=(13,3.3))
for name,(sub,col) in runs.items():
    r=metrics(sub)
    ev=[(x["step"],x["success_rate"]) for x in r if x.get("success_rate") is not None]
    if ev: ax[0].plot(*zip(*ev),label=name,color=col,lw=2)
    en=[(x["step"],x["policy_entropy"]) for x in r if x.get("policy_entropy") is not None]
    if en: ax[1].plot(*zip(*en),color=col,lw=1.3)
    vl=[(x["step"],x["value_loss"]) for x in r if x.get("value_loss") is not None][:60]
    if vl: ax[2].plot(*zip(*vl),color=col,lw=1.3)
ax[0].set(title="(a) eval success rate",xlabel="env steps",ylabel="success",ylim=(-.05,1.05)); ax[0].legend(fontsize=7,loc="center right")
ax[1].set(title="(b) policy entropy",xlabel="env steps",ylabel="entropy (max 1.386)")
ax[2].set(title="(c) value loss, first 60 updates",xlabel="env steps",ylabel="value loss")
fig.suptitle("Fig 1 — The phenomenon: a random-init encoder SOLVES real LS20-L1; both JEPA-encoder variants get 0%",y=1.05,fontsize=11,weight="bold")
figs["fig1"]=b64(fig)

# ---------- Fig 2 : entropy collapses w/o reward; deleting critic rescues ----------
cv=load("curves.json")
if cv:
    lab={"random_base":"random enc + critic  (solves @18)","resc_base":"JEPA enc + critic  (fails)",
         "froz_base":"frozen JEPA enc + critic  (fails)","resc_nobase":"JEPA enc, NO critic  (rescued @34)"}
    col={"random_base":"#1b9e77","resc_base":"#7570b3","froz_base":"#e7298a","resc_nobase":"#111"}
    fig,ax=plt.subplots(1,2,figsize=(11,3.5))
    for k in lab:
        if k not in cv: continue
        e=cv[k]["entropy"]; c=cv[k]["cum_success"]; x=range(1,len(e)+1)
        ls="--" if k=="resc_nobase" else "-"
        ax[0].plot(x,e,label=lab[k],color=col[k],lw=2,ls=ls); ax[1].plot(x,c,color=col[k],lw=2,ls=ls)
    ax[0].axhline(1.386,color="grey",ls=":",lw=.8); ax[0].set(title="(a) policy entropy vs update  (terminal-only reward)",xlabel="PPO update",ylabel="entropy"); ax[0].legend(fontsize=7)
    ax[1].set(title="(b) cumulative successes",xlabel="PPO update",ylabel="cum. successes (symlog)",yscale="symlog")
    fig.suptitle("Fig 2 — Structured-encoder entropy collapses with ZERO reward present; removing the value critic rescues it",y=1.05,fontsize=10.5,weight="bold")
    figs["fig2"]=b64(fig)

# ---------- Fig 3 : phantom-advantage mechanism ----------
ph=load("phantom.json"); rows=ph["rows"]; kinds=["random","raw","resc"]; C={"random":"#1b9e77","raw":"#d95f02","resc":"#7570b3"}
disp=["random","JEPA\n(raw)","JEPA\n(norm-matched)"]   # 'resc' = norm-matched, NOT 'rescue'
def agg(k,c): i=ph["columns"].index(c); return float(np.mean([r[i] for r in rows if r[0]==k]))
fig,ax=plt.subplots(1,3,figsize=(12,3.5)); xs=np.arange(3)
ax[0].bar(xs,[agg(k,"feat_cos") for k in kinds],color=[C[k] for k in kinds]); ax[0].set(title="(a) raw feature mean-cosine\n1.0 = state-invariant component dominates",xticks=xs,xticklabels=disp,ylim=(0,1.05))
ax[1].bar(xs,[agg(k,"V_std") for k in kinds],color=[C[k] for k in kinds]); ax[1].set(title="(b) std of V(s) over states\nphantom value structure (zero reward)",xticks=xs,xticklabels=disp)
ax[2].bar(xs,[-agg(k,"dent_1upd") for k in kinds],color=[C[k] for k in kinds]); ax[2].set(title="(c) entropy lost in ONE update\n(one zero-reward PPO update)",xticks=xs,xticklabels=disp,ylabel="-Δ entropy")
fig.suptitle("Fig 3 — Mechanism: an informative encoder lets the critic paint state-structured V(s) from noise → phantom advantages",y=1.06,fontsize=10.3,weight="bold")
figs["fig3"]=b64(fig)

# ---------- Fig 4 : L2 transfer probe ----------
probe=None
for l in open(DATA/"sweep_results.txt"):
    if l.startswith("[probe]"): probe=json.loads(l.split(":",1)[1])
if probe:
    encs=["random","jepa","ppo_l1"]; w=.35; fig,ax=plt.subplots(1,2,figsize=(11,3.5))
    cmap={0:["#bbb","#f0a050","#b0a8d8"],1:["#777","#d95f02","#7570b3"]}
    for j,lvl in enumerate(["L1","L2"]):
        ax[0].bar(np.arange(3)+(j-.5)*w,[probe[lvl][e]["idm_acc"] for e in encs],w,label=lvl,color=cmap[j])
        ax[1].bar(np.arange(3)+(j-.5)*w,[probe[lvl][e]["fwd_r2"] for e in encs],w,label=lvl,color=cmap[j])
    ax[0].axhline(.25,color="red",ls=":",lw=1); ax[0].set(title="(a) inverse-dynamics action accuracy\n(frozen enc; chance=0.25)",xticks=range(3),xticklabels=encs,ylim=(0,1)); ax[0].legend(fontsize=8)
    ax[1].set(title="(b) forward-prediction R²\n(frozen enc; higher = dynamics preserved)",xticks=range(3),xticklabels=encs,ylim=(0,1)); ax[1].legend(fontsize=8)
    fig.suptitle("Fig 4 — L2 transfer probe: only the PPO-task encoder loses forward-dynamics structure on the new level",y=1.06,fontsize=10.3,weight="bold")
    figs["fig4"]=b64(fig)

# ---------- Fig 6 : phantom advantage directly shapes the policy (early training) ----------
psj=load("policy_shaping.json")
if psj:
    runs=list(psj.values()); kinds=["random","raw","resc"]
    col={"random":"#1b9e77","raw":"#d95f02","resc":"#7570b3"}
    name={"random":"random CNN","raw":"JEPA (raw)","resc":"JEPA (norm-matched)"}
    fig,ax=plt.subplots(1,3,figsize=(13,3.6))
    for k in kinds:
        rs=[r for r in runs if r["kind"]==k]
        H=np.array([r["H"] for r in rs]); KL=np.array([r["KL"] for r in rs]); x=np.arange(H.shape[1])
        ax[0].plot(x,H.mean(0),color=col[k],lw=2,label=name[k]); ax[0].fill_between(x,H.min(0),H.max(0),color=col[k],alpha=0.15)
        ax[1].plot(x,KL.mean(0),color=col[k],lw=2,label=name[k]); ax[1].fill_between(x,KL.min(0),KL.max(0),color=col[k],alpha=0.15)
        # scatter cumulative per-action advantage vs Δlogit (4 actions x 3 seeds)
        xs=np.concatenate([r["advA_cum"] for r in rs]); ys=np.concatenate([r["dlogit"] for r in rs])
        ax[2].scatter(xs,ys,color=col[k],s=26,alpha=0.8,label=name[k])
    ax[0].axhline(1.386,color="grey",ls=":",lw=.8); ax[0].set(title="(a) policy entropy vs update\n(zero reward — no success yet)",xlabel="PPO update",ylabel="entropy"); ax[0].legend(fontsize=7)
    ax[1].set(title="(b) KL(marginal policy ‖ uniform)\nhow far the policy has been pushed",xlabel="PPO update",ylabel="KL"); ax[1].legend(fontsize=7)
    ax[2].axhline(0,color="grey",lw=.6); ax[2].axvline(0,color="grey",lw=.6)
    ax[2].set(title="(c) the driver: per-action\nΣ normalized advantage  vs  Δ logit",xlabel="cumulative normalized advantage (per action)",ylabel="Δ logit (per action)"); ax[2].legend(fontsize=7)
    fig.suptitle("Fig 6 — Direct evidence: the critic's phantom advantage systematically reinforces actions for JEPA, ~3× more than for random",y=1.05,fontsize=10.2,weight="bold")
    figs["fig6"]=b64(fig)

for k,v in figs.items(): open(FIG/f"{k}.png","wb").write(base64.b64decode(v))
p5=FIG/"fig5_value_hist.png"   # produced by 08_value_histogram.py
if p5.exists(): figs["fig5"]=base64.b64encode(open(p5,"rb").read()).decode()
p7=FIG/"fig7_value_spread_matched.png"   # produced by 11_value_spread_matched_norm.py
if p7.exists(): figs["fig7"]=base64.b64encode(open(p7,"rb").read()).decode()
p8=FIG/"fig8_advantage_dist.png"   # produced by 12_advantage_distribution.py
if p8.exists(): figs["fig8"]=base64.b64encode(open(p8,"rb").read()).decode()
p9=FIG/"fig9_credit_localization.png"   # produced by 14_credit_localization.py
if p9.exists(): figs["fig9"]=base64.b64encode(open(p9,"rb").read()).decode()
p10=FIG/"fig10_baseline_differentiation.png"   # produced by 16_baseline_differentiation
if p10.exists(): figs["fig10"]=base64.b64encode(open(p10,"rb").read()).decode()

# ---------------- assemble HTML ----------------
def img(k): return f'<img src="data:image/png;base64,{figs[k]}" style="width:100%;max-width:1100px;border:1px solid #ddd;border-radius:6px"/>' if k in figs else "<em>(figure pending)</em>"
sw={}
for l in open(DATA/"sweep_results.txt"):
    if l.startswith("[ppo]"):
        m=re.match(r"\[ppo\] (\S+):",l); sw[m.group(1)]=json.loads(l.split(":",1)[1])
def cell(tag):
    d=sw.get(tag);
    if not d: return "–"
    return f'{d["first_succ"]}' if d["first_succ"] else "—"
P=probe
HTML=f"""<!doctype html><html><head><meta charset="utf-8"><title>Phantom Advantages — exp_010</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:960px;margin:32px auto;padding:0 20px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:25px;line-height:1.25;margin-bottom:2px}} h2{{font-size:19px;margin-top:34px;border-bottom:2px solid #eee;padding-bottom:4px}}
h3{{font-size:15px;margin-top:22px;color:#333}} .sub{{color:#666;font-size:14px;margin-top:4px}}
.abs{{background:#f7f7f9;border-left:4px solid #7570b3;padding:14px 18px;border-radius:4px;font-size:14.5px}}
.k{{background:#eef7f1;border-left:4px solid #1b9e77;padding:10px 16px;border-radius:4px;margin:14px 0}}
figure{{margin:22px 0;text-align:center}} figcaption{{font-size:12.5px;color:#555;margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}} th,td{{border:1px solid #ddd;padding:6px 9px;text-align:center}} th{{background:#f3f3f6}}
code{{background:#f0f0f3;padding:1px 5px;border-radius:3px;font-size:13px}} .win{{color:#1b9e77;font-weight:600}} .fail{{color:#c0392b;font-weight:600}}
.tag{{display:inline-block;background:#7570b3;color:#fff;font-size:11px;padding:1px 7px;border-radius:10px;vertical-align:middle}}
</style></head><body>

<h1>Phantom Advantages: why a “good” world-model representation is a <i>bad</i> initialization for sparse-reward RL</h1>
<div class="sub">An automated-research dive into experiment&nbsp;010 (real LS20 Level&nbsp;1, terminal-only reward). All artifacts in <code>exp_010_ls20_cnn_ppo_jepa/exp_010_3/</code>. Single seed unless noted; key results replicated over 3 seeds.</div>

<div class="abs"><b>Abstract.</b> On the real LS20 game a vanilla CNN+PPO agent with a <i>randomly-initialized</i> encoder reliably solves Level&nbsp;1 under terminal-only reward, yet warm-starting the same agent from a JEPA world-model encoder — pretrained to predict the environment’s own dynamics — drops success to <b>0%</b>. The intuitive culprit (the JEPA features are “bad”/mis-scaled) is wrong: we falsify three such hypotheses in turn (feature content, feature norm, encoder drift). The real cause is a single, simple mechanism we call the <b>phantom-advantage trap</b>. Under terminal-only reward, <i>before the first success there is no task signal</i>, so the only gradient acting on the policy comes through the <b>value critic</b>. An <i>informative</i> encoder gives the critic enough expressive power to fit its own approximation noise into a <i>state-structured</i> value function V(s); GAE + advantage-normalization convert that structure into <b>persistent, state-consistent phantom advantages</b>; PPO “exploits” them and collapses the exploratory policy — entropy falls even though the agent has <i>never seen a reward</i> — so the rare goal is never stumbled upon. A random encoder — whose features are dominated by a large <i>state-invariant</i> component (only ~1% of the feature norm varies across states; note they are <i>not</i> low-rank — mean-centered, their pairwise cosine is ≈0 just like JEPA’s) — yields V(s)≈const, so the critic fits V≈0 immediately, no phantom advantages arise, and its policy stays a stable explorer and finds the reward. We confirm the mechanism causally: <b>deleting the value baseline</b> (Monte-Carlo advantages) rescues the JEPA encoder, which then solves at every seed (u≈21–34). Finally we show the deeper tension: in a frozen-encoder probe the JEPA representation is the <i>more transferable</i> one to Level&nbsp;2 (it preserves forward-dynamics structure that the task-specific PPO encoder discards) — i.e. <b>the very richness that makes a representation transferable is what makes it sabotage sparse-reward exploration.</b></div>

<h2>1 · The phenomenon</h2>
<p>Three variants share an identical architecture, PPO recipe, and seed; only the encoder treatment differs (system card §2). <b>10_0</b> trains the CNN from scratch; <b>10_1</b> adds an online JEPA(+IDM) loss to the shared encoder; <b>10_2</b> initializes the encoder from a JEPA model pretrained on random-policy data, then fine-tunes with PPO. Only 10_0 ever clears the level.</p>
<figure>{img("fig1")}<figcaption><b>Fig 1.</b> (a) 10_0 → 100%; 10_1 and 10_2 → 0%. (b) 10_0’s entropy stays near-uniform until it finds reward, then exploits; the JEPA variants’ entropy slides down with no reward. (c) 10_2’s value loss spikes to ~9.5 at init (large-norm pretrained features) — the clue that first sent us down a wrong path.</figcaption></figure>

<h2>2 · Three intuitive hypotheses — all falsified</h2>
<div class="k">Single-seed controlled comparison (seed 0, identical seeded policy/value heads; <b>only</b> the encoder differs). Outcome = whether PPO ever observes a success in 140 updates.</div>
<table>
<tr><th>Encoder treatment</th><th>u1 grad / value-loss</th><th>1st success</th><th>entropy w/o reward</th><th>outcome</th></tr>
<tr><td>random-init</td><td>0.34 / 0.008</td><td>update 18</td><td>pinned ≈1.386</td><td class="win">SOLVES</td></tr>
<tr><td>JEPA, as-is (=10_2)</td><td>29 / 9.5</td><td>never</td><td>collapses</td><td class="fail">FAILS</td></tr>
<tr><td>JEPA, <b>norm-matched</b> (quiet)</td><td>0.10 / 0.001</td><td>never</td><td>1.37 → 0.61</td><td class="fail">FAILS</td></tr>
<tr><td>JEPA, <b>frozen</b></td><td>30.8 / 11.1</td><td>never</td><td>1.37 → 0.68</td><td class="fail">FAILS</td></tr>
</table>
<p style="background:#fff8e6;border-left:4px solid #e0a800;padding:8px 14px;border-radius:4px;font-size:13.5px"><b>Note on “frozen” — two different experiments.</b> The table’s <b>“JEPA, frozen”</b> row is a full <b>140-update PPO training run</b> (encoder frozen, but the policy <i>and value heads are trained</i>); it fails. The H1 evidence below instead comes from a <b>critic-free coverage rollout</b>: the <i>untrained, never-updated</i> initial policy just sampling actions (no PPO, no value head in the loop). These probe different things — representational <i>capacity</i> vs optimization <i>dynamics</i> — so they do not contradict.</p>
<p><b>(H1) “The JEPA features are bad / capture the wrong latents.”</b> Falsified by the critic-free coverage rollout: the untrained initial policy on the frozen JEPA encoder visits about the same number of distinct states as on the random encoder (~100–120) and still reaches the goal occasionally — so the features <i>can</i> represent and reach the goal; they are <i>not</i> incapable. (The “JEPA, frozen” training row fails for a different reason — its trainable critic still hallucinates advantages over the frozen features — which is evidence <i>for</i> the §3 mechanism, not for H1.) <b>(H2) “The 66× feature-norm mismatch mis-calibrates the value head and floods the encoder with gradients.”</b> This was even baked into the repo’s <code>analyze_runs.py</code>. Falsified: rescaling the JEPA features to the random-init norm makes the run perfectly “quiet” (grad 0.10, value-loss 0.001) yet it <i>still never succeeds</i>, and its entropy <i>still</i> slides down. <b>(H3) “A trainable structured encoder drifts under value-fitting.”</b> Falsified: <i>freezing</i> the encoder doesn’t help either. The common thread of all three failures: the encoder is <b>informative</b> (feature-cosine ≈ 0.6, not the random encoder’s ≈0.99), and entropy collapses <i>with zero reward ever seen</i>.</p>

<h2>3 · The mechanism: the phantom-advantage trap</h2>
<div style="background:#f7f7f9;border:1px solid #e2e2e8;border-radius:5px;padding:10px 16px;font-size:13px">
<b>Reading guide / definitions.</b>
<ul style="margin:6px 0 0 0;padding-left:18px">
<li><b>zero-reward update</b> — a PPO update on a rollout in which no success occurred, so every reward is 0; the only thing driving the actor is the critic.</li>
<li><b>entropy lost in one update (Fig 3c)</b> — start from the untrained policy, do <i>one</i> zero-reward PPO update, measure the drop in mean action entropy. Isolates what the critic alone does to exploration in a single step.</li>
<li><b>advantage normalization</b> — PPO replaces the advantages in each minibatch by their z-score, <code>Â ← (Â − mean)/(std+ε)</code>. Mean-subtraction kills a <i>constant</i> advantage; dividing by std rescales whatever pattern remains to unit size, so a tiny-but-structured signal is <b>amplified</b> to the same influence as a large one — only the <i>shape</i> (which states/actions are above average) matters, not the magnitude.</li>
<li><b>state-consistent</b> — the advantage is a reproducible function of (state,action), not per-sample noise, so its push on the policy does not average to zero across samples/updates. Evidence: the <i>monotone</i> entropy decline in Fig 2 (noise would random-walk, not steadily fall), and a positive split-half policy-gradient cosine (gradients computed on two disjoint halves of a zero-reward rollout agree: ~0.48 for JEPA vs ~0.33 for random).</li>
<li><b>“resc” / norm-matched</b> — the JEPA encoder with its output rescaled to the random encoder’s feature norm. It is <i>not</i> the “rescue” run (that is <code>resc_nobase</code> in Fig 2, the no-critic variant).</li>
</ul></div>
<p>The PPO actor gradient is <code>E[ ∇log π(a|s) · Â ]</code>: the reward <i>never</i> enters the actor directly — only through the advantage <code>Â_t = Σ (γλ)^l δ_{{t+l}}</code>, with <code>δ_t = r_t + γV(s_{{t+1}}) − V(s_t)</code> (and through having trained the critic, which itself touches the actor only via Â). With terminal-only reward and no success yet, every <code>r_t = 0</code>, so the advantage collapses to a <b>pure function of the critic</b>: <code>Â_t = Σ (γλ)^l (γV(s_{{t+1+l}}) − V(s_{{t+l}}))</code> — the whole actor gradient is now built from value <i>differences</i> with zero reward content (hence “phantom”). This does <i>not</i> vanish because the reward is 0; it vanishes only when <b>V(s) is state-independent</b>, for two reasons: (i) PPO normalizes advantages by subtracting the batch mean (<code>ppo.py</code>), so a constant V → constant δ → constant Â → ≈0; and (ii) the score function has zero mean, <code>E[∇log π(a|s)]=0</code>, so a state-independent advantage produces no consistent update regardless. When V(s) <i>varies</i> across states, Â becomes <b>state-(and action-)correlated</b>, survives normalization, and the policy reliably climbs the critic’s spurious value landscape: it exploits pure value-function noise. What sets V(s)’s expressiveness is the <i>state-varying fraction</i> of the (raw, pre-head) features: for the random encoder only ~1% of the feature norm varies across states (V_std≈0.02), so the critic drives V→0 almost instantly and no phantom signal persists; for the JEPA encoder ~11% varies (V_std≈5.5), giving the critic a real state-structured target it fits and re-generates over many updates. (This is <i>not</i> a rank/“collapse” distinction — mean-centered, both encoders’ features are full-rank; it is purely how large the state-dependent component is relative to a common state-invariant one.)</p>
<figure>{img("fig3")}<figcaption><b>Fig 3.</b> One untrained, zero-reward rollout (mean of 3 seeds). “JEPA (raw)” and “JEPA (norm-matched)” (= <code>resc</code>, the rescaled encoder — <i>not</i> “rescue”) have low raw feature-cosine (a) — their features genuinely swing across states (~11% of norm) — and produce a state-structured V(s) (b: V_std up to ~5.5 vs random’s ~0.02), and lose entropy after a single zero-reward PPO update (c). The random encoder’s features are dominated by a state-invariant component (~1% varies), so V≈const and it loses no entropy.</figcaption></figure>
<figure>{img("fig5")}<figcaption><b>Fig 5.</b> The mechanism made visual. Roll out a uniform-random policy, collect the <i>same</i> visited states for every encoder, and evaluate V(s) with an <b>untrained, identically-seeded</b> value head. (a) Raw scale: the random CNN gives a sharp spike at V≈0 (V_std 0.018) — there is no state-structure for the critic to exploit — while the JEPA encoder produces a broad V(s) spanning ≈[−21,+4] (V_std 5.2). (b) Zoomed: even the <b>norm-matched</b> JEPA encoder (same feature norm 2.2 as random) gives a V(s) ~4× wider than random (V_std 0.078 vs 0.018) — so the spread is created by the feature <i>directions</i> (state structure), not the feature norm. This state-structured V(s) is exactly what GAE turns into the phantom advantages of §3.</figcaption></figure>
<h3>3.1 · Does the phantom advantage actually shape the policy? (direct evidence)</h3>
<p>Fig 5 shows the <i>source</i> (V varies for JEPA, not random). Fig 6 closes the loop to the <i>consequence</i>. The chain, in the simplest 1-step terms: with r=0, the advantage of action a in state s is <code>Â(s,a) ∝ γV(s′) − V(s)</code>. If neighbouring states have different value (JEPA: V_std 5.2), some actions get a systematically positive Â and are reinforced; if V≈const (random), no action is preferentially reinforced. We run 25 zero-reward updates (no success occurs) and, on a fixed probe set, watch the policy; from each rollout we log the whole-batch-<b>normalized</b> advantage broken down by action — the literal scalar that multiplies <code>∇log π</code>.</p>
<figure>{img("fig6")}<figcaption><b>Fig 6.</b> 3 seeds/encoder, terminal-only reward, no success in this window. (a) Entropy falls fastest for JEPA(raw) (→1.00) vs random (→1.25). (b) The <i>marginal</i> action distribution is pushed ~1.6× further from uniform for JEPA (KL 0.22 vs 0.13). (c) The driver: each point is one action’s <b>cumulative normalized advantage</b> (x) vs the change in its logit (y) over the 25 updates. Points lie on a positive diagonal for <i>all</i> encoders — the actions that accumulate positive advantage are exactly the ones whose logits rise (the advantage causally moves the policy) — but JEPA’s points spread ~3× further along x (max |cum adv| 1.31 vs random’s 0.41): the critic injects a much stronger, systematic, sign-consistent advantage. So the phantom advantage is real, it is state/action-structured, and it demonstrably reshapes the policy — strongly for JEPA, weakly for random (which is why random is fragile, not immune).</figcaption></figure>
<p><b>Marginal bias vs state-dependence — the metric that really separates them.</b> The marginal KL in Fig 6b looks almost the same for random and norm-matched JEPA, which seems to undercut the story. It doesn’t — KL(marginal‖uniform) only measures a <i>global</i> action bias and is blind to <i>state-conditional</i> shaping (per-state preferences that cancel in the average). Decompose the per-state entropy exactly: <code>mean_s H(π(·|s)) = log4 − KL(π̄‖uniform) − I(S;A)</code>, where <code>I(S;A)</code> is the state–action mutual information = how state-dependent the policy became.</p>
<table>
<tr><th>encoder</th><th>per-state entropy</th><th>KL(marginal‖uniform)</th><th>I(S;A) — state-dependence</th></tr>
<tr><td>random</td><td>1.253</td><td>0.133</td><td class="win">0.000</td></tr>
<tr><td>JEPA (raw)</td><td>1.000</td><td>0.216</td><td>0.170</td></tr>
<tr><td>JEPA (norm-matched)</td><td>1.109</td><td>0.149</td><td class="fail">0.128</td></tr>
</table>
<p>random and norm-matched JEPA have nearly equal <i>marginal</i> KL (0.133 vs 0.149), but their <b>state-dependence differs categorically: I(S;A)=0.000 for random vs 0.128 for JEPA</b> (≈ raw’s 0.170). random’s drift is a harmless uniform “biased coin” applied identically in every state — it does not change <i>where</i> exploration goes. The JEPA-driven phantom advantage instead makes the policy genuinely <i>state-dependent</i> — different states pushed toward different actions — bending the random walk into a state-conditioned path that gets trapped away from the goal. That state-conditional structure is exactly what the marginal KL cannot see, and it is the real signature of the phantom-advantage trap.</p>
<p><b>Where does the difference live, before vs after PPO’s advantage normalization?</b> A natural worry is that normalization should wash the phantom out. Fig 8 traces the GAE advantage itself through normalization. (a) The <i>raw</i> advantage scale differs ~250× (std 0.023 random, 5.6 JEPA-raw). (b) After normalization (mean 0, std 1 for everyone), the <b>marginal distributions are ~identical</b> — and, checked over 8 seeds, even the higher moments don’t robustly differ (skew/kurtosis ≈ 0 for all; the per-seed skew we first saw was noise). So normalization erases not just the scale but the marginal <i>shape</i> too — the difference is genuinely <b>not</b> in the advantage’s marginal distribution. (c) What it cannot erase is the <i>state-conditional</i> structure: the same I(S;A) as above (0.000 random vs 0.128/0.170 JEPA). <b>Conclusion:</b> normalization removes everything <i>except</i> the dependence of the advantage on the state — and that residual is precisely what reshapes the policy. The phantom survives normalization as <i>structure</i>, not as a fatter histogram.</p>
<figure>{img("fig8")}<figcaption><b>Fig 8.</b> GAE advantage through PPO’s normalization, zero-reward rollouts pooled over 8 seeds. (a) raw std differs ~250×; (b) normalized marginals overlap almost exactly (the difference is NOT in the marginal distribution — normalization equalizes scale and shape); (c) the surviving difference is the policy’s state-dependence I(S;A) (0.000 vs 0.128 vs 0.170). The phantom advantage is normalization-invariant because it lives in <i>which</i> states/actions are favoured, not in the spread of the advantage values.</figcaption></figure>
<h3>3.3 · The actual mechanism: credit <i>generalization</i>, not advantage value</h3>
<p>A natural objection: at a given state <code>s</code> the advantage value is the <i>same</i> for both encoders (Fig 8b), so how can the representation change what gets reinforced? Because the policy is <code>π(a|s)=softmax(W·h(s))</code>: a gradient step that reinforces action <code>a</code> at <code>s</code> changes <code>π(a|s′)</code> at <i>every other</i> state <code>s′</code> by an amount set by how similar <code>h(s′)</code> is to <code>h(s)</code>. The advantage at <code>s</code> is identical; what differs is <b>how the update generalizes to other states</b>. Fig 9 measures this directly (matched feature norm): reinforce one action at one state, then look at the induced <code>Δπ</code> everywhere as a function of representational similarity.</p>
<figure>{img("fig9")}<figcaption><b>Fig 9.</b> Reinforcing ONE action at ONE state, frozen encoders at matched norm (≈2.3). <b>random</b> (left): every state sits at cos-similarity 0.97–1.0 to the target — the encoder can barely tell states apart — so the change is essentially <b>uniform</b> across all states (CV 0.01): credit is <i>smeared into a global policy shift</i>. <b>norm-matched JEPA</b> (right): states span similarity 0.2–1.0 and <code>Δπ</code> rises with similarity (corr 0.96, CV 0.28): credit is <i>localized</i> to states resembling the target. (raw JEPA, norm 150, is omitted — one step saturates the softmax everywhere, a scale artifact.)</figcaption></figure>
<p>This resolves the mechanism precisely. <b>Pre-reward:</b> the critic’s state-correlated phantom advantage, generalized through a <i>localizing</i> (JEPA) representation, writes <i>state-specific</i> action biases → it bends the exploratory walk into a trapped, state-conditioned path. Through a <i>smearing</i> (random) representation the same phantom can only shift the global policy — a harmless biased coin — so undirected coverage survives. <b>Post-reward exploitation</b> is the mirror image: localized credit assignment is exactly what you want to reproduce a winning trajectory. We tested it (script 13): behavior-cloning the 13-step winning L1 trajectory into a fresh head on a frozen encoder, <i>both</i> random and JEPA reproduce it and solve on replay — on L1 the states are already separable enough (random features are full-rank) that the representation gives <b>no</b> exploitation advantage. So on L1 the representation’s effect is confined to the pre-reward exploration phase. <b>On the harder level (L2) it is not.</b></p>
<div class="k"><b>On L2, the JEPA representation beats random (exploitation).</b> We replayed the 60-action Go-Explore L2 solution to get a winning trajectory (all 60 frames pixel-distinct, but only 49 distinct once the step-counter UI rows are masked → ~11 frames share player-state), and behavior-cloned it into a fresh head on each <b>frozen</b> encoder (5 head-init seeds). Frozen <b>JEPA</b>: BC acc 1.000, reproduces the win <b>5/5</b>. Frozen <b>random</b>: BC acc 0.900, <b>0/5</b> — it cannot linearly separate the ~6 near-collinear states the win needs, so greedy replay diverges. (With a <i>trainable</i> encoder both solve.) So when the task requires distinguishing subtly-different states, JEPA’s structured representation is the deciding factor — the same localizing property that traps it pre-reward pays off for credit assignment once a reward exists.</div>
<h3>3.4 · Why the random encoder works: cancellation pre-reward, differentiation post-reward</h3>
<p>The policy-head gradient is <code>Σ_t Â_t (π(·|s_t) − e_{{a_t}}) ⊗ h(s_t)</code>. Write <code>h(s)=h̄+δ(s)</code> (a big shared component + a small state-varying part). It splits into a <b>marginal</b> push (the <code>h̄</code> part, a global shift of all logits) and a <b>state-specific</b> push (the <code>δ(s)</code> part). For the <b>random</b> encoder pre-reward, <code>Â</code> is sign-balanced noise: the marginal push averages to ≈0 (mean-zero advantage), and the state-specific push needs <code>Â</code> to <i>correlate with</i> <code>δ(s)</code> — which it doesn’t (V≈const), so it also averages to ≈0. <b>Both terms cancel across states</b> → the policy stays uniform. This is exactly the cancellation intuition, and it is what <code>I(S;A)=0.000</code> measures. For <b>JEPA</b>, <code>Â</code> (built from <code>V=w·h</code>) <i>is</i> correlated with <code>δ(s)</code>, so the state-specific term does <b>not</b> cancel → state-dependence emerges with no reward (<code>I(S;A)=0.128</code>) — the phantom.</p>
<p><b>What breaks the cancellation? A real reward.</b> Once a success is sampled, the advantage acquires a <i>consistent</i>, reward-correlated component (the states/actions on the path to the goal), no longer noise. That component does not cancel — it backpropagates through the head into the <i>encoder</i> and amplifies the features <code>δ(s)</code> that separate reward-relevant states. Fig 10 is the direct evidence: in the solving baseline the encoder’s feature mean-cosine falls from 0.99 (states ~identical) toward 0.73 (states distinct) — and pairwise feature distance grows 26→65 — as eval success consolidates to 100%. So the random encoder is <i>self-organising</i>: it starts state-blind (good — undirected exploration, Fig 9 left), and the reward it stumbles onto is what carves the state structure it needs to exploit.</p>
<figure>{img("fig10")}<figcaption><b>Fig 10.</b> Baseline (random-init, trainable encoder) over training. Feature mean-cosine (purple) drops 0.99→0.73: the encoder differentiates states. This is driven by reward — it proceeds as eval success (green) is established, not before. The causal loop: state-blind init → unbiased exploration → stumble on reward → reward-correlated (non-cancelling) gradient → encoder differentiates → state-specific policy → solve.</figcaption></figure>
<h3>3.2 · Why doesn’t rescaling the JEPA features fix it? (it’s structure, not norm)</h3>
<p>Rescaling cut the cumulative per-action advantage from 1.31 (raw) to 0.59 (norm-matched) — closer to random’s 0.41 — so it is tempting to think shrinking the features kills the phantom. It doesn’t, for a precise reason: the phantom comes from the feature <i>directions</i> (which states look different), and <b>advantage normalization divides the magnitude back out</b>. Rescaling can only weaken the signal’s <i>consistency</i> slightly; it cannot remove it. Fig 7 isolates this: with the JEPA encoder rescaled to the random encoder’s feature norm (≈2.3) and averaged over 40 independent random value heads, V(s) is still <b>~5× more spread</b> than random’s (V_std 0.086±0.022 vs 0.017±0.003; bands disjoint). So the spread is created purely by the encoder’s <i>structure</i>, at matched magnitude and not from a lucky head.</p>
<figure>{img("fig7")}<figcaption><b>Fig 7.</b> The control suggested by the “is it just the norm?” question. (a) A single random value head, JEPA encoder rescaled to random’s feature norm: random is a tight spike (V_std 0.018), norm-matched JEPA is broad (V_std 0.10). (b) Per-head std of V(s) over 40 random value heads: random clusters at ~0.017, norm-matched JEPA at ~0.086 — non-overlapping. <b>Conclusion:</b> JEPA’s feature <i>structure</i> alone (not its norm) produces a state-spread value function. That is why rescaling leaves norm-matched JEPA on the worse side of the fragile boundary (V_std still ~5× random → still collapses over the 140-update horizon, 1/3 seeds), and why the genuine fix is to remove the critic (advantage = reward-to-go = exactly 0 pre-reward), not to shrink the encoder.</figcaption></figure>

<h2>4 · Causal confirmation: delete the critic, rescue the encoder</h2>
<p>The mechanism makes a sharp prediction: if the <i>value baseline</i> is the source of the phantom gradient, then removing it — using Monte-Carlo advantages (advantage = discounted reward-to-go, which is exactly 0 until a real success) — should restore stable exploration for the JEPA encoder. It does, and reproducibly.</p>
<figure>{img("fig2")}<figcaption><b>Fig 2.</b> Same JEPA (norm-matched) encoder: <b>with</b> the critic, entropy decays under zero reward and the goal is never found; <b>without</b> the critic (dashed), entropy stays pinned at the uniform maximum until update ≈34, then the agent stumbles on the goal and learns normally.</figcaption></figure>
<h3>Robustness (seeds 0–2): “first-success update” (— = never)</h3>
<table>
<tr><th>condition</th><th>seed 0</th><th>seed 1</th><th>seed 2</th><th>summary</th></tr>
<tr><td>JEPA enc + critic (<code>resc_base</code>)</td><td>—</td><td>{cell("resc_base_s1")}</td><td>{cell("resc_base_s2")}</td><td class="fail">1/3, and only late</td></tr>
<tr><td>random enc + critic (<code>random_base</code>)</td><td>18</td><td>{cell("random_base_s1")}</td><td>{cell("random_base_s2")}</td><td>2/3, fragile</td></tr>
<tr><td><b>JEPA enc, NO critic</b> (<code>resc_nobase</code>)</td><td>34</td><td>{cell("resc_nobase_s1")}</td><td>{cell("resc_nobase_s2")}</td><td class="win">3/3, early</td></tr>
</table>
<p>The value critic makes sparse-reward exploration <i>fragile and seed-dependent for both encoders</i> (the random encoder still has a small state-varying component, so it too occasionally manufactures enough phantom structure to fail — e.g. seed 2). Removing the critic makes the rich JEPA encoder the <i>most reliable and fastest</i> explorer. This is the within-encoder controlled result: <code>resc_base</code> vs <code>resc_nobase</code> differ only in the presence of the value baseline.</p>
<div class="k"><b>The right axis is critic vs no-critic, not encoder.</b> It is tempting (and the seed-0 table in §2 invites it) to read this as “random encoder good, JEPA encoder bad.” That over-states it. The three regimes form a <b>continuum in phantom-gradient strength</b>: <b>JEPA + critic</b> = strong, state-consistent phantom (V_std≈5.5, −0.015 entropy/update) → reliably collapses; <b>random + critic</b> = <i>weak</i> phantom (V_std≈0.02, −0.0009/update) → behaves <i>almost</i> like pure random exploration but occasionally accumulates into a collapse (hence 2/3, fragile); <b>JEPA + no critic</b> = phantom is <i>exactly</i> 0 (the advantage is the reward-to-go, which is 0 until a real success) → genuinely pure max-entropy random exploration → reliably finds the reward, then learns. So “random + critic” and “JEPA + no critic” are indeed <i>similar</i> in spirit (both ≈ undirected exploration); they differ only in that the former carries a small destabilising critic term and the latter carries none. The encoder’s informativeness merely sets <i>how strong</i> the critic’s phantom signal is.</div>

<h2>5 · The transfer twist (your Level-2 question)</h2>
<p>Is a JEPA representation actually <i>more transferable</i> than a task-specific PPO one? We froze three encoders — random, JEPA-pretrained-on-L1, and the encoder of the PPO agent that <i>solved</i> L1 — and probed them on held-out Level-2 frames (reached via the engine <code>set_level</code> API) with two label-free read-outs: inverse-dynamics action decoding and forward-dynamics prediction.</p>
<figure>{img("fig4")}<figcaption><b>Fig 4.</b> On L2, action-decodability (a) is similar for all three (a random CNN is a strong baseline — Johnson–Lindenstrauss). But forward-prediction R² (b) cleanly separates them: random {P['L2']['random']['fwd_r2']:.2f} ≈ JEPA {P['L2']['jepa']['fwd_r2']:.2f} ≫ <b>PPO-L1 {P['L2']['ppo_l1']['fwd_r2']:.2f}</b>. The task-specific PPO encoder is the only one whose dynamics structure degrades on the new level (its L1→L2 R² falls {P['L1']['ppo_l1']['fwd_r2']:.2f}→{P['L2']['ppo_l1']['fwd_r2']:.2f}).</figcaption></figure>
<p><b>Why the consensus (“self-supervised reps transfer better than task-specific RL reps”) holds — and where it breaks.</b> It holds <i>against the PPO encoder</i>: maximizing a single task’s return is a compression pressure that discards everything not needed for that task’s reward, so the PPO-L1 encoder keeps L1-goal features and loses general dynamics — exactly what fails to transfer. The JEPA objective, by predicting the environment’s own dynamics, has no such pressure and preserves transferable structure. <b>But it breaks in two honest ways here:</b> (i) a <i>random</i> CNN transfers just as well as JEPA on these low-complexity ARC frames, so “learned SSL features” buy little over a random projection — the consensus is real but oversold at this scale; and (ii) — the central irony of this report — <b>that same transferable richness is precisely what triggers the phantom-advantage trap.</b> A representation’s mutual information with the latent state is simultaneously what makes it portable <i>and</i> what lets a critic hallucinate value structure before any reward exists.</p>

<h2>6 · Takeaways</h2>
<div class="k"><b>1.</b> “Representation health” metrics (effective rank, feature diversity, world-model accuracy) <i>do not</i> predict RL usefulness: 10_2 had the highest effective rank and failed. &nbsp; <b>2.</b> Under sparse reward, the diagnostic that matters is <b>entropy stability before the first reward</b> — if entropy falls with zero reward seen, the critic is hallucinating advantages. &nbsp; <b>3.</b> The cheap fix is to <b>silence the critic until the first reward</b> (Monte-Carlo / no-baseline warm-up, or critic-free exploration), <i>not</i> to re-scale or freeze the encoder. &nbsp; <b>4.</b> Transferability and RL-init quality can be <b>anti-correlated</b>: the better the world-model, the worse the cold-start explorer.</div>

<h2>7 · Reproduce</h2>
<p>From repo root. Scripts in <code>{DBG.name}/scripts/</code>, data in <code>{DBG.name}/data/</code>, figures in <code>{DBG.name}/figures/</code>.</p>
<pre style="background:#f5f5f7;padding:12px;border-radius:5px;font-size:12.5px;overflow-x:auto">
uv run python .../exp_010_3/scripts/01_phantom_advantage_probe.py    # Fig 3 data
uv run python .../exp_010_3/scripts/02_quiet_frozen_control.py        # H2/H3 table
uv run python .../exp_010_3/scripts/03_nobaseline_rescue.py           # seed-0 rescue
uv run python .../exp_010_3/scripts/04_parallel_sweep.py              # seeds 1-2 + L2 probe (parallel)
uv run python .../exp_010_3/scripts/06_figure_curves.py              # Fig 2 hero curves
uv run python .../exp_010_3/scripts/07_make_report.py                # figures + this report
</pre>
<div class="sub">Caveats: single game (LS20), Level 1→2 only, ≤3 seeds, 70–140 PPO updates per run; the L2 probe uses self-supervised read-outs, not end-to-end L2 solving (terminal-only PPO does not solve L2 from any init — an exploration limit, consistent with the phantom-advantage account). Conclusions are mechanistic and within-game; treat cross-game generality as a hypothesis.</div>
</body></html>"""
open(DBG/"report.html","w").write(HTML)
print("wrote",DBG/"report.html","figs:",list(figs))
