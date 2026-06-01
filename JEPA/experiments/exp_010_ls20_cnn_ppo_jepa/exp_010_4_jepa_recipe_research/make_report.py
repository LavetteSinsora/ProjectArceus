"""Build the exp_010_4 findings report (single self-contained HTML) from
/tmp/jepa4/metrics.json, and copy the best-recipe encoder into ./artifacts/.
All numbers are real, produced by study.py on the real LS20 env."""
from __future__ import annotations
import base64, io, json, os, shutil, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open("/tmp/jepa4/metrics.json"))
os.makedirs(f"{HERE}/artifacts", exist_ok=True)

RECIPES = [r for r in ["baseline_sg", "vicreg", "sigreg", "ema"] if r in M]
REFS = [r for r in ["random_init", "exp_010_2"] if r in M]
COL = {"baseline_sg": "#c0392b", "vicreg": "#1b9e77", "sigreg": "#e67e22",
       "ema": "#7570b3", "random_init": "#999", "exp_010_2": "#444"}
LAB = {"baseline_sg": "stop-grad only", "vicreg": "VICReg", "sigreg": "SIGReg (mine)",
       "ema": "EMA target", "random_init": "random init", "exp_010_2": "exp_010_2 (old)"}
def g(r, k): return M[r]["final"][k]

def b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(b.getvalue()).decode()

# fig1: effective-rank dynamics
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for r in RECIPES:
    c = M[r]["curve"]
    if c: ax.plot([p["step"] for p in c], [p["rank"] for p in c], "-o", ms=3,
                  color=COL[r], label=LAB[r], lw=2)
for r in REFS:
    ax.axhline(g(r, "rank"), ls="--", lw=1, color=COL[r], label=LAB[r])
ax.set_xlabel("JEPA step"); ax.set_ylabel("effective rank of trunk (/256)")
ax.set_title("Effective rank during training"); ax.legend(fontsize=8); fig1 = b64(fig)

allk = RECIPES + REFS
# fig2: rank + per-dim std (std exposes the 'high-rank but near-zero' collapse)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].bar([LAB[r] for r in allk], [g(r,"rank") for r in allk], color=[COL[r] for r in allk])
ax[0].set_ylabel("effective rank /256"); ax[0].set_title("Effective rank (scale-free)")
ax[0].tick_params(axis='x', rotation=30, labelsize=8)
ax[1].bar([LAB[r] for r in allk], [max(g(r,"std"),1e-4) for r in allk], color=[COL[r] for r in allk])
ax[1].set_yscale("log"); ax[1].set_ylabel("mean per-dim std (log)")
ax[1].set_title("Per-dim std — exposes near-zero collapse"); ax[1].tick_params(axis='x', rotation=30, labelsize=8)
fig2 = b64(fig)

# fig3: usefulness as a prior — IDM action-decodability (the discriminative metric)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].bar([LAB[r] for r in allk], [g(r,"idm_acc") for r in allk], color=[COL[r] for r in allk])
ax[0].axhline(0.25, ls=":", color="k"); ax[0].axhline(g("random_init","idm_acc"), ls="--", color="#999")
ax[0].set_ylabel("linear IDM accuracy"); ax[0].set_title("Action-decodability (chance .25; dashed=random-init)")
ax[0].tick_params(axis='x', rotation=30, labelsize=8)
ax[1].bar([LAB[r] for r in allk], [g(r,"cos") for r in allk], color=[COL[r] for r in allk])
ax[1].set_ylim(0,1.02); ax[1].set_ylabel("mean pairwise cosine"); ax[1].set_title("Anisotropy (→1 = collapse)")
ax[1].tick_params(axis='x', rotation=30, labelsize=8); fig3 = b64(fig)

# best recipe: must be non-collapsed (std>0.05 AND rank>=8), then max IDM accuracy
def ok(r): return g(r,"std") > 0.05 and g(r,"rank") >= 8
cands = [r for r in RECIPES if ok(r)]
best = max(cands, key=lambda r: g(r,"idm_acc")) if cands else max(RECIPES, key=lambda r: g(r,"idm_acc"))
src = f"/tmp/jepa4/encoder_{best}.pt"
if os.path.exists(src): shutil.copy(src, f"{HERE}/artifacts/encoder_best_{best}.pt")

def row(r):
    f = M[r]["final"]
    return (f"<tr><td style='text-align:left'>{LAB[r]}</td><td>{f['rank']:.1f}</td>"
            f"<td>{f['std']:.4f}</td><td>{f['cos']:.3f}</td><td>{f['norm']:.1f}</td>"
            f"<td>{f['idm_acc']:.3f}</td><td>{f['loc_r2']:.2f}</td></tr>")
rows = "".join(row(r) for r in allk)
ri_idm = g("random_init","idm_acc")

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>exp_010_4 — best JEPA prior for LS20 L1</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 20px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:23px;margin-bottom:2px}} h2{{font-size:17px;margin-top:28px;border-bottom:2px solid #eee;padding-bottom:4px}}
.sub{{color:#666;font-size:13.5px}} .tldr{{background:#f7f7f9;border-left:4px solid #1b9e77;padding:12px 16px;border-radius:4px;font-size:14px}}
.k{{background:#eef7f1;border-left:4px solid #1b9e77;padding:9px 14px;border-radius:4px;margin:12px 0;font-size:13.8px}}
.w{{background:#fdf2ef;border-left:4px solid #c0392b;padding:9px 14px;border-radius:4px;margin:12px 0;font-size:13.8px}}
figure{{margin:16px 0;text-align:center}} img{{max-width:100%;border:1px solid #eee;border-radius:4px}}
figcaption{{font-size:12px;color:#555;margin-top:4px}} code{{background:#f0f0f3;padding:1px 5px;border-radius:3px;font-size:12.5px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}} th,td{{border:1px solid #ddd;padding:5px 8px;text-align:center}} th{{background:#f3f3f6}}
.win{{color:#1b9e77;font-weight:600}} .bad{{color:#c0392b;font-weight:600}}</style></head><body>

<h1>Finding the best JEPA <i>prior</i> for LS20 Level&nbsp;1</h1>
<div class="sub">exp_010_4. The world model exists to hand PPO the <b>best frozen state representation</b>. We
compare anti-collapse recipes for an action-conditioned forward-JEPA on the <b>real</b> LS20 env (60k random
transitions), with the <b>fixed</b> exp_010 CNN trunk (256-d, post-ReLU) + a 128-d projector; prediction in
projector space; eval on the trunk. 1000 steps each on MPS, shared init &amp; data order. All numbers real
(study.py). Modern grounding: LeJEPA/SIGReg (Balestriero &amp; LeCun, arXiv:2511.08544) — isotropic-Gaussian
embeddings are provably the best prior for linear probing.</div>

<div class="tldr"><b>TL;DR.</b> A frozen <b>random</b> CNN trunk is already a surprisingly strong probe baseline
(action-decodability {ri_idm:.2f}). Naive stop-grad JEPA <b>makes it worse</b> (anisotropic collapse,
IDM&nbsp;{g('baseline_sg','idm_acc'):.2f}). Among the recipes, <b class="win">VICReg is the clear winner</b> and
the <b>only one that beats random init</b>: effective rank {g('vicreg','rank'):.0f}/256, IDM
{g('vicreg','idm_acc'):.2f} (vs {ri_idm:.2f} random), healthy per-dim std. <b>My SIGReg and EMA variants
collapsed</b> under these settings (SIGReg → near-zero embeddings, IDM at chance; EMA → rank&nbsp;{g('ema','rank'):.1f}).
Artifact: <code>artifacts/encoder_best_{best}.pt</code>.</div>

<h2>1 · Effective rank during training</h2>
<figure><img src="data:image/png;base64,{fig1}"><figcaption>Only VICReg lifts and holds the effective rank
(to {g('vicreg','rank'):.0f}, above random-init's {g('random_init','rank'):.0f}). EMA's rank crashes to ~1–2.
Stop-grad partially collapses. SIGReg's rank looks mid (~18) — but that is <b>misleading</b> (next panel).</figcaption></figure>
<div class="w"><b>Effective rank is scale-free, so it can lie.</b> SIGReg keeps a spread <i>shape</i>
(rank≈{g('sigreg','rank'):.0f}) while the embedding shrinks to near-zero magnitude
(std&nbsp;{g('sigreg','std'):.4f}, ‖h‖&nbsp;{g('sigreg','norm'):.1f}) — a collapse rank alone misses. Always
pair rank with per-dim std and a downstream probe.</div>

<h2>2 · Representation health: rank vs magnitude</h2>
<figure><img src="data:image/png;base64,{fig2}"><figcaption>Left: effective rank. Right: mean per-dim std
(log) — VICReg keeps real variance; stop-grad and SIGReg shrink to ~0; EMA inflates a tiny subspace.</figcaption></figure>

<h2>3 · What matters: usefulness as a prior</h2>
<figure><img src="data:image/png;base64,{fig3}"><figcaption>Left: frozen linear inverse-dynamics probe (can the
action be read off (h_t,h_t+1)?; chance 0.25, dashed = random-init {ri_idm:.2f}). Right: anisotropy. VICReg is
the only recipe that <b>improves</b> action-decodability over a random encoder; the collapsed recipes fall to /
below random.</figcaption></figure>

<h2>4 · All numbers (final, step 1000)</h2>
<table><tr><th>recipe</th><th>eff. rank /256</th><th>per-dim std</th><th>anisotropy cos</th><th>‖h‖</th>
<th>IDM acc</th><th>agent-loc R²</th></tr>{rows}</table>
<div class="sub">Agent-loc R² (frame-diff proxy) is ~0.40 for every non-collapsed encoder including random
init — it is saturated/non-discriminative here; the discriminating metrics are rank, per-dim std, anisotropy,
and IDM accuracy.</div>

<h2>5 · Findings &amp; the artifact</h2>
<ul>
<li><b>Random init is a strong baseline</b> (IDM {ri_idm:.2f}) — any recipe must beat it to be worth using.</li>
<li><b>Naive stop-grad forward-JEPA degrades the representation</b> (anisotropic, IDM {g('baseline_sg','idm_acc'):.2f}
&lt; random) — consistent with the exp_010_2 failure.</li>
<li><b class="win">VICReg wins and is the deliverable</b> → <code>artifacts/encoder_best_{best}.pt</code>
(fixed trunk, drop-in PPO encoder): rank {g('vicreg','rank'):.0f}, IDM {g('vicreg','idm_acc'):.2f}.</li>
<li><b>My SIGReg variant collapsed</b> (near-zero embeddings) under λ=1, applied in the prediction space with no
per-slice standardisation. This is most likely an <i>implementation/weighting</i> issue, not a refutation of
LeJEPA — fixing SIGReg (stronger λ, separate projector, standardised slices) is the top follow-up, since its
isotropic-Gaussian objective is the principled target.</li>
<li><b>Data is the next ceiling:</b> random data reaches the goal ~1/1159 lives, so no recipe can learn goal
dynamics it never sees — goal-aware data is the highest-leverage follow-up.</li>
</ul>
<div class="sub">Reproduce: <code>study.py</code> → <code>make_report.py</code>. Eval harness: effective rank,
per-dim std, anisotropy, frozen linear IDM (action-decodability), frame-diff agent-loc probe.</div>
</body></html>"""

open(f"{HERE}/report.html", "w").write(html)
print(f"best={best}  wrote report.html ({len(html)} bytes) + artifacts/encoder_best_{best}.pt")
