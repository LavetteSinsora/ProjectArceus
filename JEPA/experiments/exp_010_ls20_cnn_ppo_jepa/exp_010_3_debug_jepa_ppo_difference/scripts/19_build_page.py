"""Build a polished, self-contained index.html (figures embedded as base64).
Hand-designed layout. Run: uv run python JEPA/experiments/exp_010_ls20_cnn_ppo_jepa/exp_010_3/scripts/19_build_page.py
"""
import base64
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]; FIG=BASE/"figures"   # scripts/ -> exp_010_3
def b(name):
    p=FIG/name
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""
A,B,C=b("web_value_spread.png"),b("web_rescue.png"),b("web_advantage.png")

HTML=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Why a "smart" encoder makes sparse-reward RL fail</title>
<style>
:root{{--ink:#23272e;--muted:#5b636e;--line:#e7e9ee;--violet:#6c5ce7;--green:#2a9d8f;--coral:#e76f51;--bg:#eef0f4;--card:#fff;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.72 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:860px;margin:40px auto;background:var(--card);border-radius:16px;box-shadow:0 1px 3px rgba(20,23,28,.06),0 12px 40px rgba(20,23,28,.07);padding:54px 60px 46px}}
.kick{{color:var(--violet);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12.5px;margin:0 0 14px}}
h1{{font-size:33px;line-height:1.18;font-weight:800;letter-spacing:-.02em;margin:0 0 14px}}
.lede{{font-size:19px;color:var(--muted);line-height:1.6;margin:0 0 6px}}
h2{{font-size:21px;font-weight:750;letter-spacing:-.01em;margin:42px 0 12px;padding-top:6px}}
h2 .n{{color:var(--violet);font-weight:800;margin-right:.5em}}
h3{{font-size:16px;margin:26px 0 8px}}
p{{margin:13px 0}} b{{color:#14171c}} a{{color:var(--violet)}}
hr{{border:0;border-top:1px solid var(--line);margin:38px 0}}
.rule{{height:4px;width:54px;background:var(--violet);border-radius:3px;margin:22px 0 32px}}
.puzzle{{background:linear-gradient(180deg,#f6f4ff,#f3f6ff);border:1px solid #e4e1ff;border-radius:14px;padding:20px 24px;margin:26px 0;font-size:18px}}
.puzzle b{{color:var(--violet)}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0}}
.def{{background:#fafbfc;border:1px solid var(--line);border-radius:12px;padding:15px 17px}}
.def h4{{margin:0 0 6px;font-size:14px;color:var(--violet);letter-spacing:.02em}}
.def p{{margin:0;font-size:14.5px;color:#444;line-height:1.5}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0}}
.col{{border-radius:13px;padding:18px 20px;border:1px solid}}
.col.win{{background:#effaf6;border-color:#c7ece1}} .col.fail{{background:#fdf1ec;border-color:#f6d8cc}}
.col .tag{{display:inline-block;font-weight:800;font-size:12.5px;letter-spacing:.05em;padding:3px 10px;border-radius:20px;margin-bottom:8px}}
.col.win .tag{{background:var(--green);color:#fff}} .col.fail .tag{{background:var(--coral);color:#fff}}
.col h4{{margin:0 0 6px;font-size:15.5px}} .col p{{margin:0;font-size:15px;line-height:1.55;color:#3a3f47}}
figure{{margin:34px 0;}}
figure img{{width:100%;display:block;border:1px solid var(--line);border-radius:12px;background:#fff}}
figcaption{{font-size:14px;color:var(--muted);line-height:1.55;margin-top:12px;padding-left:2px}}
figcaption b{{color:var(--violet)}}
.callout{{background:#f7f8fa;border-left:4px solid var(--violet);border-radius:0 10px 10px 0;padding:16px 22px;margin:24px 0}}
.callout h3{{margin:0 0 6px;color:var(--violet);font-size:15px}}
ul{{margin:12px 0;padding-left:22px}} li{{margin:7px 0}}
.quote{{font-size:21px;line-height:1.45;font-weight:600;color:#14171c;border-left:4px solid var(--violet);padding:6px 0 6px 22px;margin:30px 0 6px}}
.foot{{margin-top:40px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13.5px;line-height:1.6}}
.chip{{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:baseline;margin-right:4px}}
@media(max-width:680px){{.wrap{{padding:30px 22px;margin:14px}}.grid,.cols{{grid-template-columns:1fr}}h1{{font-size:27px}}}}
</style></head><body><div class="wrap">

<p class="kick">Deep RL · experiment 010 · mechanism study</p>
<h1>Why a “smart” pretrained encoder makes sparse-reward RL fail — and a random one succeeds</h1>
<p class="lede">A world-model encoder is supposed to be a <i>better</i> starting point than random weights. On a sparse-reward maze it does the opposite. Here is exactly why, in plain terms, with the three experiments that pin it down.</p>
<div class="rule"></div>

<h2><span class="n">1</span>The setup, in plain words</h2>
<div class="grid">
  <div class="def"><h4>THE GAME</h4><p>An agent moves in a small maze, seeing the screen as a tiny image with 4 move buttons. Reward is <b>+1 only when the level is finished</b>, 0 otherwise — so until it finishes once, it gets <b>no feedback at all</b>.</p></div>
  <div class="def"><h4>THE AGENT (PPO)</h4><p>A network in 3 parts: an <b>encoder</b> turns the screen into 256 numbers (a “representation”), a <b>policy</b> picks a button, and a <b>critic</b> guesses how good the state is.</p></div>
  <div class="def"><h4>THE COMPARISON</h4><p>Two encoders: a <b>random</b> one (untrained, just noise) and a <b>JEPA</b> one (pretrained to predict how the screen changes — a representation that is <i>supposed to be good</i>).</p></div>
</div>

<div class="puzzle">The random encoder lets the agent learn to solve the level. The “good” JEPA encoder gives <b>0%</b> — it never solves it. <b>The better representation makes things worse.</b> Why?</div>

<h2><span class="n">2</span>What it is <i>not</i></h2>
<ul>
<li><b>Not the feature scale.</b> JEPA’s numbers are ~66× bigger, but shrinking them to match the random encoder’s scale doesn’t help.</li>
<li><b>Not the encoder drifting during training.</b> Freezing it doesn’t help either.</li>
<li><b>Not “the features are broken.”</b> They’re rich and full-rank — on a harder level they’re actually <i>better</i> (see §6). The random encoder’s features are full-rank too; it’s <i>not</i> that it “can’t tell states apart.”</li>
</ul>

<h2><span class="n">3</span>The cause: the critic invents a fake reward</h2>
<p>The only thing that ever pushes PPO’s policy is the <b>advantage</b> — a number, computed from the <b>critic’s</b> value guesses, meaning “was this action better than expected?” Before the agent has <i>ever</i> seen a reward, every reward is 0, so the advantage is built <b>entirely from the critic’s guesses.</b> And the two encoders feed the critic very differently:</p>
<div class="cols">
  <div class="col win"><span class="tag">RANDOM → SOLVES</span><h4>Nothing to grab onto</h4><p>It maps almost every state to <i>nearly the same</i> 256 numbers (~1% varies). So the critic’s value barely changes across states → the advantage is tiny, directionless noise → the policy stays ~uniform → it <b>wanders and stumbles onto the exit</b>. That first real reward then teaches it.</p></div>
  <div class="col fail"><span class="tag">JEPA → 0%</span><h4>A fake landscape to chase</h4><p>It maps different states to <i>distinct</i> numbers, so the critic paints a richly varying value landscape — <b>out of its own untrained noise</b>. PPO chases it as if real, bending the policy into a state-specific pattern <b>before any reward exists</b> → it commits to a made-up plan and <b>stops exploring</b> before it can find the exit.</p></div>
</div>
<p>We call the made-up signal a <b>phantom advantage</b>: it has the <i>form</i> of useful guidance but carries no real information about the game.</p>

<h2><span class="n">4</span>The evidence</h2>
<p><span class="chip" style="background:var(--green)"></span><b>random</b> &nbsp; <span class="chip" style="background:var(--violet)"></span><b>JEPA (matched scale)</b> &nbsp; <span class="chip" style="background:var(--coral)"></span><b>JEPA (raw)</b></p>

<figure><img src="data:image/png;base64,{A}"/>
<figcaption><b>Fig 1 · The keystone.</b> Give both encoders the <i>same</i> output scale and ask untrained critics to score states. The JEPA representation produces a value landscape ~<b>5× wider</b> than random’s, and across 40 different random critics the two never overlap. Same scale, same critic — the only difference is the representation’s <i>structure</i>. That structure is the entire story.</figcaption></figure>

<figure><img src="data:image/png;base64,{B}"/>
<figcaption><b>Fig 2 · The decisive test.</b> Replace the critic-based advantage with the plain reward-to-go (exactly 0 until a real win, so no phantom can exist). The <i>same</i> JEPA encoder that scored 0% now keeps exploring at full entropy (left) and <b>solves</b> (right), at every seed. Put the critic back and it breaks again — so the value critic, fed an informative representation, is what manufactures the failure.</figcaption></figure>

<figure><img src="data:image/png;base64,{C}"/>
<figcaption><b>Fig 3 · Why normalization doesn’t save it.</b> PPO rescales advantages to a standard size each step. That erases the scale difference (left) <i>and</i> even the overall shape (middle, ~identical). What it cannot remove is <b>state-dependence</b> — how much the chosen action depends on the state (right): <b>exactly 0 for random</b>, nonzero for JEPA, all before any reward. Small in absolute terms, but categorically different from zero — and enough to bend exploration.</figcaption></figure>

<h2><span class="n">5</span>So should it explore “truly randomly” before any reward?</h2>
<div class="callout"><h3>The short answer: no — but don’t let the critic invent a direction either.</h3>
<p>Truly-uniform, state-blind action is <i>not</i> the ideal way to explore. Provably-good exploration is <b>directed</b>: it biases actions toward parts of the world the agent <b>hasn’t seen yet</b> (novelty / low visit-counts). A uniform random walk is slow; it works here only because this level is solvable by a <i>short</i> wander.</p></div>
<p>The real principle is about <b>what a bias is conditioned on</b>:</p>
<ul>
<li>A bias toward <b>genuinely under-explored states</b> (novelty, counts, prediction-error) is <b>good</b> — it’s real information about the environment.</li>
<li>A bias toward states the <b>untrained critic</b> happens to score highly is <b>bad</b> — that “information” is pure noise from the network’s geometry, correlated with nothing real. It has the <i>shape</i> of a plan but the <i>content</i> of static.</li>
</ul>
<p>So uniform exploration is a <b>safe fallback</b>, correct only when your one candidate bias is noise; it becomes a <i>limitation</i> the moment you have a real novelty signal. The lesson isn’t “don’t be directed.” It’s: <b>don’t let the critic synthesize a direction out of representation structure before it has ever seen a reward.</b></p>

<h2><span class="n">6</span>The flip side: same property, opposite sign</h2>
<p>The very thing that traps JEPA <i>before</i> a reward — distinct, separable state representations — is exactly what you <i>want</i> <b>after</b> a reward, to credit the right state. On a harder level (L2), given a known winning sequence, a <b>frozen</b> JEPA encoder can reproduce it where a <b>frozen</b> random encoder cannot separate the look-alike states. (The gap is specific to a frozen encoder; let either one adapt and both succeed.) JEPA’s representation isn’t “bad” — it’s <i>powerful</i>, and that power cuts both ways: harmful for cold-start exploration, helpful for exploiting a found solution.</p>

<p class="quote">The failure is not “directed exploration” — it’s the value critic manufacturing a content-free direction out of an informative representation before any reward exists. Remove the critic, and the “bad” representation works fine.</p>

<div class="foot">
<b>Honest limits.</b> One game, levels 1–2, ≤3 seeds, short runs. The bare “solves / fails” counts are too few to be statistically significant on their own (the random encoder itself fails on 1 of 3 seeds). The conclusions rest on the <i>continuous, norm-controlled</i> measurements — value-landscape spread, state→action dependence, and the critic-removal rescue — which are consistent across seeds. Treat generalization beyond this game as a hypothesis.<br><br>
Full investigation: <code>report.html</code> (same folder). All data in <code>data/</code>, all generating code in <code>scripts/</code>. This page is rebuilt with <code>scripts/19_build_page.py</code>; its figures with <code>scripts/18_web_figures.py</code>.
</div>
</div></body></html>"""
(BASE/"index.html").write_text(HTML)
print("wrote",BASE/"index.html","bytes",len(HTML))
