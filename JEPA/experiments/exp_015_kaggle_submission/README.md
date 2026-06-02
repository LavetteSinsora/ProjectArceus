# exp_015 — ARC-AGI-3 (ARC Prize 2026) Kaggle Submission Design

**Goal:** Convert our intrinsic-exploration agent (leaky-RND + CNN encoder + PPO policy, `exp_013` style) into a submission-ready ARC-AGI-3 agent for the *ARC Prize 2026 — ARC-AGI-3* Kaggle competition.

Every factual claim below is cited with a source URL. Where the official sources do **not** confirm a detail, it is flagged **[UNCONFIRMED]** rather than guessed.

> **Confidence note.** The agent *interface* (Sections 1, 4) is read directly from the official `arcprize/ARC-AGI-3-Agents` source code and `docs.arcprize.org`, so it is high-confidence. The Kaggle *packaging mechanics* (Section 1.2) and exact compute limits (Section 2) are partly assembled from arcprize.org + secondary write-ups because the Kaggle competition page is JS-rendered and could not be fetched verbatim by the research tool; those are flagged. **Verify the Kaggle "Code Requirements" / "Evaluation" tabs before relying on the packaging specifics.**

---

## 1. What is expected to be submitted (submission format)

### 1.1 The agent interface (CONFIRMED from source)

An ARC-AGI-3 agent is a **Python class that subclasses `agents.agent.Agent`** and implements two abstract methods. This is the canonical interface in the official agents repo.

Source: `agents/agent.py` in <https://github.com/arcprize/ARC-AGI-3-Agents> and <https://docs.arcprize.org/> (Agents quickstart / Create-agent pages).

```python
class Agent(ABC):
    MAX_ACTIONS: int = 80   # default per-game action cap (looping guard)

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool: ...

    @abstractmethod
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction: ...
```

The framework drives the episode loop for you (`Agent.main()`, verbatim from `agents/agent.py`):

```python
while (not self.is_done(self.frames, self.frames[-1])
       and self.action_counter <= self.MAX_ACTIONS):
    action = self.choose_action(self.frames, latest_frame)
    if frame := self.take_action(action):
        self.append_frame(frame)
    self.action_counter += 1
```

So **you do not write the training/eval loop or the env wrappers** — you only implement `choose_action` (pick the next `GameAction`, fill its data) and `is_done` (when to stop). The harness handles the HTTP `step`, frame conversion, scorecard, and recording. (Source: `agents/agent.py`, `main.py` in the repo.)

`FrameData` fields (CONFIRMED from `_convert_raw_frame_data` in `agents/agent.py`):

| Field | Meaning |
|---|---|
| `game_id` | string id of the current game |
| `frame` | `list[list[list[int]]]` — a **list of 2-D grids**, each grid 64×64 of integer color values (0–15). Multiple grids = stacked layers/frames. |
| `state` | `GameState` enum: `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER` (names confirmed in `random_agent.py` / `llm_agents.py`). |
| `levels_completed` | int — the scoring quantity (was named `score` pre-v0.9.2). |
| `win_levels` | int — number of levels needed to win. |
| `available_actions` | `list[GameAction]` — **which actions are legal in the current frame** (varies per game/frame). |
| `guid`, `full_reset` | session/reset bookkeeping. |

Source: `agents/agent.py` (`_convert_raw_frame_data`), and FrameData rename note in repo README.

The agent is selected/run via CLI; the harness can also fan out a "swarm" over all games:

```bash
uv run main.py --agent=<your_agent_name> --game=<game_id>   # one game
uv run main.py --agent=<your_agent_name>                    # swarm: all available games
```

Source: <https://docs.arcprize.org/> (Agents quickstart) and `main.py` argparse (`--agent`, `--game`; "If none specified, an agent swarm will play all available games").

### 1.2 Kaggle packaging (PARTIALLY CONFIRMED — verify on Kaggle)

- Submissions are made **through the designated Kaggle competition**; **all code/methods must be open-sourced** (MIT or CC0 / MIT-0) **before** receiving private scores. Source: <https://arcprize.org/competitions/2026/arc-agi-3> ("Submissions must be made through the designated Kaggle competition"; "All code and methods must be open sourced to be eligible for prizes"); reinforced by ARC Prize 2026 overview <https://arcprize.org/competitions/2026>.
- Evaluation is a **sandboxed Kaggle environment with no internet** — no calls to hosted LLM APIs (OpenAI/Anthropic/Google) during scoring. Source: <https://arcprize.org/competitions/2026/arc-agi-3> ("No internet access during evaluation"); secondary confirmation: <https://www.datacamp.com/blog/arc-agi-3>.
- Because there is no internet, the ARC-AGI-3 environment is **served locally inside the container** at eval time and your agent talks to it over the same `ROOT_URL`/HTTP `step` interface (`SCHEME/HOST/PORT` env vars, default `http://localhost:8001`, in `main.py`). The "you can't block the counter" remark in the task is consistent with this: at eval you talk to the real env server through `step()` and the action counter advances every turn — there is no offline wrapper you control. **[UNCONFIRMED in official Kaggle docs — confirm whether the submission is a Kaggle Notebook that imports your agent class, or a repo/Docker bundle. The standard ARC-Prize-on-Kaggle pattern (ARC-AGI-1/2) is a Code Competition Notebook that runs offline against an attached dataset; ARC-AGI-3 adds the local env server.]** Source for local/offline + ROOT_URL: `main.py` in the agents repo; offline requirement: arcprize.org above.

**Net:** Treat the deliverable as *"a `choose_action`/`is_done` agent class that runs fully offline, plus any pre-trained weights shipped as a file (e.g. a Kaggle Dataset / `data/models/` dir), packaged per the Kaggle Code Competition rules."* The "trained models / data directories for replays and trained models" packaging is referenced by community reference repos (e.g. <https://github.com/ssppsy/arc-agi-3>), not by an official template we could verify — **flagged**.

---

## 2. Submission rules / constraints

| Constraint | Value | Source |
|---|---|---|
| Internet at eval | **OFF** (sandboxed). No hosted-LLM API calls. | <https://arcprize.org/competitions/2026/arc-agi-3> |
| Open-source | Mandatory before private scoring; MIT or CC0/MIT-0. | <https://arcprize.org/competitions/2026/arc-agi-3>, <https://arcprize.org/competitions/2026> |
| Hardware / compute | "Hardware and compute limits will be announced with the competition launch." Kaggle Code competitions historically give a **single P100/T4 GPU and ≤12 h runtime**. **[compute exact numbers UNCONFIRMED — read the Kaggle "Code Requirements" tab]** | arcprize.org (quote above); 12h/P100 figure is the typical Kaggle pattern reported by <https://www.datacamp.com/blog/arc-agi-3> — **treat as indicative, not official.** |
| Public-API budget (only relevant if you used an API path, which we will NOT) | "< $1K to reproduce on the 5 public/private games within 8 hrs." | secondary: <https://www.datacamp.com/blog/arc-agi-3> |
| Per-game action cap | `MAX_ACTIONS = 80` is the harness default guard, **but this is editable by the agent** and is *not* the scoring rule. Actions taken are what the score is computed from (see §3). | `agents/agent.py` |
| Timeline | Opened 2026-03-25; final submission 2026-11-02; results 2026-12-04. **[dates from secondary source — confirm on Kaggle]** | <https://www.datacamp.com/blog/arc-agi-3> |

### "You can't block the counter" — what env manipulation is / isn't allowed
At eval the agent only interacts through `choose_action` → `take_action` → `arc_env.step(action)`. Every returned frame increments `action_counter`, and **the score is a function of how many actions you took** (§3). There is no eval-time hook to pause, rewind, or freeze the environment's action/step counter, and no offline wrapper of ours sits in between — the env is the authoritative server. So any training-time trick that relied on manipulating our own env's step timer / reward shaping **does not transfer to eval**. What you *can* do at eval: call `RESET` (counts as an action), read `available_actions`, and choose any legal action. (Grounded in `agents/agent.py` loop + `do_action_request`/`step`.)

---

## 3. Levels / games tested & scoring

- **Dataset split** (from the ARC-AGI-3 paper, <https://arxiv.org/html/2603.24621v2>):
  - Public demo: **25 environments** (demonstration only, excluded from official scoring).
  - Semi-private: **55 environments** (external API testing).
  - **Fully private: 55 environments** (the official ARC Prize competition set).
  - **Games are held-out / unseen at eval** — the agent meets each environment for the first time; the benchmark explicitly tests *skill acquisition*, not memorization. Source: <https://arxiv.org/html/2603.24621v2>, <https://arcprize.org/competitions/2026/arc-agi-3>.
- **Levels:** each environment has **multiple levels (≥6 cited)**, with later levels weighted more (e.g. in a 5-level env, level *k* contributes weight *k*/15). Source: <https://arxiv.org/html/2603.24621v2>.
- **Scoring metric — RHAE (Relative Human Action Efficiency):** for level *l* in env *e*, with agent actions `a_{l,e}` and human-baseline actions `h_{l,e}`, the level score is

  `S_{l,e} = min(1.15, h_{l,e} / a_{l,e})²` (capped/squared), aggregated and capped at 100%.

  Humans solve 100% of included environments; median ≈ 7.4 min/env. Source: <https://arxiv.org/html/2603.24621v2>; metric summary echoed at <https://arcprize.org/arc-agi/3>.

**Implication for us:** the score rewards *completing levels in few actions*, not just first-reward. Pure undirected exploration that eventually wins still scores poorly if `a >> h`. Our intrinsic agent must (a) actually reach `WIN`/level completions, and (b) be reasonably action-efficient. First-reward speed (what `exp_013` optimizes) is a *necessary precursor* but not the final metric.

---

## 4. Action space

CONFIRMED from `docs.arcprize.org/actions`, `agents/templates/random_agent.py`, and `agents/templates/llm_agents.py`.

| Action | Type | Meaning |
|---|---|---|
| `RESET` | control | Initialize / restart the game or level. |
| `ACTION1` | simple | up |
| `ACTION2` | simple | down |
| `ACTION3` | simple | left |
| `ACTION4` | simple | right |
| `ACTION5` | simple | interact / select / rotate / execute (game-defined) |
| `ACTION6` | **complex** | **click/point at (x, y), each in 0–63 → a 64×64 = 4096-cell coordinate grid** |
| `ACTION7` | simple | undo |

Sources: <https://docs.arcprize.org/actions> (RESET, ACTION1–7, "ACTION6 = complex action requiring x,y coordinates (0–63 range)"); `random_agent.py` shows `x,y = randint(0,63)`.

**How the API represents an action (CONFIRMED, `random_agent.py` / `llm_agents.py`):** actions are members of a `GameAction` enum, looked up by `GameAction.from_id(...)` / `GameAction.from_name(...)`. Simple actions carry no data. ACTION6 (the only "complex" one, `action.is_complex()`) requires a data dict:

```python
action = GameAction.ACTION6
action.set_data({"x": 32, "y": 32})   # center click; x,y ∈ [0,63]
```

So it is **action-enum + (optional x,y data dict)**, *not* a single flat discrete index, and *not* a continuous coordinate.

**Does action count vary per game?** **Yes.** Each frame returns `available_actions: list[GameAction]`, and "each game explicitly defines the set of available actions." A game may expose 4 (just moves), 5 (+interact), or 6 (+click) actions, and `available_actions` can change per frame. Crucially, when ACTION6 is available the API tells you *that* it's available but **not which (x,y) cells are active** — you must discover that. Sources: <https://docs.arcprize.org/actions>; `FrameData.available_actions` in `agents/agent.py`; `llm_agents.py` action descriptions.

---

## 5. Adapting leaky-RND to a submission

**Key constraints that shape the answer:**
1. No internet, no offline pre-training loop *at eval* — only `choose_action` per frame against the live env.
2. Games are **unseen** → a frozen, pre-trained zero-shot policy will not generalize to a novel game's controls/goal. The benchmark is explicitly designed to defeat memorized policies.
3. Scoring is **action-efficiency relative to human** → we get a finite action budget per level and must convert exploration into level completions.

**Does online learning during eval work?** The competition is built around agents that *learn within the episode* (it tests exploration + skill acquisition, not a pre-trained mapping). **[The Kaggle rules do not, in the sources we could verify, explicitly forbid in-process gradient updates during eval — confirm on the Kaggle rules tab.]** Nothing in the `Agent` interface prevents `choose_action` from running an optimizer step: you receive the full frame history (`frames`) each call and may update weights between actions. So **online learning is the intended mode, and leaky-RND fits as an in-episode exploration driver** — *provided* the update budget fits within wall-clock + the per-level action budget.

**Why leaky-RND specifically fits.** RND's intrinsic reward needs gradient updates to a predictor network, but it is *robust to non-stationarity* and needs **no env-supplied reward** — exactly the regime here (reward = sparse level completion, mostly 0). Our `exp_013` "leaky" variant (warm-up + EMA normalizer) cured the ICM-style collapse, which matters more here than on LS20 because every eval game is new and the novelty signal must stay alive for hundreds of actions. (Internal: `finding_phi_drift.md`, `project_exp013_protocol.md`.)

**Concrete porting plan (in-episode online RL inside `choose_action`):**
1. **Encoder.** Reuse the `exp_013` CNN encoder, but input is the ARC-AGI-3 `frame` = `list[64×64]` integer grids (0–15). Map ints → embedding/one-hot channels (16 channels) and stack the grids as additional channels. *Do not* assume a fixed grid count; pad/truncate to a max number of layers. This replaces our LS20 obs wrapper.
2. **Two-network RND.** Frozen random target `f_target(s)` + trainable predictor `f_pred(s)`; intrinsic reward `r^i = ||f_pred(s') - f_target(s')||²`, normalized by the running EMA std (the leaky/warm-up normalizer from `exp_013`). No extrinsic reward except `Δlevels_completed` (a strong sparse +1).
3. **Policy = PPO over the action head in §6.** Buffer transitions inside the agent across `choose_action` calls; run a PPO + RND-predictor update every *N* actions (small *N*, e.g. 8–16, because the per-level action budget is small and human-relative). Keep nets tiny (M3-Pro/MPS-sized per `hardware_macbook.md`; on Kaggle a single GPU).
4. **Reset handling.** On `state ∈ {NOT_PLAYED, GAME_OVER}` emit `RESET` (as in `random_agent.py`). Treat `GAME_OVER` as a terminal with 0 extrinsic reward; `WIN`/`Δlevels_completed` as the goal signal.
5. **`is_done`.** Stop on `WIN` (or when out of useful budget). Do **not** burn actions after a level is solved if the harness moves on automatically — extra actions hurt RHAE.
6. **Weights at submission.** We may ship a *pre-trained encoder/predictor init* (faster warm-up) as a packaged weights file, but the **adaptation must happen online** because games are unseen. Frozen zero-shot is expected to underperform.

**Risk / honest caveat:** in-episode RL with only ~tens–low-hundreds of actions per level (RHAE penalizes `a >> h`, human ≈ minutes) is a *very* tight budget for PPO to converge from scratch on a novel game. Leaky-RND keeps exploration efficient, but realistically this agent targets *reaching first level completions on a subset of games*, not topping the leaderboard. This is the central design tension and should be stated in any report.

---

## 6. Variable action count + the click action — action-head design

### Verdict on the proposed hierarchical/autoregressive head
The user's proposal — (a) softmax over the (up to) 6 actions; (b) if ACTION6, a second head picks `x∈[0,64)`; (c) a third head takes `[rep ⊕ x]` and picks `y∈[0,64)` — is **sound and is the recommended design**, with two refinements:

- **Why autoregressive y|x beats independent x,y.** Active click targets are spatially structured (objects, buttons), so `x` and `y` are correlated. Factoring `P(y|x, rep)` lets the head represent that; independent `P(x)P(y)` cannot and tends to "average" onto dead cells. Cost: only `64 + 64 = 128` logits, two small heads.
- **Why this beats a flat `6 + 4096` softmax.** A flat 4102-way softmax wastes capacity (4096 click logits dominate, mostly invalid), gives no gradient structure over the 2-D grid, and is awkward to mask. The hierarchical head keeps the top-level decision (which of 6 actions) cleanly separated from the spatial sub-decision.
- **vs. a spatial/pointer head (64×64 conv → softmax over the map).** This is the main alternative and is arguably *better* when the click target is "a location in the frame," because a fully-convolutional 64×64 logit map shares spatial weights and aligns clicks with pixels. **Recommendation:** use the **hierarchical head as the primary design for its simplicity and clean masking, but consider the spatial conv-map head as a drop-in for the (x,y) sub-policy** if click-grounding is poor — they are interchangeable at interface level (both ultimately output an (x,y)).

### Recommended structure
```
rep = Encoder(frame)                       # shared CNN features
a   ~ Categorical(softmax(W_a · rep))      # over 6 action logits, MASKED to available_actions (+ RESET)
if a == ACTION6:
    x ~ Categorical(softmax(W_x · rep))            # 64 logits
    y ~ Categorical(softmax(W_y · [rep ⊕ emb(x)])) # 64 logits, conditioned on x
    action = ACTION6; action.set_data({"x": x, "y": y})
else:
    action = a                                     # simple action / RESET
```

### Masking scheme (handles variable n_actions 4/5/6 cleanly)
- Build a fixed 7-slot logit vector for `{RESET, ACTION1..ACTION6}` (ACTION7/undo optional). Each frame, read `latest_frame.available_actions` and set logits of **unavailable actions to −∞ before softmax** (additive mask `-1e9`). This makes 4-, 5-, and 6-action games share one network with zero architecture change — the policy simply never samples a masked action, and PPO log-probs are computed over the masked distribution. (Grounded in `available_actions` being per-frame, `agents/agent.py`.)
- **Click masking:** the API does **not** reveal which (x,y) are active (<https://docs.arcprize.org/actions>). So we **cannot** hard-mask the 64×64 grid. Options: (i) leave it unmasked and let RND/policy learn live cells; (ii) heuristically down-weight cells that are background color / unchanged across frames. Recommend starting with (i) + the RND novelty bonus to drive click exploration, since "frame didn't change ⇒ probably invalid" is itself a useful learned signal (the LLM template literally tells the model this).
- **PPO correctness with masking:** apply the same `available_actions` mask at action-selection *and* at log-prob/entropy computation so the importance ratio is over the valid support. For ACTION6, the joint log-prob is `log P(a=ACTION6) + log P(x) + log P(y|x)`.

---

## 7. Open items to confirm on Kaggle before finalizing
1. Exact submission packaging: Code Competition **Notebook** vs repo/Docker; where pre-trained weights attach (Kaggle Dataset?). (Section 1.2)
2. Official compute limits & per-submission wall-clock (the 12h/P100 figure is indicative only). (Section 2)
3. Whether in-process gradient updates during eval are explicitly allowed (we believe yes by design; confirm rules tab). (Section 5)
4. Whether the local env server is provided in the container and on which `HOST:PORT`. (Section 1.2)

---

## Sources
- Action space / ACTION6 / per-game availability: <https://docs.arcprize.org/actions>
- Agent interface, FrameData, episode loop, ACTION6 `set_data`, CLI: `agents/agent.py`, `agents/templates/random_agent.py`, `agents/templates/llm_agents.py`, `main.py` in <https://github.com/arcprize/ARC-AGI-3-Agents>
- Quickstart / connection / API key: <https://docs.arcprize.org/> (Agents quickstart)
- Competition rules (Kaggle, open-source, no-internet, hardware-TBD): <https://arcprize.org/competitions/2026/arc-agi-3>, <https://arcprize.org/competitions/2026>
- Kaggle competition landing page (JS-rendered; not fetched verbatim): <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3>
- Dataset split, levels, RHAE scoring, human baseline, held-out games: <https://arxiv.org/html/2603.24621v2> (ARC-AGI-3 paper), <https://arcprize.org/arc-agi/3>
- Secondary (indicative compute/timeline/packaging, flagged): <https://www.datacamp.com/blog/arc-agi-3>, reference repo <https://github.com/ssppsy/arc-agi-3>
