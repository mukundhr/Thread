# Thread

**A belief formation engine. Not a chatbot.**

Thread takes any question, searches the web for real sources, and runs three completely independent analyses through different intellectual lenses — structural, psychological, and material. Each lens forms its own belief, interrogates it from four angles, and produces its own verdict. Then Thread compares all three, showing where they agree (most trustworthy), where they diverge (genuine uncertainty), and what only appears from one lens.

The result isn't a summary. It's a map of how hard a question actually is.

---

## Why this is different from asking ChatGPT

When you ask an LLM a question, you get an answer — confident, well-written, no visible reasoning. You don't know what it considered and rejected, how certain it actually is, or where the knowledge breaks down. You also only get one perspective, framed by whatever the model's training data overrepresents.

Thread does something structurally different:

- It **starts uncertain** and has to earn confidence through evidence
- It **searches the real web** — grounded in current articles, not just training data
- It **runs three independent lenses** with no cross-contamination, so you see genuine disagreement
- It **actively tries to break its own beliefs** — four rounds of interrogation per lens
- It **uses two different models**: DeepSeek generates beliefs, Gemini Flash critiques them
- It **shows every step** — what changed, what didn't, and why
- It **tells you what it doesn't know** — open questions, divergence points, lens-only findings

ChatGPT tells you what to think. Thread shows you how to think about it.

---

## System Architecture

### Pipeline Overview

```
                          ┌─────────────────────┐
                          │   User Question     │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
        ┌───────────▼────────┐  ┌────▼────────────┐  │
        │  Web Search Layer  │  │  Searcher.py    │  │
        └────────────────────┘  └─────────────────┘  │
                    │                                 │
        ┌───────────▼────────────────────────────────▼──────────┐
        │         Three Parallel Epistemic Arcs                │
        └─┬──────────────────┬────────────────────┬─────────────┘
          │                  │                    │
   ┌──────▼────────┐  ┌──────▼────────┐  ┌──────▼─────────┐
   │  Structural   │  │  Agent /      │  │  Material  /   │
   │  Lens         │  │  Psychological│  │  Economic      │
   │               │  │  Lens         │  │  Lens          │
   └──────┬────────┘  └──────┬────────┘  └──────┬─────────┘
          │                  │                  │
          │        ┌─────────▼────────────┐     │
          └───────▶│  4-Round Cycle      │◀────┘
                   │  (per lens)         │
                   └─────────┬───────────┘
                             │
                   ┌─────────▼─────────┐
                   │ Round 1: Mechanism│
                   │ → Web search      │
                   │ → Form belief     │
                   └─────────┬─────────┘
                             │
                   ┌─────────▼──────────┐
                   │ Round 2: Critique  │
                   │ → Counterexamples  │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼─────────────┐
                   │ Round 3: Evidence Qlt │
                   └─────────┬─────────────┘
                             │
                   ┌─────────▼────────────┐
                   │ Round 4: Consistency │
                   └─────────┬────────────┘
                             │
                   ┌─────────▼─────────────────────┐
                   │ Final Verdict per Lens        │
                   │ (confidence + belief state)   │
                   └─────────┬─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼─────┐         ┌────▼─────┐        ┌────▼─────┐
   │Structural│         │Agent/Psy. │        │ Material │
   │ Verdict  │         │ Verdict   │        │ Verdict  │
   └────┬─────┘         └────┬──────┘        └────┬─────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   Comparison Layer         │
              │   Arc Comparison Engine    │
              └──────────────┬──────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼──────────┐  ┌──────▼──────────┐  ┌────▼──────────┐
   │  Convergence  │  │  Divergence     │  │  Lens-Only    │
   │  (agreement)  │  │  (disagreement)  │  │  (unique)     │
   └───────────────┘  └─────────────────┘  └───────────────┘
```

### Component Details

#### 1. **Searcher Layer** (`searcher.py`)
- Uses DuckDuckGo API for web search (via `ddgs` library)
- **Domain filtering**:
  - **Blocked**: social media, aggregators, low-quality blogs
  - **Preferred**: news, academic, think tanks, historical journals
- Returns tiered sources based on domain authority
- Called once per interrogation round per lens (12 searches total per run)

#### 2. **Three Epistemic Frameworks** (in `thread.py`)
Each framework uses the same generator model (DeepSeek V3) but with distinct prompt instructions:

| Lens | Focus | Query Angle |
|------|-------|-----------|
| **Structural** | Systems, institutions, path dependency, historical forces | "What structural conditions made this inevitable?" |
| **Agent/Psychological** | Individual decisions, psychology, cognition, leadership | "What did people choose? Why did they believe what they believed?" |
| **Material/Economic** | Resources, incentives, costs, power distribution | "Follow the money. Who gains? What are the material incentives?" |

Each lens independently:
1. Searches the web from its own perspective
2. Forms an initial belief with confidence estimate
3. Runs through 4 interrogation rounds (see below)
4. Updates belief based on adversarial critique
5. Generates final verdict with computed confidence

#### 3. **Interrogation System** (4 rounds per lens)
Each round involves:
- **New web search** (context-aware for that interrogation type)
- **Generator model** (DeepSeek) answers the round's question
- **Critic model** (Gemini Flash) reviews the answer and flags weaknesses
- **Belief state update** based on new evidence and critique

| Round | Question | Purpose |
|-------|----------|---------|
| **Mechanism** | "Why is that actually true?" | Expose hidden assumptions, trace causal chains |
| **Counterexample** | "Where does this fail completely?" | Find bitter contradictions and edge cases |
| **Evidence Quality** | "How good is the evidence really?" | Evaluate source credibility, incentives, methodology |
| **Consistency** | "Does this contradict itself?" | Identify internal tensions and logical failures |

#### 4. **Belief State Tracking**
Every belief is represented as:
```json
{
  "core_claim": "the current best explanation",
  "confidence": 0.65,
  "supporting_evidence": ["point [source]"],
  "contradicting_evidence": ["counter [source]"],
  "open_questions": ["what remains unresolved"],
  "used_sources": [1, 3, 4]
}
```

State evolves through 4 rounds → final verdict with **two confidence metrics**:
- **LLM confidence**: model's self-reported certainty
- **Computed confidence**: weighted by source tiers, evidence balance, and model report

#### 5. **Database Persistence** (`thread.db`)
SQLite schema tracks the complete reasoning trajectory:
- `runs` — unique query executions
- `arcs` — one structural/agent/material arc per run
- `belief_states` — snapshot at each of 4 rounds + final
- `interrogations` — every question asked and answer given
- `articles` — sources fetched, tiered by authority
- `final_views` — synthesized verdicts per arc
- `arc_comparisons` — convergence/divergence/lens-only findings

#### 6. **Comparison Engine**
After all three lenses complete:
- **Convergence**: what all three agree on (most epistemically robust)
- **Divergence**: where lenses reached fundamentally different conclusions
- **Lens-Only**: what only appeared from one lens's perspective

#### 7. **Flask Dashboard** (`server.py`)
Web UI with 6 views, all backed by the same database:
- Summary view (final verdicts + confidence labels per arc)
- Disagreement map (parallel arc evolution + convergence analysis)
- Interrogation transcript (all questions/answers in sequence)
- Source inventory (articles tiered by quality, grouped by round)
- Topic comparison (overlay confidence trajectories)
- Longitudinal view (same topic across multiple runs/days)

### Model Selection

- **Generator**: DeepSeek V3.2 (via OpenRouter)
  - Reason: cheap, capable, good reasoning over long contexts
  - Role: forms initial beliefs, answers interrogation rounds

- **Critic**: Gemini Flash 2.5 Lite (via OpenRouter)
  - Reason: different training distribution from DeepSeek creates adversarial pressure
  - Role: interrogates and critiques each belief, flags weaknesses

### Confidence Labeling

Computed confidence combines three signals (weighted):
- **Source tier quality** (35%): Are the sources authoritative?
- **Evidence balance** (25%): Is there genuine disagreement in sources?
- **LLM self-report** (40%): How certain did the model claim to be?

Results in honest labels (not probabilities):
- "Strongly supported" (≥80%)
- "Well supported" (65-79%)
- "Moderately supported" (50-64%)
- "Contested" (35-49%)
- "Poorly supported" (<35%)

---

## Setup

**Requirements:** Python 3.11+, an [OpenRouter](https://openrouter.ai) API key

```bash
git clone https://github.com/mukundhr/thread.git
cd thread
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENROUTER_API_KEY=your_key_here
```

**Models:** Thread uses DeepSeek V3 as the generator and Gemini Flash lite as the critic. Both are configurable in `thread.py`. DeepSeek V3 was chosen because it is cheap.

---

## Usage

### Run a topic
```bash
python thread.py "why do democracies fail"
python thread.py "does foreign aid make poverty worse"
python thread.py "why wars happen"
```

### Skip web search (faster, uses model knowledge only)
```bash
python thread.py "why wars happen" --no-search
```

### Compare multiple topics
```bash
python thread.py "why wars happen" "why civilizations collapse" "why democracies fail"
```

### View belief drift over time
```bash
python thread.py "why wars happen" --longitudinal
```

### Launch the dashboard
```bash
python server.py
# open http://localhost:5050
```

---

## Dashboard

Six views, all connected to the same SQLite database:

| View | What it shows |
|---|---|
| **Summary** | Final verdict + confidence label + full belief arc per lens |
| **Disagreement Map** | Three lens arcs side by side — convergence, divergence, lens-only findings |
| **How it questioned itself** | Every question asked and every answer given, in sequence |
| **Sources read** | Every article fetched, grouped by round, tiered by quality |
| **Compare topics** | Overlay confidence trajectories across multiple topics |
| **Belief over time** | Same topic across different days — watch beliefs shift as sources change |

---

## Good topics to try

Thread works best on questions where smart people genuinely disagree and evidence exists on both sides:

- `why do civilizations collapse`
- `does capitalism create inequality by design`
- `do we have free will`
- `why did the Soviet Union fall`
- `does foreign aid make poverty worse`
- `are humans naturally violent or peaceful`
- `why wars happen`

Avoid questions with single factual answers. Thread is built for contested, complex questions where the reasoning process is the point.

---

## Database schema

```
runs              — one row per topic run
arcs              — one row per lens arc (3 per run)
articles          — every article fetched, with tier + source_type
belief_states     — belief at each iteration, llm_confidence + computed_confidence
interrogations    — every question + answer per round
final_views       — synthesized verdict per arc
arc_comparisons   — convergence, divergence, lens-only findings per run
```

All queryable directly with SQLite.
The database (`thread.db`) is created automatically on first run.

---

## Known limitations

These are honest architectural constraints, not bugs to fix later:

**Three lenses, not true independence.** The structural, agent, and material arcs use the same underlying model with different framing prompts. They produce genuinely different perspectives, but they share training data, RLHF tuning, and base reasoning patterns. True epistemic independence would require models with fundamentally different world models.

**Confidence labels are computed, not calibrated.** The label ("Moderately supported", "Contested") is derived from source tier quality (35%), evidence balance (25%), and the LLM's self-report (40%). It describes the epistemic state honestly, but it is not a probability and does not predict real-world accuracy.

**Evidence quality analysis is LLM-assisted, not peer review.** Thread classifies sources into tiers and asks the LLM to distinguish RCTs from opinion pieces. It cannot check sample sizes, verify statistical methods, or detect p-hacking. Better than nothing — not a substitute for methodological scrutiny.

**Search results are non-reproducible.** The same topic run on different days may find different sources and reach different conclusions. Every source used is visible and tiered, but Thread outputs are not replicable in the scientific sense.

Thread is best understood as a **structured reasoning aid**, not a research instrument. The value is in making the reasoning process visible and the disagreements explicit — not in the outputs being ground truth.

---

## License

MIT