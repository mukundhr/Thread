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

## How it works

```
You give it a topic
        ↓
Three independent lenses run — no cross-contamination:

  [Structural]          [Agent / Psychological]      [Material / Economic]
  institutions,         individuals, decisions,       resources, incentives,
  history, systems      psychology, leadership        costs, power

  Each lens independently:
    ↓ searches the web (different angle per round)
    ↓ forms an initial belief
    ↓ Round 1 — Mechanism: "Why is that actually true?"
    ↓ Round 2 — Counterexample: "Where does this completely break?"
    ↓ Round 3 — Evidence quality: "How good is the evidence really?"
    ↓ Round 4 — Consistency: "Does this contradict itself?"
    ↓ Final verdict per lens
        ↓
Comparison: where do the three lenses agree? Where do they split?
        ↓
Disagreement Map:
  Convergence  — what all three agreed on (most robust)
  Divergence   — where they reached different conclusions
  Lens-only    — what only appeared from one lens
```

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