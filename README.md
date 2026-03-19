# Thread

**A belief formation engine. Not a chatbot.**

Thread takes any question, searches the web for real sources, forms an initial belief, then systematically interrogates that belief from four angles : mechanism, counterexample, evidence quality, and internal consistency. It tracks how the belief evolves, what changed it, and what's still unresolved. At the end, it produces a plain-English verdict with a confidence score.

The result isn't a summary. It's a record of reasoning.

---

## Why this is different from asking ChatGPT

When you ask an LLM a question, you get an answer - confident, well-written, no visible reasoning. You don't know what it considered and rejected, how certain it actually is, or where the knowledge breaks down.

Thread does something structurally different:

- It **starts uncertain** and has to earn confidence through evidence
- It **searches the real web**, grounded in current articles, not just training data
- It **actively tries to break its own belief**, four rounds, four different attacks
- It **shows every step**, what changed, what didn't, and why
- It **tells you what it doesn't know**, the open questions are the honest residue after interrogation

ChatGPT tells you what to think. Thread shows you how to think about it.

---

## How it works

```
You give it a topic
        ↓
Searches the web (DuckDuckGo, no API key needed)
Different search angles per round, not the same sources repeated
        ↓
Forms an initial belief
Uncertain by design. Confidence starts low.
        ↓
Round 1 — Mechanism
"Why is that actually true? Trace the causal chain."
        ↓
Round 2 — Counterexample
"Find a real historical case where this completely breaks."
        ↓
Round 3 — Evidence quality
"How good is the evidence? Who claims it? What would falsify it?"
        ↓
Round 4 — Consistency
"Does this belief contradict itself or other accepted ideas?"
        ↓
Final verdict
Plain English. Confidence score. Strongest for/against. Still unresolved.
        ↓
Everything saved, belief at every step, every source, every answer
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

**Note on models:** In `thread.py`, you can select any model you like. I chose DeepSeek V3 because it is cheap.

---

## Usage

### Run a topic from the CLI

```bash
python thread.py "why do democracies fail"
python thread.py "does foreign aid make poverty worse"
python thread.py "why wars happen"
```

### Compare multiple topics

```bash
python thread.py "why wars happen" "why civilizations collapse" "why democracies fail"
```

### View belief drift over time (run same topic on different days)

```bash
python thread.py "why wars happen" --longitudinal
```

### Launch the dashboard

```bash
python server.py
# open http://localhost:5050
```

---

## Good topics to try

Thread works best on questions where smart people genuinely disagree and evidence exists on both sides:

- `why do civilizations collapse`
- `does capitalism create inequality by design`
- `do we have free will`

Avoid questions with single factual answers, Thread is built for contested, complex questions where the reasoning process is the point.

---

The database (`thread.db`) is created automatically on first run and stores everything, runs, belief states at each iteration, every article fetched, every question asked and answered, and the final verdict.

---

## The database schema

```
runs              — one row per topic run
articles          — every article fetched, tagged by round type
belief_states     — belief at each iteration with cited_article_ids
interrogations    — every question + answer per round
final_views       — the synthesized final verdict per run
```

All queryable directly with SQLite if you want to do your own analysis.

---


## License

MIT