import os
import re
import json
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import datetime
import requests
import argparse
from typing import Optional

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL   = "deepseek/deepseek-chat-v3-0324"
DB_PATH = "thread.db"

INTERROGATION_ROUNDS = [
    ("mechanism",       "Why is that actually true? Trace the causal chain behind the claim. Don't restate it — go one level deeper into the mechanism."),
    ("counterexample",  "What's the strongest counterexample? Find a real historical case or scenario where this explanation completely fails or produces the opposite outcome."),
    ("evidence",        "How good is the evidence really? Where does it come from, who claims it, what are their incentives, and what would it take to falsify it?"),
    ("consistency",     "Does this belief contradict itself, or conflict with other widely accepted ideas? If two parts are in tension, which one has to give?"),
]

BELIEF_SCHEMA = """
Return ONLY valid JSON, no markdown fences, no extra text:
{
  "core_claim": "one clear sentence: the current best belief",
  "confidence": 0.55,
  "supporting_evidence": [
    "evidence point with source citation like [1][3]",
    "another point [2]"
  ],
  "contradicting_evidence": [
    "counter-point [4]"
  ],
  "open_questions": [
    "what remains unresolved"
  ],
  "used_sources": [1, 3, 4]
}

Rules:
- confidence: float 0.0 (total uncertainty) to 1.0 (absolute certainty)
- cite sources inline using [N] in evidence strings
- used_sources: list of source indices you actually drew from
- 3-5 items per evidence array, 2-4 open questions
- early beliefs should have low confidence
"""


# ── LLM ──────────────────────────────────────────────────────────────────────
def call_llm(messages: list, expect_json: bool = False) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY env var")
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    if expect_json:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload, timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── Database ──────────────────────────────────────────────────────────────────
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id      TEXT PRIMARY KEY,
            topic       TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            use_search  INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS articles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            round_type  TEXT,
            title       TEXT,
            url         TEXT,
            snippet     TEXT,
            source      TEXT,
            published   TEXT,
            fetched_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS belief_states (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL,
            topic               TEXT NOT NULL,
            iteration           INTEGER NOT NULL,
            core_claim          TEXT NOT NULL,
            confidence          REAL NOT NULL,
            supporting          TEXT NOT NULL,
            contradicting       TEXT NOT NULL,
            open_questions      TEXT NOT NULL,
            cited_article_ids   TEXT NOT NULL DEFAULT '[]',
            timestamp           TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS interrogations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        TEXT NOT NULL,
            iteration     INTEGER NOT NULL,
            question_type TEXT NOT NULL,
            question      TEXT NOT NULL,
            answer        TEXT,
            timestamp     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS final_views (
            run_id           TEXT PRIMARY KEY,
            topic            TEXT NOT NULL,
            verdict          TEXT NOT NULL,
            strongest_for    TEXT NOT NULL,
            strongest_against TEXT NOT NULL,
            what_changed     TEXT NOT NULL,
            still_uncertain  TEXT NOT NULL,
            confidence       REAL NOT NULL,
            plain_english    TEXT NOT NULL,
            timestamp        TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


def new_run_id(topic: str) -> str:
    ts   = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:28]
    return f"{slug}__{ts}"


def save_articles(conn, run_id: str, articles: list[dict],
                  round_type: Optional[str]) -> list[int]:
    """Store articles, return their DB ids."""
    ids = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for a in articles:
        cur = conn.execute("""
            INSERT INTO articles (run_id, round_type, title, url, snippet, source, published, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, round_type, a.get("title",""), a.get("url",""),
              a.get("snippet",""), a.get("source",""), a.get("published",""), now))
        ids.append(cur.lastrowid)
    conn.commit()
    return ids


def save_belief(conn, run_id: str, topic: str, iteration: int,
                belief: dict, article_ids: list[int]):
    """
    Save a belief state. article_ids = DB ids of articles fetched for this round.
    used_sources from the LLM are indices (1-based) into that round's article list.
    We map them → actual DB article ids.
    """
    used_indices  = [i - 1 for i in belief.get("used_sources", []) if isinstance(i, int)]
    cited_db_ids  = [article_ids[i] for i in used_indices if 0 <= i < len(article_ids)]

    conn.execute("""
        INSERT INTO belief_states
            (run_id, topic, iteration, core_claim, confidence,
             supporting, contradicting, open_questions, cited_article_ids, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, topic, iteration,
        belief["core_claim"],
        belief["confidence"],
        json.dumps(belief.get("supporting_evidence", [])),
        json.dumps(belief.get("contradicting_evidence", [])),
        json.dumps(belief.get("open_questions", [])),
        json.dumps(cited_db_ids),
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ))
    conn.commit()


def save_interrogation(conn, run_id: str, iteration: int,
                       q_type: str, question: str, answer: str):
    conn.execute("""
        INSERT INTO interrogations (run_id, iteration, question_type, question, answer, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (run_id, iteration, q_type, question, answer,
          datetime.datetime.now(datetime.timezone.utc).isoformat()))
    conn.commit()


# ── DB reads ──────────────────────────────────────────────────────────────────
def load_belief_history(conn, run_id: str) -> list[dict]:
    rows = conn.execute("""
        SELECT iteration, core_claim, confidence, supporting,
               contradicting, open_questions, cited_article_ids, timestamp
        FROM belief_states WHERE run_id = ? ORDER BY iteration
    """, (run_id,)).fetchall()
    return [{
        "iteration":              r[0],
        "core_claim":             r[1],
        "confidence":             r[2],
        "supporting_evidence":    json.loads(r[3]),
        "contradicting_evidence": json.loads(r[4]),
        "open_questions":         json.loads(r[5]),
        "cited_article_ids":      json.loads(r[6]),
        "timestamp":              r[7],
    } for r in rows]


def load_interrogations(conn, run_id: str) -> list[dict]:
    rows = conn.execute("""
        SELECT iteration, question_type, question, answer, timestamp
        FROM interrogations WHERE run_id = ? ORDER BY iteration
    """, (run_id,)).fetchall()
    return [{
        "iteration":     r[0],
        "question_type": r[1],
        "question":      r[2],
        "answer":        r[3],
        "timestamp":     r[4],
    } for r in rows]


def load_articles_for_run(conn, run_id: str) -> list[dict]:
    rows = conn.execute("""
        SELECT id, round_type, title, url, snippet, source, published
        FROM articles WHERE run_id = ? ORDER BY id
    """, (run_id,)).fetchall()
    return [{
        "id": r[0], "round_type": r[1], "title": r[2],
        "url": r[3], "snippet": r[4], "source": r[5], "published": r[6],
    } for r in rows]


def load_articles_by_ids(conn, ids: list[int]) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(f"""
        SELECT id, title, url, source, published, snippet
        FROM articles WHERE id IN ({placeholders})
    """, ids).fetchall()
    return [{"id": r[0], "title": r[1], "url": r[2],
             "source": r[3], "published": r[4], "snippet": r[5]} for r in rows]


def load_runs_for_topic(conn, topic: str) -> list[dict]:
    rows = conn.execute("""
        SELECT run_id, started_at FROM runs WHERE topic = ? ORDER BY started_at
    """, (topic,)).fetchall()
    return [{"run_id": r[0], "started_at": r[1]} for r in rows]


def load_all_topics(conn) -> list[str]:
    rows = conn.execute("SELECT DISTINCT topic FROM runs ORDER BY topic").fetchall()
    return [r[0] for r in rows]


def load_final_belief_per_run(conn, topic: str) -> list[dict]:
    results = []
    for run in load_runs_for_topic(conn, topic):
        h = load_belief_history(conn, run["run_id"])
        if h:
            f = h[-1].copy()
            f["run_id"]     = run["run_id"]
            f["started_at"] = run["started_at"]
            results.append(f)
    return results


# ── Final View ────────────────────────────────────────────────────────────────
FINAL_VIEW_SCHEMA = """
Return ONLY valid JSON, no markdown:
{
  "verdict": "one bold declarative sentence — the clearest answer the evidence supports",
  "plain_english": "3-5 sentences explaining this to someone who knows nothing about the topic. No jargon. Write like you're talking to a curious friend.",
  "strongest_for": ["top argument supporting the belief", "second argument"],
  "strongest_against": ["top argument against or complicating the belief", "second argument"],
  "what_changed": "one sentence on what most changed from initial belief to final, and why",
  "still_uncertain": ["biggest unresolved question", "second open question"],
  "confidence": 0.0
}
confidence: your honest final confidence 0.0–1.0. For complex topics this is rarely above 0.75.
"""

def generate_final_view(topic: str, history: list, interrogations: list) -> Optional[dict]:
    """
    One final LLM call after all interrogation rounds.
    Synthesizes the full arc into a plain-English verdict.
    """
    if not history:
        return None

    # build a compact arc summary for the prompt
    arc_lines = []
    q_map = {q["iteration"]: q for q in interrogations}
    for h in history:
        q = q_map.get(h["iteration"])
        label = "Initial" if h["iteration"] == 0 else f"Round {h['iteration']} [{q['question_type'] if q else ''}]"
        arc_lines.append(f"{label}: confidence={h['confidence']:.2f}")
        arc_lines.append(f"  Claim: {h['core_claim']}")
        if q and q.get("answer"):
            arc_lines.append(f"  What changed: {q['answer']}")
        arc_lines.append("")

    arc_text = "\n".join(arc_lines)

    messages = [
        {"role": "system", "content": (
            "You are producing a final synthesis after a rigorous multi-round investigation. "
            "Your job is to give someone who knows nothing about this topic the clearest, "
            "most honest answer the evidence supports — including where it's still murky. "
            "Do not hedge everything into meaninglessness. Take a position, then qualify it honestly."
        )},
        {"role": "user", "content": (
            f"Topic: \"{topic}\"\n\n"
            f"Full belief arc:\n{arc_text}\n\n"
            f"Final supporting evidence: {json.dumps(history[-1].get('supporting_evidence', []))}\n"
            f"Final contradicting evidence: {json.dumps(history[-1].get('contradicting_evidence', []))}\n"
            f"Still open: {json.dumps(history[-1].get('open_questions', []))}\n\n"
            "Produce the final view. Be honest, be clear, be useful to someone learning this for the first time.\n\n"
            f"{FINAL_VIEW_SCHEMA}"
        )},
    ]

    raw = call_llm(messages, expect_json=True)
    return safe_parse(raw)


def save_final_view(conn, run_id: str, topic: str, fv: dict):
    conn.execute("""
        INSERT OR REPLACE INTO final_views
            (run_id, topic, verdict, strongest_for, strongest_against,
             what_changed, still_uncertain, confidence, plain_english, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, topic,
        fv.get("verdict", ""),
        json.dumps(fv.get("strongest_for", [])),
        json.dumps(fv.get("strongest_against", [])),
        fv.get("what_changed", ""),
        json.dumps(fv.get("still_uncertain", [])),
        fv.get("confidence", 0.0),
        fv.get("plain_english", ""),
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ))
    conn.commit()


def load_final_view(conn, run_id: str) -> Optional[dict]:
    row = conn.execute("""
        SELECT verdict, strongest_for, strongest_against,
               what_changed, still_uncertain, confidence, plain_english, timestamp
        FROM final_views WHERE run_id = ?
    """, (run_id,)).fetchone()
    if not row:
        return None
    return {
        "verdict":          row[0],
        "strongest_for":    json.loads(row[1]),
        "strongest_against":json.loads(row[2]),
        "what_changed":     row[3],
        "still_uncertain":  json.loads(row[4]),
        "confidence":       row[5],
        "plain_english":    row[6],
        "timestamp":        row[7],
    }


def print_final_view(fv: dict):
    print(f"\n{'═'*66}")
    print("  FINAL VIEW")
    print(f"{'═'*66}")
    print(f"\n  Verdict: {fv['verdict']}")
    print(f"\n  {fv['plain_english']}")
    print(f"\n  Confidence: {conf_bar(fv['confidence'])} {fv['confidence']:.2f}")
    print(f"\n  Strongest for:")
    for a in fv.get("strongest_for", []):
        print(f"    + {a}")
    print(f"\n  Strongest against:")
    for a in fv.get("strongest_against", []):
        print(f"    - {a}")
    print(f"\n  What changed: {fv['what_changed']}")
    print(f"\n  Still uncertain:")
    for q in fv.get("still_uncertain", []):
        print(f"    ? {q}")
    print(f"{'═'*66}\n")


# ── Prompts ───────────────────────────────────────────────────────────────────
def research_prompt(topic: str, sources_block: str) -> list:
    return [
        {"role": "system", "content": (
            "You are forming an initial belief about a broad, general topic. "
            "CRITICAL:\n"
            "- Your core_claim must answer the topic BROADLY — not fixate on one region or example. "
            "If a source only covers one case (e.g. Middle East conflicts), treat it as one data "
            "point in supporting evidence, never as the definition of the whole answer.\n"
            "- Draw on your full knowledge: history, political science, economics, philosophy. "
            "Sources supplement your reasoning — they do not replace it.\n"
            "- Address multiple competing theories: resource competition, ideology, nationalism, "
            "security dilemmas, historical grievances, domestic politics, miscalculation.\n"
            "- Cite sources as [1][2] where relevant. No single source should dominate.\n"
            "- Confidence should be 0.3–0.55 for an initial belief. Stay genuinely uncertain."
        )},
        {"role": "user", "content": (
            f"Topic: \"{topic}\"\n\n"
            f"{sources_block}\n\n"
            "Form a broad general belief addressing the full scope of this topic. "
            f"Do not anchor on any single source or example.\n{BELIEF_SCHEMA}"
        )},
    ]


def interrogation_prompt(topic: str, belief: dict,
                         q_type: str, question: str,
                         sources_block: str) -> list:
    prev_supporting  = [e.split("[")[0].strip() for e in belief.get("supporting_evidence", [])]
    prev_contra      = [e.split("[")[0].strip() for e in belief.get("contradicting_evidence", [])]
    prev_open        = belief.get("open_questions", [])

    return [
        {"role": "system", "content": (
            "You are a rigorous intellectual critic doing a deep interrogation of a belief. "
            "STRICT RULES:\n"
            "1. You MUST produce NEW evidence points. Do not reuse or rephrase any of the "
            "previous supporting_evidence or contradicting_evidence. Every item in your output "
            "must make a point not already present in the belief.\n"
            "2. The core_claim MUST change wording if anything new was discovered — even slightly. "
            "Identical claims across rounds means you failed.\n"
            "3. If sources are irrelevant, ignore them and reason from your own deep knowledge.\n"
            "4. This round's focus is specifically: {q_type}. Stay on that angle.\n"
            "5. Be genuinely critical — find the real weakness in this belief for this round type."
        ).replace("{q_type}", q_type)},
        {"role": "user", "content": (
            f"Topic: \"{topic}\"\n\n"
            f"Current belief (DO NOT copy these into your output):\n"
            f"Claim: {belief.get('core_claim', '')}\n"
            f"Confidence: {belief.get('confidence', 0.5)}\n"
            f"Previous supporting points (DO NOT reuse): {prev_supporting}\n"
            f"Previous contradicting points (DO NOT reuse): {prev_contra}\n"
            f"Previous open questions (DO NOT reuse): {prev_open}\n\n"
            f"Interrogation type [{q_type.upper()}]: {question}\n\n"
            f"Sources (use if relevant, ignore if not):\n{sources_block}\n\n"
            "Produce an updated belief with ENTIRELY NEW evidence points specific to this "
            f"[{q_type}] challenge. The claim should evolve to reflect what this round revealed. "
            "Confidence goes up if the belief survives, down if a real gap was found.\n"
            "Add 'interrogation_result': one sentence — what specifically changed this round and why.\n"
            f"{BELIEF_SCHEMA}\n"
            '"interrogation_result": "what specifically changed this round and why"'
        )},
    ]


# ── Core run ──────────────────────────────────────────────────────────────────
def run_thread(topic: str, use_search: bool = True) -> str:
    conn   = init_db()
    run_id = new_run_id(topic)

    conn.execute(
        "INSERT INTO runs (run_id, topic, started_at, use_search) VALUES (?,?,?,?)",
        (run_id, topic, datetime.datetime.now(datetime.timezone.utc).isoformat(), int(use_search))
    )
    conn.commit()

    print(f"\n{'═'*66}")
    print(f"  THREAD  |  \"{topic}\"")
    print(f"  run_id  |  {run_id}")
    print(f"{'═'*66}\n")

    # track all fetched URLs across rounds to avoid repeating sources
    seen_urls = set()

    # ── initial research ──────────────────────────────────────────────────────
    articles, article_ids, sources_block = _fetch_sources(
        conn, run_id, topic, round_type=None, use_search=use_search, seen_urls=seen_urls
    )
    seen_urls.update(a["url"] for a in articles if a.get("url"))

    print("📚  Forming initial belief...\n")
    raw    = call_llm(research_prompt(topic, sources_block), expect_json=True)
    belief = safe_parse(raw)
    if not belief:
        return run_id

    print_belief(belief, iteration=0)
    save_belief(conn, run_id, topic, 0, belief, article_ids)
    save_interrogation(conn, run_id, 0, "initial", f'What is the initial belief about "{topic}"?',
                       belief.get("core_claim", ""))

    # ── interrogation rounds ──────────────────────────────────────────────────
    for i, (q_type, question) in enumerate(INTERROGATION_ROUNDS, start=1):
        print(f"\n{'─'*66}")
        print(f"🔍  Round {i}/4 — {q_type.upper()}")
        print(f"    {question[:72]}...")
        print(f"{'─'*66}\n")

        articles, article_ids, sources_block = _fetch_sources(
            conn, run_id, topic, round_type=q_type, use_search=use_search, seen_urls=seen_urls
        )
        seen_urls.update(a["url"] for a in articles if a.get("url"))

        raw        = call_llm(
            interrogation_prompt(topic, belief, q_type, question, sources_block),
            expect_json=True
        )
        new_belief = safe_parse(raw)
        if not new_belief:
            continue

        answer = new_belief.pop("interrogation_result", f"{q_type} probe")
        delta  = new_belief["confidence"] - belief["confidence"]
        belief = new_belief

        print_belief(belief, iteration=i, delta=delta, trigger=answer)
        save_belief(conn, run_id, topic, i, belief, article_ids)
        save_interrogation(conn, run_id, i, q_type, question, answer)

    # ── final view ────────────────────────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("🧠  Synthesizing final view...\n")
    history = load_belief_history(conn, run_id)
    qs      = load_interrogations(conn, run_id)
    fv      = generate_final_view(topic, history, qs)
    if fv:
        save_final_view(conn, run_id, topic, fv)
        print_final_view(fv)

    # ── trajectory summary ────────────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print("  TRAJECTORY")
    print(f"{'═'*66}")
    for h in history:
        label = f"[{h['iteration']}]"
        src   = f"  ({len(h['cited_article_ids'])} sources)" if h["cited_article_ids"] else ""
        print(f"  {label} {conf_bar(h['confidence'])} {h['confidence']:.2f}{src}")
        print(f"      {h['core_claim'][:70]}")

    all_articles = load_articles_for_run(conn, run_id)
    print(f"\n  Total sources fetched : {len(all_articles)}")
    print(f"  Saved to              : {DB_PATH}  (run_id={run_id})")
    print(f"{'═'*66}\n")

    conn.close()
    return run_id


def _fetch_sources(conn, run_id, topic, round_type, use_search, seen_urls=None):
    """Fetch + store articles for a round. Returns (articles, db_ids, sources_block)."""
    articles = []
    if use_search:
        try:
            from searcher import build_sources_for_round, format_sources_for_prompt
            articles = build_sources_for_round(
                topic, round_type, verbose=True,
                seen_urls=seen_urls or set()
            )
        except Exception as e:
            import traceback
            print(f"  [searcher] FULL ERROR:")
            traceback.print_exc()

    article_ids = save_articles(conn, run_id, articles, round_type)

    try:
        from searcher import format_sources_for_prompt
        sources_block = format_sources_for_prompt(articles)
    except Exception:
        sources_block = "(No external sources)"

    return articles, article_ids, sources_block


# ── Multi-topic comparison ────────────────────────────────────────────────────
def run_compare(topics: list[str], use_search: bool = True):
    print(f"\n{'═'*66}")
    print(f"  THREAD COMPARE  |  {len(topics)} topics")
    print(f"{'═'*66}\n")

    conn    = init_db()
    results = {}
    for topic in topics:
        run_id         = run_thread(topic, use_search=use_search)
        results[topic] = load_belief_history(conn, run_id)

    print(f"\n{'═'*66}")
    print("  CONVERGENCE SUMMARY")
    print(f"{'═'*66}")
    print(f"  {'TOPIC':<36} {'INIT':>5} {'FINAL':>6} {'Δ':>6} {'FLIPS':>6}")
    print(f"  {'─'*36} {'─'*5} {'─'*6} {'─'*6} {'─'*6}")
    for topic, history in results.items():
        if not history:
            continue
        s = history[0]["confidence"]
        e = history[-1]["confidence"]
        d = e - s
        f = count_flips(history)
        label = (topic[:34] + "..") if len(topic) > 36 else topic
        print(f"  {label:<36} {s:>5.2f} {e:>6.2f} {d:>+6.2f} {f:>6}")
    print(f"{'═'*66}\n")
    conn.close()


# ── Longitudinal ──────────────────────────────────────────────────────────────
def show_longitudinal(topic: str):
    conn = init_db()
    runs = load_final_belief_per_run(conn, topic)
    conn.close()

    if not runs:
        print(f"  No runs found for: '{topic}'")
        return

    print(f"\n{'═'*66}")
    print(f"  LONGITUDINAL  |  \"{topic}\"  ({len(runs)} run(s))")
    print(f"{'═'*66}")
    for r in runs:
        ts = r["started_at"][:10]
        print(f"\n  [{ts}]  {conf_bar(r['confidence'])} {r['confidence']:.2f}")
        print(f"  Claim   : {r['core_claim']}")
        if r.get("open_questions"):
            print(f"  Open    : {r['open_questions'][0]}")
    print(f"\n{'═'*66}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_parse(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        print("  ⚠️  JSON parse failed:\n", raw[:300])
        return None


def conf_bar(c: float) -> str:
    n = round(c * 10)
    return f"[{'█'*n}{'░'*(10-n)}]"


def conf_color(c: float) -> str:
    return "high" if c >= 0.65 else ("mid" if c >= 0.4 else "low")


def count_flips(history: list) -> int:
    flips, prev = 0, None
    for i in range(1, len(history)):
        d   = history[i]["confidence"] - history[i-1]["confidence"]
        cur = "up" if d > 0.01 else ("down" if d < -0.01 else None)
        if cur and prev and cur != prev:
            flips += 1
        if cur:
            prev = cur
    return flips


def print_belief(belief: dict, iteration: int,
                 delta: Optional[float] = None, trigger: Optional[str] = None):
    c  = belief["confidence"]
    dv = f"  ({'↑' if delta >= 0 else '↓'} {abs(delta):.2f})" if delta is not None else ""
    print(f"  Confidence : {conf_bar(c)} {c:.2f}{dv}")
    print(f"  Claim      : {belief['core_claim']}\n")
    if trigger:
        print(f"  Changed    : {trigger}\n")
    print("  Supporting:")
    for e in belief.get("supporting_evidence", []):
        print(f"    + {e}")
    print("\n  Contradicting:")
    for e in belief.get("contradicting_evidence", []):
        print(f"    - {e}")
    print("\n  Open questions:")
    for q in belief.get("open_questions", []):
        print(f"    ? {q}")
    used = belief.get("used_sources", [])
    if used:
        print(f"\n  Sources used : {used}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("topics", nargs="*", default=["why wars happen"])
    p.add_argument("--no-search",    action="store_true")
    p.add_argument("--longitudinal", action="store_true")
    args = p.parse_args()

    use_search = not args.no_search

    if args.longitudinal:
        show_longitudinal(args.topics[0])
    elif len(args.topics) > 1:
        run_compare(args.topics, use_search=use_search)
    else:
        run_thread(args.topics[0], use_search=use_search)