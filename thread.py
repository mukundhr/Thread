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
# Two different models = genuine adversarial pressure
# Generator forms beliefs; Critic interrogates them from a different training distribution
GENERATOR_MODEL = "deepseek/deepseek-v3.2"       # forms beliefs
CRITIC_MODEL    = "google/gemini-2.5-flash-lite"               # interrogates them
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

# ── Epistemic frameworks — three independent lenses ───────────────────────────
FRAMEWORKS = {
    "structural": {
        "label":       "Structural",
        "description": "systems, institutions, historical forces, path dependency",
        "prompt":      (
            "You reason exclusively through a STRUCTURAL lens. "
            "Explanations must focus on: systemic forces, institutional constraints, "
            "historical path dependency, structural incentives, and social/political systems. "
            "Individual actors matter only insofar as they are shaped by structures. "
            "Do not invoke individual psychology, personal decisions, or cultural values "
            "as primary causes — always trace back to structural conditions."
        ),
    },
    "agent": {
        "label":       "Agent / Psychological",
        "description": "individuals, decisions, psychology, leadership, cognition",
        "prompt":      (
            "You reason exclusively through an AGENT / PSYCHOLOGICAL lens. "
            "Explanations must focus on: individual decisions, leadership psychology, "
            "cognitive biases, misperception, group identity, cultural norms, and "
            "the role of specific actors at critical junctures. "
            "Do not invoke impersonal structural forces as primary causes — "
            "always trace back to human decisions and psychology."
        ),
    },
    "material": {
        "label":       "Material / Economic",
        "description": "resources, incentives, economics, costs, power distribution",
        "prompt":      (
            "You reason exclusively through a MATERIAL / ECONOMIC lens. "
            "Explanations must focus on: resource distribution, economic incentives, "
            "cost-benefit calculations, material interests, power as measurable capability, "
            "and economic interdependence or competition. "
            "Do not invoke ideology, psychology, or institutional norms as primary causes — "
            "always trace back to material interests and resource logic."
        ),
    },
}

# ── Confidence label — honest description instead of fake precision ────────────
def confidence_label(score: float) -> str:
    if score >= 0.80: return "Strongly supported"
    if score >= 0.65: return "Well supported"
    if score >= 0.50: return "Moderately supported"
    if score >= 0.35: return "Contested"
    return "Poorly supported"



# ── LLM ──────────────────────────────────────────────────────────────────────
def call_llm(messages: list, expect_json: bool = False,
             model: str = None) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY env var")
    payload = {
        "model": model or GENERATOR_MODEL,
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
            tier        INTEGER NOT NULL DEFAULT 3,
            source_type TEXT NOT NULL DEFAULT 'unknown',
            fetched_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS belief_states (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL,
            topic               TEXT NOT NULL,
            iteration           INTEGER NOT NULL,
            core_claim          TEXT NOT NULL,
            llm_confidence      REAL NOT NULL,
            computed_confidence REAL NOT NULL,
            confidence_breakdown TEXT NOT NULL DEFAULT '{}',
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

        CREATE TABLE IF NOT EXISTS arcs (
            arc_id      TEXT PRIMARY KEY,
            run_id      TEXT NOT NULL,
            topic       TEXT NOT NULL,
            framework   TEXT NOT NULL,
            model       TEXT NOT NULL,
            started_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS arc_comparisons (
            run_id              TEXT PRIMARY KEY,
            topic               TEXT NOT NULL,
            convergence_points  TEXT NOT NULL,
            divergence_points   TEXT NOT NULL,
            framework_dependent TEXT NOT NULL,
            most_robust         TEXT NOT NULL,
            plain_english       TEXT NOT NULL,
            timestamp           TEXT NOT NULL
        );
    """)
    conn.commit()

    # ── Schema migrations — add columns that didn't exist in older DBs ──────────
    migrations = [
        ("articles",     "tier",                 "INTEGER NOT NULL DEFAULT 3"),
        ("articles",     "source_type",          "TEXT NOT NULL DEFAULT 'unknown'"),
        ("belief_states","llm_confidence",        "REAL NOT NULL DEFAULT 0.5"),
        ("belief_states","computed_confidence",   "REAL NOT NULL DEFAULT 0.5"),
        ("belief_states","confidence_breakdown",  "TEXT NOT NULL DEFAULT '{}'"),
    ]
    existing_cols = {}
    for table, col, definition in migrations:
        if table not in existing_cols:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            existing_cols[table] = {r[1] for r in rows}
        if col not in existing_cols[table]:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
            existing_cols[table].add(col)
    conn.commit()

    return conn


def new_run_id(topic: str) -> str:
    ts   = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:28]
    return f"{slug}__{ts}"


def save_articles(conn, run_id: str, articles: list[dict],
                  round_type: Optional[str]) -> list[int]:
    """Store articles with tier + source_type. Returns DB ids."""
    ids = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for a in articles:
        cur = conn.execute("""
            INSERT INTO articles
                (run_id, round_type, title, url, snippet, source, published, tier, source_type, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, round_type, a.get("title",""), a.get("url",""),
              a.get("snippet",""), a.get("source",""), a.get("published",""),
              a.get("tier", 3), a.get("source_type", "unknown"), now))
        ids.append(cur.lastrowid)
    conn.commit()
    return ids


def save_belief(conn, run_id: str, topic: str, iteration: int,
                belief: dict, article_ids: list[int],
                articles: list[dict] = None):
    """
    Save a belief state with both LLM-reported and computed confidence.
    article_ids = DB ids for this round; articles = raw dicts for tier scoring.
    """
    used_indices = [i - 1 for i in belief.get("used_sources", []) if isinstance(i, int)]
    cited_db_ids = [article_ids[i] for i in used_indices if 0 <= i < len(article_ids)]

    # use cited articles for quality scoring; fall back to all round articles
    cited_articles = [articles[i] for i in used_indices if articles and 0 <= i < len(articles)]
    if not cited_articles and articles:
        cited_articles = articles

    llm_conf  = float(belief.get("confidence", 0.5))
    breakdown = compute_confidence(llm_conf, cited_articles, belief)
    computed  = breakdown["computed"]

    conn.execute("""
        INSERT INTO belief_states
            (run_id, topic, iteration, core_claim,
             llm_confidence, computed_confidence, confidence_breakdown,
             supporting, contradicting, open_questions, cited_article_ids, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, topic, iteration,
        belief["core_claim"],
        llm_conf, computed,
        json.dumps(breakdown),
        json.dumps(belief.get("supporting_evidence", [])),
        json.dumps(belief.get("contradicting_evidence", [])),
        json.dumps(belief.get("open_questions", [])),
        json.dumps(cited_db_ids),
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ))
    conn.commit()
    return breakdown


# ── Confidence computation ────────────────────────────────────────────────────
TIER_SCORES = {1: 1.0, 2: 0.55, 3: 0.15}
SOURCE_TYPE_SCORES = {
    "rct": 1.0, "meta-analysis": 1.0, "systematic-review": 0.95,
    "peer-reviewed": 0.85, "preprint": 0.65,
    "government-data": 0.80, "primary-data": 0.80,
    "think-tank": 0.60, "journalism": 0.50,
    "expert-opinion": 0.45, "blog": 0.20, "unknown": 0.25,
}

def compute_confidence(llm_conf: float, articles: list, belief: dict) -> dict:
    """
    Derive confidence from observable quantities rather than trusting
    the LLM's self-reported number alone.

    Components:
      - llm_conf (40%)  : LLM self-report — useful signal, not gospel
      - source_quality (35%) : average tier score of cited articles
      - evidence_ratio (25%) : supporting / (supporting + contradicting)
      - open_penalty         : subtract for unresolved questions
    """
    # source quality from cited articles
    if articles:
        avg_tier  = sum(TIER_SCORES.get(a.get("tier", 3), 0.15) for a in articles) / len(articles)
        avg_stype = sum(SOURCE_TYPE_SCORES.get(a.get("source_type","unknown"), 0.25)
                        for a in articles) / len(articles)
        source_quality = (avg_tier + avg_stype) / 2
    else:
        source_quality = 0.10  # no sources = honest uncertainty

    # evidence balance
    n_sup = len(belief.get("supporting_evidence", []))
    n_con = len(belief.get("contradicting_evidence", []))
    total = n_sup + n_con
    evidence_ratio = (n_sup / total) if total > 0 else 0.5

    # open questions penalty (each unresolved question = -4%, max -20%)
    n_open       = len(belief.get("open_questions", []))
    open_penalty = min(n_open * 0.04, 0.20)

    computed = (
        0.40 * llm_conf +
        0.35 * source_quality +
        0.25 * evidence_ratio -
        open_penalty
    )
    computed = round(max(0.05, min(0.95, computed)), 2)

    return {
        "llm_reported":     round(llm_conf, 2),
        "computed":         computed,
        "source_quality":   round(source_quality, 2),
        "evidence_ratio":   round(evidence_ratio, 2),
        "open_penalty":     round(open_penalty, 2),
        "n_sources":        len(articles),
        "n_supporting":     n_sup,
        "n_contradicting":  n_con,
        "n_open":           n_open,
    }


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
    # handle both old schema (confidence) and new (llm_confidence + computed_confidence)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(belief_states)").fetchall()]
    if "llm_confidence" in cols:
        rows = conn.execute("""
            SELECT iteration, core_claim, llm_confidence, computed_confidence,
                   confidence_breakdown, supporting, contradicting, open_questions,
                   cited_article_ids, timestamp
            FROM belief_states WHERE run_id = ? ORDER BY iteration
        """, (run_id,)).fetchall()
        return [{
            "iteration":              r[0],
            "core_claim":             r[1],
            "llm_confidence":         r[2],
            "confidence":             r[3],   # computed — this is what UI shows
            "confidence_breakdown":   json.loads(r[4] or "{}"),
            "supporting_evidence":    json.loads(r[5]),
            "contradicting_evidence": json.loads(r[6]),
            "open_questions":         json.loads(r[7]),
            "cited_article_ids":      json.loads(r[8]),
            "timestamp":              r[9],
        } for r in rows]
    else:
        # old schema fallback
        rows = conn.execute("""
            SELECT iteration, core_claim, confidence, supporting,
                   contradicting, open_questions, cited_article_ids, timestamp
            FROM belief_states WHERE run_id = ? ORDER BY iteration
        """, (run_id,)).fetchall()
        return [{
            "iteration":              r[0],
            "core_claim":             r[1],
            "llm_confidence":         r[2],
            "confidence":             r[2],
            "confidence_breakdown":   {},
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
    cols = [r[1] for r in conn.execute("PRAGMA table_info(articles)").fetchall()]
    if "tier" in cols:
        rows = conn.execute("""
            SELECT id, round_type, title, url, snippet, source, published, tier, source_type
            FROM articles WHERE run_id = ? ORDER BY id
        """, (run_id,)).fetchall()
        return [{"id":r[0],"round_type":r[1],"title":r[2],"url":r[3],
                 "snippet":r[4],"source":r[5],"published":r[6],
                 "tier":r[7],"source_type":r[8]} for r in rows]
    else:
        rows = conn.execute("""
            SELECT id, round_type, title, url, snippet, source, published
            FROM articles WHERE run_id = ? ORDER BY id
        """, (run_id,)).fetchall()
        return [{"id":r[0],"round_type":r[1],"title":r[2],"url":r[3],
                 "snippet":r[4],"source":r[5],"published":r[6],
                 "tier":3,"source_type":"unknown"} for r in rows]


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


# ── Arc DB functions ──────────────────────────────────────────────────────────
def save_arc(conn, arc_id: str, run_id: str, topic: str, framework: str, model: str):
    conn.execute(
        "INSERT INTO arcs (arc_id, run_id, topic, framework, model, started_at) VALUES (?,?,?,?,?,?)",
        (arc_id, run_id, topic, framework, model,
         datetime.datetime.now(datetime.timezone.utc).isoformat())
    )
    conn.commit()


def load_arcs_for_run(conn, run_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT arc_id, framework, model, started_at FROM arcs WHERE run_id=? ORDER BY started_at",
        (run_id,)
    ).fetchall()
    return [{"arc_id":r[0],"framework":r[1],"model":r[2],"started_at":r[3]} for r in rows]


def load_arc_history(conn, arc_id: str) -> list[dict]:
    """Load belief history scoped to a single arc (arc_id stored as run_id in belief_states)."""
    return load_belief_history(conn, arc_id)


def _to_str_list(val) -> list:
    """Ensure list items are strings — LLMs sometimes return objects instead."""
    if not isinstance(val, list):
        return []
    return [v if isinstance(v, str) else json.dumps(v) for v in val]


def save_arc_comparison(conn, run_id: str, topic: str, comp: dict):
    conn.execute("""
        INSERT OR REPLACE INTO arc_comparisons
            (run_id, topic, convergence_points, divergence_points,
             framework_dependent, most_robust, plain_english, timestamp)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        run_id, topic,
        json.dumps(_to_str_list(comp.get("convergence_points", []))),
        json.dumps(_to_str_list(comp.get("divergence_points", []))),
        json.dumps(_to_str_list(comp.get("framework_dependent", []))),
        str(comp.get("most_robust", "")),
        str(comp.get("plain_english", "")),
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ))
    conn.commit()


def load_arc_comparison(conn, run_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT convergence_points, divergence_points, framework_dependent, most_robust, plain_english, timestamp "
        "FROM arc_comparisons WHERE run_id=?", (run_id,)
    ).fetchone()
    if not row:
        return None
    return {
        "convergence_points":  json.loads(row[0]),
        "divergence_points":   json.loads(row[1]),
        "framework_dependent": json.loads(row[2]),
        "most_robust":         row[3],
        "plain_english":       row[4],
        "timestamp":           row[5],
    }


ARC_COMPARISON_SCHEMA = """
Return ONLY valid JSON:
{
  "convergence_points": [
    "claim that ALL three frameworks agreed on"
  ],
  "divergence_points": [
    "specific point where frameworks reached different conclusions — describe the disagreement"
  ],
  "framework_dependent": [
    "finding that is only true from one framework's lens — name which lens"
  ],
  "most_robust": "one sentence: what claim survives across all three frameworks and is most trustworthy",
  "plain_english": "3-4 sentences for a non-expert: what did the three lenses agree on, where did they split, and what does that tell us about the limits of our knowledge"
}
"""


def generate_arc_comparison(topic: str, arc_results: list[dict]) -> Optional[dict]:
    """
    arc_results: list of {framework, label, final_claim, confidence_label, key_points}
    Sends all three arc summaries to a neutral model and asks where they converge/diverge.
    """
    if len(arc_results) < 2:
        return None

    arc_summaries = []
    for a in arc_results:
        arc_summaries.append(
            f"=== {a['label'].upper()} LENS ===\n"
            f"Final claim: {a['final_claim']}\n"
            f"Confidence: {a['confidence_label']}\n"
            f"Key supporting points: {'; '.join(a.get('key_supporting', [])[:3])}\n"
            f"Key contradictions: {'; '.join(a.get('key_contradicting', [])[:2])}\n"
            f"Open questions: {'; '.join(a.get('open_questions', [])[:2])}"
        )

    messages = [
        {"role": "system", "content": (
            "You are a neutral meta-analyst comparing three independent belief arcs "
            "about the same topic. Each arc used a different analytical lens with no "
            "cross-contamination. Your job is to identify: what they all agree on "
            "(most trustworthy), where they diverge (genuine uncertainty), and what "
            "findings only appear from one lens (framework-dependent). "
            "Be specific about the disagreements — vague 'different perspectives' is useless. "
            "Name the exact claim and exactly how the lenses differ on it."
        )},
        {"role": "user", "content": (
            f"Topic: \"{topic}\"\n\n"
            f"Three independent analyses:\n\n"
            + "\n\n".join(arc_summaries)
            + f"\n\n{ARC_COMPARISON_SCHEMA}"
        )},
    ]
    raw = call_llm(messages, expect_json=True, model=CRITIC_MODEL)
    return safe_parse(raw)


# ── Final View ────────────────────────────────────────────────────────────────
# ── Evidence classification schema ────────────────────────────────────────────
EVIDENCE_CLASSIFICATION_SCHEMA = """
Before evaluating evidence quality, classify each source. Return ONLY valid JSON:
{
  "source_classifications": [
    {
      "index": 1,
      "source_type": "peer-reviewed|meta-analysis|rct|systematic-review|preprint|government-data|primary-data|think-tank|journalism|expert-opinion|blog|unknown",
      "study_design": "what kind of evidence is this? (e.g. RCT, observational cohort, survey, opinion, historical analysis)",
      "sample_or_scope": "what population/scope does this cover?",
      "key_limitation": "biggest methodological weakness",
      "weight": "high|medium|low"
    }
  ],
  "core_claim": "updated claim after evaluating source quality",
  "confidence": 0.0,
  "supporting_evidence": ["point citing sources [N]"],
  "contradicting_evidence": ["counter citing sources [N]"],
  "open_questions": ["what would require better evidence to resolve"],
  "used_sources": [1, 2],
  "interrogation_result": "what changed after scrutinising evidence quality"
}
"""


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
def research_prompt(topic: str, sources_block: str,
                    framework: dict = None) -> list:
    framework_instruction = (
        f"\n\nFRAMEWORK CONSTRAINT: {framework['prompt']}"
        if framework else ""
    )
    framework_label = f" [{framework['label']} lens]" if framework else ""
    return [
        {"role": "system", "content": (
            f"You are forming an initial belief about a broad topic{framework_label}. "
            "RULES:\n"
            "- Your core_claim must answer the topic BROADLY — not fixate on one example.\n"
            "- Draw on your full knowledge. Sources supplement reasoning, not replace it.\n"
            "- Cite sources as [1][2] where relevant. No single source should dominate.\n"
            "- Confidence should be 0.3–0.55 for an initial belief. Stay genuinely uncertain."
            f"{framework_instruction}"
        )},
        {"role": "user", "content": (
            f"Topic: \"{topic}\"\n\n"
            f"{sources_block}\n\n"
            "Form a broad general belief addressing the full scope of this topic "
            f"{'through the ' + framework['label'] + ' lens' if framework else ''}. "
            f"Do not anchor on any single source or example.\n{BELIEF_SCHEMA}"
        )},
    ]


def interrogation_prompt(topic: str, belief: dict,
                         q_type: str, question: str,
                         sources_block: str) -> tuple[list, str]:
    """Returns (messages, model_to_use). Critic uses CRITIC_MODEL."""
    prev_supporting = [e.split("[")[0].strip() for e in belief.get("supporting_evidence", [])]
    prev_contra     = [e.split("[")[0].strip() for e in belief.get("contradicting_evidence", [])]
    prev_open       = belief.get("open_questions", [])

    # evidence round uses structured source classification
    if q_type == "evidence":
        schema = EVIDENCE_CLASSIFICATION_SCHEMA
        extra  = (
            "Classify EVERY source before evaluating it. "
            "Distinguish between RCTs, observational studies, expert opinion, and journalism. "
            "Downweight low-quality sources explicitly. "
            "Only update confidence if the best-quality sources support or undermine the claim."
        )
    else:
        schema = BELIEF_SCHEMA + '\n"interrogation_result": "what specifically changed this round and why"'
        extra  = ""

    messages = [
        {"role": "system", "content": (
            f"You are a rigorous critic interrogating a belief from the [{q_type.upper()}] angle. "
            "You were trained differently from the model that generated this belief — use that difference. "
            "RULES:\n"
            "1. Every evidence point must be NEW — do not reuse or rephrase previous points.\n"
            "2. The core_claim must change wording if this round revealed something new.\n"
            "3. Ignore irrelevant sources. Reason from knowledge when sources are weak.\n"
            f"4. {extra if extra else 'Find the genuine weakness specific to this angle.'}"
        )},
        {"role": "user", "content": (
            f"Topic: \"{topic}\"\n\n"
            f"Current belief — DO NOT copy into output:\n"
            f"  Claim: {belief.get('core_claim', '')}\n"
            f"  Confidence: {belief.get('confidence', 0.5)}\n"
            f"  Previous supporting (DO NOT reuse): {prev_supporting}\n"
            f"  Previous contradicting (DO NOT reuse): {prev_contra}\n"
            f"  Previous open questions (DO NOT reuse): {prev_open}\n\n"
            f"Interrogation [{q_type.upper()}]: {question}\n\n"
            f"Sources:\n{sources_block}\n\n"
            f"{schema}"
        )},
    ]
    return messages, CRITIC_MODEL


# ── Core run ──────────────────────────────────────────────────────────────────
def run_arc(conn, run_id: str, topic: str,
            framework: dict, use_search: bool = True) -> dict:
    """
    Run one independent belief arc for a single framework lens.
    Stores beliefs under arc_id (not run_id) so arcs don't pollute each other.
    Returns a summary dict for comparison.
    """
    arc_id = f"{run_id}__{framework['label'].lower()}"
    save_arc(conn, arc_id, run_id, topic, framework["label"], GENERATOR_MODEL)

    fw_label = framework["label"]
    print(f"\n  {'─'*60}")
    print(f"  🔭  Arc: {fw_label.upper()} lens  [{framework['description']}]")
    print(f"  {'─'*60}\n")

    seen_urls: set = set()

    articles, article_ids, sources_block = _fetch_sources(
        conn, arc_id, topic, round_type=None, use_search=use_search, seen_urls=seen_urls
    )
    seen_urls.update(a["url"] for a in articles if a.get("url"))

    print(f"  📚  Forming initial belief [{fw_label} lens]...\n")
    raw    = call_llm(research_prompt(topic, sources_block, framework),
                      expect_json=True, model=GENERATOR_MODEL)
    belief = safe_parse(raw)
    if not belief:
        return {"framework": fw_label, "failed": True}

    save_belief(conn, arc_id, topic, 0, belief, article_ids, articles)
    save_interrogation(conn, arc_id, 0, "initial",
                       f'What is the [{fw_label}] belief about "{topic}"?',
                       belief.get("core_claim", ""))

    for i, (q_type, question) in enumerate(INTERROGATION_ROUNDS, start=1):
        print(f"  🔍  [{fw_label}] Round {i}/4 — {q_type.upper()}")
        articles, article_ids, sources_block = _fetch_sources(
            conn, arc_id, topic, round_type=q_type, use_search=use_search, seen_urls=seen_urls
        )
        seen_urls.update(a["url"] for a in articles if a.get("url"))

        messages, critic_model = interrogation_prompt(topic, belief, q_type, question, sources_block)
        raw        = call_llm(messages, expect_json=True, model=critic_model)
        new_belief = safe_parse(raw)
        if not new_belief:
            continue

        new_belief.pop("source_classifications", None)
        answer = new_belief.pop("interrogation_result", f"{q_type} probe")
        delta  = new_belief["confidence"] - belief["confidence"]
        belief = new_belief

        save_belief(conn, arc_id, topic, i, belief, article_ids, articles)
        save_interrogation(conn, arc_id, i, q_type, question, answer)

    # per-arc final view
    history = load_belief_history(conn, arc_id)
    qs      = load_interrogations(conn, arc_id)
    fv      = generate_final_view(topic, history, qs)
    if fv:
        save_final_view(conn, arc_id, topic, fv)

    final = history[-1] if history else belief
    clabel = confidence_label(final["confidence"])

    print(f"  ✓  [{fw_label}] Final: {clabel} — {final['core_claim'][:70]}\n")

    return {
        "arc_id":           arc_id,
        "framework":        fw_label,
        "label":            fw_label,
        "final_claim":      final["core_claim"],
        "confidence":       final["confidence"],
        "confidence_label": clabel,
        "key_supporting":   final.get("supporting_evidence", []),
        "key_contradicting":final.get("contradicting_evidence", []),
        "open_questions":   final.get("open_questions", []),
    }


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
    print(f"  models  |  generator={GENERATOR_MODEL.split('/')[-1]}  critic={CRITIC_MODEL.split('/')[-1]}")
    print(f"{'═'*66}\n")

    # ── run 3 independent arcs ────────────────────────────────────────────────
    arc_results = []
    for fw in FRAMEWORKS.values():
        result = run_arc(conn, run_id, topic, fw, use_search=use_search)
        if not result.get("failed"):
            arc_results.append(result)

    # ── compare arcs ──────────────────────────────────────────────────────────
    if len(arc_results) >= 2:
        print(f"\n{'─'*66}")
        print("🔀  Comparing arcs — finding convergence and divergence...\n")
        comp = generate_arc_comparison(topic, arc_results)
        if comp:
            save_arc_comparison(conn, run_id, topic, comp)
            print(f"  Most robust finding: {comp.get('most_robust', '—')}\n")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print("  FINAL COMPARISON")
    print(f"{'═'*66}")
    for r in arc_results:
        print(f"  [{r['label']:10}]  {r['confidence_label']:22}  {r['final_claim'][:55]}")

    print(f"\n  Saved: {DB_PATH}  (run_id={run_id})")
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