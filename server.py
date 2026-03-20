import os
import json
import threading
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify, request, send_from_directory
from thread import (
    init_db, load_all_topics, load_runs_for_topic,
    load_belief_history, load_interrogations,
    load_articles_for_run, load_articles_by_ids,
    load_final_belief_per_run, load_final_view,
    load_arcs_for_run, load_arc_history, load_arc_comparison,
    confidence_label, FRAMEWORKS,
    count_flips, run_thread,
)

app = Flask(__name__, static_folder=".")

# ── Static ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


# ── API ───────────────────────────────────────────────────────────────────────
@app.route("/api/topics")
def api_topics():
    conn   = init_db()
    topics = load_all_topics(conn)
    conn.close()
    return jsonify(topics)


@app.route("/api/runs")
def api_runs():
    topic = request.args.get("topic", "")
    conn  = init_db()
    runs  = load_runs_for_topic(conn, topic)
    conn.close()
    return jsonify(runs)


@app.route("/api/history")
def api_history():
    run_id = request.args.get("run_id", "")
    conn   = init_db()
    data   = load_belief_history(conn, run_id)
    conn.close()
    return jsonify(data)


@app.route("/api/interrogations")
def api_interrogations():
    run_id = request.args.get("run_id", "")
    conn   = init_db()
    data   = load_interrogations(conn, run_id)
    conn.close()
    return jsonify(data)



@app.route("/api/final_view")
def api_final_view():
    run_id = request.args.get("run_id", "")
    conn   = init_db()
    data   = load_final_view(conn, run_id)
    conn.close()
    return jsonify(data)

@app.route("/api/arcs")
def api_arcs():
    run_id = request.args.get("run_id", "")
    conn   = init_db()
    arcs   = load_arcs_for_run(conn, run_id)
    # enrich each arc with its final belief + confidence label
    result = []
    for arc in arcs:
        history = load_arc_history(conn, arc["arc_id"])
        fv      = load_final_view(conn, arc["arc_id"])
        final   = history[-1] if history else {}
        result.append({
            **arc,
            "final_claim":      final.get("core_claim", ""),
            "confidence":       final.get("confidence", 0),
            "confidence_label": confidence_label(final.get("confidence", 0)),
            "final_view":       fv,
            "history":          history,
        })
    conn.close()
    return jsonify(result)


@app.route("/api/arc_history")
def api_arc_history():
    arc_id = request.args.get("arc_id", "")
    conn   = init_db()
    data   = load_arc_history(conn, arc_id)
    qs     = load_interrogations(conn, arc_id)
    conn.close()
    return jsonify({"history": data, "interrogations": qs})


@app.route("/api/arc_comparison")
def api_arc_comparison():
    run_id = request.args.get("run_id", "")
    conn   = init_db()
    data   = load_arc_comparison(conn, run_id)
    conn.close()
    return jsonify(data)


@app.route("/api/articles")
def api_articles():
    run_id = request.args.get("run_id", "")
    conn   = init_db()
    data   = load_articles_for_run(conn, run_id)
    conn.close()
    return jsonify(data)


@app.route("/api/compare")
def api_compare():
    raw    = request.args.get("topics", "")
    topics = [t.strip() for t in raw.split("||") if t.strip()]
    conn   = init_db()
    result = {}
    for topic in topics:
        runs = load_runs_for_topic(conn, topic)
        if runs:
            hist = load_belief_history(conn, runs[-1]["run_id"])
            result[topic] = {
                "history": hist,
                "flips":   count_flips(hist),
            }
    conn.close()
    return jsonify(result)


@app.route("/api/longitudinal")
def api_longitudinal():
    topic = request.args.get("topic", "")
    conn  = init_db()
    data  = load_final_belief_per_run(conn, topic)
    conn.close()
    return jsonify(data)


# run jobs: topic → status
_jobs = {}

@app.route("/api/run", methods=["POST"])
def api_run():
    body       = request.get_json(force=True)
    topic      = body.get("topic", "").strip()
    use_search = body.get("use_search", True)

    if not topic:
        return jsonify({"error": "topic required"}), 400

    job_id = f"job_{len(_jobs)}"
    _jobs[job_id] = {"status": "running", "run_id": None, "error": None}

    def do_run():
        try:
            run_id = run_thread(topic, use_search=use_search)
            _jobs[job_id] = {"status": "done", "run_id": run_id, "error": None}
        except Exception as e:
            _jobs[job_id] = {"status": "error", "run_id": None, "error": str(e)}

    threading.Thread(target=do_run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    return jsonify(_jobs.get(job_id, {"status": "not_found"}))


if __name__ == "__main__":
    print("\n  THREAD server starting...")
    print("  Open http://localhost:5050 in your browser\n")
    app.run(port=5050, debug=False)