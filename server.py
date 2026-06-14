import os, json
from datetime import datetime
from flask import Flask, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def read_json(path):
    if not os.path.exists(path): return None
    try:
        with open(path, encoding="utf-8") as f: return json.load(f)
    except: return None

def read_log(n=30):
    if not os.path.exists("kalshi_alpha.log"): return []
    try:
        with open("kalshi_alpha.log", encoding="utf-8") as f:
            lines = f.readlines()
        return [l.strip() for l in lines[-n:] if l.strip()]
    except: return []

@app.route("/api/status")
def status():
    s = read_json("bot_state.json") or {}
    t = read_json("kalshi_trades.json") or []
    return jsonify({
        "running": s.get("running", False),
        "cycle": s.get("cycle", 0),
        "last_ticker": s.get("last_ticker", "—"),
        "last_verdict": s.get("last_verdict", "—"),
        "last_edge": s.get("last_edge", 0),
        "last_ev": s.get("last_ev", 0),
        "last_grade": s.get("last_grade", "—"),
        "last_reason": s.get("last_reason", ""),
        "last_risk": s.get("last_risk", ""),
        "last_update": s.get("last_update", ""),
        "total_trades": len(t),
        "recent_trades": t[-5:][::-1],
        "logs": read_log(25)[::-1],
        "scores": {
            "qualite_donnees": s.get("score_qualite", 0),
            "confiance_statistique": s.get("score_confiance", 0),
            "risque": s.get("score_risque", 0),
            "volatilite": s.get("score_volatilite", 0),
            "edge_score": s.get("score_edge", 0),
        },
        "scenarios": s.get("scenarios", []),
        "metrics": {
            "prob_reelle": s.get("prob_reelle", 0),
            "prob_marche": s.get("prob_marche", 0),
            "ev_brute": s.get("ev_brute", 0),
            "confiance": s.get("last_conf", 0),
            "risque": s.get("last_risque", 0),
            "taille_position": s.get("last_size", "—"),
            "risque_exogene": s.get("risque_exogene", "—"),
        }
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

@app.route("/")
def index():
    return jsonify({"status": "Kalshi Alpha Engine V3 — online"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Serveur sur port {port}")
    app.run(host="0.0.0.0", port=port)
