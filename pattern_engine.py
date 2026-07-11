"""
pattern_engine.py  --  v1
Detecte les patterns dans l'historique des trades resolus et
ajuste les parametres du modele en consequence.

PATTERNS DETECTES :
1. Heure de la journee      -- certaines heures sont plus predictibles
2. Distance au strike       -- plus loin = plus facile a predire
3. Volatilite               -- marche calme vs agite
4. Momentum recent          -- tendance haussiere/baissiere
5. Heure d'ouverture NY/London -- sessions de marche actives

USAGE :
    from pattern_engine import get_adjusted_params, update_patterns
    params = get_adjusted_params(current_features)
"""

import json
import os
import math
import time
import logging
from typing import Dict, Optional

log = logging.getLogger("PatternEngine")

TRADE_RESULTS_FILE = "btc_trade_results.json"
PATTERNS_FILE      = "btc_patterns.json"
MIN_SAMPLES        = 10   # minimum de trades par pattern pour etre fiable


# ── Chargement ────────────────────────────────────────────────────────────────

def _load_results() -> list:
    if not os.path.exists(TRADE_RESULTS_FILE):
        return []
    try:
        with open(TRADE_RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_patterns(patterns: dict):
    try:
        with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
            json.dump(patterns, f, indent=2)
    except Exception as e:
        log.warning(f"Erreur sauvegarde patterns: {e}")

def _load_patterns() -> dict:
    if not os.path.exists(PATTERNS_FILE):
        return {}
    try:
        with open(PATTERNS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Feature extraction ────────────────────────────────────────────────────────

def _hour_bucket(timestamp: float) -> str:
    """Regroupe les heures en sessions de marche."""
    from datetime import datetime, timezone
    dt   = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    hour = dt.hour
    if 13 <= hour < 17:   return "NY_open"       # 9h-13h EST
    if 17 <= hour < 21:   return "NY_afternoon"  # 13h-17h EST
    if 21 <= hour < 24:   return "NY_close_Asia" # 17h-20h EST + debut Asie
    if 0  <= hour < 3:    return "Asia_early"
    if 3  <= hour < 8:    return "Asia_late"
    if 8  <= hour < 13:   return "London"        # ouverture Londres
    return "other"

def _edge_bucket(edge: float) -> str:
    e = abs(edge)
    if e < 0.05: return "low"
    if e < 0.10: return "medium"
    return "high"

def _volatility_bucket(vol: float) -> str:
    """Classe la volatilite en 3 niveaux."""
    if vol < 0.002:  return "low"
    if vol < 0.005:  return "medium"
    return "high"

def _distance_bucket(price: float, strike: float) -> str:
    """Distance relative prix/strike."""
    if price <= 0 or strike <= 0:
        return "unknown"
    dist = abs(price - strike) / strike
    if dist < 0.003:  return "at_money"    # < 0.3% du strike
    if dist < 0.010:  return "near_money"  # 0.3-1%
    return "away"                           # > 1%


# ── Analyse des patterns ──────────────────────────────────────────────────────

def update_patterns() -> dict:
    """
    Analyse tous les trades resolus et calcule le win rate
    pour chaque combinaison de features (pattern).
    Sauvegarde le resultat dans btc_patterns.json.
    """
    results = _load_results()
    if len(results) < MIN_SAMPLES:
        log.info(f"[Patterns] Pas assez de donnees ({len(results)}/{MIN_SAMPLES})")
        return {}

    # Buckets : {pattern_key: {"wins": int, "total": int}}
    buckets: dict = {}

    for r in results:
        ts      = r.get("timestamp", time.time())
        edge    = r.get("edge", 0)
        won     = r.get("won", False)
        side    = r.get("side", "yes")

        hour_b  = _hour_bucket(ts)
        edge_b  = _edge_bucket(edge)

        # Pattern 1 : heure seule
        _add(buckets, f"hour:{hour_b}", won)

        # Pattern 2 : edge seul
        _add(buckets, f"edge:{edge_b}", won)

        # Pattern 3 : cote (YES vs NO)
        _add(buckets, f"side:{side}", won)

        # Pattern 4 : heure + edge
        _add(buckets, f"hour:{hour_b}|edge:{edge_b}", won)

        # Pattern 5 : heure + cote
        _add(buckets, f"hour:{hour_b}|side:{side}", won)

        # Pattern 6 : edge + cote
        _add(buckets, f"edge:{edge_b}|side:{side}", won)

    # Calcule le win rate et le signal pour chaque pattern
    patterns = {}
    for key, data in buckets.items():
        total = data["total"]
        if total < MIN_SAMPLES:
            continue
        wr = data["wins"] / total
        # Signal : deviation par rapport au win rate global
        global_wr = sum(1 for r in results if r.get("won")) / len(results)
        deviation = wr - global_wr

        patterns[key] = {
            "win_rate":   round(wr, 4),
            "total":      total,
            "deviation":  round(deviation, 4),
            # Signal : +1 favorable, -1 defavorable, 0 neutre
            "signal":     1 if deviation > 0.08 else (-1 if deviation < -0.08 else 0),
        }

    _save_patterns(patterns)
    log.info(f"[Patterns] {len(patterns)} patterns calcules sur {len(results)} trades.")
    return patterns


def _add(buckets: dict, key: str, won: bool):
    if key not in buckets:
        buckets[key] = {"wins": 0, "total": 0}
    buckets[key]["total"] += 1
    if won:
        buckets[key]["wins"] += 1


# ── Application des patterns aux decisions ────────────────────────────────────

def get_adjusted_params(
    timestamp: float = None,
    edge: float = 0.05,
    side: str = "yes",
    current_min_edge: float = 0.04,
) -> dict:
    """
    Retourne les parametres ajustes en fonction des patterns detectes.

    Retourne :
    {
        "min_edge_adjusted": float,   # seuil d'edge ajuste
        "confidence_boost":  float,   # bonus/malus de confiance (-2 a +2)
        "should_skip":       bool,    # True si le pattern est tres defavorable
        "reason":            str,     # explication
    }
    """
    patterns = _load_patterns()
    if not patterns:
        return {
            "min_edge_adjusted": current_min_edge,
            "confidence_boost":  0,
            "should_skip":       False,
            "reason":            "Pas encore de patterns (donnees insuffisantes)",
        }

    if timestamp is None:
        timestamp = time.time()

    hour_b = _hour_bucket(timestamp)
    edge_b = _edge_bucket(edge)

    # Collecte les signaux des patterns applicables
    signals = []
    reasons = []

    checks = [
        f"hour:{hour_b}",
        f"edge:{edge_b}",
        f"side:{side}",
        f"hour:{hour_b}|edge:{edge_b}",
        f"hour:{hour_b}|side:{side}",
        f"edge:{edge_b}|side:{side}",
    ]

    for key in checks:
        if key in patterns:
            p = patterns[key]
            if p["total"] >= MIN_SAMPLES:
                signals.append(p["signal"])
                if p["signal"] != 0:
                    reasons.append(
                        f"{key}: WR={p['win_rate']:.1%} "
                        f"({'favorable' if p['signal'] > 0 else 'defavorable'})"
                    )

    if not signals:
        return {
            "min_edge_adjusted": current_min_edge,
            "confidence_boost":  0,
            "should_skip":       False,
            "reason":            "Aucun pattern applicable",
        }

    avg_signal = sum(signals) / len(signals)

    # Ajuste le seuil d'edge selon les patterns
    if avg_signal > 0.5:
        # Conditions favorables -> on peut baisser le seuil un peu
        min_edge_adj = max(current_min_edge * 0.85, 0.025)
        conf_boost   = +1
    elif avg_signal < -0.5:
        # Conditions defavorables -> on exige plus d'edge
        min_edge_adj = current_min_edge * 1.30
        conf_boost   = -2
    else:
        min_edge_adj = current_min_edge
        conf_boost   = 0

    should_skip = avg_signal < -0.8 and len([s for s in signals if s < 0]) >= 3

    return {
        "min_edge_adjusted": round(min_edge_adj, 4),
        "confidence_boost":  conf_boost,
        "should_skip":       should_skip,
        "reason":            " | ".join(reasons) if reasons else "Patterns neutres",
        "avg_signal":        round(avg_signal, 3),
        "patterns_used":     len(signals),
    }


def print_patterns():
    """Affiche les patterns detectes de facon lisible."""
    patterns = _load_patterns()
    results  = _load_results()
    if not patterns:
        print("Aucun pattern detecte (pas assez de donnees).")
        return

    global_wr = sum(1 for r in results if r.get("won")) / len(results) if results else 0
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  PATTERNS DETECTES  (WR global: {global_wr:.1%}  |  {len(results)} trades)")
    print(sep)

    # Tri par deviation absolue (les plus significatifs en premier)
    sorted_p = sorted(patterns.items(), key=lambda x: abs(x[1]["deviation"]), reverse=True)
    for key, p in sorted_p[:20]:
        icon = "✓" if p["signal"] > 0 else ("✗" if p["signal"] < 0 else "~")
        print(f"  {icon}  {key:<40s}  WR={p['win_rate']:.1%}  n={p['total']:3d}  dev={p['deviation']:+.1%}")
    print(sep)
