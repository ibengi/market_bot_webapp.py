"""
btc_context.py  --  v4  (Learning Edition)
Integre :
- Persistance historique prix (v3)
- Auto-calibration drift_weight (v3)
- Detection de patterns (pattern_engine)
- Modele ML progressif (ml_model)
- Resolution automatique des trades (trade_resolver)
"""

import time, math, json, os, logging, statistics
from typing import Optional
import requests

log = logging.getLogger("BTCContext")

PRICE_HISTORY_FILE  = "btc_price_history.json"
TRADE_RESULTS_FILE  = "btc_trade_results.json"
MAX_HISTORY_MINUTES = 240
CALIBRATION_WINDOW  = 50

_price_history  = []
_history_loaded = False

# ── Imports optionnels des modules ML ────────────────────────────────────────
try:
    from pattern_engine import get_adjusted_params, update_patterns
    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False
    def get_adjusted_params(**kw):
        return {"min_edge_adjusted": kw.get("current_min_edge", 0.04),
                "confidence_boost": 0, "should_skip": False, "reason": "pattern_engine absent"}
    def update_patterns():
        return {}

try:
    from ml_model import predict_probability, train_model, get_model_status
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    def predict_probability(**kw):
        return {"probability": kw.get("bs_probability", 0.5), "phase": 1,
                "confidence": 0.5, "source": "ml_model absent"}
    def train_model(**kw):
        return {}
    def get_model_status():
        return "ml_model absent"

# ── Persistance historique ────────────────────────────────────────────────────

def _load_price_history():
    global _price_history, _history_loaded
    if _history_loaded:
        return
    _history_loaded = True
    if not os.path.exists(PRICE_HISTORY_FILE):
        return
    try:
        with open(PRICE_HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        cutoff = time.time() - MAX_HISTORY_MINUTES * 60
        _price_history = [(t, p) for t, p in data if t >= cutoff]
        log.info(f"[BTCv4] Historique charge: {len(_price_history)} points")
    except Exception as e:
        log.warning(f"[BTCv4] Erreur chargement historique: {e}")

def _save_price_history():
    try:
        with open(PRICE_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_price_history, f)
    except Exception as e:
        log.warning(f"[BTCv4] Erreur sauvegarde historique: {e}")

def _record_price(price: float):
    _load_price_history()
    now = time.time()
    _price_history.append((now, price))
    cutoff = now - MAX_HISTORY_MINUTES * 60
    while _price_history and _price_history[0][0] < cutoff:
        _price_history.pop(0)
    if len(_price_history) % 5 == 0:
        _save_price_history()

# ── Resultats trades ──────────────────────────────────────────────────────────

def record_trade_result(verdict: str, edge: float, won: bool, pnl: float = 0.0):
    history = _load_trade_results()
    history.append({"timestamp": time.time(), "verdict": verdict,
                     "edge": edge, "won": won, "pnl": pnl})
    if len(history) > 500:
        history = history[-500:]
    try:
        with open(TRADE_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        # Reentraine le modele ML apres chaque nouveau resultat
        if ML_AVAILABLE:
            train_model()
        # Recalcule les patterns tous les 10 nouveaux resultats
        if PATTERNS_AVAILABLE and len(history) % 10 == 0:
            update_patterns()
    except Exception as e:
        log.warning(f"[BTCv4] Erreur sauvegarde trade results: {e}")

def _load_trade_results() -> list:
    if not os.path.exists(TRADE_RESULTS_FILE):
        return []
    try:
        with open(TRADE_RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def get_performance_stats() -> dict:
    history = _load_trade_results()
    if not history:
        return {"total": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "drift_weight": 0.35, "ml_status": get_model_status()}
    total    = len(history)
    wins     = sum(1 for t in history if t.get("won"))
    win_rate = wins / total
    total_pnl = sum(t.get("pnl", 0) for t in history)
    recent   = history[-CALIBRATION_WINDOW:]
    recent_wr = sum(1 for t in recent if t.get("won")) / len(recent) if recent else win_rate
    return {
        "total":            total,
        "win_rate":         win_rate,
        "recent_win_rate":  recent_wr,
        "total_pnl":        total_pnl,
        "drift_weight":     _calibrated_drift_weight(history),
        "ml_status":        get_model_status(),
    }

# ── Auto-calibration drift_weight ────────────────────────────────────────────

def _calibrated_drift_weight(history: list = None) -> float:
    if history is None:
        history = _load_trade_results()
    recent = history[-CALIBRATION_WINDOW:] if len(history) >= 10 else history
    if len(recent) < 10:
        return 0.35
    win_rate = sum(1 for t in recent if t.get("won")) / len(recent)
    if win_rate > 0.60:
        weight = 0.35 + (win_rate - 0.60) * 1.25
    elif win_rate < 0.40:
        weight = 0.35 - (0.40 - win_rate) * 1.25
    else:
        weight = 0.35
    return round(max(0.10, min(0.60, weight)), 3)

# ── Exchanges ─────────────────────────────────────────────────────────────────

def _fetch_coinbase() -> Optional[tuple]:
    try:
        r = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=3)
        r.raise_for_status()
        return float(r.json()["data"]["amount"]), 1.0
    except Exception as e:
        log.debug(f"Coinbase: {e}"); return None

def _fetch_kraken() -> Optional[tuple]:
    try:
        r = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=3)
        r.raise_for_status()
        d = r.json()["result"][list(r.json()["result"].keys())[0]]
        return float(d["c"][0]), float(d["v"][1])
    except Exception as e:
        log.debug(f"Kraken: {e}"); return None

def _fetch_gemini() -> Optional[tuple]:
    try:
        r = requests.get("https://api.gemini.com/v2/ticker/btcusd", timeout=3)
        r.raise_for_status()
        d = r.json()
        return float(d["close"]), float(d.get("volume", {}).get("BTC", 1.0))
    except Exception as e:
        log.debug(f"Gemini: {e}"); return None

def _fetch_bitstamp() -> Optional[tuple]:
    try:
        r = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=3)
        r.raise_for_status()
        d = r.json()
        return float(d["last"]), float(d["volume"])
    except Exception as e:
        log.debug(f"Bitstamp: {e}"); return None

def get_btc_price() -> Optional[float]:
    _load_price_history()
    results = [r for r in [_fetch_coinbase(), _fetch_kraken(),
                            _fetch_gemini(), _fetch_bitstamp()] if r]
    if not results:
        log.warning("Aucun exchange accessible."); return None
    if len(results) == 1:
        return results[0][0]
    prices = [p for p, v in results]
    med    = statistics.median(prices)
    filt   = [(p, v) for p, v in results if abs(p - med) / med < 0.01] or results
    tw     = sum(v for p, v in filt) or 1.0
    wmean  = sum(p * v / tw for p, v in filt)
    log.info(f"BTC price ({len(results)} exchanges): ${wmean:,.2f}")
    return round(wmean, 2)

# ── Volatilite & Momentum ─────────────────────────────────────────────────────

def _recent_volatility_pct(window_minutes: int = 60) -> float:
    if len(_price_history) < 3:
        return 0.50 * math.sqrt(15 / (365 * 24 * 60))
    now    = time.time()
    recent = [(t, p) for t, p in _price_history if t >= now - window_minutes * 60]
    if len(recent) < 3:
        recent = _price_history[-5:]
    returns = [math.log(recent[i][1] / recent[i-1][1])
               for i in range(1, len(recent)) if recent[i-1][1] > 0]
    if len(returns) < 2:
        return 0.50 * math.sqrt(15 / (365 * 24 * 60))
    stdev = statistics.stdev(returns)
    span  = (recent[-1][0] - recent[0][0]) / 60 or 1
    per_obs = span / max(len(recent) - 1, 1)
    return stdev * math.sqrt(15 / per_obs) if per_obs > 0 else stdev

def _recent_momentum(window_minutes: float = 5.0) -> float:
    if len(_price_history) < 3:
        return 0.0
    now    = time.time()
    recent = [(t, p) for t, p in _price_history if t >= now - window_minutes * 60]
    if len(recent) < 3:
        recent = _price_history[-5:]
    if len(recent) < 2:
        return 0.0
    t0, p0 = recent[0]; t1, p1 = recent[-1]
    span = (t1 - t0) / 60
    return math.log(p1 / p0) / span if span > 0 and p0 > 0 else 0.0

# ── Black-Scholes ─────────────────────────────────────────────────────────────

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def estimate_probability(current_price, strike_price, minutes_remaining,
                          volatility_15min=None, drift_weight=None):
    if volatility_15min is None: volatility_15min = _recent_volatility_pct()
    if drift_weight     is None: drift_weight     = _calibrated_drift_weight()
    if current_price <= 0 or strike_price <= 0 or minutes_remaining <= 0:
        return 0.5
    t     = max(minutes_remaining / 15.0, 0.01)
    sigma = max(volatility_15min, 0.0001)
    mu    = _recent_momentum() * drift_weight
    try:
        d2 = (math.log(current_price / strike_price) + mu * minutes_remaining
              - 0.5 * sigma**2 * t) / (sigma * math.sqrt(t))
        return max(0.0, min(1.0, _norm_cdf(d2)))
    except Exception:
        return 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────

def _position_size(edge):
    if edge >= 0.15: return "2%"
    if edge >= 0.10: return "1%"
    return "0.5%"

def _grade(edge):
    if edge > 0.10: return "A"
    if edge > 0.06: return "B"
    if edge > 0.04: return "C"
    return "D"

# ── Interface principale ──────────────────────────────────────────────────────

def get_btc_context(target_price=0, minutes=15):
    price = get_btc_price()
    if price is None: return "Donnees BTC indisponibles."
    _record_price(price)
    vol   = _recent_volatility_pct()
    prob  = estimate_probability(price, target_price or price, minutes, vol)
    stats = get_performance_stats()
    return (
        f"Prix BTC: ${price:,.2f} | Strike: ${target_price:,.2f} | Vol: {vol:.2%}\n"
        f"P(above): {prob:.1%} | Historique: {len(_price_history)} pts\n"
        f"Trades: {stats['total']} | WR: {stats['win_rate']:.1%} | "
        f"PnL: ${stats['total_pnl']:.2f}\n"
        f"Modele: {stats['ml_status']}"
    )


def evaluate_btc_trade(
    strike_price: float,
    market_yes_price_cents: int,
    minutes_remaining: float,
    min_edge: float = 0.04,
    min_history_points: int = 3,
    market_no_price_cents: int = None,
) -> dict:
    """
    Decision de trade BTC 15min -- v4 Learning Edition.

    Nouveautes :
    - Probabilite ajustee par le modele ML (si assez de donnees)
    - Parametres ajustes par les patterns detectes
    - Raison detaillee incluant phase ML et patterns
    """
    price = get_btc_price()
    if price is None:
        return {"verdict": "AUCUN TRADE",
                "raison_principale": "Prix BTC indisponible.", "confiance": 0, "edge": 0.0}

    _record_price(price)

    if len(_price_history) < min_history_points:
        return {"verdict": "AUCUN TRADE",
                "raison_principale": f"Historique insuffisant ({len(_price_history)}/{min_history_points}).",
                "confiance": 0, "edge": 0.0,
                "prob_reelle": 0.5, "prob_marche": market_yes_price_cents / 100.0}

    vol          = _recent_volatility_pct()
    momentum     = _recent_momentum()
    drift_weight = _calibrated_drift_weight()

    # ── Probabilite Black-Scholes (base) ──────────────────────────────────────
    bs_prob_yes  = estimate_probability(price, strike_price, minutes_remaining, vol, drift_weight)
    bs_prob_no   = 1.0 - bs_prob_yes

    # ── Amelioration ML (si disponible) ──────────────────────────────────────
    from datetime import datetime, timezone
    hour_utc = datetime.now(tz=timezone.utc).hour

    ml_result_yes = predict_probability(
        price=price, strike=strike_price,
        minutes_remaining=minutes_remaining,
        volatility=vol, momentum_per_min=momentum,
        side="yes", bs_probability=bs_prob_yes,
    )
    ml_result_no = predict_probability(
        price=price, strike=strike_price,
        minutes_remaining=minutes_remaining,
        volatility=vol, momentum_per_min=momentum,
        side="no", bs_probability=bs_prob_no,
    )

    prob_yes_final = ml_result_yes["probability"]
    prob_no_final  = ml_result_no["probability"]
    ml_phase       = ml_result_yes["phase"]
    ml_source      = ml_result_yes["source"]

    # ── Prix marche ───────────────────────────────────────────────────────────
    market_yes = market_yes_price_cents / 100.0
    market_no  = (market_no_price_cents / 100.0) if market_no_price_cents is not None \
                 else (1.0 - market_yes)

    edge_yes = prob_yes_final - market_yes
    edge_no  = prob_no_final  - market_no

    log.debug(f"[BTCv4] P(yes)={prob_yes_final:.1%} P(no)={prob_no_final:.1%} "
              f"edge_yes={edge_yes:+.1%} edge_no={edge_no:+.1%} phase={ml_phase}")

    # ── Ajustement patterns ───────────────────────────────────────────────────
    # On evalue d'abord le cote le plus prometteur
    candidate_side  = "yes" if edge_yes >= edge_no else "no"
    candidate_edge  = max(edge_yes, edge_no)

    pattern_params = get_adjusted_params(
        timestamp=time.time(),
        edge=candidate_edge,
        side=candidate_side,
        current_min_edge=min_edge,
    )

    if pattern_params.get("should_skip"):
        return {
            "verdict":           "AUCUN TRADE",
            "raison_principale": f"Pattern defavorable: {pattern_params['reason']}",
            "confiance": 0, "edge": candidate_edge,
            "prob_reelle": prob_yes_final, "prob_marche": market_yes,
            "ev_brute": 0.0, "ev_nette": 0.0,
            "taille_position": "0%", "grade": "D", "risque": 10,
        }

    effective_min_edge = pattern_params["min_edge_adjusted"]
    conf_boost         = pattern_params.get("confidence_boost", 0)

    # ── Selection du cote ────────────────────────────────────────────────────
    trade_yes = edge_yes >= effective_min_edge
    trade_no  = edge_no  >= effective_min_edge

    if trade_yes and trade_no:
        if edge_yes >= edge_no:
            verdict, edge_taken = "ACHETER YES", edge_yes
            prob_r, prob_m = prob_yes_final, market_yes
        else:
            verdict, edge_taken = "ACHETER NO", edge_no
            prob_r, prob_m = prob_no_final, market_no
    elif trade_yes:
        verdict, edge_taken = "ACHETER YES", edge_yes
        prob_r, prob_m = prob_yes_final, market_yes
    elif trade_no:
        verdict, edge_taken = "ACHETER NO", edge_no
        prob_r, prob_m = prob_no_final, market_no
    else:
        verdict, edge_taken = "AUCUN TRADE", max(edge_yes, edge_no)
        prob_r, prob_m = prob_yes_final, market_yes

    # ── Confiance ────────────────────────────────────────────────────────────
    t_scaled     = max(minutes_remaining / 15.0, 0.01)
    sigma_scaled = max(vol, 0.0001) * math.sqrt(t_scaled)
    try:
        d2 = abs(math.log(price / strike_price) - 0.5 * vol**2 * t_scaled) / sigma_scaled
    except Exception:
        d2 = 0.0
    confiance = min(10, max(1, int(d2 * 3) + conf_boost))

    # ── EV ────────────────────────────────────────────────────────────────────
    ev_brute = prob_r * (1.0 - prob_m) - (1.0 - prob_r) * prob_m
    ev_nette = prob_r * (1.0 - prob_m) * 0.9755 - (1.0 - prob_r) * prob_m

    stats     = get_performance_stats()
    direction = "au-dessus" if verdict == "ACHETER YES" else "en-dessous"
    raison = (
        f"[{ml_source}] prix=${price:,.0f} strike=${strike_price:,.0f} "
        f"vol={vol:.2%} t={minutes_remaining:.1f}min | "
        f"P({direction})={prob_r:.1%} vs mkt={prob_m:.1%} edge={edge_taken:+.1%} | "
        f"WR={stats['win_rate']:.1%}({stats['total']}T) | "
        f"Pattern: {pattern_params.get('reason', 'N/A')[:40]}"
    )

    return {
        "verdict":           verdict,
        "prob_reelle":       round(prob_r, 4),
        "prob_marche":       round(prob_m, 4),
        "edge":              round(edge_taken, 4),
        "ev_brute":          round(ev_brute, 4),
        "ev_nette":          round(ev_nette, 4),
        "confiance":         confiance,
        "risque":            10 - confiance,
        "grade":             _grade(edge_taken),
        "raison_principale": raison,
        "risque_principal":  "Volatilite BTC peut devier brutalement.",
        "risque_exogene":    "Slippage si carnet d'ordres fin.",
        "taille_position":   _position_size(edge_taken) if verdict != "AUCUN TRADE" else "0%",
        "ml_phase":          ml_phase,
        "drift_weight_used": drift_weight,
        "history_points":    len(_price_history),
        "pattern_signal":    pattern_params.get("avg_signal", 0),
    }
