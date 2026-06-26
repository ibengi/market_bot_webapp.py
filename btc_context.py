"""
btc_context.py  --  v5  (Simple Probability Mode)

LOGIQUE SIMPLE ET CLAIRE :
- Calcule la probabilite que BTC finisse AU-DESSUS du strike (prob_yes)
- prob_no = 1 - prob_yes
- Si prob_yes >= 60% → ACHETER YES
- Si prob_no  >= 60% → ACHETER NO (i.e. prob_yes <= 40%)
- Sinon → AUCUN TRADE

Plus de Black-Scholes complexe -- juste la distance au strike,
le temps restant, et le momentum recent.
"""

import time, math, json, os, logging, statistics
from typing import Optional
import requests

log = logging.getLogger("BTCContext")

PRICE_HISTORY_FILE  = "btc_price_history.json"
TRADE_RESULTS_FILE  = "btc_trade_results.json"
MAX_HISTORY_MINUTES = 240

_price_history  = []
_history_loaded = False

# ── Seuil de decision ────────────────────────────────────────────────────────
THRESHOLD_BUY = 0.60   # achete YES si prob_yes >= 60%
                        # achete NO  si prob_yes <= 40% (prob_no >= 60%)

# ── Persistance historique prix ───────────────────────────────────────────────

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
        log.info(f"[BTC] Historique charge: {len(_price_history)} points")
    except Exception as e:
        log.warning(f"[BTC] Erreur chargement historique: {e}")

def _save_price_history():
    try:
        with open(PRICE_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_price_history, f)
    except Exception:
        pass

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
    except Exception as e:
        log.warning(f"[BTC] Erreur sauvegarde: {e}")

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
        return {"total": 0, "win_rate": 0.0, "total_pnl": 0.0}
    total    = len(history)
    wins     = sum(1 for t in history if t.get("won"))
    total_pnl = sum(t.get("pnl", 0) for t in history)
    yes_trades = [t for t in history if "YES" in t.get("verdict", "")]
    no_trades  = [t for t in history if "NO"  in t.get("verdict", "")]
    yes_wr = sum(1 for t in yes_trades if t.get("won")) / len(yes_trades) if yes_trades else 0
    no_wr  = sum(1 for t in no_trades  if t.get("won")) / len(no_trades)  if no_trades  else 0
    return {
        "total":      total,
        "win_rate":   wins / total,
        "total_pnl":  total_pnl,
        "yes_trades": len(yes_trades),
        "no_trades":  len(no_trades),
        "yes_wr":     yes_wr,
        "no_wr":      no_wr,
    }


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
        d = r.json()["result"]
        k = list(d.keys())[0]
        return float(d[k]["c"][0]), float(d[k]["v"][1])
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


# ── Calcul de probabilite ─────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _recent_volatility() -> float:
    """Volatilite recente ramenee a 15min."""
    if len(_price_history) < 3:
        # Fallback : ~50% annualisee ramenee a 15min
        return 0.50 * math.sqrt(15 / (365 * 24 * 60))
    now    = time.time()
    recent = [(t, p) for t, p in _price_history if t >= now - 3600]
    if len(recent) < 3:
        recent = _price_history[-10:]
    returns = [math.log(recent[i][1] / recent[i-1][1])
               for i in range(1, len(recent)) if recent[i-1][1] > 0]
    if len(returns) < 2:
        return 0.50 * math.sqrt(15 / (365 * 24 * 60))
    stdev = statistics.stdev(returns)
    span  = (recent[-1][0] - recent[0][0]) / 60 or 1
    per_obs = span / max(len(recent) - 1, 1)
    return stdev * math.sqrt(15 / per_obs) if per_obs > 0 else stdev

def _recent_momentum(window_minutes: float = 5.0) -> float:
    """Drift recent en log-return/minute."""
    if len(_price_history) < 2:
        return 0.0
    now    = time.time()
    recent = [(t, p) for t, p in _price_history if t >= now - window_minutes * 60]
    if len(recent) < 2:
        recent = _price_history[-5:]
    if len(recent) < 2:
        return 0.0
    t0, p0 = recent[0]; t1, p1 = recent[-1]
    span = (t1 - t0) / 60
    return math.log(p1 / p0) / span if span > 0 and p0 > 0 else 0.0

def compute_prob_yes(price: float, strike: float, minutes: float) -> float:
    """
    Calcule la probabilite que BTC finisse AU-DESSUS du strike.

    Utilise Black-Scholes binaire avec momentum recent comme derive.
    prob_no = 1 - prob_yes (toujours coherent, somme = 100%).
    """
    if price <= 0 or strike <= 0 or minutes <= 0:
        return 0.5

    vol      = _recent_volatility()
    momentum = _recent_momentum()
    sigma    = max(vol, 0.0001)
    t        = max(minutes / 15.0, 0.01)

    # Derive : momentum recente ponderee a 35%
    mu = momentum * 0.35

    try:
        d2 = (math.log(price / strike) + mu * minutes - 0.5 * sigma**2 * t) \
             / (sigma * math.sqrt(t))
        prob = _norm_cdf(d2)
    except Exception:
        prob = 0.5

    return round(max(0.01, min(0.99, prob)), 4)


# ── Decision principale ───────────────────────────────────────────────────────

def evaluate_btc_trade(
    strike_price: float,
    market_yes_price_cents: int,
    minutes_remaining: float,
    min_edge: float = 0.04,
    min_history_points: int = 3,
    market_no_price_cents: int = None,
) -> dict:
    """
    Logique de decision v5 -- simple et bidirectionnelle :

    1. Calcule prob_yes = P(BTC > strike a expiration)
    2. prob_no = 1 - prob_yes
    3. Si prob_yes >= 60% ET edge_yes >= min_edge  → ACHETER YES
    4. Si prob_no  >= 60% ET edge_no  >= min_edge  → ACHETER NO
    5. Sinon → AUCUN TRADE

    Les deux directions sont evaluees exactement de la meme facon.
    """
    price = get_btc_price()
    if price is None:
        return {
            "verdict": "AUCUN TRADE",
            "raison_principale": "Prix BTC indisponible.",
            "confiance": 0, "edge": 0.0,
            "ev_brute": 0.0, "ev_nette": 0.0,
        }

    _record_price(price)

    if len(_price_history) < min_history_points:
        return {
            "verdict": "AUCUN TRADE",
            "raison_principale": (
                f"Historique insuffisant ({len(_price_history)}/{min_history_points} points)."
            ),
            "confiance": 0, "edge": 0.0,
            "ev_brute": 0.0, "ev_nette": 0.0,
            "prob_reelle": 0.5,
            "prob_marche": market_yes_price_cents / 100.0,
        }

    # ── Probabilites modele ───────────────────────────────────────────────────
    prob_yes = compute_prob_yes(price, strike_price, minutes_remaining)
    prob_no  = round(1.0 - prob_yes, 4)

    # ── Prix marche ───────────────────────────────────────────────────────────
    mkt_yes = market_yes_price_cents / 100.0
    mkt_no  = (market_no_price_cents / 100.0) if market_no_price_cents is not None \
              else round(1.0 - mkt_yes, 4)

    # ── Edges ────────────────────────────────────────────────────────────────
    edge_yes = round(prob_yes - mkt_yes, 4)
    edge_no  = round(prob_no  - mkt_no,  4)

    log.info(
        f"[BTC] price=${price:,.0f} strike=${strike_price:,.0f} "
        f"t={minutes_remaining:.1f}min | "
        f"P(yes)={prob_yes:.1%} P(no)={prob_no:.1%} | "
        f"mkt_yes={mkt_yes:.1%} mkt_no={mkt_no:.1%} | "
        f"edge_yes={edge_yes:+.1%} edge_no={edge_no:+.1%}"
    )

    # ── Decision : seuil 60% ─────────────────────────────────────────────────
    can_buy_yes = (prob_yes >= THRESHOLD_BUY) and (edge_yes >= min_edge)
    can_buy_no  = (prob_no  >= THRESHOLD_BUY) and (edge_no  >= min_edge)

    if can_buy_yes and can_buy_no:
        # Les deux depassent 60% -- impossible en theorie (somme=100%)
        # mais si ca arrive on prend le meilleur edge
        if edge_yes >= edge_no:
            can_buy_no = False
        else:
            can_buy_yes = False

    if can_buy_yes:
        verdict      = "ACHETER YES"
        edge_taken   = edge_yes
        prob_reelle  = prob_yes
        prob_marche  = mkt_yes
    elif can_buy_no:
        verdict      = "ACHETER NO"
        edge_taken   = edge_no
        prob_reelle  = prob_no
        prob_marche  = mkt_no
    else:
        verdict      = "AUCUN TRADE"
        edge_taken   = max(edge_yes, edge_no)
        prob_reelle  = prob_yes
        prob_marche  = mkt_yes

    # ── Raison claire ────────────────────────────────────────────────────────
    if verdict == "AUCUN TRADE":
        if prob_yes >= THRESHOLD_BUY and edge_yes < min_edge:
            raison = (f"YES probable ({prob_yes:.1%}) mais edge insuffisant "
                      f"({edge_yes:+.1%} < {min_edge:.1%})")
        elif prob_no >= THRESHOLD_BUY and edge_no < min_edge:
            raison = (f"NO probable ({prob_no:.1%}) mais edge insuffisant "
                      f"({edge_no:+.1%} < {min_edge:.1%})")
        else:
            raison = (f"Probabilites insuffisantes -- P(yes)={prob_yes:.1%} "
                      f"P(no)={prob_no:.1%} (seuil: {THRESHOLD_BUY:.0%})")
    else:
        direction = "AU-DESSUS" if verdict == "ACHETER YES" else "EN-DESSOUS"
        raison = (
            f"P({direction} strike)={prob_reelle:.1%} >= {THRESHOLD_BUY:.0%} | "
            f"prix=${price:,.0f} strike=${strike_price:,.0f} | "
            f"edge={edge_taken:+.1%} mkt={prob_marche:.1%}"
        )

    # ── EV ───────────────────────────────────────────────────────────────────
    ev_brute = prob_reelle * (1.0 - prob_marche) - (1.0 - prob_reelle) * prob_marche
    ev_nette = prob_reelle * (1.0 - prob_marche) * 0.9755 \
               - (1.0 - prob_reelle) * prob_marche

    # ── Confiance basee sur distance a 50% ───────────────────────────────────
    distance_from_50 = abs(prob_reelle - 0.5)
    confiance = min(10, max(1, int(distance_from_50 * 40)))  # 60%->4, 70%->8, 75%->10

    # ── Taille position ───────────────────────────────────────────────────────
    if verdict == "AUCUN TRADE":
        taille = "0%"
    elif prob_reelle >= 0.75:
        taille = "2%"
    elif prob_reelle >= 0.65:
        taille = "1%"
    else:
        taille = "0.5%"

    # ── Grade ─────────────────────────────────────────────────────────────────
    if edge_taken > 0.15:   grade = "A"
    elif edge_taken > 0.08: grade = "B"
    elif edge_taken > 0.04: grade = "C"
    else:                   grade = "D"

    stats = get_performance_stats()

    return {
        "verdict":           verdict,
        "prob_reelle":       prob_reelle,
        "prob_marche":       prob_marche,
        "edge":              edge_taken,
        "ev_brute":          round(ev_brute, 4),
        "ev_nette":          round(ev_nette, 4),
        "confiance":         confiance,
        "risque":            10 - confiance,
        "grade":             grade,
        "raison_principale": raison,
        "risque_principal":  "Volatilite BTC peut devier brutalement (news, cascade).",
        "risque_exogene":    "Slippage si carnet d'ordres fin.",
        "taille_position":   taille,
        # Infos supplementaires pour le log
        "prob_yes_model":    prob_yes,
        "prob_no_model":     prob_no,
        "edge_yes":          edge_yes,
        "edge_no":           edge_no,
        "mkt_yes":           mkt_yes,
        "mkt_no":            mkt_no,
        "history_points":    len(_price_history),
        "perf_total":        stats["total"],
        "perf_wr":           stats["win_rate"],
        "perf_pnl":          stats["total_pnl"],
    }


def get_btc_context(target_price: float = 0, minutes: int = 15) -> str:
    price = get_btc_price()
    if price is None:
        return "Donnees BTC indisponibles."
    _record_price(price)
    prob_yes = compute_prob_yes(price, target_price or price, minutes)
    prob_no  = 1.0 - prob_yes
    stats    = get_performance_stats()
    return (
        f"Prix BTC: ${price:,.2f} | Strike: ${target_price:,.2f}\n"
        f"P(YES/above): {prob_yes:.1%}  P(NO/below): {prob_no:.1%}\n"
        f"Seuil decision: {THRESHOLD_BUY:.0%} dans les deux sens\n"
        f"Trades: {stats['total']} | WR: {stats['win_rate']:.1%} | "
        f"PnL: ${stats['total_pnl']:.2f}\n"
        f"YES: {stats['yes_trades']} trades WR={stats['yes_wr']:.1%} | "
        f"NO: {stats['no_trades']} trades WR={stats['no_wr']:.1%}"
    )
