"""
btc_context.py  --  v6 SIMPLE
Logique : si le marche dit YES >= 60c -> achete YES
          si le marche dit NO  >= 60c -> achete NO
          sinon -> aucun trade

Pas de modele, pas de Black-Scholes.
On suit le prix de marche directement.
"""

import time, json, os, logging, statistics
from typing import Optional
import requests

log = logging.getLogger("BTCContext")

VERSION = "v7-fix-2026-07-02"

PRICE_HISTORY_FILE = "btc_price_history.json"
TRADE_RESULTS_FILE = "btc_trade_results.json"

_price_history  = []
_history_loaded = False

THRESHOLD = 60  # cents -- seuil de decision


# ── Persistance prix (pour les stats) ────────────────────────────────────────

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
        cutoff = time.time() - 240 * 60
        _price_history = [(t, p) for t, p in data if t >= cutoff]
    except Exception:
        pass

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
    cutoff = now - 240 * 60
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
    except Exception:
        pass

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
                "yes_trades": 0, "no_trades": 0, "yes_wr": 0.0, "no_wr": 0.0}
    total     = len(history)
    wins      = sum(1 for t in history if t.get("won"))
    total_pnl = sum(t.get("pnl", 0) for t in history)
    yes_t = [t for t in history if "YES" in t.get("verdict", "")]
    no_t  = [t for t in history if "NO"  in t.get("verdict", "")]
    return {
        "total":      total,
        "win_rate":   wins / total,
        "total_pnl":  total_pnl,
        "yes_trades": len(yes_t),
        "no_trades":  len(no_t),
        "yes_wr":     sum(1 for t in yes_t if t.get("won")) / len(yes_t) if yes_t else 0,
        "no_wr":      sum(1 for t in no_t  if t.get("won")) / len(no_t)  if no_t  else 0,
    }


# ── Prix BTC (pour le log) ────────────────────────────────────────────────────

def _fetch_coinbase() -> Optional[float]:
    try:
        r = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=3)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])
    except Exception:
        return None

def _fetch_kraken() -> Optional[float]:
    try:
        r = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=3)
        r.raise_for_status()
        d = r.json()["result"]
        return float(d[list(d.keys())[0]]["c"][0])
    except Exception:
        return None

def get_btc_price() -> Optional[float]:
    _load_price_history()
    prices = [p for p in [_fetch_coinbase(), _fetch_kraken()] if p]
    if not prices:
        return None
    price = sum(prices) / len(prices)
    log.info(f"BTC price ({len(prices)} exchanges): ${price:,.2f}")
    return round(price, 2)

def get_btc_context(target_price: float = 0, minutes: int = 15) -> str:
    price = get_btc_price()
    stats = get_performance_stats()
    # FIX : si Coinbase ET Kraken echouent, price vaut None et
    # f"${price:,.2f}" levait un TypeError qui faisait planter le cycle.
    price_txt = f"${price:,.2f}" if price is not None else "indisponible"
    return (
        f"Prix BTC: {price_txt} | Seuil decision: {THRESHOLD}c\n"
        f"Trades: {stats['total']} | WR: {stats['win_rate']:.1%} | PnL: ${stats['total_pnl']:.2f}\n"
        f"YES: {stats['yes_trades']} trades WR={stats['yes_wr']:.1%} | "
        f"NO: {stats['no_trades']} trades WR={stats['no_wr']:.1%}"
    )


# ── DECISION PRINCIPALE -- LOGIQUE SIMPLE ────────────────────────────────────

def evaluate_btc_trade(
    strike_price: float,
    market_yes_price_cents: int,
    minutes_remaining: float,
    min_edge: float = 0.04,           # garde pour compatibilite mais non utilise
    min_history_points: int = 1,      # reduit a 1 car on n'a plus besoin d'historique
    market_no_price_cents: int = None,
) -> dict:
    """
    Logique v6 ULTRA-SIMPLE :

    yes_price >= 60c  ->  ACHETER YES
    no_price  >= 60c  ->  ACHETER NO
    sinon             ->  AUCUN TRADE

    On suit le prix de marche directement, sans modele mathematique.
    Le marche Kalshi encode deja l'opinion collective des traders.
    Si 60% des traders pensent que le BTC va monter -> on achete YES.
    Si 60% pensent qu'il va baisser             -> on achete NO.
    """
    price = get_btc_price()
    if price:
        _record_price(price)
    # FIX : si les deux APIs de prix echouent, price est None et les
    # f-strings "${price:,.0f}" plus bas levaient un TypeError -> crash du
    # cycle complet. On formate le prix une seule fois, de facon sure.
    price_txt = f"${price:,.0f}" if price is not None else "N/A"

    yes_cents = int(market_yes_price_cents)
    no_cents  = int(market_no_price_cents) if market_no_price_cents is not None \
                else (100 - yes_cents)

    # Garde-fou : yes + no doit valoir ~100c. Un gros ecart signale des
    # donnees de carnet incoherentes -> on ne trade pas dessus.
    if abs((yes_cents + no_cents) - 100) > 15:
        log.warning(f"[BTC] Carnet incoherent: yes={yes_cents}c + no={no_cents}c "
                    f"!= 100c -- AUCUN TRADE ce cycle.")
        yes_cents = no_cents = 50  # force le verdict AUCUN TRADE plus bas

    log.info(
        f"[BTC Simple] yes={yes_cents}c no={no_cents}c "
        f"strike=${strike_price:,.2f} t={minutes_remaining:.1f}min | "
        f"seuil={THRESHOLD}c dans les deux sens"
    )

    # ── Decision ─────────────────────────────────────────────────────────────
    if yes_cents >= THRESHOLD:
        verdict    = "ACHETER YES"
        prob_r     = yes_cents / 100.0
        prob_m     = yes_cents / 100.0
        edge       = 0.0   # on suit le marche, pas d'edge calcule
        raison     = (f"YES a {yes_cents}c >= seuil {THRESHOLD}c "
                      f"-> marche faveur UP | BTC={price_txt} strike=${strike_price:,.0f}")

    elif no_cents >= THRESHOLD:
        verdict    = "ACHETER NO"
        prob_r     = no_cents / 100.0
        prob_m     = no_cents / 100.0
        edge       = 0.0
        raison     = (f"NO a {no_cents}c >= seuil {THRESHOLD}c "
                      f"-> marche faveur DOWN | BTC={price_txt} strike=${strike_price:,.0f}")

    else:
        verdict    = "AUCUN TRADE"
        prob_r     = yes_cents / 100.0
        prob_m     = yes_cents / 100.0
        edge       = 0.0
        raison     = (f"Ni YES ({yes_cents}c) ni NO ({no_cents}c) n'atteint "
                      f"le seuil de {THRESHOLD}c")

    # Confiance basee sur la distance au seuil
    if verdict != "AUCUN TRADE":
        active_cents = yes_cents if verdict == "ACHETER YES" else no_cents
        distance     = active_cents - THRESHOLD          # ex: 72c -> distance=12
        confiance    = min(10, max(3, 3 + distance // 3)) # 60c->3, 69c->6, 90c->10
    else:
        confiance = 0

    # Taille position selon le prix
    if verdict == "AUCUN TRADE":
        taille = "0%"
    elif (yes_cents if verdict == "ACHETER YES" else no_cents) >= 80:
        taille = "2%"
    elif (yes_cents if verdict == "ACHETER YES" else no_cents) >= 70:
        taille = "1%"
    else:
        taille = "0.5%"

    stats = get_performance_stats()

    return {
        "verdict":           verdict,
        "prob_reelle":       round(prob_r, 4),
        "prob_marche":       round(prob_m, 4),
        "prob_yes_model":    yes_cents / 100.0,
        "prob_no_model":     no_cents  / 100.0,
        "edge":              edge,
        "ev_brute":          0.0,
        "ev_nette":          0.0,
        "confiance":         confiance,
        "risque":            10 - confiance,
        "grade":             "A" if confiance >= 8 else "B" if confiance >= 5 else "C",
        "raison_principale": raison,
        "risque_principal":  "Le marche peut se retourner rapidement sur BTC 15min.",
        "risque_exogene":    "News ou liquidation cascade peuvent invalider le signal.",
        "taille_position":   taille,
        "yes_cents":         yes_cents,
        "no_cents":          no_cents,
        "btc_price":         price or 0,
        "perf_total":        stats["total"],
        "perf_wr":           stats["win_rate"],
        "perf_pnl":          stats["total_pnl"],
    }
