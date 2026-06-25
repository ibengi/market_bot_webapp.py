"""
btc_context.py
Module de contexte BTC pour le mode --btc de kalshi_alpha_bot.py.

OBJECTIF :
Fournir un prix BTC fiable et une estimation de probabilite "above strike"
SANS appeler Claude -- calcul mathematique instantane, zero cout API,
zero latence, adapte a l'echelle de temps des marches Kalshi 15min.

PRINCIPE :
Kalshi regle ses marches BTC 15min contre le CME CF Bitcoin Real-Time Index
(BRTI), qui est une mediane ponderee par volume des prix sur plusieurs
exchanges regules (Coinbase, Kraken, Gemini, Bitstamp, itBit/Paxos).
L'acces direct a l'API BRTI necessite une licence payante CF Benchmarks.
Ce module replique l'approche en agregeant les memes exchanges sources
publiquement accessibles, ce qui donne une approximation tres proche du
BRTI reel sans cout de licence.

La probabilite "above strike" a l'expiration est estimee via le modele
de Black-Scholes pour option binaire (cash-or-nothing) : la probabilite
qu'un mouvement brownien geometrique termine au-dessus d'un seuil donne,
fonction de la distance au strike, du temps restant et de la volatilite.

USAGE (depuis kalshi_alpha_bot.py) :
    from btc_context import get_btc_price, get_btc_context, estimate_probability

    price = get_btc_price()
    ctx   = get_btc_context(target_price=65000, minutes=15)

CORRECTIONS v2 :
- evaluate_btc_trade accepte maintenant market_no_price_cents en parametre
  pour calculer l'edge NO avec le vrai prix du marche (et non 1 - yes_price)
- confiance calculee correctement pour les deux cotes
- taille_position adaptee a l'edge reel
"""

import time
import math
import logging
import statistics
from typing import Optional

import requests

log = logging.getLogger("BTCContext")

# ── Exchanges sources (memes constituants que BRTI) ──────────────────────────
# Chaque fonction retourne (prix, volume_24h_usd) ou None si echec.

def _fetch_coinbase() -> Optional[tuple]:
    try:
        r = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=3)
        r.raise_for_status()
        price = float(r.json()["data"]["amount"])
        return price, 1.0  # Coinbase spot endpoint ne donne pas le volume ici
    except Exception as e:
        log.debug(f"Coinbase fetch echoue: {e}")
        return None

def _fetch_kraken() -> Optional[tuple]:
    try:
        r = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=3)
        r.raise_for_status()
        data = r.json()["result"]
        pair_key = list(data.keys())[0]
        ticker = data[pair_key]
        price  = float(ticker["c"][0])   # dernier prix trade
        volume = float(ticker["v"][1])   # volume 24h
        return price, volume
    except Exception as e:
        log.debug(f"Kraken fetch echoue: {e}")
        return None

def _fetch_gemini() -> Optional[tuple]:
    try:
        r = requests.get("https://api.gemini.com/v2/ticker/btcusd", timeout=3)
        r.raise_for_status()
        data = r.json()
        price  = float(data["close"])
        volume = float(data.get("volume", {}).get("BTC", 1.0))
        return price, volume
    except Exception as e:
        log.debug(f"Gemini fetch echoue: {e}")
        return None

def _fetch_bitstamp() -> Optional[tuple]:
    try:
        r = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=3)
        r.raise_for_status()
        data = r.json()
        price  = float(data["last"])
        volume = float(data["volume"])
        return price, volume
    except Exception as e:
        log.debug(f"Bitstamp fetch echoue: {e}")
        return None


def get_btc_price() -> Optional[float]:
    """
    Recupere le prix BTC actuel en agregeant plusieurs exchanges via une
    mediane ponderee par volume -- approximation du CME CF BRTI utilise
    par Kalshi pour le reglement des marches BTC 15min.

    Retourne None si aucun exchange n'a pu etre interroge (panne reseau totale).
    """
    fetchers = [_fetch_coinbase, _fetch_kraken, _fetch_gemini, _fetch_bitstamp]
    results = []
    for fetch in fetchers:
        r = fetch()
        if r is not None:
            results.append(r)

    if not results:
        log.warning("Aucun exchange BTC accessible -- prix indisponible.")
        return None

    if len(results) == 1:
        return results[0][0]

    # Mediane ponderee par volume (meme principe que BRTI)
    prices  = [p for p, v in results]
    weights = [v for p, v in results]
    total_w = sum(weights) or 1.0
    weighted_median = sum(p * (w / total_w) for p, w in results)

    # Filtre les outliers : si un exchange devie de >1% de la mediane simple,
    # on l'exclut (protection contre une donnee corrompue d'un seul exchange)
    simple_median = statistics.median(prices)
    filtered = [(p, v) for p, v in results if abs(p - simple_median) / simple_median < 0.01]
    if filtered and len(filtered) < len(results):
        prices  = [p for p, v in filtered]
        weights = [v for p, v in filtered]
        total_w = sum(weights) or 1.0
        weighted_median = sum(p * (w / total_w) for p, w in filtered)

    log.info(f"BTC price ({len(results)} exchanges): ${weighted_median:,.2f}")
    return round(weighted_median, 2)


# ── Volatilite recente (pour le modele de probabilite) ──────────────────────

_price_history = []  # [(timestamp, price), ...] -- historique en memoire du process

def _record_price(price: float):
    now = time.time()
    _price_history.append((now, price))
    # Garde seulement les 60 dernieres minutes d'historique
    cutoff = now - 3600
    while _price_history and _price_history[0][0] < cutoff:
        _price_history.pop(0)

def _recent_volatility_pct(window_minutes: int = 60) -> float:
    """
    Estime la volatilite recente du BTC (ecart-type des rendements log)
    sur la fenetre donnee, annualisee puis ramenee a une base horaire.
    Si l'historique est insuffisant, retourne une volatilite par defaut
    raisonnable basee sur la volatilite typique du BTC (~40-60% annualisee).
    """
    if len(_price_history) < 3:
        # Pas assez de donnees -- fallback sur une volatilite typique BTC
        # ~50% annualisee -> ramenee a une fenetre de 15 min
        annual_vol = 0.50
        minutes_per_year = 365 * 24 * 60
        return annual_vol * math.sqrt(15 / minutes_per_year)

    now = time.time()
    cutoff = now - (window_minutes * 60)
    recent = [(t, p) for t, p in _price_history if t >= cutoff]
    if len(recent) < 3:
        recent = _price_history[-5:]

    returns = []
    for i in range(1, len(recent)):
        p0 = recent[i - 1][1]
        p1 = recent[i][1]
        if p0 > 0:
            returns.append(math.log(p1 / p0))

    if len(returns) < 2:
        annual_vol = 0.50
        minutes_per_year = 365 * 24 * 60
        return annual_vol * math.sqrt(15 / minutes_per_year)

    stdev = statistics.stdev(returns)
    # Ramene a l'echelle de 15 minutes
    n_obs = len(recent)
    span_minutes = (recent[-1][0] - recent[0][0]) / 60 or 1
    per_obs_minutes = span_minutes / max(n_obs - 1, 1)
    scale_to_15min = math.sqrt(15 / per_obs_minutes) if per_obs_minutes > 0 else 1
    return stdev * scale_to_15min


def _recent_momentum(window_minutes: float = 5.0) -> float:
    """
    Calcule la derive (drift) recente du prix BTC, en log-return par minute,
    sur une fenetre courte (5min par defaut). Une valeur negative signifie
    une tendance baissiere recente, positive une tendance haussiere.
    """
    if len(_price_history) < 3:
        return 0.0

    now = time.time()
    cutoff = now - (window_minutes * 60)
    recent = [(t, p) for t, p in _price_history if t >= cutoff]
    if len(recent) < 3:
        recent = _price_history[-5:]
    if len(recent) < 2:
        return 0.0

    t0, p0 = recent[0]
    t1, p1 = recent[-1]
    span_minutes = (t1 - t0) / 60
    if span_minutes <= 0 or p0 <= 0:
        return 0.0

    log_return = math.log(p1 / p0)
    return log_return / span_minutes  # drift par minute


# ── Modele de probabilite (Black-Scholes binaire / cash-or-nothing) ─────────

def _norm_cdf(x: float) -> float:
    """Fonction de repartition de la loi normale standard (approximation erf)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def estimate_probability(current_price: float, strike_price: float,
                          minutes_remaining: float, volatility_15min: float = None,
                          drift_weight: float = 0.35) -> float:
    """
    Estime la probabilite que le prix BTC cloture AU-DESSUS du strike
    a l'expiration, via le modele cash-or-nothing binaire AVEC un terme
    de derive (drift) base sur le momentum recent.

    Retourne une probabilite entre 0.0 et 1.0.
    """
    if volatility_15min is None:
        volatility_15min = _recent_volatility_pct()

    if current_price <= 0 or strike_price <= 0 or minutes_remaining <= 0:
        return 0.5

    t = max(minutes_remaining / 15.0, 0.01)
    sigma = max(volatility_15min, 0.0001)

    momentum_per_min = _recent_momentum()
    mu = momentum_per_min * drift_weight

    try:
        d2 = (math.log(current_price / strike_price) + mu * minutes_remaining
              - 0.5 * sigma**2 * t) / (sigma * math.sqrt(t))
        prob_above = _norm_cdf(d2)
    except (ValueError, ZeroDivisionError):
        prob_above = 0.5

    return max(0.0, min(1.0, prob_above))


# ── Interface principale utilisee par kalshi_alpha_bot.py ───────────────────

def get_btc_context(target_price: float = 0, minutes: int = 15) -> str:
    """
    Construit un resume textuel du contexte BTC pour supervision.
    """
    price = get_btc_price()
    if price is None:
        return "Donnees BTC indisponibles (tous les exchanges sources ont echoue)."

    _record_price(price)
    vol = _recent_volatility_pct()
    prob = estimate_probability(price, target_price or price, minutes, vol)

    return (
        f"Prix BTC agrege (multi-exchanges): ${price:,.2f}\n"
        f"Strike cible: ${target_price:,.2f}\n"
        f"Volatilite recente (echelle {minutes}min): {vol:.2%}\n"
        f"Probabilite modele (above strike): {prob:.1%}\n"
        f"Note: estimation mathematique (Black-Scholes binaire), pas une "
        f"prediction qualitative -- a utiliser comme aide a la supervision."
    )


def _position_size(edge: float) -> str:
    """Taille de position proportionnelle a l'edge detecte."""
    if edge >= 0.15:
        return "2%"
    if edge >= 0.10:
        return "1%"
    return "0.5%"


def _grade(edge: float) -> str:
    if edge > 0.10:
        return "A"
    if edge > 0.06:
        return "B"
    if edge > 0.04:
        return "C"
    return "D"


def evaluate_btc_trade(
    strike_price: float,
    market_yes_price_cents: int,
    minutes_remaining: float,
    min_edge: float = 0.04,
    min_history_points: int = 5,
    market_no_price_cents: int = None,   # ← NOUVEAU : prix NO reel du marche
) -> dict:
    """
    Decision de trade BTC 15min SANS appel Claude -- calcul instantane.

    CORRECTION v2 :
    - Accepte market_no_price_cents pour calculer l'edge NO avec le vrai
      prix Kalshi (et non 1 - yes_price, qui ignore le spread bid/ask).
    - Si market_no_price_cents est absent (backward compat), fallback sur
      100 - market_yes_price_cents (comportement precedent).
    - prob_reelle / prob_marche toujours coherents avec le cote trade.
    - confiance calculee sur d2 mis a l'echelle du temps restant (fix existant
      conserve).
    """
    price = get_btc_price()
    if price is None:
        return {
            "verdict": "AUCUN TRADE",
            "raison_principale": "Prix BTC indisponible (echec reseau exchanges).",
            "confiance": 0, "edge": 0.0,
        }

    _record_price(price)

    if len(_price_history) < min_history_points:
        return {
            "verdict": "AUCUN TRADE",
            "raison_principale": (
                f"Historique insuffisant ({len(_price_history)}/{min_history_points} "
                f"points) -- volatilite/momentum pas encore fiables."
            ),
            "confiance": 0, "edge": 0.0,
            "prob_reelle": 0.5,
            "prob_marche": market_yes_price_cents / 100.0,
        }

    vol = _recent_volatility_pct()

    # ── Probabilites modele ───────────────────────────────────────────────────
    prob_yes_model = estimate_probability(price, strike_price, minutes_remaining, vol)
    prob_no_model  = 1.0 - prob_yes_model

    # ── Prix marche des deux cotes (en fraction, pas en cents) ───────────────
    market_yes = market_yes_price_cents / 100.0

    # Utilise le vrai prix NO si fourni, sinon fallback sur complement du YES
    # (le complement sous-estime l'edge NO quand le spread est large)
    if market_no_price_cents is not None:
        market_no = market_no_price_cents / 100.0
    else:
        market_no = 1.0 - market_yes

    # ── Edges independants sur chaque cote ───────────────────────────────────
    edge_yes = prob_yes_model - market_yes   # positif => YES est sous-paye
    edge_no  = prob_no_model  - market_no    # positif => NO est sous-paye

    log.debug(
        f"[BTC] strike=${strike_price:,.0f} price=${price:,.0f} "
        f"P(yes)={prob_yes_model:.1%} | "
        f"mkt_yes={market_yes:.1%} edge_yes={edge_yes:+.1%} | "
        f"mkt_no={market_no:.1%}  edge_no={edge_no:+.1%}"
    )

    # ── Selection du cote : meilleur edge POSITIF au-dessus du seuil ─────────
    # Si les deux cotes sont sous le seuil -> AUCUN TRADE
    # Si les deux cotes depassent le seuil -> on prend celui avec le plus grand edge
    trade_yes = edge_yes >= min_edge
    trade_no  = edge_no  >= min_edge

    if trade_yes and trade_no:
        # Les deux ont un edge -- on prend le meilleur
        if edge_yes >= edge_no:
            verdict, edge_taken = "ACHETER YES", edge_yes
            prob_reelle_trade, prob_marche_trade = prob_yes_model, market_yes
        else:
            verdict, edge_taken = "ACHETER NO", edge_no
            prob_reelle_trade, prob_marche_trade = prob_no_model, market_no
    elif trade_yes:
        verdict, edge_taken = "ACHETER YES", edge_yes
        prob_reelle_trade, prob_marche_trade = prob_yes_model, market_yes
    elif trade_no:
        verdict, edge_taken = "ACHETER NO", edge_no
        prob_reelle_trade, prob_marche_trade = prob_no_model, market_no
    else:
        verdict, edge_taken = "AUCUN TRADE", max(edge_yes, edge_no)
        prob_reelle_trade, prob_marche_trade = prob_yes_model, market_yes

    # ── Confiance : normalisee sur le cote trade ──────────────────────────────
    t_scaled     = max(minutes_remaining / 15.0, 0.01)
    sigma_scaled = max(vol, 0.0001) * math.sqrt(t_scaled)
    try:
        d2 = abs(math.log(price / strike_price) - 0.5 * vol**2 * t_scaled) / sigma_scaled
    except (ValueError, ZeroDivisionError):
        d2 = 0.0
    confiance = min(10, max(1, int(d2 * 3)))

    # ── EV ────────────────────────────────────────────────────────────────────
    # Gain si gagne = (1 - prix_paye) ; perte si perd = prix_paye
    p_gain     = prob_reelle_trade
    p_perte    = 1.0 - prob_reelle_trade
    gain       = 1.0 - prob_marche_trade
    perte      = prob_marche_trade
    ev_brute   = p_gain * gain - p_perte * perte
    ev_nette   = p_gain * gain * 0.9755 - p_perte * perte  # apres frais Kalshi

    direction = "au-dessus" if verdict == "ACHETER YES" else "en-dessous"
    raison = (
        f"Modele BS binaire: prix=${price:,.2f} strike=${strike_price:,.2f} "
        f"vol={vol:.2%} t={minutes_remaining:.1f}min | "
        f"P({direction})={prob_reelle_trade:.1%} vs mkt={prob_marche_trade:.1%} "
        f"edge={edge_taken:+.1%}"
    )

    return {
        "verdict":           verdict,
        "prob_reelle":       round(prob_reelle_trade, 4),
        "prob_marche":       round(prob_marche_trade, 4),
        "edge":              round(edge_taken, 4),
        "ev_brute":          round(ev_brute, 4),
        "ev_nette":          round(ev_nette, 4),
        "confiance":         confiance,
        "risque":            10 - confiance,
        "grade":             _grade(edge_taken),
        "raison_principale": raison,
        "risque_principal":  "Volatilite BTC peut devier brutalement (news, liquidation cascade).",
        "risque_exogene":    "Slippage d'execution si le carnet d'ordres est fin.",
        "taille_position":   _position_size(edge_taken) if verdict != "AUCUN TRADE" else "0%",
    }
