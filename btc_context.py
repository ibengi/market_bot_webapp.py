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
L'acces direct a l'API BRTI necessite une licence payante CF Benchmarks.//
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
    # Ramene a l'echelle de 15 minutes (en supposant des observations ~reguliere
    # sur la fenetre observee)
    n_obs = len(recent)
    span_minutes = (recent[-1][0] - recent[0][0]) / 60 or 1
    per_obs_minutes = span_minutes / max(n_obs - 1, 1)
    scale_to_15min = math.sqrt(15 / per_obs_minutes) if per_obs_minutes > 0 else 1
    return stdev * scale_to_15min


# ── Modele de probabilite (Black-Scholes binaire / cash-or-nothing) ─────────

def _norm_cdf(x: float) -> float:
    """Fonction de repartition de la loi normale standard (approximation erf)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def estimate_probability(current_price: float, strike_price: float,
                          minutes_remaining: float, volatility_15min: float = None) -> float:
    """
    Estime la probabilite que le prix BTC cloture AU-DESSUS du strike
    a l'expiration, via le modele cash-or-nothing binaire :

        P(S_T > K) = N(d2)
        d2 = [ln(S/K) - 0.5*sigma^2*t] / (sigma*sqrt(t))

    ou S = prix actuel, K = strike, sigma = volatilite (echelle de la
    fenetre), t = fraction de la fenetre totale restante (normalisee a 1
    pour une fenetre de 15min).

    Retourne une probabilite entre 0.0 et 1.0.
    """
    if volatility_15min is None:
        volatility_15min = _recent_volatility_pct()

    if current_price <= 0 or strike_price <= 0 or minutes_remaining <= 0:
        return 0.5

    # t = fraction du temps restant par rapport a la fenetre de reference (15min)
    t = max(minutes_remaining / 15.0, 0.01)
    sigma = max(volatility_15min, 0.0001)  # evite division par zero

    try:
        d2 = (math.log(current_price / strike_price) - 0.5 * sigma**2 * t) / (sigma * math.sqrt(t))
        prob_above = _norm_cdf(d2)
    except (ValueError, ZeroDivisionError):
        prob_above = 0.5

    return max(0.0, min(1.0, prob_above))


# ── Interface principale utilisee par kalshi_alpha_bot.py ───────────────────

def get_btc_context(target_price: float = 0, minutes: int = 15) -> str:
    """
    Construit un resume textuel du contexte BTC pour fournir a Claude en
    supervision (PAS pour decision de trade temps reel -- voir
    evaluate_btc_trade ci-dessous pour la decision sans Claude).
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


def evaluate_btc_trade(strike_price: float, market_yes_price_cents: int,
                        minutes_remaining: float, min_edge: float = 0.04) -> dict:
    """
    Decision de trade BTC 15min SANS appel Claude -- calcul instantane.
    A appeler directement dans la boucle de trading pour eviter toute
    latence sur un marche qui bouge en quelques secondes.

    Retourne un dict compatible avec le format 'phase10' utilise ailleurs
    dans le bot, pour rester homogene avec le reste du pipeline.
    """
    price = get_btc_price()
    if price is None:
        return {
            "verdict": "AUCUN TRADE",
            "raison_principale": "Prix BTC indisponible (echec reseau exchanges).",
            "confiance": 0, "edge": 0.0,
        }

    _record_price(price)
    vol = _recent_volatility_pct()
    prob_yes = estimate_probability(price, strike_price, minutes_remaining, vol)
    market_prob_yes = market_yes_price_cents / 100.0

    edge_yes = prob_yes - market_prob_yes
    edge_no  = (1 - prob_yes) - (1 - market_prob_yes)

    if edge_yes >= min_edge and edge_yes >= edge_no:
        verdict = "ACHETER YES"
        edge    = edge_yes
    elif edge_no >= min_edge:
        verdict = "ACHETER NO"
        edge    = edge_no
    else:
        verdict = "AUCUN TRADE"
        edge    = max(edge_yes, edge_no)

    # Confiance basee sur |d2| (distance au strike normalisee par la vol ET
    # le temps restant -- meme echelle que le calcul de probabilite ci-dessus).
    # BUG CORRIGE : l'ancienne version utilisait la volatilite brute sans la
    # mettre a l'echelle du temps restant, ce qui rendait le modele
    # systematiquement trop prudent en toute fin de fenetre (la ou il devrait
    # au contraire etre le plus confiant, puisqu'il reste tres peu de temps
    # pour que le prix s'eloigne du strike).
    t_scaled = max(minutes_remaining / 15.0, 0.01)
    sigma_scaled = max(vol, 0.0001) * math.sqrt(t_scaled)
    try:
        d2 = abs(math.log(price / strike_price) - 0.5 * vol**2 * t_scaled) / sigma_scaled
    except (ValueError, ZeroDivisionError):
        d2 = 0.0
    confiance = min(10, max(1, int(d2 * 3)))

    return {
        "verdict":           verdict,
        "prob_reelle":       prob_yes if verdict != "ACHETER NO" else 1 - prob_yes,
        "prob_marche":       market_prob_yes if verdict != "ACHETER NO" else 1 - market_prob_yes,
        "edge":              edge,
        "confiance":         confiance,
        "risque":            10 - confiance,
        "grade":             "A" if edge > 0.10 else "B" if edge > 0.06 else "C" if edge > 0.04 else "D",
        "raison_principale": (
            f"Modele math (BS binaire): prix=${price:,.2f} strike=${strike_price:,.2f} "
            f"vol={vol:.2%} t={minutes_remaining:.1f}min -> P(yes)={prob_yes:.1%}"
        ),
        "risque_principal":  "Volatilite BTC peut deviée brutalement (news, liquidation cascade).",
        "risque_exogene":    "Slippage d'execution si le carnet d'ordres est fin.",
        "taille_position":   "1%" if edge > 0.10 else "0.5%",
    }
