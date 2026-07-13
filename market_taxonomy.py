"""
market_taxonomy.py — v1 (2026-07-13)
Classification DETERMINISTE des marches Kalshi en types precis.

Ordre de classification (fixe, teste) :
    1. series_ticker   2. event_ticker   3. ticker
    4. categorie native   5. title/subtitle   6. unknown

Chaque etape applique les MEMES tables de regles ; la premiere etape qui
produit un type l'emporte. Aucune donnee externe, aucun aleatoire, aucun
appel reseau : memes entrees => meme sortie (teste).

Ce module NE DECIDE RIEN : il ne route pas, ne trade pas, ne cree aucun
ordre. Il etiquette.
"""

from typing import Optional

MARKET_TYPES = (
    "btc_above_strike_15m",
    "btc_above_strike_1h",
    "btc_above_strike_daily",      # extension (serie KXBTCD vue en demo)
    "eth_above_strike_15m",
    "eth_above_strike_1h",         # extension
    "sports_moneyline",
    "sports_spread",
    "sports_total",
    "sports_player_prop",
    "cpi_above_threshold",
    "fed_rate_decision",
    "jobs_report",
    "weather_high_temperature",
    "election_winner",
    "unknown",
)

# Ligues/sports reconnus (prefixe de serie Kalshi et mots-cles)
_SPORT_LEAGUES = ("MLB", "NBA", "NFL", "NHL", "NCAA", "UFC", "MMA", "EPL",
                  "SOCCER", "FIFA", "TENNIS", "GOLF", "F1", "NASCAR", "WNBA")

# Marqueurs de sous-type sportif (ordre de test FIXE : spread, total, prop,
# puis moneyline — un spread contient souvent le nom du match, donc les
# sous-types specifiques passent AVANT le generique)
_SPORT_SPREAD = ("SPREAD", "HANDICAP", "COVERS")
_SPORT_TOTAL = ("TOTAL", "OVERUNDER", "OVER/UNDER", "O/U", "COMBINED")
_SPORT_PROP = ("HIT", "HITS", "HR", "HOMERUN", "STRIKEOUT", "OUTS", "POINTS",
               "ASSISTS", "REBOUNDS", "YARDS", "TOUCHDOWN", "GOALS", "SAVES",
               "PROP", "PLAYER")
_SPORT_MONEYLINE = ("GAME", "WIN", "WINNER", "ML", "MONEYLINE", "BEAT",
                    "MATCH", "OUTS")  # KXMLBOUTS: serie de match MLB


def _norm(v) -> str:
    return str(v or "").upper()


def _classify_text(text: str) -> Optional[str]:
    """Applique les tables de regles a UNE chaine. None si rien ne matche."""
    t = _norm(text)
    if not t:
        return None

    # ── Crypto (du plus specifique au moins specifique) ──
    if "KXBTC15M" in t or ("BTC" in t and "15M" in t):
        return "btc_above_strike_15m"
    if "KXETH15M" in t or ("ETH" in t and "15M" in t):
        return "eth_above_strike_15m"
    if "KXBTC1H" in t or "KXBTCH" in t or ("BTC" in t and ("1H" in t or "HOURLY" in t)):
        return "btc_above_strike_1h"
    if "KXETH1H" in t or "KXETHH" in t or ("ETH" in t and ("1H" in t or "HOURLY" in t)):
        return "eth_above_strike_1h"
    if "KXBTCD" in t or ("BTC" in t and ("DAILY" in t or " EOD" in t)):
        return "btc_above_strike_daily"

    # ── Sports : ligue requise + sous-type ──
    if any(lg in t for lg in _SPORT_LEAGUES):
        if any(k in t for k in _SPORT_SPREAD):
            return "sports_spread"
        if any(k in t for k in _SPORT_TOTAL):
            return "sports_total"
        if any(k in t for k in _SPORT_PROP):
            return "sports_player_prop"
        if any(k in t for k in _SPORT_MONEYLINE):
            return "sports_moneyline"
        return None          # ligue seule sans sous-type: on laisse la main
                             # aux etapes suivantes (title...) sinon unknown

    # ── Economie ──
    if "CPI" in t or "INFLATION" in t:
        return "cpi_above_threshold"
    if "FED" in t or "FOMC" in t or "RATE DECISION" in t or "RATE CUT" in t \
            or "RATE HIKE" in t:
        return "fed_rate_decision"
    if "PAYROLL" in t or "NFP" in t or "JOBS REPORT" in t or "UNEMPLOYMENT" in t \
            or "JOBLESS" in t:
        return "jobs_report"

    # ── Meteo ──
    if ("HIGH" in t and ("TEMP" in t or "°" in t or " T" in t)) \
            or "KXHIGH" in t or "HIGHTEMP" in t or t.startswith("HIGH"):
        return "weather_high_temperature"

    # ── Elections ──
    if any(k in t for k in ("PRES", "SENATE", "GOVERNOR", "MAYOR", "ELECT",
                            "PRIMARY", "POTUS", "HOUSE RACE", "WINNER OF THE")):
        return "election_winner"

    return None


# Categories natives Kalshi -> type UNIQUEMENT quand la categorie suffit a
# elle seule (rare) ; sinon None et l'etape suivante tranche.
_NATIVE_HINTS = {
    # aucune categorie native ne suffit seule a donner un market_type precis
    # (ex: "Crypto" ne dit pas 15m/1h) -> table volontairement vide, presente
    # pour rendre l'etape 4 explicite et extensible.
}


def classify_market_type(market: dict) -> str:
    """Classification deterministe. Ordre impose par le cahier des charges."""
    if not isinstance(market, dict):
        return "unknown"
    # 1..3 : identifiants, du plus structurel au moins structurel
    for key in ("series_ticker", "event_ticker", "ticker"):
        mt = _classify_text(market.get(key))
        if mt:
            return mt
    # 4 : categorie native (voir _NATIVE_HINTS)
    native = _norm(market.get("category"))
    if native in _NATIVE_HINTS:
        return _NATIVE_HINTS[native]
    # 5 : texte libre
    for key in ("title", "subtitle", "yes_sub_title"):
        mt = _classify_text(market.get(key))
        if mt:
            return mt
    # 6
    return "unknown"


def classification_report(snapshots: list) -> dict:
    """Comptage par market_type sur un univers de snapshots (rapport JSON)."""
    counts = {}
    for s in snapshots:
        mt = getattr(s, "market_type", None) or "unknown"
        counts[mt] = counts.get(mt, 0) + 1
    total = len(snapshots)
    return {"total": total,
            "by_market_type": dict(sorted(counts.items(),
                                          key=lambda kv: -kv[1])),
            "unknown": counts.get("unknown", 0),
            "unknown_pct": round(100.0 * counts.get("unknown", 0)
                                 / total, 2) if total else 0.0}
