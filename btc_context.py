"""
btc_context.py — v2 (2026-07-12)
Acquisition et normalisation des donnees BTC pour le modele 15 minutes.

CE MODULE NE PREND AUCUNE DECISION DE TRADING. Il retourne un objet
structure (BtcMarketContext) decrivant l'etat du marche et la QUALITE des
donnees. Si moins de deux sources spot valides sont disponibles, le contexte
est invalide et AUCUNE probabilite ne pourra etre calculee en aval.

Sources par defaut (publiques, sans cle) : Coinbase, Kraken, Bitstamp pour
le spot ; Binance pour les bougies 1 minute. Toutes les sources sont
INJECTABLES pour les tests hors-ligne (aucun test ne touche le reseau).

Aucun secret dans ce fichier.
"""

import math
import time
import logging
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

log = logging.getLogger("BTCCTX")

# ── Parametres (env-surchargables cote appelant si besoin) ───────────────────
HTTP_TIMEOUT_S      = 5.0
MAX_RETRIES         = 1            # retry LIMITE par source
CACHE_TTL_S         = 10.0         # cache court : donnees "ultra fraiches"
MAX_PRICE_AGE_S     = 90.0         # au-dela : donnee PERIMEE
MAX_DISPERSION_PCT  = 0.5          # ecart max entre exchanges (aberrant sinon)
MIN_VALID_SOURCES   = 2
MIN_KLINES          = 11           # pour rendement 10m + vol realisee

_cache = {}                        # key -> (value, ts)


def _cached(key, ttl, fn):
    now = time.time()
    if key in _cache:
        val, ts = _cache[key]
        if now - ts < ttl:
            return val
    val = fn()
    if val is not None:
        _cache[key] = (val, now)
    return val


# ── Fetchers par defaut (reseau) — remplacables par injection ────────────────

def _http_get_json(url, params=None):
    import requests
    last = None
    for _ in range(1 + MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        except Exception as e:            # noqa: BLE001 — retry limite
            last = e
    log.debug(f"source {url}: {last}")
    return None


def fetch_coinbase() -> Optional[dict]:
    d = _http_get_json("https://api.coinbase.com/v2/prices/BTC-USD/spot")
    if not d:
        return None
    try:
        return {"source": "coinbase", "price": float(d["data"]["amount"]),
                "ts": time.time()}
    except (KeyError, TypeError, ValueError):
        return None


def fetch_kraken() -> Optional[dict]:
    d = _http_get_json("https://api.kraken.com/0/public/Ticker",
                       {"pair": "XBTUSD"})
    try:
        return {"source": "kraken",
                "price": float(d["result"]["XXBTZUSD"]["c"][0]),
                "ts": time.time()}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def fetch_bitstamp() -> Optional[dict]:
    d = _http_get_json("https://www.bitstamp.net/api/v2/ticker/btcusd/")
    try:
        # bitstamp fournit son propre timestamp -> validation de fraicheur
        return {"source": "bitstamp", "price": float(d["last"]),
                "ts": float(d.get("timestamp", time.time()))}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def fetch_klines_binance(limit: int = 30) -> Optional[list]:
    d = _http_get_json("https://api.binance.com/api/v3/klines",
                       {"symbol": "BTCUSDT", "interval": "1m", "limit": limit})
    if not d:
        return None
    try:
        return [{"ts": k[0] / 1000.0, "open": float(k[1]), "high": float(k[2]),
                 "low": float(k[3]), "close": float(k[4]),
                 "volume": float(k[5])} for k in d]
    except (TypeError, ValueError, IndexError):
        return None


DEFAULT_SPOT_SOURCES = (fetch_coinbase, fetch_kraken, fetch_bitstamp)


# ── Objet retourne ───────────────────────────────────────────────────────────

@dataclass
class BtcMarketContext:
    valid: bool
    reason: str
    generated_ts: float
    spot: Optional[float] = None            # consensus (mediane des sources)
    sources: list = field(default_factory=list)   # [{source, price, ts}]
    n_valid_sources: int = 0
    dispersion_pct: Optional[float] = None  # (max-min)/mediane * 100
    strike: Optional[float] = None
    distance: Optional[float] = None        # spot - strike ($)
    distance_norm: Optional[float] = None   # ln(spot/strike)
    minutes_remaining: Optional[float] = None
    returns: dict = field(default_factory=dict)   # {"1m","3m","5m","10m"} log
    realized_vol_1m: Optional[float] = None # ecart-type des log-rendements 1m
    momentum_per_min: Optional[float] = None
    klines_count: int = 0
    data_quality_score: float = 0.0         # 0..100
    quality_flags: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ── Validation des sources spot ──────────────────────────────────────────────

def _validate_sources(raw: list, now: float) -> (list, list):
    """Filtre : prix positifs, timestamps frais, aberrants exclus (ecart a
    la mediane > MAX_DISPERSION_PCT)."""
    flags = []
    fresh = []
    for s in raw:
        if not s or not isinstance(s.get("price"), (int, float)):
            continue
        if s["price"] <= 0 or math.isnan(s["price"]) or math.isinf(s["price"]):
            flags.append(f"{s.get('source','?')}:prix_invalide")
            continue
        age = now - float(s.get("ts", 0))
        if age > MAX_PRICE_AGE_S or age < -30:
            flags.append(f"{s.get('source','?')}:perime({age:.0f}s)")
            continue
        fresh.append(s)
    if len(fresh) >= 2:
        med = statistics.median(x["price"] for x in fresh)
        kept = []
        for s in fresh:
            dev = abs(s["price"] - med) / med * 100
            if dev > MAX_DISPERSION_PCT:
                flags.append(f"{s['source']}:aberrant({dev:.2f}%)")
            else:
                kept.append(s)
        fresh = kept
    return fresh, flags


# ── Construction du contexte ─────────────────────────────────────────────────

def get_btc_context(strike: Optional[float] = None,
                    minutes_remaining: Optional[float] = None,
                    spot_sources: tuple = None,
                    klines_fn: Callable = None,
                    now: Optional[float] = None,
                    use_cache: bool = True) -> BtcMarketContext:
    """Recupere, valide et normalise. spot_sources/klines_fn injectables
    (tests hors-ligne). Retourne TOUJOURS un BtcMarketContext ; valid=False
    avec 'reason' explicite si les donnees sont insuffisantes."""
    now = now if now is not None else time.time()
    spot_sources = spot_sources or DEFAULT_SPOT_SOURCES
    klines_fn = klines_fn or fetch_klines_binance

    def pull():
        return [f() for f in spot_sources]
    raw = _cached("spot_sources", CACHE_TTL_S, pull) if use_cache else pull()
    sources, flags = _validate_sources(raw or [], now)

    ctx = BtcMarketContext(valid=False, reason="", generated_ts=now,
                           sources=sources, n_valid_sources=len(sources),
                           quality_flags=flags,
                           strike=strike, minutes_remaining=minutes_remaining)

    if len(sources) < MIN_VALID_SOURCES:
        ctx.reason = (f"sources_spot_insuffisantes "
                      f"({len(sources)}/{MIN_VALID_SOURCES})")
        return ctx

    prices = [s["price"] for s in sources]
    ctx.spot = statistics.median(prices)
    ctx.dispersion_pct = round((max(prices) - min(prices)) / ctx.spot * 100, 4)

    kl = (_cached("klines", CACHE_TTL_S, lambda: klines_fn())
          if use_cache else klines_fn()) or []
    # validation des timestamps des bougies (chronologie + fraicheur)
    kl = [k for k in kl if isinstance(k.get("close"), (int, float))
          and k["close"] > 0]
    if kl and any(kl[i]["ts"] >= kl[i + 1]["ts"] for i in range(len(kl) - 1)):
        flags.append("klines:timestamps_non_monotones")
        kl = []
    if kl and now - kl[-1]["ts"] > 3 * 60:
        flags.append("klines:perimees")
        kl = []
    ctx.klines_count = len(kl)

    if len(kl) >= MIN_KLINES:
        closes = [k["close"] for k in kl]
        def lr(n):  # log-rendement sur n minutes
            return math.log(closes[-1] / closes[-1 - n])
        ctx.returns = {"1m": lr(1), "3m": lr(3), "5m": lr(5), "10m": lr(10)}
        rets = [math.log(closes[i + 1] / closes[i])
                for i in range(len(closes) - 1)]
        ctx.realized_vol_1m = statistics.pstdev(rets) if len(rets) >= 2 else None
        ctx.momentum_per_min = (closes[-1] - closes[-6]) / 5 \
            if len(closes) >= 6 else None
    else:
        flags.append(f"klines:insuffisantes({len(kl)}/{MIN_KLINES})")

    if strike is not None and strike > 0 and ctx.spot:
        ctx.distance = round(ctx.spot - strike, 2)
        ctx.distance_norm = math.log(ctx.spot / strike)

    # ── data_quality_score (0..100), documente ──
    score = 0.0
    score += 40.0 * min(1.0, len(sources) / 3)              # nb de sources
    if ctx.dispersion_pct is not None:                      # accord entre elles
        score += 20.0 * max(0.0, 1.0 - ctx.dispersion_pct / MAX_DISPERSION_PCT)
    if ctx.realized_vol_1m is not None and ctx.realized_vol_1m > 0:
        score += 25.0                                       # vol mesurable
    if ctx.klines_count >= MIN_KLINES:
        score += 15.0                                       # historique 1m
    ctx.data_quality_score = round(score, 1)

    if ctx.realized_vol_1m is None or ctx.realized_vol_1m <= 0:
        ctx.reason = "volatilite_indisponible_ou_nulle"
        return ctx

    ctx.valid = True
    ctx.reason = "ok"
    return ctx


def get_btc_price(spot_sources: tuple = None) -> Optional[float]:
    """Compat : spot consensus, ou None si < 2 sources valides."""
    ctx = get_btc_context(spot_sources=spot_sources)
    return ctx.spot if ctx.n_valid_sources >= MIN_VALID_SOURCES else None


def clear_cache():
    _cache.clear()
