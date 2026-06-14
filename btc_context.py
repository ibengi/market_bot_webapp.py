"""
BTC CONTEXT MODULE — Kalshi Alpha Engine V3
Recupere les donnees crypto temps reel pour l'analyse des marches BTC Kalshi.

Sources gratuites :
- CoinGecko API (prix, market cap, sentiment)
- Binance API (orderbook, funding rate, klines)
- Alternative.me (Fear & Greed Index)

USAGE:
    from btc_context import get_btc_context, get_btc_ticker
    context = get_btc_context(target_price=65145.57, minutes=15)
    ticker  = get_btc_ticker()  # ex: KXBTC15M-14JUN-T65145
"""

import os, json, time, logging
from datetime import datetime, timezone
from typing import Optional
import requests

log = logging.getLogger("BTCContext")

# CF Benchmarks RTI (source officielle Kalshi)
CF_BASE = "https://www.cfbenchmarks.com/data/indices"

def get_cf_rti_price() -> Optional[float]:
    """
    Recupere le prix CF Benchmarks RTI — source officielle Kalshi.
    Calcule une approximation via moyenne ponderee multi-exchanges.
    """
    # CF Benchmarks RTI = moyenne ponderee de Coinbase, Kraken, Bitstamp, itBit, Gemini
    exchanges = [
        ("https://api.coinbase.com/v2/prices/BTC-USD/spot", lambda d: float(d["data"]["amount"])),
        ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", lambda d: float(d["result"]["XXBTZUSD"]["c"][0])),
        ("https://www.bitstamp.net/api/v2/ticker/btcusd/", lambda d: float(d["last"])),
    ]
    prices = []
    for url, parser in exchanges:
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            prices.append(parser(r.json()))
        except Exception as e:
            log.debug(f"Exchange {url}: {e}")

    if not prices:
        return None
    # Moyenne simple des exchanges disponibles (approximation RTI)
    rti = sum(prices) / len(prices)
    return round(rti, 2), prices

def get_spread_analysis(target: float, rti: float, spot: float) -> dict:
    """
    Analyse l'ecart entre prix spot, RTI et target Kalshi.
    Crucial pour determiner la probabilite reelle de resolution.
    """
    rti_vs_target  = rti - target
    spot_vs_target = spot - target
    rti_vs_spot    = rti - spot

    return {
        "rti_price":       rti,
        "spot_price":      spot,
        "target":          target,
        "rti_vs_target":   round(rti_vs_target, 2),
        "spot_vs_target":  round(spot_vs_target, 2),
        "rti_vs_spot":     round(rti_vs_spot, 2),
        "rti_above":       rti > target,
        "spread_pct":      round(rti_vs_target / target * 100, 4),
        "risk_zone":       abs(rti_vs_target) < 100,  # Moins de $100 du target = zone risquee
    }


BINANCE_BASE  = "https://api.binance.com/api/v3"
BINFUT_BASE   = "https://fapi.binance.com/fapi/v1"
COINGECKO     = "https://api.coingecko.com/api/v3"
FEARGREED     = "https://api.alternative.me/fng/"

_cache = {}
CACHE_TTL = 30  # 30 secondes pour crypto (ultra-frais)

def _get(url, params=None, ttl=CACHE_TTL):
    key = url + str(params)
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < ttl:
            return data
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        _cache[key] = (data, now)
        return data
    except Exception as e:
        log.warning(f"Erreur {url}: {e}")
        return None

def get_btc_price() -> Optional[float]:
    """Prix BTC spot temps reel (Binance)."""
    d = _get(f"{BINANCE_BASE}/ticker/price", {"symbol": "BTCUSDT"})
    return float(d["price"]) if d else None

def get_btc_klines(interval: str = "1m", limit: int = 15) -> list:
    """
    Bougies BTC recentes.
    interval: 1m, 5m, 15m, 1h
    Retourne liste de {open, high, low, close, volume}
    """
    d = _get(f"{BINANCE_BASE}/klines", {
        "symbol": "BTCUSDT", "interval": interval, "limit": limit
    })
    if not d:
        return []
    return [{
        "open":   float(k[1]),
        "high":   float(k[2]),
        "low":    float(k[3]),
        "close":  float(k[4]),
        "volume": float(k[5]),
        "time":   datetime.fromtimestamp(k[0]/1000, tz=timezone.utc).strftime("%H:%M"),
    } for k in d]

def get_orderbook_imbalance() -> Optional[float]:
    """
    Desequilibre orderbook BTC (bid vs ask).
    > 0 = pression acheteuse, < 0 = pression vendeuse
    """
    d = _get(f"{BINANCE_BASE}/depth", {"symbol": "BTCUSDT", "limit": 20})
    if not d:
        return None
    bid_vol = sum(float(b[1]) for b in d["bids"])
    ask_vol = sum(float(a[1]) for a in d["asks"])
    total = bid_vol + ask_vol
    if total == 0:
        return None
    return round((bid_vol - ask_vol) / total * 100, 2)

def get_funding_rate() -> Optional[float]:
    """Funding rate BTC perps (sentiment institutionnel)."""
    d = _get(f"{BINFUT_BASE}/fundingRate", {
        "symbol": "BTCUSDT", "limit": 1
    }, ttl=300)
    if d and len(d) > 0:
        return float(d[0]["fundingRate"]) * 100
    return None

def get_fear_greed() -> Optional[dict]:
    """Fear & Greed Index (0=peur extreme, 100=avidite extreme)."""
    d = _get(FEARGREED, {"limit": 1}, ttl=3600)
    if d and d.get("data"):
        item = d["data"][0]
        return {
            "value":       int(item["value"]),
            "label":       item["value_classification"],
            "timestamp":   item["timestamp"],
        }
    return None

def get_btc_dominance() -> Optional[float]:
    """Dominance BTC sur le marche crypto total."""
    d = _get(f"{COINGECKO}/global", ttl=3600)
    if d:
        return round(d.get("data", {}).get("market_cap_percentage", {}).get("btc", 0), 1)
    return None

def calc_rsi(closes: list, period: int = 14) -> Optional[float]:
    """Calcule le RSI sur les dernieres bougies."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)

def calc_vwap(klines: list) -> Optional[float]:
    """Calcule le VWAP sur les bougies fournies."""
    if not klines:
        return None
    num = sum(((k["high"] + k["low"] + k["close"]) / 3) * k["volume"] for k in klines)
    den = sum(k["volume"] for k in klines)
    return round(num / den, 2) if den > 0 else None

def get_btc_context(target_price: float, minutes: int = 15) -> str:
    """
    Genere le contexte BTC complet pour l'analyse Claude.
    
    Args:
        target_price: Prix seuil du marche Kalshi
        minutes:      Duree du contrat (15, 60, etc.)
    
    Returns:
        Contexte textuel formate
    """
    log.info(f"Recuperation contexte BTC (target=${target_price:,.2f}, {minutes}min)...")

    # ── Donnees temps reel ────────────────────────────────────────────────────
    btc_price  = get_btc_price()

    # CF Benchmarks RTI (source officielle Kalshi)
    rti_result = get_cf_rti_price()
    rti_price  = rti_result[0] if rti_result else btc_price
    exch_prices = rti_result[1] if rti_result else []
    spread     = get_spread_analysis(target_price, rti_price or btc_price, btc_price or 0) if btc_price else None
    klines_1m  = get_btc_klines("1m",  limit=30)
    klines_5m  = get_btc_klines("5m",  limit=12)
    ob_imbal   = get_orderbook_imbalance()
    funding    = get_funding_rate()
    fg         = get_fear_greed()
    dominance  = get_btc_dominance()

    if not btc_price:
        return "Donnees BTC non disponibles — verifiez la connexion internet."

    # ── Calculs techniques ────────────────────────────────────────────────────
    closes_1m = [k["close"] for k in klines_1m]
    closes_5m = [k["close"] for k in klines_5m]
    rsi_1m    = calc_rsi(closes_1m, 14)
    rsi_5m    = calc_rsi(closes_5m, 9)
    vwap_15m  = calc_vwap(klines_1m[-15:]) if len(klines_1m) >= 15 else None

    # Distance au target
    dist_pct  = ((btc_price - target_price) / target_price) * 100
    above     = btc_price > target_price
    dist_abs  = abs(btc_price - target_price)

    # Momentum 5 dernières minutes
    momentum  = None
    if len(klines_1m) >= 5:
        momentum = klines_1m[-1]["close"] - klines_1m[-5]["close"]

    # Volume moyen vs dernier volume
    vol_avg   = None
    vol_ratio = None
    if len(klines_1m) >= 10:
        vol_avg   = sum(k["volume"] for k in klines_1m[-10:]) / 10
        last_vol  = klines_1m[-1]["volume"]
        vol_ratio = round(last_vol / vol_avg, 2) if vol_avg > 0 else None

    # High / Low des 15 dernières minutes
    hi15 = max(k["high"]  for k in klines_1m[-15:]) if len(klines_1m) >= 15 else None
    lo15 = min(k["low"]   for k in klines_1m[-15:]) if len(klines_1m) >= 15 else None

    # ── Formate le contexte ───────────────────────────────────────────────────
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    lines = [
        f"=== CONTEXTE BTC TEMPS REEL — {now_str} ===",
        f"Marche Kalshi : BTC {minutes}min | Target : ${target_price:,.2f}",
        "",
        "--- PRIX CF BENCHMARKS (SOURCE OFFICIELLE KALSHI) ---",
        f"  CF Benchmarks RTI  : ${rti_price:,.2f}" if rti_price else "  CF Benchmarks RTI  : N/A (Binance utilise)",
        f"  Prix Binance spot  : ${btc_price:,.2f}" if btc_price else "",
        f"  Ecart RTI vs Spot  : {spread['rti_vs_spot']:+.2f}$" if spread else "",
        f"  RTI vs Target      : {spread['rti_vs_target']:+.2f}$ ({'AU-DESSUS' if spread and spread['rti_above'] else 'EN-DESSOUS'})" if spread else "",
        f"  ZONE RISQUE        : {'⚠️ OUI — moins de $100 du target' if spread and spread['risk_zone'] else 'NON — marge suffisante'}" if spread else "",
        "",
        "--- PRIX & POSITION ---",
        f"  Prix BTC actuel    : ${btc_price:,.2f}",
        f"  Target Kalshi      : ${target_price:,.2f}",
        f"  Position vs target : {'AU-DESSUS' if above else 'EN-DESSOUS'} de {dist_pct:+.3f}% (${dist_abs:,.2f})",
        f"  VWAP 15min         : ${vwap_15m:,.2f}" if vwap_15m else "  VWAP 15min         : N/A",
        f"  High 15min         : ${hi15:,.2f}" if hi15 else "",
        f"  Low  15min         : ${lo15:,.2f}" if lo15 else "",
        "",
        "--- MOMENTUM & TECHNIQUE ---",
        f"  RSI 1min (14)      : {rsi_1m}" if rsi_1m else "  RSI 1min          : N/A",
        f"  RSI 5min (9)       : {rsi_5m}"  if rsi_5m else "  RSI 5min          : N/A",
        f"  Momentum 5min      : {momentum:+.2f}$" if momentum is not None else "  Momentum 5min     : N/A",
        f"  Volume ratio       : {vol_ratio}x vs moyenne" if vol_ratio else "  Volume ratio      : N/A",
        "",
        "--- SENTIMENT & ORDERBOOK ---",
        f"  Orderbook imbalance: {ob_imbal:+.1f}% ({'ACHETEURS' if (ob_imbal or 0) > 0 else 'VENDEURS'} dominants)" if ob_imbal is not None else "  Orderbook         : N/A",
        f"  Funding rate       : {funding:+.4f}% ({'long squeeze risque' if (funding or 0) > 0.05 else 'neutre/short' if (funding or 0) < 0 else 'neutre'})" if funding is not None else "  Funding rate      : N/A",
        f"  Fear & Greed       : {fg['value']}/100 — {fg['label']}" if fg else "  Fear & Greed      : N/A",
        f"  Dominance BTC      : {dominance}%" if dominance else "  Dominance BTC     : N/A",
        "",
        "--- ANALYSE RAPIDE ---",
    ]

    # Signaux automatiques
    signals = []

    if above:
        signals.append(f"BTC est AU-DESSUS du target de ${dist_abs:,.2f} — favorise UP")
    else:
        signals.append(f"BTC est EN-DESSOUS du target de ${dist_abs:,.2f} — favorise DOWN")

    if rsi_1m:
        if rsi_1m > 70:
            signals.append(f"RSI 1min suracheté ({rsi_1m}) — risque de retournement baissier")
        elif rsi_1m < 30:
            signals.append(f"RSI 1min survendu ({rsi_1m}) — risque de rebond haussier")
        else:
            signals.append(f"RSI 1min neutre ({rsi_1m}) — pas de signal fort")

    if momentum is not None:
        if momentum > 50:
            signals.append(f"Momentum 5min positif ({momentum:+.0f}$) — tendance haussière")
        elif momentum < -50:
            signals.append(f"Momentum 5min negatif ({momentum:+.0f}$) — tendance baissière")

    if ob_imbal is not None:
        if ob_imbal > 20:
            signals.append(f"Forte pression acheteuse orderbook ({ob_imbal:+.1f}%)")
        elif ob_imbal < -20:
            signals.append(f"Forte pression vendeuse orderbook ({ob_imbal:+.1f}%)")

    if vwap_15m and btc_price:
        if btc_price > vwap_15m:
            signals.append(f"Prix au-dessus du VWAP 15min (${vwap_15m:,.2f}) — momentum haussier")
        else:
            signals.append(f"Prix sous le VWAP 15min (${vwap_15m:,.2f}) — pression baissière")

    # Signal de convergence
    bull_signals = sum(1 for s in signals if "haussier" in s.lower() or "haussière" in s.lower() or "UP" in s or "DESSUS" in s)
    bear_signals = sum(1 for s in signals if "baissier" in s.lower() or "baissière" in s.lower() or "DOWN" in s or "DESSOUS" in s)

    for s in signals:
        lines.append(f"  → {s}")

    lines.append("")
    lines.append(f"  CONVERGENCE : {bull_signals} signaux UP vs {bear_signals} signaux DOWN")
    lines.append(f"  VERDICT TECHNIQUE : {'PLUTOT UP' if bull_signals > bear_signals else 'PLUTOT DOWN' if bear_signals > bull_signals else 'NEUTRE'}")

    # Avertissement critique CF Benchmarks
    if spread and spread["risk_zone"]:
        lines.append("")
        lines.append("  ⚠️  ATTENTION : Prix RTI a moins de $100 du target")
        lines.append(f"  La resolution Kalshi utilise la MOYENNE des 60 dernieres")
        lines.append(f"  cotations CF Benchmarks — pas le prix spot Binance.")
        lines.append(f"  Ecart actuel RTI vs target : {spread['rti_vs_target']:+.2f}$")
        lines.append(f"  Une volatilite de {abs(spread['rti_vs_target']):.0f}$ suffit a inverser la resolution.")

    return "\n".join(l for l in lines if l is not None)


def get_btc_ticker(target_price: float, minutes: int = 15) -> str:
    """
    Genere le ticker Kalshi probable pour ce marche BTC.
    Format: KXBTC15M-DDMMM-TXXXXX
    """
    now = datetime.now()
    date_str = now.strftime("%d%b").upper()
    price_str = str(int(target_price))
    return f"KXBTC{minutes}M-{date_str}-T{price_str}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print("\n=== TEST BTC CONTEXT MODULE ===\n")
    btc = get_btc_price()
    print(f"Prix BTC actuel : ${btc:,.2f}" if btc else "Prix non disponible")
    print()
    ctx = get_btc_context(target_price=65145.57, minutes=15)
    print(ctx)
