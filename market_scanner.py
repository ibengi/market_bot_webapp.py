"""
market_scanner.py — v1 (2026-07-11)
Scanner d'univers de marches Kalshi. INDEPENDANT du moteur d'execution :
ce module ne connait ni OrderManager, ni RiskManager, ni PositionManager,
ni TradeLogger, et ne contient AUCUN appel d'envoi d'ordre.

Il ne fait que : paginer /markets, normaliser, classer, filtrer, rapporter.

Le client passe en argument (duck typing) : tout objet exposant
    _req(method, path, params=...) -> dict          (pagination)
suffit — le KalshiClient v11 convient tel quel.

AVERTISSEMENT D'INTEGRITE : le nom du parametre de pagination ("cursor",
"limit") et le nom des champs de carnet suivent la documentation Kalshi v2
telle que connue ; la premiere page brute est loggee ([RAW:markets_page])
pour verification humaine au premier scan reel.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Callable

from market_taxonomy import classify_market_type

log = logging.getLogger("SCANNER")

# ── Configuration (surchargeables par variables d'environnement) ─────────────

def _env_f(name, default):
    try: return float(os.getenv(name, str(default)))
    except ValueError: return default

def _env_i(name, default):
    try: return int(os.getenv(name, str(default)))
    except ValueError: return default

def _env_b(name, default):
    return os.getenv(name, "1" if default else "0").strip().lower() \
        not in ("0", "false", "no", "non")

def _env_list(name):
    raw = os.getenv(name, "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else []

class ScanConfig:
    MIN_MINUTES        = _env_f("SCANNER_MIN_MINUTES", 5.0)
    MAX_MINUTES        = _env_f("SCANNER_MAX_MINUTES", 60 * 24 * 90)  # 90 jours
    MIN_VOLUME         = _env_i("SCANNER_MIN_VOLUME", 0)
    MIN_OPEN_INTEREST  = _env_i("SCANNER_MIN_OPEN_INTEREST", 0)
    MAX_SPREAD_CENTS   = _env_i("SCANNER_MAX_SPREAD_CENTS", 10)
    REQUIRE_LIQUIDITY  = _env_b("SCANNER_REQUIRE_LIQUIDITY", True)
    ALLOWED_CATEGORIES = _env_list("SCANNER_ALLOWED_CATEGORIES")   # vide = toutes
    EXCLUDED_CATEGORIES= _env_list("SCANNER_EXCLUDED_CATEGORIES")
    PAGE_LIMIT         = _env_i("SCANNER_PAGE_LIMIT", 200)
    MAX_PAGES          = _env_i("SCANNER_MAX_PAGES", 200)  # garde-fou anti-boucle
    UNIVERSE_FILE      = os.getenv("SCANNER_UNIVERSE_FILE", "market_universe.json")
    REPORT_FILE        = os.getenv("SCANNER_REPORT_FILE", "market_scanner_report.json")

SCFG = ScanConfig()

# ── Normalisation des nombres (cents / *_dollars / *_fp / strings / null) ────

def parse_cents(market: dict, base: str) -> Optional[int]:
    """Extrait un prix en cents 1..99 depuis toutes les variantes connues :
       base (int cents), base_fp ("2.00" = cents a virgule fixe),
       base_dollars (0.12 = dollars). 0/None = cote vide -> None.
       Ne JAMAIS inventer de liquidite."""
    for key, is_dollars in ((base, False), (f"{base}_fp", False),
                            (f"{base}_dollars", True)):
        v = market.get(key)
        if v is None or v == "":
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        cents = int(round(x * 100)) if is_dollars else int(round(x))
        if 1 <= cents <= 99:
            return cents
        # 0 = cote vide ; >99/negatif = invalide -> on continue de chercher
    return None

def parse_number(market: dict, *keys) -> Optional[float]:
    """Nombre generique (volume, OI, liquidite) : int, float, string, *_fp."""
    for k in keys:
        for cand in (k, f"{k}_fp", f"{k}_dollars"):
            v = market.get(cand)
            if v is None or v == "":
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None

def parse_time(v) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None

# ── Carnet : derivation YES<->NO sans inventer de liquidite ──────────────────

def normalize_book(m: dict) -> dict:
    """Retourne {'ok': bool, 'reason': str|None, yes_bid.., spread_yes..}.
       0/None = cote vide. Derivation NO=100-YES uniquement quand l'autre
       cote existe reellement. Carnet croise -> invalid_book."""
    yb, ya = parse_cents(m, "yes_bid"), parse_cents(m, "yes_ask")
    nb, na = parse_cents(m, "no_bid"),  parse_cents(m, "no_ask")
    # derivations mathematiquement valides (complementarite YES/NO)
    if nb is None and ya is not None: nb = 100 - ya
    if na is None and yb is not None: na = 100 - yb
    if yb is None and na is not None: yb = 100 - na
    if ya is None and nb is not None: ya = 100 - nb
    out = {"yes_bid": yb, "yes_ask": ya, "no_bid": nb, "no_ask": na,
           "yes_mid": None, "no_mid": None,
           "spread_yes": None, "spread_no": None}
    if yb is None and ya is None and nb is None and na is None:
        return {"ok": False, "reason": "no_liquidity", **out}
    if yb is None or ya is None or nb is None or na is None:
        # une seule jambe cotee et non derivable : inutilisable, pas invente
        return {"ok": False, "reason": "no_liquidity", **out}
    if ya < yb or na < nb:
        return {"ok": False, "reason": "invalid_book", **out}
    out.update({"yes_mid": round((yb + ya) / 2), "no_mid": round((nb + na) / 2),
                "spread_yes": ya - yb, "spread_no": na - nb})
    return {"ok": True, "reason": None, **out}

# ── Classification par categorie ─────────────────────────────────────────────

CATEGORIES = ["Crypto", "Sports", "Politics", "Elections", "Economics",
              "Finance", "Commodities", "Climate", "Tech & Science",
              "Culture", "Entertainment", "Other"]

# Prefixes/mots-cles de series et d'events Kalshi (heuristique documentee ;
# le champ 'category' natif de l'API, quand il existe, prime toujours).
_SERIES_RULES = [
    ("Crypto",        ("KXBTC", "KXETH", "BTC", "ETH", "CRYPTO", "SOL", "DOGE")),
    ("Elections",     ("PRES", "SENATE", "HOUSE", "GOV", "MAYOR", "ELECT",
                       "PRIMARY", "POTUS", "EC-")),
    ("Politics",      ("SCOTUS", "CABINET", "IMPEACH", "SHUTDOWN", "VETO",
                       "CONGRESS", "EXEC", "TREATY", "POLI")),
    ("Economics",     ("CPI", "GDP", "FED", "FOMC", "PAYROLL", "NFP", "JOBS",
                       "UNRATE", "INFLATION", "RECESS", "RATE")),
    ("Finance",       ("INX", "NASDAQ", "SP500", "S&P", "DOW", "DJIA", "VIX",
                       "TSLA", "AAPL", "EARNINGS", "IPO", "STOCK")),
    ("Commodities",   ("OIL", "WTI", "BRENT", "GOLD", "SILVER", "GAS",
                       "WHEAT", "CORN", "COMMOD")),
    ("Climate",       ("HIGH", "LOW", "TEMP", "RAIN", "SNOW", "HURRICANE",
                       "CLIMATE", "WEATHER", "HEAT", "STORM")),
    ("Sports",        ("NFL", "NBA", "MLB", "NHL", "NCAA", "UFC", "F1",
                       "TENNIS", "GOLF", "SOCCER", "EPL", "FIFA", "OLYMP",
                       "SUPERBOWL", "WORLDCUP")),
    ("Tech & Science",("AI", "OPENAI", "SPACEX", "NASA", "LAUNCH", "APPLE",
                       "GOOGLE", "TECH", "CHIP", "QUANTUM", "FDA")),
    ("Entertainment", ("OSCAR", "EMMY", "GRAMMY", "BOXOFFICE", "MOVIE",
                       "NETFLIX", "ALBUM", "TVRATINGS")),
    ("Culture",       ("TIME-", "PERSONOF", "WORD", "MISS", "ROYAL",
                       "CELEB", "TWITTER", "TIKTOK")),
]

_NATIVE_MAP = {  # categories natives Kalshi -> nos categories
    "crypto": "Crypto", "cryptocurrency": "Crypto",
    "sports": "Sports",
    "politics": "Politics", "world": "Politics",
    "elections": "Elections",
    "economics": "Economics", "economy": "Economics",
    "financials": "Finance", "finance": "Finance", "companies": "Finance",
    "commodities": "Commodities",
    "climate": "Climate", "climate and weather": "Climate", "weather": "Climate",
    "science and technology": "Tech & Science", "technology": "Tech & Science",
    "science": "Tech & Science", "health": "Tech & Science",
    "entertainment": "Entertainment",
    "culture": "Culture", "social": "Culture",
}

def classify(m: dict) -> str:
    """category native > series_ticker > event_ticker > titre. Jamais 'invente' :
       si rien ne matche -> Other."""
    native = str(m.get("category") or "").strip().lower()
    if native in _NATIVE_MAP:
        return _NATIVE_MAP[native]
    hay = " ".join(str(m.get(k) or "") for k in
                   ("series_ticker", "event_ticker", "ticker",
                    "title", "subtitle")).upper()
    for cat, keys in _SERIES_RULES:
        if any(k in hay for k in keys):
            return cat
    return "Other"

# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    ticker: str
    event_ticker: Optional[str]
    series_ticker: Optional[str]
    title: Optional[str]
    subtitle: Optional[str]
    category: str
    market_type: str
    status: Optional[str]
    open_time: Optional[str]
    close_time: Optional[str]
    expiration_time: Optional[str]
    minutes_remaining: Optional[float]
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    yes_mid: Optional[int]
    no_mid: Optional[int]
    spread_yes: Optional[int]
    spread_no: Optional[int]
    volume: Optional[float]
    volume_24h: Optional[float]
    open_interest: Optional[float]
    liquidity: Optional[float]
    last_price: Optional[int]
    market_url: Optional[str]
    included: bool = True
    exclusion_reason: Optional[str] = None
    raw_market: dict = field(default_factory=dict, repr=False)

# ── Construction d'un snapshot + filtres ─────────────────────────────────────

def build_snapshot(m: dict, now: Optional[datetime] = None,
                   cfg: ScanConfig = SCFG) -> MarketSnapshot:
    now = now or datetime.now(timezone.utc)
    close_dt = parse_time(m.get("close_time"))
    mins = ((close_dt - now).total_seconds() / 60.0) if close_dt else None
    book = normalize_book(m)
    snap = MarketSnapshot(
        ticker=str(m.get("ticker") or ""),
        event_ticker=m.get("event_ticker"),
        series_ticker=m.get("series_ticker"),
        title=m.get("title"), subtitle=m.get("subtitle") or m.get("yes_sub_title"),
        category=classify(m),
        market_type=classify_market_type(m),
        status=m.get("status"),
        open_time=m.get("open_time"), close_time=m.get("close_time"),
        expiration_time=m.get("expiration_time"),
        minutes_remaining=round(mins, 2) if mins is not None else None,
        yes_bid=book["yes_bid"], yes_ask=book["yes_ask"],
        no_bid=book["no_bid"], no_ask=book["no_ask"],
        yes_mid=book["yes_mid"], no_mid=book["no_mid"],
        spread_yes=book["spread_yes"], spread_no=book["spread_no"],
        volume=parse_number(m, "volume"),
        volume_24h=parse_number(m, "volume_24h"),
        open_interest=parse_number(m, "open_interest"),
        liquidity=parse_number(m, "liquidity"),
        last_price=parse_cents(m, "last_price"),
        market_url=f"https://kalshi.com/markets/{m.get('ticker')}"
                   if m.get("ticker") else None,
        raw_market=m,
    )
    snap.exclusion_reason = _exclusion_reason(snap, book, cfg)
    snap.included = snap.exclusion_reason is None
    return snap

def _exclusion_reason(s: MarketSnapshot, book: dict,
                      cfg: ScanConfig) -> Optional[str]:
    if not s.ticker:
        return "unsupported"
    if s.status and str(s.status).lower() not in ("open", "active"):
        return "expired"
    if s.minutes_remaining is not None and s.minutes_remaining <= 0:
        return "expired"
    if s.minutes_remaining is not None and s.minutes_remaining < cfg.MIN_MINUTES:
        return "closes_too_soon"
    if s.minutes_remaining is not None and s.minutes_remaining > cfg.MAX_MINUTES:
        return "unsupported"
    if book["reason"] == "invalid_book":
        return "invalid_book"
    if book["reason"] == "no_liquidity":
        return "no_liquidity" if cfg.REQUIRE_LIQUIDITY else None
    if book["ok"] and s.spread_yes is not None \
            and s.spread_yes > cfg.MAX_SPREAD_CENTS:
        return "spread_too_wide"
    if cfg.MIN_VOLUME and (s.volume or 0) < cfg.MIN_VOLUME:
        return "low_volume"
    if cfg.MIN_OPEN_INTEREST and (s.open_interest or 0) < cfg.MIN_OPEN_INTEREST:
        return "low_volume"
    if cfg.ALLOWED_CATEGORIES and s.category not in cfg.ALLOWED_CATEGORIES:
        return "unsupported"
    if s.category in cfg.EXCLUDED_CATEGORIES:
        return "unsupported"
    return None

# ── Pagination complete de /markets ──────────────────────────────────────────

def fetch_all_markets(client, status: str = "open",
                      cfg: ScanConfig = SCFG) -> (list, int, int):
    """Pagine GET /markets par curseur. Retourne (markets, n_pages, n_errors).
       Le parametre 'cursor' suit la doc Kalshi v2 ; la premiere page brute
       est loggee pour verification ([RAW:markets_page])."""
    markets, cursor, pages, errors = [], None, 0, 0
    while pages < cfg.MAX_PAGES:
        params = {"status": status, "limit": cfg.PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        try:
            r = client._req("GET", "/markets", params=params)
        except Exception as e:                      # KalshiAPIError ou autre
            errors += 1
            log.error(f"Pagination page {pages + 1}: {e}")
            break
        pages += 1
        if pages == 1 and hasattr(client, "_log_raw_once"):
            client._log_raw_once("markets_page",
                                 {k: r.get(k) for k in ("cursor",)} |
                                 {"markets_sample": (r.get("markets") or [])[:1]})
        batch = r.get("markets", []) or []
        markets.extend(batch)
        cursor = r.get("cursor")
        if not cursor or not batch:
            break
    else:
        log.warning(f"Garde-fou MAX_PAGES={cfg.MAX_PAGES} atteint -- "
                    f"pagination interrompue (univers peut-etre incomplet).")
    return markets, pages, errors

# ── Scan complet + rapports ──────────────────────────────────────────────────

def _top(snaps: list, key: str, n: int = 20) -> list:
    rows = [s for s in snaps if getattr(s, key) is not None]
    rows.sort(key=lambda s: getattr(s, key), reverse=True)
    return [{"ticker": s.ticker, "title": s.title, "category": s.category,
             key: getattr(s, key)} for s in rows[:n]]

def run_scan(client, cfg: ScanConfig = SCFG, save: bool = True,
             now: Optional[datetime] = None) -> dict:
    """Scan complet. AUCUN ordre : ce module n'expose ni n'appelle aucune
    fonction d'envoi d'ordre (garanti par test statique)."""
    raw_markets, pages, errors = fetch_all_markets(client, cfg=cfg)
    snaps = [build_snapshot(m, now=now, cfg=cfg) for m in raw_markets]
    valid = [s for s in snaps if s.included]
    excl_by_reason = {}
    for s in snaps:
        if s.exclusion_reason:
            excl_by_reason[s.exclusion_reason] = \
                excl_by_reason.get(s.exclusion_reason, 0) + 1
    if errors:
        excl_by_reason["api_error_pages"] = errors
    by_cat = {}
    for s in snaps:
        by_cat[s.category] = by_cat.get(s.category, 0) + 1
    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_markets_received": len(raw_markets),
        "api_pages": pages,
        "api_error_pages": errors,
        "valid_markets": len(valid),
        "excluded_by_reason": excl_by_reason,
        "by_category": by_cat,
        "top20_volume": _top(valid, "volume"),
        "top20_liquidity": _top(valid, "liquidity"),
        "top20_open_interest": _top(valid, "open_interest"),
        "empty_book_markets": [s.ticker for s in snaps
                               if s.exclusion_reason == "no_liquidity"][:200],
        "filters": {k: getattr(cfg, k) for k in
                    ("MIN_MINUTES", "MAX_MINUTES", "MIN_VOLUME",
                     "MIN_OPEN_INTEREST", "MAX_SPREAD_CENTS",
                     "REQUIRE_LIQUIDITY", "ALLOWED_CATEGORIES",
                     "EXCLUDED_CATEGORIES")},
    }
    if save:
        _save_json(cfg.UNIVERSE_FILE,
                   [_snap_dict(s) for s in snaps])
        _save_json(cfg.REPORT_FILE, report)
        log.info(f"Univers: {len(snaps)} marches ({len(valid)} valides) -> "
                 f"{cfg.UNIVERSE_FILE} | rapport -> {cfg.REPORT_FILE}")
    return {"snapshots": snaps, "report": report}

def _snap_dict(s: MarketSnapshot) -> dict:
    d = asdict(s)
    d.pop("raw_market", None)          # l'univers reste lisible ; le brut
    return d                           # est disponible en memoire si besoin

def _save_json(path: str, data):
    """Ecriture atomique locale (tmp + replace) — sans dependre du moteur."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        log.error(f"Ecriture {path}: {e}")
