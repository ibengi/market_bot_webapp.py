#!/usr/bin/env python3
"""
KALSHI ALPHA ENGINE  --  v11.0-pro (2026-07-10)
================================================
Reconstruction professionnelle du moteur de trading.

PRINCIPES :
  - AUCUN dry-run, AUCUNE simulation : tous les ordres sont reellement
    envoyes a l'API Kalshi de l'environnement selectionne (demo ou prod).
  - Un trade n'est enregistre QUE s'il a reellement ete execute
    (fill verifie aupres de l'API), jamais sur simple envoi d'ordre.
  - Le stop-loss quotidien repose UNIQUEMENT sur le PnL REALISE.
  - Aucun edge fictif : pour la strategie de suivi de marche, l'edge
    est honnetement 0 (prob. modele = prob. marche) et affiche comme tel.
  - Persistance atomique + sauvegardes + checksums : aucun JSON corrompu.
  - Recuperation apres crash : ordres et positions reconcilies au demarrage.

ARCHITECTURE :
  Config, JsonStore, KalshiClient, FeeModel, TradeLogger, PositionManager,
  OrderManager, RiskManager, PositionSizer, MarketValidator, SignalValidator,
  StatsEngine, BtcStrategy, ExecutionEngine.

AVERTISSEMENTS D'INTEGRITE (a verifier lors des premiers ordres reels) :
  - Les noms exacts des champs de la reponse "ordre" de l'API Kalshi
    (compteur de fills, quantite restante, prix moyen) sont extraits de
    facon TOLERANTE (plusieurs noms candidats) car la documentation peut
    differer de la realite. La reponse brute est loggee en DEBUG.
  - La formule de frais (0.07 x C x P x (1-P), arrondi au cent sup.)
    doit etre verifiee contre le bareme officiel Kalshi.
  - L'environnement demo (demo-api.kalshi.co) peut ne pas lister les
    marches courants (constate le 2026-07-04) et requiert normalement
    des cles API DEMO distinctes (KALSHI_DEMO_KEY_ID / KALSHI_DEMO_PRIVATE_KEY).
"""

import os, sys, json, time, math, uuid, base64, hashlib, logging, argparse, shutil
from datetime import datetime, timezone, date
from urllib.parse import urlparse
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*a, **k): return False
import requests

load_dotenv()

ENGINE_VERSION = "v11.0-pro-2026-07-10"

# ══════════════════════════════════════════════════════════════════════════
# S1. CONFIGURATION (centralisee, surchargeables par variables d'env)
# ══════════════════════════════════════════════════════════════════════════

def _env_f(name, default): 
    try: return float(os.getenv(name, str(default)))
    except ValueError: return default
def _env_i(name, default):
    try: return int(os.getenv(name, str(default)))
    except ValueError: return default
def _env_b(name, default=True):
    return os.getenv(name, "1" if default else "0").strip().lower() not in ("0","false","no","non")

class Config:
    # Environnements
    PROD_URL  = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL  = "https://demo-api.kalshi.co/trade-api/v2"
    # Identifiants (prod) et identifiants demo distincts si fournis
    KEY_ID        = os.getenv("KALSHI_KEY_ID", "")
    PRIV_KEY      = os.getenv("KALSHI_PRIVATE_KEY", "").replace("\\n", "\n")
    DEMO_KEY_ID   = os.getenv("KALSHI_DEMO_KEY_ID", "")
    DEMO_PRIV_KEY = os.getenv("KALSHI_DEMO_PRIVATE_KEY", "").replace("\\n", "\n")
    # Strategie / protections
    SERIES            = os.getenv("BTC_SERIES", "KXBTC15M")
    MAX_ENTRY_CENTS   = _env_i("MAX_ENTRY_CENTS", 85)
    ONE_TRADE_PER_MKT = _env_b("ONE_TRADE_PER_MARKET", True)
    MIN_MINUTES       = _env_f("BTC_MIN_MINUTES", 5.0)
    MAX_SPREAD_PAY    = _env_i("MAX_SPREAD_PAY", 5)
    # Risque
    MAX_DAILY_LOSS    = _env_f("MAX_DAILY_LOSS", 50.0)      # $ PnL REALISE
    MAX_TRADES_CYCLE  = _env_i("MAX_TRADES_CYCLE", 3)
    MAX_POS_PCT       = _env_f("MAX_POSITION_PCT", 1.0)     # % capital / position (plafond dur)
    RISK_BUDGET_PCT   = _env_f("RISK_BUDGET_PCT", 5.0)      # % capital en risque ouvert total
    DD_THROTTLE_PCT   = _env_f("DD_THROTTLE_PCT", 10.0)     # au-dela: taille /2
    # Ordres
    ORDER_TTL_SECONDS = _env_i("ORDER_TTL_SECONDS", 45)
    ORDER_POLL_START  = 1.0
    ORDER_POLL_MAX    = 5.0
    # Frais (A VERIFIER contre le bareme officiel Kalshi)
    FEE_RATE          = _env_f("KALSHI_FEE_RATE_TRADING", 0.07)
    # Fichiers
    DATA_DIR    = os.getenv("DATA_DIR", ".")
    TRADES_FILE     = "kalshi_trades.json"
    POSITIONS_FILE  = "positions_state.json"
    ORDERS_FILE     = "orders_state.json"
    RISK_FILE       = "risk_state.json"
    CURVE_FILE      = "capital_curve.json"
    REPORT_DIR      = "reports"
    BACKUPS         = 3

CFG = Config()

def _p(name: str) -> str:
    return os.path.join(CFG.DATA_DIR, name)

# ══════════════════════════════════════════════════════════════════════════
# S2. LOGGING (canaux BOT / API / TRADE / RISK / POSITION / STATS)
# ══════════════════════════════════════════════════════════════════════════

_FMT = "%(asctime)s  %(levelname)-7s [%(name)s] %(message)s"
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format=_FMT, datefmt="%Y-%m-%d %H:%M:%S")
log      = logging.getLogger("BOT")
log_api  = logging.getLogger("API")
log_trd  = logging.getLogger("TRADE")
log_rsk  = logging.getLogger("RISK")
log_pos  = logging.getLogger("POSITION")
log_sts  = logging.getLogger("STATS")

# ══════════════════════════════════════════════════════════════════════════
# S3. PERSISTANCE ATOMIQUE (tmp + os.replace, checksum, rotation de backups)
# ══════════════════════════════════════════════════════════════════════════

class JsonStore:
    """Ecriture atomique, checksum sha256, rotation de sauvegardes,
    et lecture avec reprise automatique sur backup en cas de corruption."""

    @staticmethod
    def _sha(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def save(cls, path: str, data) -> bool:
        try:
            payload = json.dumps(data, indent=1, ensure_ascii=False).encode()
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(payload); f.flush(); os.fsync(f.fileno())
            # rotation des backups AVANT remplacement
            if os.path.exists(path):
                for i in range(CFG.BACKUPS - 1, 0, -1):
                    src, dst = f"{path}.bak{i}", f"{path}.bak{i+1}"
                    if os.path.exists(src): shutil.copy2(src, dst)
                shutil.copy2(path, f"{path}.bak1")
            os.replace(tmp, path)
            with open(path + ".sha256", "w") as f:
                f.write(cls._sha(payload))
            return True
        except Exception as e:
            log.error(f"JsonStore.save({path}): {e}")
            return False

    @classmethod
    def load(cls, path: str, default):
        candidates = [path] + [f"{path}.bak{i}" for i in range(1, CFG.BACKUPS + 1)]
        for cand in candidates:
            if not os.path.exists(cand):
                continue
            try:
                raw = open(cand, "rb").read()
                data = json.loads(raw.decode())
                if cand == path and os.path.exists(path + ".sha256"):
                    want = open(path + ".sha256").read().strip()
                    if want and want != cls._sha(raw):
                        log.warning(f"JsonStore: checksum invalide pour {path} "
                                    f"-- tentative sur backup.")
                        continue
                if cand != path:
                    log.warning(f"JsonStore: {path} corrompu/absent -- "
                                f"recupere depuis {cand}.")
                return data
            except Exception:
                continue
        return default

# ══════════════════════════════════════════════════════════════════════════
# S4. ERREURS API + RETRY/BACKOFF
# ══════════════════════════════════════════════════════════════════════════

class KalshiAPIError(Exception):
    def __init__(self, status: int, message: str, body: str = ""):
        self.status, self.body = status, body[:500]
        super().__init__(f"HTTP {status}: {message}")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

def pick(d: dict, *names, default=None):
    """Extraction tolerante : retourne la premiere cle presente et non nulle."""
    for n in names:
        if isinstance(d, dict) and d.get(n) is not None:
            return d[n]
    return default

def pick_int(d: dict, *names, default=0) -> int:
    v = pick(d, *names, default=None)
    try:    return int(float(v))
    except (TypeError, ValueError): return default

# ══════════════════════════════════════════════════════════════════════════
# S5. CLIENT KALSHI (env demo/prod, signature RSA, retry/backoff)
# ══════════════════════════════════════════════════════════════════════════

class KalshiClient:
    """Client HTTP signe. env='demo' -> demo-api (cles demo si fournies),
    env='prod' -> production. TOUT (donnees, ordres, reglements) passe par
    le MEME environnement, condition de coherence d'un vrai broker."""

    def __init__(self, env: str = "demo"):
        self.env      = env
        self.base_url = CFG.DEMO_URL if env == "demo" else CFG.PROD_URL
        if env == "demo" and CFG.DEMO_KEY_ID:
            self.key_id, key_pem = CFG.DEMO_KEY_ID, CFG.DEMO_PRIV_KEY
            self.cred_src = "cles DEMO dediees"
        else:
            self.key_id, key_pem = CFG.KEY_ID, CFG.PRIV_KEY
            self.cred_src = ("cles PROD (ATTENTION: l'environnement demo requiert "
                             "normalement des cles demo distinctes -- a verifier)"
                             if env == "demo" else "cles PROD")
        self.session = requests.Session()
        self._pk = self._load_key(key_pem)
        self._raw_logged = set()   # types de reponses deja loggees en brut

    # -- Signature ----------------------------------------------------------
    def _load_key(self, key_pem: str):
        try:
            from cryptography.hazmat.primitives import serialization
            key_text = key_pem.strip()
            if not key_text.startswith("-----"):
                log_api.warning("Cle privee absente ou non-PEM -- requetes non signees.")
                return None
            return serialization.load_pem_private_key(key_text.encode(), password=None)
        except Exception as e:
            log_api.warning(f"Chargement cle RSA impossible: {e}")
            return None

    def _sign_headers(self, method: str, url: str) -> dict:
        if not self._pk or not self.key_id:
            return {"Content-Type": "application/json"}
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts  = str(int(time.time() * 1000))
        msg = f"{ts}{method.upper()}{urlparse(url).path}".encode()
        sig = self._pk.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "Content-Type":            "application/json",
            "KALSHI-ACCESS-KEY":       self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    # -- Requete avec retry/backoff ------------------------------------------
    def _req(self, method: str, path: str, *, retries: int = 3, **kw) -> dict:
        url = self.base_url + path
        attempt, delay = 0, 1.0
        while True:
            attempt += 1
            try:
                r = self.session.request(method.upper(), url,
                                         headers=self._sign_headers(method, url),
                                         timeout=15, **kw)
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt > retries:
                    raise KalshiAPIError(0, f"reseau: {e}")
                log_api.warning(f"{method} {path}: {type(e).__name__} -- "
                                f"retry {attempt}/{retries} dans {delay:.0f}s")
                time.sleep(delay); delay = min(delay * 2, 8); continue

            if r.status_code in RETRYABLE_STATUS and attempt <= retries:
                wait = delay
                if r.status_code == 429:
                    try: wait = max(wait, float(r.headers.get("Retry-After", delay)))
                    except ValueError: pass
                log_api.warning(f"{method} {path}: HTTP {r.status_code} -- "
                                f"retry {attempt}/{retries} dans {wait:.0f}s")
                time.sleep(wait); delay = min(delay * 2, 8); continue

            if r.status_code >= 400:
                raise KalshiAPIError(r.status_code, f"{method} {path}", r.text)

            try:
                return r.json() if r.text.strip() else {}
            except ValueError:
                raise KalshiAPIError(r.status_code, f"{method} {path}: JSON invalide", r.text)

    def _log_raw_once(self, kind: str, payload: dict):
        """Logge UNE FOIS la reponse brute de chaque type d'appel critique,
        pour verifier les noms de champs reels de l'API."""
        if kind not in self._raw_logged:
            self._raw_logged.add(kind)
            log_api.info(f"[RAW:{kind}] {json.dumps(payload, ensure_ascii=False)[:800]}")

    # -- Endpoints -----------------------------------------------------------
    def get_markets(self, series: str, status: str = "open", limit: int = 50) -> list:
        try:
            r = self._req("GET", "/markets",
                          params={"series_ticker": series, "status": status, "limit": limit})
            return r.get("markets", []) or []
        except KalshiAPIError as e:
            log_api.error(f"get_markets({series}): {e}")
            return []

    def get_market(self, ticker: str) -> dict:
        try:
            r = self._req("GET", f"/markets/{ticker}")
            return r.get("market", r) or {}
        except KalshiAPIError as e:
            log_api.warning(f"get_market({ticker}): {e}")
            return {}

    def get_balance(self) -> Optional[float]:
        """Solde du compte en $. Champ 'balance' attendu en cents (a verifier)."""
        try:
            r = self._req("GET", "/portfolio/balance")
            self._log_raw_once("balance", r)
            cents = pick_int(r, "balance", "available_balance", default=-1)
            return cents / 100.0 if cents >= 0 else None
        except KalshiAPIError as e:
            log_api.warning(f"get_balance: {e}")
            return None

    def create_order(self, ticker: str, side: str, count: int, price_cents: int) -> dict:
        payload = {
            "ticker":          ticker,
            "client_order_id": f"alpha_{uuid.uuid4().hex}",
            "action":          "buy",
            "side":            side,                    # "yes" | "no"
            "type":            "limit",
            "count":           int(count),
        }
        payload["yes_price" if side == "yes" else "no_price"] = int(price_cents)
        r = self._req("POST", "/portfolio/orders", json=payload)
        self._log_raw_once("create_order", r)
        return r.get("order", r) or {}

    def get_order(self, order_id: str) -> dict:
        r = self._req("GET", f"/portfolio/orders/{order_id}")
        self._log_raw_once("get_order", r)
        return r.get("order", r) or {}

    def cancel_order(self, order_id: str) -> dict:
        try:
            r = self._req("DELETE", f"/portfolio/orders/{order_id}")
            return r.get("order", r) or {}
        except KalshiAPIError as e:
            # 404/410 = deja execute ou deja annule : pas une erreur fatale.
            log_api.warning(f"cancel_order({order_id}): {e}")
            return {}

    def get_fills(self, order_id: str) -> list:
        try:
            r = self._req("GET", "/portfolio/fills", params={"order_id": order_id})
            self._log_raw_once("fills", r)
            return r.get("fills", []) or []
        except KalshiAPIError as e:
            log_api.warning(f"get_fills({order_id}): {e}")
            return []

# ══════════════════════════════════════════════════════════════════════════
# S6. MODELE DE FRAIS
# ══════════════════════════════════════════════════════════════════════════

class FeeModel:
    """Frais de trading Kalshi. Formule usuelle : 0.07 x C x P x (1-P),
    arrondie au cent superieur. A VERIFIER contre le bareme officiel --
    taux reglable via KALSHI_FEE_RATE_TRADING sans toucher au code."""
    @staticmethod
    def trading_fee(count: int, price_cents: int) -> float:
        p = max(1, min(99, price_cents)) / 100.0
        return math.ceil(CFG.FEE_RATE * count * p * (1 - p) * 100) / 100.0

# ══════════════════════════════════════════════════════════════════════════
# S7. TRADE LOGGER (journal complet, un enregistrement = un trade REEL)
# ══════════════════════════════════════════════════════════════════════════

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

class TradeLogger:
    """Journal des trades. REGLE ABSOLUE : on n'enregistre un trade que si
    filled_count > 0 (execution verifiee). Les anciens enregistrements
    'dry_run' sont archives a part, jamais melanges aux vrais."""

    SCHEMA = "v11"

    def __init__(self):
        self.path = _p(CFG.TRADES_FILE)
        raw = JsonStore.load(self.path, [])
        legacy = [t for t in raw if t.get("schema") != self.SCHEMA]
        self.trades = [t for t in raw if t.get("schema") == self.SCHEMA]
        if legacy:
            legacy_path = _p("kalshi_trades_legacy.json")
            old = JsonStore.load(legacy_path, [])
            JsonStore.save(legacy_path, old + legacy)
            JsonStore.save(self.path, self.trades)
            log_trd.warning(f"{len(legacy)} enregistrement(s) heritee(s) "
                            f"(dry-run/ancien schema) archives dans "
                            f"kalshi_trades_legacy.json -- exclus des statistiques.")

    def open_trade(self, *, ticker, market_title, side, req_price, avg_price,
                   req_count, filled_count, spread, fees, edge, ev, confidence,
                   grade, reason, analysis, order_id, order_status) -> dict:
        rec = {
            "schema": self.SCHEMA, "trade_id": uuid.uuid4().hex[:12],
            "timestamp": now_iso(), "ticker": ticker, "market": market_title,
            "side": side, "requested_price": req_price, "avg_fill_price": avg_price,
            "requested_count": req_count, "filled_count": filled_count,
            "spread": spread, "fees": round(fees, 2), "edge": edge, "ev": ev,
            "confidence": confidence, "grade": grade, "reason": reason,
            "analysis": analysis, "order_id": order_id, "order_status": order_status,
            "state": "open",       # open -> settled | expired
            "result": None, "won": None,
            "gross_pnl": None, "net_pnl": None, "roi": None,
            "holding_seconds": None, "settled_at": None,
        }
        self.trades.append(rec)
        self.flush()
        log_trd.info(f"OUVERT {ticker} {side.upper()} {filled_count}/{req_count} "
                     f"@ {avg_price}c (frais {fees:.2f}$) ordre={order_id}")
        return rec

    def settle_trade(self, trade_id: str, result: str, won: bool,
                     gross_pnl: float, net_pnl: float):
        for t in self.trades:
            if t["trade_id"] == trade_id:
                opened = datetime.fromisoformat(t["timestamp"])
                t.update({
                    "state": "settled", "result": result, "won": won,
                    "gross_pnl": round(gross_pnl, 2), "net_pnl": round(net_pnl, 2),
                    "roi": round(net_pnl / max(0.01, t["avg_fill_price"] / 100.0
                                               * t["filled_count"]), 4),
                    "settled_at": now_iso(),
                    "holding_seconds": int((datetime.now(timezone.utc) - opened)
                                           .total_seconds()),
                })
                self.flush()
                log_trd.info(f"REGLE  {t['ticker']} -> {result.upper()} | "
                             f"{'GAGNE' if won else 'PERDU'} | net {net_pnl:+.2f}$")
                return t
        log_trd.error(f"settle_trade: trade_id {trade_id} introuvable.")
        return None

    def has_open_on(self, ticker: str) -> bool:
        return any(t["ticker"] == ticker and t["state"] == "open" for t in self.trades)

    def open_trades(self) -> list:
        return [t for t in self.trades if t["state"] == "open"]

    def settled_trades(self) -> list:
        return [t for t in self.trades if t["state"] == "settled"]

    def flush(self):
        JsonStore.save(self.path, self.trades)

# ══════════════════════════════════════════════════════════════════════════
# S8. POSITION MANAGER (positions ouvertes, reglement, PnL realise)
# ══════════════════════════════════════════════════════════════════════════

class PositionManager:
    def __init__(self, client: KalshiClient, trade_log: TradeLogger):
        self.client, self.tlog = client, trade_log
        self.positions = JsonStore.load(_p(CFG.POSITIONS_FILE), {})   # ticker -> pos

    def flush(self):
        JsonStore.save(_p(CFG.POSITIONS_FILE), self.positions)

    def open_position(self, trade: dict):
        self.positions[trade["ticker"]] = {
            "trade_id": trade["trade_id"], "side": trade["side"],
            "count": trade["filled_count"], "avg_price": trade["avg_fill_price"],
            "fees": trade["fees"], "opened_at": trade["timestamp"],
        }
        self.flush()
        log_pos.info(f"{trade['ticker']}: {trade['side'].upper()} "
                     f"x{trade['filled_count']} @ {trade['avg_fill_price']}c")

    def open_risk(self) -> float:
        """Capital en risque = cout total des positions ouvertes (perte max)."""
        return sum(p["count"] * p["avg_price"] / 100.0 for p in self.positions.values())

    def unrealized_pnl(self, mid_price_lookup=None) -> float:
        """PnL latent estime au prix mid courant (0 si donnee indisponible)."""
        total = 0.0
        for tk, p in self.positions.items():
            if not mid_price_lookup: continue
            mid = mid_price_lookup(tk, p["side"])
            if mid is None: continue
            total += p["count"] * (mid - p["avg_price"]) / 100.0
        return total

    def check_settlements(self) -> list:
        """Interroge l'API pour les marches regles ; realise le PnL.
        Champs 'result'/'status' extraits de facon tolerante."""
        realized = []
        for ticker in list(self.positions.keys()):
            m = self.client.get_market(ticker)
            if not m: continue
            result = str(pick(m, "result", default="") or "").lower()
            status = str(pick(m, "status", default="") or "").lower()
            if result not in ("yes", "no"):
                if status in ("settled", "finalized"):
                    log_pos.warning(f"{ticker}: statut '{status}' mais champ "
                                    f"'result' illisible -- verifier le schema API.")
                continue
            p    = self.positions.pop(ticker)
            won  = (result == p["side"])
            cost = p["count"] * p["avg_price"] / 100.0
            gross = (p["count"] * 1.0 - cost) if won else -cost
            net   = gross - p["fees"]
            self.flush()
            t = self.tlog.settle_trade(p["trade_id"], result, won, gross, net)
            if t: realized.append(t)
        return realized

    def reconcile_startup(self):
        """Apres crash/redemarrage : les positions persistees restent valides
        (elles vivent chez le broker) ; on les re-verifie au prochain cycle."""
        if self.positions:
            log_pos.info(f"Recovery: {len(self.positions)} position(s) ouverte(s) "
                         f"rechargee(s): {', '.join(self.positions)}")

# ══════════════════════════════════════════════════════════════════════════
# S9. ORDER MANAGER (placement, surveillance, TTL, fills partiels, recovery)
# ══════════════════════════════════════════════════════════════════════════

class ExecutionResult:
    def __init__(self, order_id, requested, filled, avg_price, status, state):
        self.order_id, self.requested, self.filled = order_id, requested, filled
        self.avg_price, self.status, self.state = avg_price, status, state
        # state: "filled" | "partial" | "cancelled" | "rejected" | "unknown"

class OrderManager:
    TERMINAL = {"executed", "canceled", "cancelled", "expired", "filled"}

    def __init__(self, client: KalshiClient):
        self.client = client
        self.open_orders = JsonStore.load(_p(CFG.ORDERS_FILE), {})  # id -> meta

    def flush(self):
        JsonStore.save(_p(CFG.ORDERS_FILE), self.open_orders)

    # -- extraction tolerante de l'etat d'un ordre ---------------------------
    @staticmethod
    def _extract(order: dict, requested: int):
        status = str(pick(order, "status", "order_status", default="") or "").lower()
        filled = pick_int(order, "taker_fill_count", "fill_count",
                          "filled_count", "filled_quantity", default=-1)
        remaining = pick_int(order, "remaining_count", "remaining_quantity", default=-1)
        if filled < 0 and remaining >= 0:
            filled = max(0, requested - remaining)
        if filled < 0:
            filled = requested if status in ("executed", "filled") else 0
        return status, min(filled, requested)

    def _avg_fill_price(self, order_id: str, side: str, fallback: int) -> int:
        fills = self.client.get_fills(order_id)
        tot_c, tot_px = 0, 0
        for f in fills:
            c  = pick_int(f, "count", "quantity", default=0)
            px = pick_int(f, f"{side}_price", "price", default=0)
            if c > 0 and 1 <= px <= 99:
                tot_c += c; tot_px += c * px
        if tot_c > 0:
            return round(tot_px / tot_c)
        log_api.warning(f"Fills indisponibles pour {order_id} -- prix moyen "
                        f"suppose = prix limite ({fallback}c).")
        return fallback

    # -- cycle de vie complet d'un ordre --------------------------------------
    def place_and_track(self, ticker: str, side: str, count: int,
                        limit_cents: int) -> ExecutionResult:
        try:
            order = self.client.create_order(ticker, side, count, limit_cents)
        except KalshiAPIError as e:
            log_api.error(f"ORDRE REFUSE {ticker} {side} x{count} @ {limit_cents}c "
                          f"-> {e} | corps: {e.body}")
            return ExecutionResult(None, count, 0, limit_cents, f"rejected:{e.status}",
                                   "rejected")
        order_id = str(pick(order, "order_id", "id", default="") or "")
        if not order_id:
            log_api.error(f"Reponse d'ordre sans identifiant -- trade NON enregistre. "
                          f"Reponse: {json.dumps(order)[:300]}")
            return ExecutionResult(None, count, 0, limit_cents, "no_id", "rejected")

        self.open_orders[order_id] = {"ticker": ticker, "side": side,
                                      "count": count, "price": limit_cents,
                                      "placed_at": now_iso()}
        self.flush()

        deadline, delay = time.time() + CFG.ORDER_TTL_SECONDS, CFG.ORDER_POLL_START
        status, filled = self._extract(order, count)
        while time.time() < deadline and status not in self.TERMINAL and filled < count:
            time.sleep(delay); delay = min(delay + 1, CFG.ORDER_POLL_MAX)
            try:
                order = self.client.get_order(order_id)
                status, filled = self._extract(order, count)
            except KalshiAPIError as e:
                log_api.warning(f"Suivi ordre {order_id}: {e}")

        if status not in self.TERMINAL and filled < count:
            log_api.info(f"TTL {CFG.ORDER_TTL_SECONDS}s atteint -- annulation "
                         f"de l'ordre {order_id} (rempli {filled}/{count}).")
            self.client.cancel_order(order_id)
            try:
                order = self.client.get_order(order_id)
                status, filled = self._extract(order, count)
            except KalshiAPIError:
                pass

        self.open_orders.pop(order_id, None); self.flush()

        if filled <= 0:
            return ExecutionResult(order_id, count, 0, limit_cents,
                                   status or "unfilled", "cancelled")
        avg = self._avg_fill_price(order_id, side, limit_cents)
        state = "filled" if filled >= count else "partial"
        return ExecutionResult(order_id, count, filled, avg, status or "executed", state)

    def reconcile_startup(self, tlog: TradeLogger, posmgr: PositionManager):
        """Recovery apres crash : verifier chaque ordre laisse ouvert."""
        if not self.open_orders: return
        log_api.warning(f"Recovery: {len(self.open_orders)} ordre(s) non conclu(s) "
                        f"au dernier arret -- reconciliation...")
        for oid, meta in list(self.open_orders.items()):
            try:
                order = self.client.get_order(oid)
                status, filled = self._extract(order, meta["count"])
                if status not in self.TERMINAL:
                    self.client.cancel_order(oid)
                    order = self.client.get_order(oid)
                    status, filled = self._extract(order, meta["count"])
                if filled > 0 and not tlog.has_open_on(meta["ticker"]):
                    avg = self._avg_fill_price(oid, meta["side"], meta["price"])
                    fees = FeeModel.trading_fee(filled, avg)
                    t = tlog.open_trade(
                        ticker=meta["ticker"], market_title="(recovery)",
                        side=meta["side"], req_price=meta["price"], avg_price=avg,
                        req_count=meta["count"], filled_count=filled,
                        spread=None, fees=fees, edge=0.0, ev=0.0, confidence=0,
                        grade="R", reason="ordre recupere apres crash",
                        analysis={}, order_id=oid, order_status=status)
                    posmgr.open_position(t)
            except KalshiAPIError as e:
                log_api.error(f"Recovery ordre {oid}: {e}")
            finally:
                self.open_orders.pop(oid, None)
        self.flush()

# ══════════════════════════════════════════════════════════════════════════
# S10. RISK MANAGER (base sur le PnL REALISE, jamais le capital investi)
# ══════════════════════════════════════════════════════════════════════════

class RiskManager:
    def __init__(self, tlog: TradeLogger, posmgr: PositionManager, capital: float):
        self.tlog, self.posmgr, self.capital = tlog, posmgr, capital
        st = JsonStore.load(_p(CFG.RISK_FILE), {})
        today = date.today().isoformat()
        if st.get("date") != today:
            st = {"date": today}
        self.state = st
        self.flush()

    def flush(self):
        JsonStore.save(_p(CFG.RISK_FILE), self.state)

    # -- agregats jour (recalcules depuis le journal : source de verite unique)
    def _today_settled(self) -> list:
        today = date.today().isoformat()
        return [t for t in self.tlog.settled_trades()
                if (t.get("settled_at") or "").startswith(today)]

    def daily_realized_pnl(self) -> float:
        return sum(t["net_pnl"] for t in self._today_settled())

    def daily_realized_loss(self) -> float:
        return sum(t["net_pnl"] for t in self._today_settled() if t["net_pnl"] < 0)

    def daily_realized_profit(self) -> float:
        return sum(t["net_pnl"] for t in self._today_settled() if t["net_pnl"] > 0)

    def trades_today(self) -> int:
        today = date.today().isoformat()
        return sum(1 for t in self.tlog.trades
                   if t["timestamp"].startswith(today))

    def rolling_drawdown(self) -> float:
        """Drawdown courant de la courbe de capital (PnL net cumule)."""
        curve, peak, dd = 0.0, 0.0, 0.0
        for t in self.tlog.settled_trades():
            curve += t["net_pnl"]; peak = max(peak, curve)
            dd = max(dd, peak - curve)
        cur_dd = peak - curve
        return cur_dd

    # -- portes de risque ------------------------------------------------------
    def can_trade(self, cycle_trades: int) -> (bool, str):
        pnl = self.daily_realized_pnl()
        if pnl <= -CFG.MAX_DAILY_LOSS:
            return False, (f"STOP JOURNALIER: PnL realise {pnl:+.2f}$ <= "
                           f"-{CFG.MAX_DAILY_LOSS:.2f}$")
        if cycle_trades >= CFG.MAX_TRADES_CYCLE:
            return False, f"max {CFG.MAX_TRADES_CYCLE} trades/cycle atteint"
        open_risk = self.posmgr.open_risk()
        budget    = self.capital * CFG.RISK_BUDGET_PCT / 100.0
        if open_risk >= budget:
            return False, (f"budget de risque ouvert atteint "
                           f"({open_risk:.2f}$ >= {budget:.2f}$)")
        return True, ""

    def snapshot(self) -> dict:
        settled = self.tlog.settled_trades()
        wins    = [t for t in settled if t["won"]]
        losses  = [t for t in settled if not t["won"]]
        gp = sum(t["net_pnl"] for t in wins)
        gl = -sum(t["net_pnl"] for t in losses)
        return {
            "capital_deployed":      round(self.posmgr.open_risk(), 2),
            "open_risk":             round(self.posmgr.open_risk(), 2),
            "realized_pnl":          round(sum(t["net_pnl"] for t in settled), 2),
            "unrealized_pnl":        round(self.posmgr.unrealized_pnl(), 2),
            "daily_realized_pnl":    round(self.daily_realized_pnl(), 2),
            "daily_realized_loss":   round(self.daily_realized_loss(), 2),
            "daily_realized_profit": round(self.daily_realized_profit(), 2),
            "gross_pnl":             round(sum(t["gross_pnl"] for t in settled), 2),
            "net_pnl":               round(sum(t["net_pnl"] for t in settled), 2),
            "fees_paid":             round(sum(t["fees"] for t in self.tlog.trades), 2),
            "win_rate":  round(len(wins) / len(settled), 4) if settled else 0.0,
            "profit_factor": round(gp / gl, 3) if gl > 0 else None,
            "rolling_drawdown": round(self.rolling_drawdown(), 2),
        }

# ══════════════════════════════════════════════════════════════════════════
# S11. STATS ENGINE (statistiques completes + rapports periodiques)
# ══════════════════════════════════════════════════════════════════════════

class StatsEngine:
    def __init__(self, tlog: TradeLogger):
        self.tlog = tlog
        self._last_report_day = None

    def compute(self) -> dict:
        settled = self.tlog.settled_trades()
        if not settled:
            return {"n": 0}
        rets   = [t["net_pnl"] for t in settled]
        wins   = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        n      = len(rets)
        wr     = len(wins) / n
        avg_w  = sum(wins) / len(wins) if wins else 0.0
        avg_l  = sum(losses) / len(losses) if losses else 0.0
        mean   = sum(rets) / n
        var    = sum((r - mean) ** 2 for r in rets) / n if n > 1 else 0.0
        std    = math.sqrt(var)
        downs  = [min(0.0, r) for r in rets]
        dvar   = sum(d ** 2 for d in downs) / n
        dstd   = math.sqrt(dvar)
        # Sharpe/Sortino PAR TRADE (pas annualises -- interpretation relative
        # uniquement ; l'annualisation sur du 15-min serait trompeuse).
        sharpe  = mean / std  if std  > 0 else None
        sortino = mean / dstd if dstd > 0 else None
        gp, gl  = sum(wins), -sum(losses)
        # Kelly (formule classique ; valable seulement si WR/gains stables)
        kelly = None
        if avg_w > 0 and avg_l < 0:
            b = avg_w / abs(avg_l)
            kelly = round(wr - (1 - wr) / b, 4)
        curve, peak, maxdd = 0.0, 0.0, 0.0
        curve_points = []
        for t in settled:
            curve += t["net_pnl"]; peak = max(peak, curve)
            maxdd = max(maxdd, peak - curve)
            curve_points.append({"t": t["settled_at"], "equity": round(curve, 2)})
        JsonStore.save(_p(CFG.CURVE_FILE), curve_points)
        durations = [t["holding_seconds"] for t in settled if t.get("holding_seconds")]
        return {
            "n": n, "win_rate": round(wr, 4),
            "profit_factor": round(gp / gl, 3) if gl > 0 else None,
            "expectancy": round(mean, 3),
            "sharpe_per_trade": round(sharpe, 3) if sharpe is not None else None,
            "sortino_per_trade": round(sortino, 3) if sortino is not None else None,
            "kelly_fraction": kelly,
            "average_win": round(avg_w, 2), "average_loss": round(avg_l, 2),
            "largest_win": round(max(rets), 2), "largest_loss": round(min(rets), 2),
            "max_drawdown": round(maxdd, 2),
            "net_pnl": round(sum(rets), 2),
            "avg_duration_s": round(sum(durations) / len(durations)) if durations else None,
        }

    def _period_report(self, label: str, day_filter) -> dict:
        settled = [t for t in self.tlog.settled_trades()
                   if day_filter((t.get("settled_at") or "")[:10])]
        pnl = sum(t["net_pnl"] for t in settled)
        wins = sum(1 for t in settled if t["won"])
        return {"period": label, "trades": len(settled), "wins": wins,
                "net_pnl": round(pnl, 2),
                "win_rate": round(wins / len(settled), 4) if settled else 0.0}

    def maybe_daily_report(self):
        """A chaque changement de jour UTC : rapport quotidien (+ hebdo le
        lundi, + mensuel le 1er), ecrit dans reports/ et logge."""
        today = date.today()
        if self._last_report_day == today:
            return
        prev = self._last_report_day
        self._last_report_day = today
        if prev is None:
            return
        os.makedirs(_p(CFG.REPORT_DIR), exist_ok=True)
        day = prev.isoformat()
        rep = {"generated": now_iso(),
               "daily":  self._period_report(day, lambda d: d == day),
               "global": self.compute()}
        if today.weekday() == 0:
            rep["weekly"] = self._period_report(
                "7 derniers jours",
                lambda d: d >= (today.toordinal() - 7 and
                                date.fromordinal(today.toordinal() - 7).isoformat()))
        if today.day == 1:
            rep["monthly"] = self._period_report(
                prev.strftime("%Y-%m"), lambda d: d.startswith(prev.strftime("%Y-%m")))
        JsonStore.save(os.path.join(_p(CFG.REPORT_DIR), f"report_{day}.json"), rep)
        log_sts.info(f"Rapport {day}: {json.dumps(rep['daily'], ensure_ascii=False)}")

    def log_summary(self):
        s = self.compute()
        if s.get("n"):
            log_sts.info(f"BILAN: n={s['n']} WR={s['win_rate']:.1%} "
                         f"PnL={s['net_pnl']:+.2f}$ PF={s['profit_factor']} "
                         f"Exp={s['expectancy']}$ MaxDD={s['max_drawdown']}$")

# ══════════════════════════════════════════════════════════════════════════
# S12. POSITION SIZER (plafond dur 1% du capital, ajuste au contexte)
# ══════════════════════════════════════════════════════════════════════════

class PositionSizer:
    @staticmethod
    def contracts(capital: float, price_cents: int, taille_str: str,
                  confidence: int, drawdown: float, open_risk: float) -> int:
        base_pct = {"0.5%": 0.5, "1%": 1.0, "2%": 2.0}.get(taille_str)
        if base_pct is None or price_cents <= 0:
            return 0
        pct = min(base_pct, CFG.MAX_POS_PCT)              # plafond dur 1%
        if confidence <= 4:
            pct *= 0.5                                     # signal faible
        if capital > 0 and drawdown / capital * 100.0 >= CFG.DD_THROTTLE_PCT:
            pct *= 0.5                                     # drawdown eleve
            log_rsk.info(f"Sizer: drawdown {drawdown:.2f}$ >= "
                         f"{CFG.DD_THROTTLE_PCT:g}% du capital -- taille reduite.")
        budget_left = capital * CFG.RISK_BUDGET_PCT / 100.0 - open_risk
        alloc = min(capital * pct / 100.0, max(0.0, budget_left))
        return max(0, int(alloc / (price_cents / 100.0)))

# ══════════════════════════════════════════════════════════════════════════
# S13. VALIDATEURS (marche + signal)
# ══════════════════════════════════════════════════════════════════════════

class MarketValidator:
    @staticmethod
    def normalize_book(m: dict) -> Optional[dict]:
        """Carnet coherent ou None. Derive le cote NO du cote YES si absent."""
        def cents(*names):
            v = pick(m, *names, default=None)
            try: return int(round(float(v))) if v is not None else None
            except (TypeError, ValueError): return None
        yb, ya = cents("yes_bid"), cents("yes_ask")
        nb, na = cents("no_bid"),  cents("no_ask")
        if nb is None and ya is not None: nb = 100 - ya
        if na is None and yb is not None: na = 100 - yb
        if yb is None and na is not None: yb = 100 - na
        if ya is None and nb is not None: ya = 100 - nb
        if yb is None or ya is None:
            return None
        clamp = lambda x: max(1, min(99, int(x)))
        yb, ya, nb, na = clamp(yb), clamp(ya), clamp(nb or 50), clamp(na or 50)
        if ya < yb or na < nb:
            return None
        mid = round((yb + ya) / 2)
        if abs((mid + (100 - mid)) - 100) > 0:      # invariant par construction
            return None
        return {"yes_bid": yb, "yes_ask": ya, "no_bid": nb, "no_ask": na,
                "yes_mid": mid, "no_mid": 100 - mid,
                "spread": ya - yb}

class SignalValidator:
    @staticmethod
    def check(verdict: str, entry_price: int, ticker: str,
              tlog: TradeLogger, posmgr: PositionManager) -> (bool, str):
        if verdict not in ("ACHETER YES", "ACHETER NO"):
            return False, "aucun signal"
        if entry_price > CFG.MAX_ENTRY_CENTS:
            return False, (f"prix d'entree {entry_price}c > plafond "
                           f"{CFG.MAX_ENTRY_CENTS}c (ratio risque/gain)")
        if entry_price < 1 or entry_price > 99:
            return False, f"prix d'entree invalide: {entry_price}c"
        if CFG.ONE_TRADE_PER_MKT and (ticker in posmgr.positions
                                      or tlog.has_open_on(ticker)):
            return False, "position deja prise sur ce marche (1 trade/marche)"
        return True, ""

# ══════════════════════════════════════════════════════════════════════════
# S14. STRATEGIE BTC (btc_context inchange -- edge HONNETE = 0)
# ══════════════════════════════════════════════════════════════════════════

try:
    from btc_context import evaluate_btc_trade, get_btc_price
    try:
        from btc_context import VERSION as BTC_CTX_VERSION
    except ImportError:
        BTC_CTX_VERSION = "inconnue"
    BTC_AVAILABLE = True
except ImportError as e:
    BTC_AVAILABLE, BTC_CTX_VERSION = False, "absente"
    log.error(f"btc_context indisponible: {e}")

class BtcStrategy:
    """Selection du marche ATM + decision par btc_context.
    Probabilite modele = probabilite marche (strategie de suivi) donc
    edge = 0.0, affiche et enregistre comme tel. Aucune valeur fictive."""

    def __init__(self, client: KalshiClient):
        self.client = client

    def _select_market(self):
        markets = self.client.get_markets(CFG.SERIES, status="open", limit=50)
        if not markets:
            log.warning(f"Aucun marche '{CFG.SERIES}' renvoye par "
                        f"l'environnement {self.client.env} -- si cela persiste, "
                        f"cet environnement ne liste pas cette serie.")
            return None, None, None
        now = datetime.now(timezone.utc)
        spot = get_btc_price() if BTC_AVAILABLE else None
        best, best_key = None, None
        diag = {"cand": len(markets), "no_ct": 0, "bad_ct": 0, "soon": 0, "dmax": None}
        for m in markets:
            ct = m.get("close_time")
            if not ct: diag["no_ct"] += 1; continue
            try:
                close_dt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
            except ValueError:
                diag["bad_ct"] += 1; continue
            mins = (close_dt - now).total_seconds() / 60.0
            diag["dmax"] = mins if diag["dmax"] is None else max(diag["dmax"], mins)
            if mins < CFG.MIN_MINUTES:
                diag["soon"] += 1; continue
            strike = pick(m, "floor_strike", "cap_strike", "strike_price", default=None)
            try: strike = float(strike)
            except (TypeError, ValueError): continue
            key = abs((spot or strike) - strike)
            if best is None or key < best_key or (key == best_key and mins < best[1]):
                best, best_key = (m, mins, strike), key
        if best is None:
            log.warning(f"Aucun marche exploitable. [DIAG] {diag}")
            return None, None, None
        m, mins, strike = best
        log.info(f"Marche ATM: {m.get('ticker')} | strike={strike:,.2f} | "
                 f"t={mins:.1f}min | spot={f'{spot:,.2f}' if spot else 'N/A'}")
        return m, mins, strike

    def signal(self):
        """Retourne (market, book, decision) ou (None, None, None)."""
        m, mins, strike = self._select_market()
        if not m: return None, None, None
        book = MarketValidator.normalize_book(m)
        if not book:
            log.warning(f"Carnet incoherent/incomplet sur {m.get('ticker')} "
                        f"-- aucun trade.")
            return None, None, None
        log.info(f"Carnet: yes {book['yes_bid']}/{book['yes_ask']}c "
                 f"(mid={book['yes_mid']}c) | no mid={book['no_mid']}c | "
                 f"spread={book['spread']}c")
        res = evaluate_btc_trade(
            strike_price=strike,
            market_yes_price_cents=book["yes_mid"],
            market_no_price_cents=book["no_mid"],
            minutes_remaining=mins,
        )
        verdict = res.get("verdict", "AUCUN TRADE")
        side = "yes" if verdict == "ACHETER YES" else "no"
        if verdict == "ACHETER YES":
            entry = min(book["yes_ask"], book["yes_bid"] + CFG.MAX_SPREAD_PAY)
        elif verdict == "ACHETER NO":
            entry = min(book["no_ask"], book["no_bid"] + CFG.MAX_SPREAD_PAY)
        else:
            entry = book["yes_mid"]
        decision = {
            "verdict": verdict, "side": side, "entry_price": entry,
            "market_prob": (book["yes_mid"] if side == "yes" else book["no_mid"]) / 100.0,
            "model_prob":  (book["yes_mid"] if side == "yes" else book["no_mid"]) / 100.0,
            "edge": 0.0,                       # suivi de marche : edge reel nul
            "ev": 0.0,
            "confidence": res.get("confiance", 0), "grade": res.get("grade", "C"),
            "taille": res.get("taille_position", "0%"),
            "reason": res.get("raison_principale", ""),
        }
        return m, book, decision

# ══════════════════════════════════════════════════════════════════════════
# S15. MOTEUR D'EXECUTION
# ══════════════════════════════════════════════════════════════════════════

class ExecutionEngine:
    def __init__(self, client: KalshiClient, capital: float):
        self.client   = client
        self.capital  = capital
        self.tlog     = TradeLogger()
        self.posmgr   = PositionManager(client, self.tlog)
        self.orders   = OrderManager(client)
        self.risk     = RiskManager(self.tlog, self.posmgr, capital)
        self.stats    = StatsEngine(self.tlog)
        self.strategy = BtcStrategy(client)
        # Recovery apres crash
        self.orders.reconcile_startup(self.tlog, self.posmgr)
        self.posmgr.reconcile_startup()

    def cycle(self, n: int) -> int:
        log.info(f"── CYCLE #{n} ─────────────────────────────────────────────")
        self.stats.maybe_daily_report()

        # 1) Reglements d'abord : le PnL realise conditionne les portes de risque
        realized = self.posmgr.check_settlements()
        for t in realized:
            log_rsk.info(f"PnL jour realise: {self.risk.daily_realized_pnl():+.2f}$ "
                         f"/ limite -{CFG.MAX_DAILY_LOSS:.2f}$")

        # 2) Portes de risque
        ok, why = self.risk.can_trade(cycle_trades=0)
        if not ok:
            log_rsk.warning(f"Trading bloque: {why}")
            self.stats.log_summary()
            return 0

        # 3) Signal
        market, book, dec = self.strategy.signal()
        if not market or dec["verdict"] == "AUCUN TRADE":
            if dec and dec.get("reason"):
                log.info(f"AUCUN TRADE -- {dec['reason']}")
            self.stats.log_summary()
            return 0
        ticker = market.get("ticker", "?")
        log.info(f"[SIGNAL] {dec['verdict']} @ {dec['entry_price']}c | "
                 f"prob_marche={dec['market_prob']:.0%} edge={dec['edge']:.1%} "
                 f"conf={dec['confidence']}/10 | {dec['reason'][:70]}")

        # 4) Validation du signal
        ok, why = SignalValidator.check(dec["verdict"], dec["entry_price"],
                                        ticker, self.tlog, self.posmgr)
        if not ok:
            log.info(f"SIGNAL REFUSE -- {why}")
            return 0

        # 5) Taille de position
        count = PositionSizer.contracts(
            self.capital, dec["entry_price"], dec["taille"],
            dec["confidence"], self.risk.rolling_drawdown(), self.posmgr.open_risk())
        if count <= 0:
            log.info("Taille de position nulle apres contraintes de risque -- pas de trade.")
            return 0

        # 6) Ordre REEL avec suivi des fills
        log_trd.info(f"ORDRE -> {ticker} {dec['side'].upper()} x{count} "
                     f"@ {dec['entry_price']}c (limite, TTL {CFG.ORDER_TTL_SECONDS}s)")
        exec_res = self.orders.place_and_track(ticker, dec["side"], count,
                                               dec["entry_price"])
        if exec_res.filled <= 0:
            log_trd.warning(f"NON EXECUTE ({exec_res.state}: {exec_res.status}) "
                            f"-- AUCUN trade enregistre.")
            return 0

        # 7) Enregistrement UNIQUEMENT du trade reellement execute
        fees = FeeModel.trading_fee(exec_res.filled, exec_res.avg_price)
        trade = self.tlog.open_trade(
            ticker=ticker, market_title=market.get("title", ""),
            side=dec["side"], req_price=dec["entry_price"],
            avg_price=exec_res.avg_price, req_count=count,
            filled_count=exec_res.filled, spread=book["spread"], fees=fees,
            edge=dec["edge"], ev=dec["ev"], confidence=dec["confidence"],
            grade=dec["grade"], reason=dec["reason"],
            analysis={"market_prob": dec["market_prob"],
                      "model_prob": dec["model_prob"]},
            order_id=exec_res.order_id, order_status=exec_res.status)
        self.posmgr.open_position(trade)
        snap = self.risk.snapshot()
        log_rsk.info(f"risque_ouvert={snap['open_risk']}$ "
                     f"pnl_jour={snap['daily_realized_pnl']}$ "
                     f"frais_cumules={snap['fees_paid']}$")
        return 1

# ══════════════════════════════════════════════════════════════════════════
# S16. CLI / BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════

def banner(client: KalshiClient, capital: float):
    bal = client.get_balance()
    log.info("=" * 62)
    log.info("  KALSHI ALPHA ENGINE  v11 PRO")
    log.info("=" * 62)
    log.info(f"Version       : {ENGINE_VERSION} | btc_context={BTC_CTX_VERSION}")
    log.info(f"Environnement : {client.env.upper()} -> {client.base_url}")
    log.info(f"               ORDRES REELS -- aucun dry-run, aucune simulation")
    log.info(f"Identifiants  : {client.cred_src}")
    log.info(f"Solde compte  : {f'{bal:,.2f}$' if bal is not None else 'indisponible (verifier cles/env)'}")
    log.info(f"Capital ref.  : {capital:,.2f}$ (sizing)")
    log.info(f"Risque        : stop jour -{CFG.MAX_DAILY_LOSS:.2f}$ (PnL REALISE) | "
             f"max {CFG.MAX_POS_PCT:g}%/position | budget ouvert {CFG.RISK_BUDGET_PCT:g}%")
    log.info(f"Protections   : entree<={CFG.MAX_ENTRY_CENTS}c | "
             f"1 trade/marche={'OUI' if CFG.ONE_TRADE_PER_MKT else 'NON'} | "
             f"min {CFG.MIN_MINUTES:g}min | TTL ordre {CFG.ORDER_TTL_SECONDS}s")
    log.info("=" * 62)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc", action="store_true", help="mode BTC 15min")
    ap.add_argument("--demo", action="store_true",
                    help="environnement Kalshi DEMO (ordres reels sur demo-api)")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--capital", type=float, default=_env_f("CAPITAL", 500.0))
    args = ap.parse_args()

    env = "demo" if args.demo else "prod"
    if env == "prod" and os.getenv("KALSHI_ENV_CONFIRM", "") != "LIVE":
        log.error("Environnement PRODUCTION demande sans confirmation. "
                  "Definir KALSHI_ENV_CONFIRM=LIVE pour autoriser les ordres reels "
                  "en production. Arret.")
        sys.exit(1)

    client = KalshiClient(env)
    banner(client, args.capital)
    if not BTC_AVAILABLE:
        log.error("btc_context.py manquant -- arret."); sys.exit(1)

    engine = ExecutionEngine(client, args.capital)
    n = 0
    while True:
        n += 1
        try:
            engine.cycle(n)
        except KalshiAPIError as e:
            log.error(f"Cycle #{n}: erreur API non recuperee: {e}")
        except Exception as e:
            log.exception(f"Cycle #{n}: erreur inattendue: {e}")
        if not args.loop:
            break
        time.sleep(max(5, args.interval))

if __name__ == "__main__":
    main()
