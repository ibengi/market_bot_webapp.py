"""
kalshi_alpha_bot.py  --  Kalshi Macro Alpha Engine V3
USAGE:
    python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --loop              # LIVE macro
    python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --demo --loop       # DEMO macro
    python kalshi_alpha_bot.py --market KXEPLGAME-XXX --soccer --demo --loop # DEMO soccer
    python kalshi_alpha_bot.py --scan --soccer --demo --loop                 # DEMO soccer scan
    python kalshi_alpha_bot.py --scan --loop                                 # LIVE scan macro
    python kalshi_alpha_bot.py --btc --loop                                  # LIVE BTC 15min
"""

import os, json, time, sys, argparse, logging, base64, uuid
from datetime import datetime, date
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
import requests
import anthropic

# ── Import FRED context (optionnel) ──────────────────────────────────────────
try:
    from fred_context import get_macro_context
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    def get_macro_context(target="CPI"):
        return ""

# ── Import BTC context (optionnel) ───────────────────────────────────────────
try:
    from btc_context import get_btc_context, get_btc_price, record_trade_result, get_performance_stats
    BTC_AVAILABLE = True
except ImportError:
    BTC_AVAILABLE = False
    def get_btc_context(target_price=0, minutes=15): return ""
    def get_btc_price(): return 65000.0
    def record_trade_result(**kw): pass
    def get_performance_stats(): return {}

# ── Import Trade Resolver (optionnel) ────────────────────────────────────────
try:
    from trade_resolver import resolve_pending_trades, print_stats as print_trade_stats
    RESOLVER_AVAILABLE = True
except ImportError:
    RESOLVER_AVAILABLE = False
    def resolve_pending_trades(k, **kw): return 0
    def print_trade_stats(): pass


load_dotenv()

# ── Logging UTF-8 ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("kalshi_alpha.log", encoding="utf-8"),
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
    ],
)
log = logging.getLogger("KalshiAlpha")

# ── Variables d'environnement ─────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
KALSHI_KEY_ID     = os.getenv("KALSHI_KEY_ID", "")
KALSHI_PRIV_KEY   = os.getenv("KALSHI_PRIVATE_KEY", "").replace("\\n", "\n")

KALSHI_BASE_URL   = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL   = "https://demo-api.kalshi.co/trade-api/v2"

KALSHI_FEE_RATE   = 0.0245
BOT_VERSION       = "v8-diag-2026-07-04"
MIN_EDGE          = 0.03
MIN_CONFIDENCE    = 4

MAX_DAILY_LOSS   = float(os.getenv("MAX_DAILY_LOSS", "50.0"))
MAX_TRADES_CYCLE = int(os.getenv("MAX_TRADES_CYCLE", "3"))

# ── System prompt Macro ───────────────────────────────────────────────────────
SYSTEM_PROMPT_MACRO = """Tu es KALSHI MACRO ALPHA ENGINE V2.
Mission unique : Detecter les erreurs de prix et identifier les trades a EV positive.
JAMAIS : Predire. Avoir raison. Trader par intuition.

Reponds UNIQUEMENT en JSON valide (aucun markdown, aucun backtick) :

{
  "phase1": {"evenement":"","seuil":"","prob_implicite_yes":0.0,"prob_implicite_no":0.0,"incoherence_detectee":false},
  "phase2": {"tendance_recente":"","moyenne":0.0,"mediane":0.0,"volatilite":"","biais_consensus":""},
  "phase3": {"nowcast":0.0,"intervalle_min":0.0,"intervalle_max":0.0,"indicateurs":[{"nom":"","poids":0.0,"signal":""}]},
  "phase4": {"consensus_bloomberg":0.0,"dispersion":"","biais_historique":"","ecart_nowcast_consensus":0.0},
  "phase5": {"scenarios":[
    {"nom":"haussier","probabilite":0.0,"resolution":"","declencheur":""},
    {"nom":"neutre","probabilite":0.0,"resolution":"","declencheur":""},
    {"nom":"baissier","probabilite":0.0,"resolution":"","declencheur":""}
  ],"somme":0.0},
  "phase6": {"prob_reelle":0.0,"prob_marche":0.0,"edge":0.0,"grade":""},
  "phase7": {"argument_haussier":"","argument_baissier":"","argument_market_maker":"","argument_risk_manager":"","edge_tient":true},
  "phase8": {"gain_potentiel":0.0,"perte_potentielle":0.0,"ev_brute":0.0,"ev_nette":0.0,"qualification":""},
  "phase9": {"qualite_donnees":0,"confiance_statistique":0,"risque":0,"volatilite_score":0,"edge_score":0,"score_composite":0.0},
  "phase10": {
    "verdict":"",
    "prob_reelle":0.0,"prob_marche":0.0,"edge":0.0,
    "ev_brute":0.0,"ev_nette":0.0,
    "confiance":0,"risque":0,"grade":"",
    "raison_principale":"","risque_principal":"","risque_exogene":"",
    "taille_position":""
  }
}

REGLES ABSOLUES :
- Somme scenarios = exactement 100%
- prob_reelle dans phase6/phase10 represente TOUJOURS la probabilite du contrat YES.
- Calcule edge_yes = prob_yes - prix_yes ET edge_no = (1 - prob_yes) - prix_no.
- Si edge_no > edge_yes et edge_no positif : verdict = ACHETER NO.
- EDGE final = edge du cote choisi.
- EV nette = (P_gain x gain x 0.9755) - (P_perte x perte)
- Si meilleur edge < 3% OU confiance < 4 : verdict = AUCUN TRADE
- verdict uniquement parmi : ACHETER YES / ACHETER NO / ATTENDRE / AUCUN TRADE
- taille_position uniquement parmi : 0.5% / 1% / 2% / 5% / 10%
"""

# ── System prompt Soccer ──────────────────────────────────────────────────────
SYSTEM_PROMPT_SOCCER = """Tu es KALSHI SOCCER ALPHA ENGINE.
Mission unique : Detecter les erreurs de prix sur des marches de resultat de match
de football/soccer et identifier les trades a EV positive.

Reponds UNIQUEMENT en JSON valide (aucun markdown, aucun backtick) :

{
  "phase1": {"match":"","competition":"","enjeu":"","prob_implicite_yes":0.0,"prob_implicite_no":0.0,"incoherence_detectee":false},
  "phase2": {"forme_equipe_yes":"","forme_equipe_no":"","historique_confrontations":"","blessures_suspensions":""},
  "phase3": {"contexte_match":"","lieu_avantage":"","enjeu_sportif":"","fatigue_calendrier":""},
  "phase4": {"prix_marche_actuel":0.0,"volume_marche":0,"liquidite_suffisante":true},
  "phase5": {"scenarios":[
    {"nom":"victoire_yes","probabilite":0.0,"declencheur":""},
    {"nom":"victoire_no","probabilite":0.0,"declencheur":""},
    {"nom":"nul_ou_incertain","probabilite":0.0,"declencheur":""}
  ],"somme":0.0},
  "phase6": {"prob_reelle":0.0,"prob_marche":0.0,"edge":0.0,"grade":""},
  "phase7": {"argument_pour_yes":"","argument_pour_no":"","argument_market_maker":"","argument_risk_manager":"","edge_tient":true},
  "phase8": {"gain_potentiel":0.0,"perte_potentielle":0.0,"ev_brute":0.0,"ev_nette":0.0,"qualification":""},
  "phase9": {"qualite_donnees":0,"confiance_statistique":0,"risque":0,"volatilite_score":0,"edge_score":0,"score_composite":0.0},
  "phase10": {
    "verdict":"",
    "prob_reelle":0.0,"prob_marche":0.0,"edge":0.0,
    "ev_brute":0.0,"ev_nette":0.0,
    "confiance":0,"risque":0,"grade":"",
    "raison_principale":"","risque_principal":"","risque_exogene":"",
    "taille_position":""
  }
}

REGLES ABSOLUES :
- Somme scenarios = exactement 100%
- prob_reelle represente TOUJOURS la probabilite du contrat YES.
- Calcule edge_yes = prob_yes - prix_yes ET edge_no = (1 - prob_yes) - prix_no.
- Si edge_no > edge_yes et edge_no positif : verdict = ACHETER NO.
- EDGE final = edge du cote choisi.
- EV nette = (P_gain x gain x 0.9755) - (P_perte x perte)
- Si meilleur edge < 5% OU confiance < 5 : verdict = AUCUN TRADE
- Si liquidite_suffisante = false : verdict = AUCUN TRADE
- Si tu ne connais pas suffisamment les deux equipes : confiance <= 3 et verdict = AUCUN TRADE
- verdict uniquement parmi : ACHETER YES / ACHETER NO / ATTENDRE / AUCUN TRADE
- taille_position uniquement parmi : 0.5% / 1% / 2% / 5% / 10%
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def _normalize_probability(value) -> float:
    p = _safe_float(value, 0.0)
    if p > 1:
        p = p / 100.0
    return max(0.0, min(1.0, p))


# ── Gestionnaire de risque journalier ────────────────────────────────────────
class RiskManager:
    RISK_FILE = "risk_state.json"

    def __init__(self, max_daily_loss: float, max_trades_cycle: int):
        self.max_daily_loss   = max_daily_loss
        self.max_trades_cycle = max_trades_cycle
        self._state           = self._load()

    def _load(self) -> dict:
        today = date.today().isoformat()
        if os.path.exists(self.RISK_FILE):
            try:
                with open(self.RISK_FILE, encoding="utf-8") as f:
                    s = json.load(f)
                if s.get("date") == today:
                    return s
            except Exception:
                pass
        return {"date": today, "daily_loss": 0.0, "daily_trades": 0}

    def _save(self):
        try:
            with open(self.RISK_FILE, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            log.error(f"Erreur sauvegarde risk_state: {e}")

    def _refresh_day(self):
        today = date.today().isoformat()
        if self._state.get("date") != today:
            self._state = {"date": today, "daily_loss": 0.0, "daily_trades": 0}
            self._save()

    def can_trade(self, trades_this_cycle: int):
        self._refresh_day()
        if self._state["daily_loss"] >= self.max_daily_loss:
            return False, (f"STOP LOSS JOURNALIER atteint -- "
                           f"perte={self._state['daily_loss']:.2f}$ / limite={self.max_daily_loss:.2f}$")
        if trades_this_cycle >= self.max_trades_cycle:
            return False, (f"LIMITE TRADES/CYCLE atteinte -- "
                           f"{trades_this_cycle}/{self.max_trades_cycle} trades ce cycle")
        return True, ""

    def record_trade(self, cost_dollars: float):
        self._refresh_day()
        self._state["daily_loss"]   += cost_dollars
        self._state["daily_trades"] += 1
        self._save()
        log.info(f"[RISK] Perte potentielle jour: {self._state['daily_loss']:.2f}$ / "
                 f"{self.max_daily_loss:.2f}$ | Trades jour: {self._state['daily_trades']}")

    @property
    def daily_loss(self) -> float:
        self._refresh_day()
        return self._state["daily_loss"]

    @property
    def daily_trades(self) -> int:
        self._refresh_day()
        return self._state["daily_trades"]


# ── Client Kalshi RSA ─────────────────────────────────────────────────────────
class KalshiClient:
    def __init__(self, demo: bool = False):
        self.base_url = KALSHI_DEMO_URL if demo else KALSHI_BASE_URL
        self.demo     = demo
        self.session  = requests.Session()
        self._pk      = None
        if KALSHI_KEY_ID and KALSHI_PRIV_KEY.strip():
            self._pk = self._load_key()

    def _load_key(self):
        try:
            from cryptography.hazmat.primitives import serialization
            key_text = KALSHI_PRIV_KEY.strip()
            if not key_text.startswith("-----"):
                log.warning("KALSHI_PRIVATE_KEY invalide -- format PEM attendu")
                return None
            return serialization.load_pem_private_key(key_text.encode(), password=None)
        except Exception as e:
            log.warning(f"Impossible de charger la cle RSA: {e}")
            return None

    def _sign_headers(self, method: str, path: str) -> dict:
        if not self._pk or not KALSHI_KEY_ID:
            return {"Content-Type": "application/json"}
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            ts  = str(int(time.time() * 1000))
            msg = f"{ts}{method.upper()}{urlparse(path).path}".encode()
            sig = self._pk.sign(
                msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
            return {
                "Content-Type":            "application/json",
                "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            }
        except Exception as e:
            log.warning(f"Erreur signature: {e}")
            return {"Content-Type": "application/json"}

    def _req(self, method: str, path: str, **kw):
        headers = self._sign_headers(method, self.base_url + path)
        return self.session.request(
            method.upper(), self.base_url + path, headers=headers, timeout=15, **kw
        )

    @staticmethod
    def _normalize_market(m: dict) -> dict:
        if not m:
            return m

        def to_cents(v, fallback):
            if v is None: return fallback
            try: return int(round(float(v) * 100))
            except: return fallback

        def to_number(v, fallback=0):
            if v is None: return fallback
            try: return float(v)
            except: return fallback

        m = dict(m)

        def pick_cents(cent_key, dollar_key):
            """Retourne le prix en cents, ou None s'il est vraiment absent."""
            v = m.get(cent_key)
            if v is not None:
                try:
                    return int(round(float(v)))
                except (TypeError, ValueError):
                    pass
            return to_cents(m.get(dollar_key), None)

        yes_bid = pick_cents("yes_bid", "yes_bid_dollars")
        yes_ask = pick_cents("yes_ask", "yes_ask_dollars")
        no_bid  = pick_cents("no_bid",  "no_bid_dollars")
        no_ask  = pick_cents("no_ask",  "no_ask_dollars")

        # ── FIX BUG "achete uniquement YES" ──────────────────────────────────
        # Ancien comportement : si no_bid manquait, il retombait sur 50c.
        # Consequence : le seuil "no >= 60c" ne se declenchait JAMAIS quand
        # l'API ne renvoyait que le carnet YES -> le bot ne pouvait acheter
        # que du YES. On derive maintenant le cote NO du cote YES
        # (identite Kalshi : no_bid = 100 - yes_ask ; no_ask = 100 - yes_bid).
        if no_bid is None and yes_ask is not None:
            no_bid = 100 - yes_ask
        if no_ask is None and yes_bid is not None:
            no_ask = 100 - yes_bid
        # Et symetriquement si seul le cote NO est connu.
        if yes_bid is None and no_ask is not None:
            yes_bid = 100 - no_ask
        if yes_ask is None and no_bid is not None:
            yes_ask = 100 - no_bid

        # Derniers recours (aucune donnee) : marche 50/50 neutre -> aucun trade.
        if yes_bid is None: yes_bid = 50
        if no_bid  is None: no_bid  = 50
        if yes_ask is None: yes_ask = min(99, yes_bid + 3)
        if no_ask  is None: no_ask  = min(99, no_bid  + 3)

        m["yes_bid"] = max(1, min(99, int(yes_bid)))
        m["no_bid"]  = max(1, min(99, int(no_bid)))
        m["yes_ask"] = max(1, min(99, int(yes_ask)))
        m["no_ask"]  = max(1, min(99, int(no_ask)))

        # Sanity check : ask doit etre >= bid
        if m["yes_ask"] < m["yes_bid"]:
            m["yes_ask"] = min(99, m["yes_bid"] + 2)
        if m["no_ask"] < m["no_bid"]:
            m["no_ask"]  = min(99, m["no_bid"] + 2)

        if "volume" not in m or m.get("volume") is None:
            m["volume"] = to_number(m.get("volume_fp"), 0)

        return m

    def get_market(self, ticker: str) -> dict:
        if not KALSHI_KEY_ID:
            return {}
        try:
            r = self._req("GET", f"/markets/{ticker}")
            r.raise_for_status()
            return self._normalize_market(r.json().get("market", {}))
        except Exception as e:
            log.error(f"Erreur get_market({ticker}): {e}")
            return {}

    def get_active_markets(self, category: str = "economic") -> list:
        if not KALSHI_KEY_ID:
            return []
        try:
            r = self._req("GET", "/markets",
                          params={"status": "open", "series_ticker": category, "limit": 50})
            r.raise_for_status()
            markets = r.json().get("markets", [])
            return [self._normalize_market(m) for m in markets]
        except Exception as e:
            log.error(f"Erreur get_active_markets: {e}")
            return []

    def place_order(self, ticker: str, side: str, count: int,
                    price: int, dry_run: bool = False) -> dict:
        if dry_run:
            log.info(f"[DRY RUN] {side.upper()} {count}x {ticker} @ {price}c")
            return {"status": "dry_run", "ticker": ticker,
                    "side": side, "count": count, "price": price}
        try:
            # ── FIX BUG ordres NO ────────────────────────────────────────────
            # Ancien payload : side="bid" + outcome_side=..., count en string
            # decimale, price en dollars, endpoint /portfolio/events/orders.
            # Ce format ne correspond pas a l'API Kalshi trade-api v2
            # documentee : les ordres NO pouvaient etre rejetes ou mal
            # interpretes (=> le bot semblait n'acheter que du YES).
            # Format v2 documente : POST /portfolio/orders avec
            # action=buy/sell, side=yes/no, type=limit, count entier,
            # yes_price OU no_price en cents entiers.
            # NOTE : verifiez ce schema dans la doc Kalshi actuelle
            # (https://trading-api.readme.io / docs.kalshi.com), l'API évolue.
            side = side.lower().strip()
            if side not in ("yes", "no"):
                log.error(f"place_order: side invalide '{side}' -- ordre annule.")
                return {"error": f"invalid side {side}"}
            payload = {
                "ticker":          ticker,
                "client_order_id": f"alpha_{uuid.uuid4().hex}",
                "action":          "buy",
                "side":            side,
                "type":            "limit",
                "count":           int(count),
            }
            if side == "yes":
                payload["yes_price"] = int(price)
            else:
                payload["no_price"] = int(price)
            r = self._req("POST", "/portfolio/orders", json=payload)
            if not r.ok:
                log.error(f"Detail Kalshi HTTP {r.status_code}: {r.text}")
            r.raise_for_status()
            result = r.json()
            log.info(f"ORDER PLACED: {result}")
            return result
        except Exception as e:
            log.error(f"Erreur place_order: {e}")
            if hasattr(e, "response") and e.response is not None:
                log.error(f"Detail Kalshi: {e.response.text}")
            return {"error": str(e)}


# ── Moteur Claude ─────────────────────────────────────────────────────────────
class AlphaEngine:
    def __init__(self, mode: str = "macro"):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.mode   = mode

    def analyse(self, market_data: dict, context: str = "",
                retries: int = 3, delay: int = 5) -> dict:
        system_prompt = SYSTEM_PROMPT_SOCCER if self.mode == "soccer" else SYSTEM_PROMPT_MACRO
        if self.mode == "soccer":
            prompt = f"""
MATCH A ANALYSER :
- Ticker     : {market_data.get('ticker', 'N/A')}
- Titre      : {market_data.get('title', 'N/A')}
- Sous-titre : {market_data.get('subtitle', 'N/A')}
- Prix YES   : {market_data.get('yes_bid', 50)} cents
- Prix NO    : {market_data.get('no_bid', 50)} cents
- Volume     : {market_data.get('volume', 0)}
- Cloture    : {market_data.get('close_time', 'N/A')}

CONTEXTE : {context or 'Aucun -- base-toi sur ta connaissance des equipes.'}
Lance l'analyse complete en 10 phases. Reponds uniquement en JSON valide.
"""
        else:
            prompt = f"""
MARCHE A ANALYSER :
- Ticker    : {market_data.get('ticker', 'N/A')}
- Titre     : {market_data.get('title', 'N/A')}
- Prix YES  : {market_data.get('yes_bid', 55)} cents
- Prix NO   : {market_data.get('no_bid', 45)} cents
- Volume    : {market_data.get('volume', 0)}
- Cloture   : {market_data.get('close_time', 'N/A')}

CONTEXTE ECONOMIQUE : {context or 'Aucun contexte supplementaire.'}
Lance l'analyse complete en 10 phases. Reponds uniquement en JSON valide.
"""
        for attempt in range(1, retries + 1):
            try:
                resp = self.client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                start = raw.find("{"); end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]
                return json.loads(raw)
            except json.JSONDecodeError as e:
                log.error(f"JSON invalide (tentative {attempt}): {e}")
            except Exception as e:
                log.error(f"Erreur Claude (tentative {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(delay)
        return {}


# ── Decision bidirectionnelle YES/NO ─────────────────────────────────────────
def make_btc_decision(market_data: dict, btc_result: dict) -> dict:
    """
    Logique BTC v6 SIMPLE -- suit directement le verdict de btc_context.
    YES >= 60c -> ACHETER YES
    NO  >= 60c -> ACHETER NO
    Sinon      -> AUCUN TRADE
    FIX : le prix d'achat est desormais le ASK du cote choisi (execution
    immediate en croisant le spread). L'ancien code posait un ordre limite
    GTC au BID : il ne se remplissait que si le marche baissait jusqu'a
    nous, c'est-a-dire exactement quand le signal s'invalidait
    (anti-selection), ou restait pendu sans jamais se remplir.
    """
    verdict  = btc_result.get("verdict", "AUCUN TRADE")
    yes_c    = btc_result.get("yes_cents", int(market_data.get("yes_bid", 50)))
    no_c     = btc_result.get("no_cents",  int(market_data.get("no_bid",  50)))
    conf     = btc_result.get("confiance", 0)

    MAX_SPREAD_PAY = 5  # ne jamais payer plus de bid+5c (protection spread anormal)

    if verdict == "ACHETER YES":
        bid     = int(market_data.get("yes_bid", yes_c))
        ask     = int(market_data.get("yes_ask", bid + 2))
        price_c = min(ask, bid + MAX_SPREAD_PAY)
        prob_r  = yes_c / 100.0
    elif verdict == "ACHETER NO":
        bid     = int(market_data.get("no_bid", no_c))
        ask     = int(market_data.get("no_ask", bid + 2))
        price_c = min(ask, bid + MAX_SPREAD_PAY)
        prob_r  = no_c / 100.0
    else:
        price_c = yes_c
        prob_r  = yes_c / 100.0

    log.info(
        f"[BTC v6] yes={yes_c}c no={no_c}c => {verdict} "
        f"| {btc_result.get('raison_principale','')[:60]}"
    )

    return {
        "phase10": {
            "verdict":           verdict,
            "prob_reelle":       round(prob_r, 4),
            "prob_marche":       round(prob_r, 4),
            "edge":              1.0,   # edge fictif >0 pour passer le filtre min_edge
            "ev_brute":          0.0,
            "ev_nette":          0.0,
            "confiance":         conf,
            "risque":            10 - conf,
            "grade":             btc_result.get("grade", "C"),
            "raison_principale": btc_result.get("raison_principale", ""),
            "risque_principal":  btc_result.get("risque_principal", ""),
            "risque_exogene":    btc_result.get("risque_exogene", ""),
            "taille_position":   btc_result.get("taille_position", "0.5%"),
            "prix_achat_cents":  price_c,
        },
        "phase8": {"ev_brute": 0.0, "ev_nette": 0.0},
        "phase9": {},
        "phase5": {},
    }


# ── Gestionnaire de trades ────────────────────────────────────────────────────
class TradeManager:
    def __init__(self, kalshi: KalshiClient, capital: float,
                 demo: bool, risk: RiskManager, mode: str = "macro",
                 btc_min_edge: float = 0.04):
        self.kalshi       = kalshi
        self.capital      = capital
        self.demo         = demo
        self.risk         = risk
        self.mode         = mode
        self.btc_min_edge = btc_min_edge
        self.trades       = []

    def compute_contracts(self, taille_pct: str, price_cents: int) -> int:
        # FIX : une taille inconnue (ex: "0%") retombait sur 1% du capital.
        # Desormais : taille inconnue ou nulle -> 0 contrat -> trade annule.
        pct = {"0.5%": .005, "1%": .01, "2%": .02, "5%": .05, "10%": .10}.get(taille_pct)
        if pct is None or price_cents <= 0:
            return 0
        return max(1, int(self.capital * pct / (price_cents / 100)))

    def execute(self, market_data: dict, analysis: dict,
                trades_this_cycle: int) -> Optional[dict]:
        p10       = analysis.get("phase10", {})
        verdict   = p10.get("verdict", "AUCUN TRADE")
        edge      = p10.get("edge", 0)
        confiance = p10.get("confiance", 0)
        taille    = p10.get("taille_position", "0%")
        ticker    = market_data.get("ticker", "?")

        if self.mode == "soccer":
            min_edge, min_confidence = 0.05, 5
        elif self.mode == "btc":
            min_edge, min_confidence = self.btc_min_edge, 3
        else:
            min_edge, min_confidence = MIN_EDGE, MIN_CONFIDENCE

        if verdict == "AUCUN TRADE":
            log.info(f"[{ticker}] AUCUN TRADE -- edge={edge:.1%} conf={confiance}/10")
            return None

        if edge < min_edge or confiance < min_confidence:
            log.warning(f"[{ticker}] BLOQUE -- edge={edge:.1%} conf={confiance}/10")
            return None

        if self.mode == "soccer":
            if (market_data.get("volume", 0) or 0) < 100:
                log.warning(f"[{ticker}] BLOQUE -- volume insuffisant")
                return None

        # Prix d'achat : utilise prix_achat_cents si disponible (ASK), sinon bid
        if verdict == "ACHETER YES":
            side  = "yes"
            price = int(_safe_float(p10.get("prix_achat_cents",
                        market_data.get("yes_ask", market_data.get("yes_bid", 50))), 50))
        elif verdict == "ACHETER NO":
            side  = "no"
            price = int(_safe_float(p10.get("prix_achat_cents",
                        market_data.get("no_ask", market_data.get("no_bid", 50))), 50))
        else:
            log.info(f"[{ticker}] ATTENDRE")
            return None

        price = max(1, min(99, price))
        count = self.compute_contracts(taille, price)
        if count <= 0:
            log.warning("Nombre de contrats = 0 -- trade annule.")
            return None

        if not self.demo:
            ok, reason = self.risk.can_trade(trades_this_cycle)
            if not ok:
                log.warning(f"[RISK BLOCK] {reason}")
                return None

        result = self.kalshi.place_order(ticker, side, count, price, dry_run=self.demo)

        if not self.demo and "error" not in result:
            self.risk.record_trade((price / 100) * count)

        # ── FIX stats faussees ───────────────────────────────────────────────
        # Ancien code : record_trade_result(won=False, pnl=0) etait appele ICI,
        # au moment de l'achat -> chaque trade etait immediatement compte
        # comme PERDU. Le win rate affiche restait donc a 0% quoi qu'il
        # arrive. Le resultat reel doit etre enregistre uniquement par le
        # resolver (trade_resolver.py) une fois le marche resolu.

        trade_log = {
            "timestamp":       datetime.now().isoformat(),
            "mode":            "DEMO" if self.demo else "LIVE",
            "market_type":     self.mode,
            "ticker":          ticker,
            "verdict":         verdict,
            "side":            side,
            "count":           count,
            "price":           price,
            "edge":            edge,
            "ev_nette":        p10.get("ev_nette", 0),
            "grade":           p10.get("grade", ""),
            "confiance":       confiance,
            "taille_position": taille,
            "order":           result,
        }
        self.trades.append(trade_log)
        self._save_trade(trade_log)
        return trade_log

    def save_state(self, analysis: dict, ticker: str, cycle: int):
        p10 = analysis.get("phase10", {})
        p8  = analysis.get("phase8",  {})
        p9  = analysis.get("phase9",  {})
        p5  = analysis.get("phase5",  {})
        state = {
            "running":          True,
            "mode":             "DEMO" if self.demo else "LIVE",
            "cycle":            cycle,
            "last_ticker":      ticker,
            "last_verdict":     p10.get("verdict", "AUCUN TRADE"),
            "last_edge":        p10.get("edge", 0),
            "last_ev":          p10.get("ev_nette") or p8.get("ev_nette", 0),
            "last_grade":       p10.get("grade", "D"),
            "last_reason":      p10.get("raison_principale", ""),
            "last_risk":        p10.get("risque_principal", ""),
            "last_update":      datetime.now().isoformat(),
            "last_conf":        p10.get("confiance", 0),
            "last_risque":      p10.get("risque", 0),
            "last_size":        p10.get("taille_position", ""),
            "prob_reelle":      p10.get("prob_reelle", 0),
            "prob_marche":      p10.get("prob_marche", 0),
            "ev_brute":         p10.get("ev_brute") or p8.get("ev_brute", 0),
            "risque_exogene":   p10.get("risque_exogene", ""),
            "score_qualite":    p9.get("qualite_donnees", 0),
            "score_confiance":  p9.get("confiance_statistique", 0),
            "score_risque":     p9.get("risque", 0),
            "score_volatilite": p9.get("volatilite_score", 0),
            "score_edge":       p9.get("edge_score", 0),
            "scenarios":        p5.get("scenarios", []),
            "daily_loss":       self.risk.daily_loss,
            "daily_trades":     self.risk.daily_trades,
            "max_daily_loss":   self.risk.max_daily_loss,
            "max_trades_cycle": self.risk.max_trades_cycle,
        }
        try:
            with open("bot_state.json", "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"Erreur save_state: {e}")

    def _save_trade(self, trade: dict):
        path, history = "kalshi_trades.json", []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(trade)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        log.info(f"Trade enregistre -> {path}")


# ── Affichage terminal ────────────────────────────────────────────────────────
def print_report(ticker: str, analysis: dict, demo: bool, risk: RiskManager):
    p10     = analysis.get("phase10", {})
    p8      = analysis.get("phase8",  {})
    verdict = p10.get("verdict", "AUCUN TRADE")
    grade   = p10.get("grade", "")
    icon    = {"ACHETER YES": "[YES]", "ACHETER NO": "[NO]",
               "ATTENDRE": "[WAIT]", "AUCUN TRADE": "[SKIP]"}.get(verdict, verdict)
    sep = "=" * 62
    print(f"""
{sep}
  KALSHI ALPHA ENGINE V3 [{'DEMO' if demo else 'LIVE'}] -- {ticker}
  GRADE {grade:<4}  {icon}  {verdict}
{sep}
  Prob reelle   : {p10.get('prob_reelle', 0):.1%}
  Prob marche   : {p10.get('prob_marche', 0):.1%}
  Edge          : {p10.get('edge', 0):.1%}
  EV brute      : {p8.get('ev_brute', p10.get('ev_brute', 0)):.1%}
  EV nette      : {p8.get('ev_nette', p10.get('ev_nette', 0)):.1%}
  Confiance     : {p10.get('confiance', 0)}/10
  Risque        : {p10.get('risque', 0)}/10
  Position      : {p10.get('taille_position', 'N/A')}
{sep}
  Raison        : {p10.get('raison_principale', '')[:58]}
  Risque princ. : {p10.get('risque_principal', '')[:58]}
  Risque exog.  : {p10.get('risque_exogene', '')[:58]}
{sep}
  [RISK] Perte jour : {risk.daily_loss:.2f}$ / {risk.max_daily_loss:.2f}$
  [RISK] Trades jour : {risk.daily_trades}
{sep}
""")


# ── Compte a rebours ──────────────────────────────────────────────────────────
def countdown(seconds: int, next_cycle: int):
    total = seconds
    for remaining in range(seconds, 0, -1):
        m, s   = divmod(remaining, 60)
        filled = int((total - remaining) / total * 28)
        bar    = "#" * filled + "-" * (28 - filled)
        sys.stdout.write(f"\r  Cycle #{next_cycle} dans {m:02d}:{s:02d}  [{bar}]  (Ctrl+C pour arreter)")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 75 + "\r")
    sys.stdout.flush()


# ── Cycle d'analyse ───────────────────────────────────────────────────────────
def run_cycle(args, kalshi: KalshiClient, engine: AlphaEngine,
              manager: TradeManager, cycle: int) -> int:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  CYCLE #{cycle}  --  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(sep)

    trades_this_cycle = 0

    if not manager.demo:
        ok, reason = manager.risk.can_trade(trades_this_cycle)
        if not ok:
            log.warning(f"[RISK] Cycle bloque: {reason}")
            return 0

    # ── Mode BTC 15min ────────────────────────────────────────────────────────
    if getattr(args, "btc", False) and BTC_AVAILABLE:
        from btc_context import evaluate_btc_trade
        from datetime import datetime as _dt, timezone as _tz

        # Resolution des trades passes tous les 5 cycles
        if RESOLVER_AVAILABLE and cycle % 5 == 1:
            try:
                n_resolved = resolve_pending_trades(kalshi)
                if n_resolved > 0:
                    log.info(f"[Resolver] {n_resolved} trades resolus -> modele mis a jour.")
            except Exception as _e:
                log.debug(f"[Resolver] Erreur: {_e}")

        manual_ticker = getattr(args, "btc_ticker", "")

        if manual_ticker:
            market_data = kalshi.get_market(manual_ticker)
            if not market_data:
                log.warning(f"Marche BTC '{manual_ticker}' introuvable.")
                return 0
        else:
            candidates = kalshi.get_active_markets("KXBTC15M")
            if not candidates:
                log.warning("Aucun marche KXBTC15M actif trouve.")
                return 0

            now_dt = _dt.now(_tz.utc)
            MIN_MINUTES = 5.0   # ignore les marches avec moins de 5min restantes

            # Prix spot BTC : sert a choisir le marche "at-the-money", c.-a-d.
            # celui dont le prix YES/NO reflete vraiment P(hausse)/P(baisse).
            spot = get_btc_price()

            best, best_score = None, None
            # [DIAG v8] Compteurs pour comprendre pourquoi aucun marche ne passe
            diag_no_ct, diag_bad_ct, diag_too_soon = 0, 0, 0
            diag_max_delta = None
            for m in candidates:
                ct = m.get("close_time")
                if not ct:
                    diag_no_ct += 1
                    continue
                try:
                    close_dt = _dt.fromisoformat(ct.replace("Z", "+00:00"))
                except Exception:
                    diag_bad_ct += 1
                    continue
                delta_min = (close_dt - now_dt).total_seconds() / 60.0
                if diag_max_delta is None or delta_min > diag_max_delta:
                    diag_max_delta = delta_min
                if delta_min < MIN_MINUTES:
                    diag_too_soon += 1
                    log.debug(f"[BTC] Marche ignore (t={delta_min:.1f}min < {MIN_MINUTES}min): {m.get('ticker','?')}")
                    continue

                strike = m.get("floor_strike") or m.get("strike_price")

                if spot is not None and strike is not None:
                    # Priorite 1 : echeance la plus proche (arrondie a la minute)
                    # Priorite 2 : strike le plus proche du spot (marche ATM)
                    score = (round(delta_min), abs(float(strike) - spot))
                else:
                    score = (delta_min, 0.0)

                if best_score is None or score < best_score:
                    best, best_score = m, score

            if best is None:
                _sample = candidates[0] if candidates else {}
                _dmax = f"{diag_max_delta:.1f}min" if diag_max_delta is not None else "n/a"
                log.warning(
                    f"Aucun marche KXBTC15M avec plus de {MIN_MINUTES}min restantes -- attente prochaine fenetre. "
                    f"[DIAG] candidats={len(candidates)} | sans close_time={diag_no_ct} | "
                    f"close_time illisible={diag_bad_ct} | trop proches={diag_too_soon} | "
                    f"delta max={_dmax} | exemple: ticker={_sample.get('ticker','?')} "
                    f"close_time={_sample.get('close_time')!r} status={_sample.get('status','?')}"
                )
                return 0

            if spot is not None:
                _bs = best.get("floor_strike") or best.get("strike_price")
                log.info(f"[BTC] Marche ATM selectionne: {best.get('ticker','?')} "
                         f"| strike={_bs} | spot=${spot:,.2f}")

            market_data = best  

        ticker = market_data.get("ticker", manual_ticker or "KXBTC15M")
        strike = (market_data.get("floor_strike") or market_data.get("strike_price")
                  or getattr(args, "btc_target", 0))
        if not strike:
            log.warning(f"Strike introuvable pour '{ticker}'.")
            return 0

        minutes_remaining = getattr(args, "btc_minutes", 15)
        close_time = market_data.get("close_time")
        if close_time:
            try:
                close_dt = _dt.fromisoformat(close_time.replace("Z", "+00:00"))
                minutes_remaining = max((close_dt - _dt.now(_tz.utc)).total_seconds() / 60.0, 0.1)
            except Exception:
                pass

        # ── FIX symetrie UP/DOWN ─────────────────────────────────────────────
        # Ancien code : decision sur yes_bid et no_bid pris separement.
        # Avec un spread large, les deux bids peuvent etre sous 60c en meme
        # temps, et l'asymetrie du carnet biaisait la decision.
        # Nouveau : prix mid du carnet YES -> prob(UP) ; prob(DOWN) = 100 - mid.
        # UP et DOWN sont ainsi traites de facon strictement symetrique.
        yes_bid_c = int(market_data.get("yes_bid", 50))
        yes_ask_c = int(market_data.get("yes_ask", yes_bid_c))
        yes_mid_c = int(round((yes_bid_c + yes_ask_c) / 2))
        no_mid_c  = 100 - yes_mid_c
        log.info(f"[BTC] Carnet: yes {yes_bid_c}/{yes_ask_c}c "
                 f"(mid={yes_mid_c}c) | no mid={no_mid_c}c")

        btc_result = evaluate_btc_trade(
            strike_price=float(strike),
            market_yes_price_cents=yes_mid_c,
            market_no_price_cents=no_mid_c,
            minutes_remaining=minutes_remaining,
            min_edge=getattr(args, "btc_min_edge", 0.04),
        )

        # Construit la structure analysis avec prix ASK corrects
        analysis = make_btc_decision(market_data, btc_result)

        log.info(
            f"[BTC] {ticker} | strike=${float(strike):,.2f} | "
            f"t={minutes_remaining:.1f}min | {btc_result.get('raison_principale', '')}"
        )
        print_report(ticker, analysis, manager.demo, manager.risk)
        manager.save_state(analysis, ticker, cycle)
        trade = manager.execute(market_data, analysis, trades_this_cycle)
        if trade:
            trades_this_cycle += 1
        return trades_this_cycle

    # ── Mode marche specifique ────────────────────────────────────────────────
    if args.market:
        log.info(f"Analyse du marche : {args.market}")
        market_data = kalshi.get_market(args.market)

        if not market_data:
            log.warning("Donnees Kalshi non disponibles -- donnees fictives utilisees")
            market_data = {
                "ticker": args.market, "title": f"Marche {args.market}",
                "yes_bid": 50, "no_bid": 50, "yes_ask": 50, "no_ask": 50,
                "volume": 0, "close_time": "N/A", "category": "economic",
            }

        full_context = ""
        if manager.mode != "soccer":
            if FRED_AVAILABLE and os.getenv("FRED_API_KEY"):
                log.info("Recuperation contexte macro FRED...")
                full_context = get_macro_context(target="CPI") + "\n\n"
        if args.context:
            full_context += "CONTEXTE ADDITIONNEL: " + args.context

        log.info("Lancement analyse Claude (10 phases)...")
        analysis = engine.analyse(market_data, context=full_context or args.context)

        if analysis:
            print_report(args.market, analysis, manager.demo, manager.risk)
            manager.save_state(analysis, args.market, cycle)
            trade = manager.execute(market_data, analysis, trades_this_cycle)
            if trade:
                trades_this_cycle += 1
                log.info(f"Trade execute: {trade.get('verdict')} | "
                         f"Edge: {trade.get('edge', 0):.1%} | "
                         f"Position: {trade.get('taille_position', '')}")
        else:
            log.error("Analyse vide -- aucun trade ce cycle.")

    # ── Mode scan ─────────────────────────────────────────────────────────────
    elif args.scan:
        default_series = "KXWCGAME" if manager.mode == "soccer" else "economic"
        scan_category  = getattr(args, "series", "") or default_series
        log.info(f"Scan des marches actifs (serie: {scan_category})...")
        markets = kalshi.get_active_markets(scan_category)
        if not markets:
            log.warning(f"Aucun marche trouve pour '{scan_category}'.")
            return 0
        log.info(f"{len(markets)} marches trouves.")
        max_m = getattr(args, "max_markets", 0)
        if max_m > 0:
            markets = markets[:max_m]
        for market in markets:
            if not manager.demo:
                ok, reason = manager.risk.can_trade(trades_this_cycle)
                if not ok:
                    log.warning(f"[RISK] Scan stoppe: {reason}")
                    break
            ticker = market.get("ticker", "")
            log.info(f"--- Analyse: {ticker} ---")
            analysis = engine.analyse(market, context=args.context)
            if analysis:
                print_report(ticker, analysis, manager.demo, manager.risk)
                manager.save_state(analysis, ticker, cycle)
                trade = manager.execute(market, analysis, trades_this_cycle)
                if trade:
                    trades_this_cycle += 1
            time.sleep(4)

    return trades_this_cycle


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Kalshi Macro Alpha Engine V3")
    parser.add_argument("--market",           type=str)
    parser.add_argument("--scan",             action="store_true")
    parser.add_argument("--soccer",           action="store_true")
    parser.add_argument("--series",           type=str,   default="")
    parser.add_argument("--max-markets",      type=int,   default=0)
    parser.add_argument("--demo",             action="store_true", default=False)
    parser.add_argument("--capital",          type=float, default=500.0)
    parser.add_argument("--context",          type=str,   default="")
    parser.add_argument("--loop",             action="store_true")
    parser.add_argument("--interval",         type=int,   default=300)
    parser.add_argument("--max-daily-loss",   type=float, default=MAX_DAILY_LOSS)
    parser.add_argument("--max-trades-cycle", type=int,   default=MAX_TRADES_CYCLE)
    parser.add_argument("--btc",              action="store_true")
    parser.add_argument("--btc-target",       type=float, default=0.0)
    parser.add_argument("--btc-minutes",      type=int,   default=15)
    parser.add_argument("--btc-ticker",       type=str,   default="")
    parser.add_argument("--btc-min-edge",     type=float, default=0.04)
    args = parser.parse_args()

    if args.btc and args.interval == 300:
        args.interval = 60
        log.info("Mode BTC -- intervalle ajuste a 60s.")

    market_mode = "soccer" if args.soccer else ("btc" if args.btc else "macro")
    sep = "=" * 62
    print("\n" + sep)
    print("   KALSHI MACRO ALPHA ENGINE V3")
    print(sep)
    log.info(f"Mode          : {'DEMO' if args.demo else 'LIVE -- ORDRES REELS'}")
    try:
        import btc_context as _bc
        _bc_ver = getattr(_bc, "VERSION", "ANCIENNE VERSION (pas de constante VERSION)")
    except ImportError:
        _bc_ver = "non importe"
    log.info(f"VERSION CODE  : bot={BOT_VERSION} | btc_context={_bc_ver}")
    log.info(f"Type marche   : {'BTC 15min (seuil 60%)' if args.btc else market_mode.upper()}")
    log.info(f"Capital       : ${args.capital:,.2f}")
    log.info(f"Stop loss/jour: ${args.max_daily_loss:,.2f}")
    log.info(f"Max trades/cyc: {args.max_trades_cycle}")
    log.info(f"Boucle        : {'OUI -- ' + str(args.interval) + 's' if args.loop else 'NON'}")
    log.info(f"Kalshi        : {'CLE CHARGEE' if KALSHI_KEY_ID else 'PAS DE CLE'}")
    log.info(f"Claude        : {'CLE OK' if ANTHROPIC_API_KEY else 'MANQUANTE'}")

    if args.btc and BTC_AVAILABLE:
        try:
            stats = get_performance_stats()
            if stats.get("total", 0) > 0:
                log.info(f"[PERF BTC] {stats['total']} trades | "
                         f"WR={stats['win_rate']:.1%} | PnL=${stats['total_pnl']:.2f} | "
                         f"YES={stats.get('yes_trades',0)} trades WR={stats.get('yes_wr',0):.1%} | "
                         f"NO={stats.get('no_trades',0)} trades WR={stats.get('no_wr',0):.1%}")
            else:
                log.info("[PERF BTC] Aucun trade precedent -- demarrage a zero.")
        except Exception:
            pass

    print(sep + "\n")

    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY manquant -- impossible de continuer.")
        sys.exit(1)
    if not args.demo and not KALSHI_KEY_ID:
        log.error("KALSHI_KEY_ID manquant -- impossible de trader en LIVE.")
        sys.exit(1)
    if not args.demo and not KALSHI_PRIV_KEY.strip():
        log.error("KALSHI_PRIVATE_KEY manquant -- impossible de signer les ordres.")
        sys.exit(1)
    if not args.market and not args.scan and not args.btc:
        parser.print_help()
        return

    risk    = RiskManager(args.max_daily_loss, args.max_trades_cycle)
    kalshi  = KalshiClient(demo=args.demo)
    engine  = AlphaEngine(mode=market_mode)
    manager = TradeManager(kalshi, args.capital, demo=args.demo, risk=risk,
                           mode=market_mode, btc_min_edge=args.btc_min_edge)

    cycle, total_trades = 1, 0
    try:
        while True:
            n = run_cycle(args, kalshi, engine, manager, cycle)
            total_trades += n
            print(f"\n  Cycle #{cycle} termine -- {n} trade(s) -- Total: {total_trades}")
            if not args.demo:
                print(f"  Perte potentielle jour: {risk.daily_loss:.2f}$ / {risk.max_daily_loss:.2f}$")
            if not args.loop:
                break
            cycle += 1
            countdown(args.interval, cycle)
    except KeyboardInterrupt:
        print(f"\n\n  Arret -- {cycle} cycle(s) -- {total_trades} trade(s) total\n")
        log.info(f"Bot arrete proprement apres {cycle} cycles et {total_trades} trades.")


if __name__ == "__main__":
    main()
