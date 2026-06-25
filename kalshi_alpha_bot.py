"""
KALSHI MACRO ALPHA ENGINE V3
Bot de trading institutionnel — version corrigee

INSTALLATION:
    pip install anthropic requests python-dotenv flask flask-cors cryptography

FICHIER .env :
    ANTHROPIC_API_KEY=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
    KALSHI_KEY_ID=486ee030-12f5-46f9-b59c-cb4903978f9c
    KALSHI_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAnWvg4YFM92F0E0c2B+dAe3jWdHEcn7B/mqC/dpVdiRaHP2sq
iWPZM7BET0irYsFXtPaYDV6qMqwbHMg9KxitxJNOT+nUrB2BaA0DPg0cjFkUHeN/
d47jKSNFXRimxL03l59mYaCsLtWRxLng98w8ujelMOELNL3no1sbyLo3ynOA6y8m
A6h5LyZGnzhuCWqcOChdRoSrLhAwS2gcDJtQs1xJzGkPvHQu6uLoUjrn2sxtTSuJ
6n5CFyrgwCrTJxEYfx+ZByY2Z9vUG5i7L+Z46HjEN5ADWCLrOi9qT0JapK70tLfX
CJrOfmmU2uFBAluVFQsiQyo6GVaFg4RQ9glKtQIDAQABAoIBAAv1NyCiHAjZmgHh
4aiRiCwo5m9zbSdzNoo2Jj0ZhQCmGbF3UESd9VCQFexI2p32jlVEexHe7EJLoQab
bkwRBJgfUW5QFpPZbOxMur+SquW9WYIYtyTLkZVdJMZ42igtMGUf2lzVoeav2fIN
5ZklJkLF+dIf8iQ4PbmCsPZbMOQ7qd/an6IIutDe/VvYXfTtY0qGuc41/e7hAHn6
v/Au1VRFadK3v8beRMKF7z5GuSZ29RG7Qn4RqyZhxXit7wiU+jnNSFjv1jDaFXlh
hlQEjSg6UtQJz8/fD6V/iXZtuAtq5iBLNUCvOOK9T+SmJKUNJMNgWVbZA53D6kZi
RlxTLdECgYEAw6FeT0vxp0beQuxcrLqmOwD2qIKeZ35q4yYowp+KGKNdiPY09tJy
Sj1EV41ZgIGAGmryI5SKeIxhTkCpl7bIdRSq2mEpc4NKoewumjdOlPMcnGEqjlou
MJgcDwbdfOEE7ivMggVnrYGdSR7Vl/jqrbm/amhD1cw87tXAtnMIbqUCgYEAzgAJ
JfqpeGl+Sw66ZC1WO1iyIfDkjBJUxDSFgiGgqi3cHVQPLOYl9d1KwcPXQ6AaYHMo
Z9KvXqEzBipTi0x+Bql8+kngx74gbs19teEI2/x7sDUV+f1vdq++sIGoZln6gFkA
O8GST31FG+ZQX3pncD7XAEMrRpOm0lp7GK//PtECgYBiD7ppf0Tzt7djzn0p7Cm1
O+doUok6kYjcsd0OqdAcR490Pw4Phy/Y/NsMFAOAQenH1EHqCeRbRurjwdABB5N1
9NUrwDZ5+57mibBWh1Cxoyd9T8t4LcYnf6fY9HUDyvugs33A0xrEQ0tnQriIhDKG
wKwtl3QhcE4+3hDKo+DfLQKBgAurwtjs/6b7yxTzi6nbS7RnDQiRPlGVREotc5bw
0spxeLQMrCNuEp6AYBjkQJDrRDNMsvBW5mqlFV/3C+6rccRs29DOWLbYVbwRVlr0
mezkvBk6mLkmG6eMw2/6mJDb7i5RXIsGJ4TrYvv2q30NUUjxtnqkU5JXES9/wtOe
PQbRAoGABoZHA/a2cgO0jZITlU7SvSaWQLMbPc9DtPMwouYDhFFA8zrO89Sm90zP
gRBpqUqbOBxClhWQh6LvRjQYUjpxYg77HnBGFk88vQCXhjxX8QNaTRqIp8oDdkHX
hfR8Z8VxNnxHQoUM5ICzhHd2/k1IAAXv4CCMyXj8DeKQi47NIK8=
-----END RSA PRIVATE KEY-----
USAGE:
    python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --loop              # LIVE macro
    python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --demo --loop       # DEMO macro
    python kalshi_alpha_bot.py --market KXEPLGAME-XXX --soccer --demo --loop # DEMO soccer
    python kalshi_alpha_bot.py --scan --soccer --demo --loop                 # DEMO soccer scan
    python kalshi_alpha_bot.py --scan --loop                                 # LIVE scan macro
    python kalshi_alpha_bot.py --btc --loop                                  # LIVE BTC 15min
"""

import os, json, time, sys, argparse, logging, base64
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
    from btc_context import get_btc_context, get_btc_price
    BTC_AVAILABLE = True
except ImportError:
    BTC_AVAILABLE = False
    def get_btc_context(target_price=0, minutes=15):
        return ""
    def get_btc_price():
        return 65000.0


load_dotenv()

# ── Logging UTF-8 (fix Windows) ───────────────────────────────────────────────
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

KALSHI_BASE_URL   = "https://api.elections.kalshi.com/trade-api/v2"   # LIVE
KALSHI_DEMO_URL   = "https://demo-api.kalshi.co/trade-api/v2"         # DEMO

KALSHI_FEE_RATE   = 0.0245
MIN_EDGE          = 0.03
MIN_CONFIDENCE    = 4

# ── Limites de securite LIVE ──────────────────────────────────────────────────
MAX_DAILY_LOSS   = float(os.getenv("MAX_DAILY_LOSS", "50.0"))
MAX_TRADES_CYCLE = int(os.getenv("MAX_TRADES_CYCLE", "3"))

# ── System prompt (Macro/CPI) ────────────────────────────────────────────────
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
- EDGE = prob_reelle - prob_marche
- EV nette = (P_gain x gain x 0.9755) - (P_perte x perte)
- Si edge < 3% OU confiance < 4 : verdict = AUCUN TRADE
- verdict uniquement parmi : ACHETER YES / ACHETER NO / ATTENDRE / AUCUN TRADE
- taille_position uniquement parmi : 0.5% / 1% / 2% / 5% / 10%
"""

# ── System prompt (Soccer / resultat de match) ───────────────────────────────
SYSTEM_PROMPT_SOCCER = """Tu es KALSHI SOCCER ALPHA ENGINE.
Mission unique : Detecter les erreurs de prix sur des marches de resultat de match
de football/soccer (1X2 -- qui gagne) et identifier les trades a EV positive.
JAMAIS : Predire par intuition ou supporterisme. Toujours raisonner par les faits
disponibles (forme recente, contexte du match, enjeux, compositions probables,
historique des confrontations, fatigue/calendrier, lieu du match).

Tu n'as PAS acces a une API de stats sportives en temps reel : base ton analyse sur
ta connaissance generale des equipes/joueurs/competitions et sur le contexte fourni
(prix de marche Kalshi, volume, titre du marche, date du match). Si tu ne connais pas
suffisamment les deux equipes pour juger, reduis fortement ta confiance plutot que
d'inventer des informations.

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
- EDGE = prob_reelle - prob_marche
- EV nette = (P_gain x gain x 0.9755) - (P_perte x perte)
- Si edge < 5% OU confiance < 5 : verdict = AUCUN TRADE (seuil plus strict que macro car
  le sport a une variance intrinseque plus elevee qu'un indicateur economique)
- Si liquidite_suffisante = false : verdict = AUCUN TRADE (impossible de sortir la position)
- Si tu ne connais pas suffisamment les deux equipes : confiance <= 3 et verdict = AUCUN TRADE
- verdict uniquement parmi : ACHETER YES / ACHETER NO / ATTENDRE / AUCUN TRADE
- taille_position uniquement parmi : 0.5% / 1% / 2% / 5% / 10%
"""


# ── Gestionnaire de risque journalier ────────────────────────────────────────
class RiskManager:
    """
    Suit les pertes journalieres et bloque les trades si MAX_DAILY_LOSS est atteint.
    Suit aussi le nombre de trades par cycle pour ne pas depasser MAX_TRADES_CYCLE.
    Etat persiste dans risk_state.json, reinitialise chaque nouveau jour.
    """
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
        """Retourne (True, '') si le trade est autorise, sinon (False, raison)."""
        self._refresh_day()

        if self._state["daily_loss"] >= self.max_daily_loss:
            return False, (
                f"STOP LOSS JOURNALIER atteint -- "
                f"perte={self._state['daily_loss']:.2f}$ / limite={self.max_daily_loss:.2f}$"
            )

        if trades_this_cycle >= self.max_trades_cycle:
            return False, (
                f"LIMITE TRADES/CYCLE atteinte -- "
                f"{trades_this_cycle}/{self.max_trades_cycle} trades ce cycle"
            )

        return True, ""

    def record_trade(self, cost_dollars: float):
        """Enregistre le cout (montant risque) d'un trade live execute."""
        self._refresh_day()
        self._state["daily_loss"]   += cost_dollars
        self._state["daily_trades"] += 1
        self._save()
        log.info(
            f"[RISK] Perte potentielle jour: {self._state['daily_loss']:.2f}$ / "
            f"{self.max_daily_loss:.2f}$ | Trades jour: {self._state['daily_trades']}"
        )

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
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
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
        """
        Normalise les champs Kalshi qui ont change de format au fil des versions API :
        - yes_bid / yes_bid_dollars (dollars en string, ex "0.5600" -> 56 cents)
        - no_bid  / no_bid_dollars
        - volume  / volume_fp (peut etre une string numerique)
        Garantit que 'yes_bid', 'no_bid' et 'volume' existent toujours avec
        des types numeriques utilisables par le reste du bot.
        """
        if not m:
            return m

        def to_cents(dollars_str, fallback_cents):
            if dollars_str is None:
                return fallback_cents
            try:
                return int(round(float(dollars_str) * 100))
            except (TypeError, ValueError):
                return fallback_cents

        def to_number(value, fallback=0):
            if value is None:
                return fallback
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        m = dict(m)  # copie pour ne pas muter l'original

        if "yes_bid" not in m or m.get("yes_bid") is None:
            m["yes_bid"] = to_cents(m.get("yes_bid_dollars"), 50)
        if "no_bid" not in m or m.get("no_bid") is None:
            m["no_bid"] = to_cents(m.get("no_bid_dollars"), 50)
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
        """
        Envoie un ordre sur Kalshi via l'endpoint V2 (/portfolio/events/orders).
        dry_run=True  -> simulation locale uniquement (aucun appel API).
        dry_run=False -> ordre reel envoye a l'API (live ou demo selon base_url).
        """
        if dry_run:
            log.info(f"[DRY RUN] {side.upper()} {count}x {ticker} @ {price}c")
            return {
                "status": "dry_run",
                "ticker": ticker,
                "side":   side,
                "count":  count,
                "price":  price,
            }
        try:
            price_dollars = f"{price / 100:.4f}"
            payload = {
                "ticker":                    ticker,
                "client_order_id":           f"alpha_{int(time.time())}",
                "side":                      "bid",
                "outcome_side":              side,
                "count":                     f"{count:.2f}",
                "price":                     price_dollars,
                "time_in_force":             "good_till_canceled",
                "self_trade_prevention_type": "taker_at_cross",
            }

            r = self._req("POST", "/portfolio/events/orders", json=payload)
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


# ── Moteur Claude avec retry ──────────────────────────────────────────────────
class AlphaEngine:
    def __init__(self, mode: str = "macro"):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.mode   = mode  # "macro" ou "soccer"

    def analyse(self, market_data: dict, context: str = "",
                retries: int = 3, delay: int = 5) -> dict:
        if self.mode == "soccer":
            system_prompt = SYSTEM_PROMPT_SOCCER
            prompt = f"""
MATCH A ANALYSER :
- Ticker     : {market_data.get('ticker', 'N/A')}
- Titre      : {market_data.get('title', 'N/A')}
- Sous-titre : {market_data.get('subtitle', 'N/A')}
- Prix YES   : {market_data.get('yes_bid', 50)} cents
- Prix NO    : {market_data.get('no_bid', 50)} cents
- Volume     : {market_data.get('volume', 0)}
- Cloture    : {market_data.get('close_time', 'N/A')}
- Categorie  : {market_data.get('category', 'N/A')}

CONTEXTE ADDITIONNEL :
{context or 'Aucun contexte supplementaire fourni -- base-toi sur ta connaissance des equipes/competition.'}

Lance l'analyse complete en 10 phases pour ce marche de resultat de match.
Reponds uniquement en JSON valide.
"""
        else:
            system_prompt = SYSTEM_PROMPT_MACRO
            prompt = f"""
MARCHE A ANALYSER :
- Ticker    : {market_data.get('ticker', 'N/A')}
- Titre     : {market_data.get('title', 'N/A')}
- Prix YES  : {market_data.get('yes_bid', 55)} cents
- Prix NO   : {market_data.get('no_bid', 45)} cents
- Volume    : {market_data.get('volume', 0)}
- Cloture   : {market_data.get('close_time', 'N/A')}
- Categorie : {market_data.get('category', 'N/A')}

CONTEXTE ECONOMIQUE :
{context or 'Aucun contexte supplementaire.'}

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
                start = raw.find("{")
                end   = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]
                return json.loads(raw)
            except json.JSONDecodeError as e:
                log.error(f"JSON invalide (tentative {attempt}): {e}")
            except Exception as e:
                log.error(f"Erreur Claude (tentative {attempt}/{retries}): {e}")
                if attempt < retries:
                    log.info(f"Retry dans {delay}s...")
                    time.sleep(delay)
        return {}


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
        pct = {"0.5%": .005, "1%": .01, "2%": .02, "5%": .05, "10%": .10}.get(taille_pct, .01)
        if price_cents <= 0:
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
            min_edge, min_confidence = getattr(self, "btc_min_edge", 0.04), 3
        else:
            min_edge, min_confidence = MIN_EDGE, MIN_CONFIDENCE

        if verdict == "AUCUN TRADE":
            log.info(f"[{ticker}] AUCUN TRADE -- edge={edge:.1%} conf={confiance}/10")
            return None

        if edge < min_edge or confiance < min_confidence:
            log.warning(f"[{ticker}] BLOQUE -- edge={edge:.1%} conf={confiance}/10")
            return None

        if self.mode == "soccer":
            volume = market_data.get("volume", 0) or 0
            if volume < 100:
                log.warning(f"[{ticker}] BLOQUE -- volume insuffisant ({volume})")
                return None

        if verdict == "ACHETER YES":
            side, price = "yes", int(market_data.get("yes_bid", 50))
        elif verdict == "ACHETER NO":
            side, price = "no",  int(market_data.get("no_bid", 50))
        else:
            log.info(f"[{ticker}] ATTENDRE")
            return None

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
        """Ecrit l'etat courant pour le dashboard."""
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
    icon    = {
        "ACHETER YES": "[YES]",
        "ACHETER NO":  "[NO]",
        "ATTENDRE":    "[WAIT]",
        "AUCUN TRADE": "[SKIP]",
    }.get(verdict, verdict)
    mode_label = "DEMO" if demo else "LIVE"
    sep = "=" * 62
    print(f"""
{sep}
  KALSHI ALPHA ENGINE V3 [{mode_label}] -- {ticker}
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

        manual_ticker = getattr(args, "btc_ticker", "")

        if manual_ticker:
            market_data = kalshi.get_market(manual_ticker)
            if not market_data:
                log.warning(f"Marche BTC '{manual_ticker}' introuvable.")
                return 0
        else:
            candidates = kalshi.get_active_markets("KXBTC15M")
            if not candidates:
                log.warning(
                    "Aucun marche KXBTC15M actif trouve -- verifie KALSHI_KEY_ID, "
                    "ou precise --btc-ticker manuellement."
                )
                return 0

            now_dt = _dt.now(_tz.utc)
            best, best_delta = None, None
            for m in candidates:
                ct = m.get("close_time")
                if not ct:
                    continue
                try:
                    close_dt = _dt.fromisoformat(ct.replace("Z", "+00:00"))
                except Exception:
                    continue
                delta = (close_dt - now_dt).total_seconds()
                if delta <= 0:
                    continue
                if best_delta is None or delta < best_delta:
                    best, best_delta = m, delta

            if best is None:
                log.warning("Aucun marche KXBTC15M avec close_time futur trouve.")
                return 0

            market_data = best

        ticker = market_data.get("ticker", manual_ticker or "KXBTC15M")
        strike = market_data.get("floor_strike") or market_data.get("strike_price") \
                 or getattr(args, "btc_target", 0)
        if not strike:
            log.warning(f"Strike introuvable pour '{ticker}' -- impossible d'evaluer.")
            return 0

        close_time = market_data.get("close_time")
        minutes_remaining = getattr(args, "btc_minutes", 15)
        if close_time:
            try:
                close_dt = _dt.fromisoformat(close_time.replace("Z", "+00:00"))
                now_dt2  = _dt.now(_tz.utc)
                minutes_remaining = max((close_dt - now_dt2).total_seconds() / 60.0, 0.1)
            except Exception:
                pass

        # ── CORRECTION v2 : passage des deux prix (YES et NO) ─────────────────
        # L'ancienne version ne passait que yes_bid, forçant btc_context a
        # calculer le prix NO comme (1 - yes_price), ce qui ignore le spread
        # bid/ask reel de Kalshi. On passe maintenant no_bid explicitement
        # pour que evaluate_btc_trade calcule l'edge NO avec le vrai prix.
        result = evaluate_btc_trade(
            strike_price=float(strike),
            market_yes_price_cents=int(market_data.get("yes_bid", 50)),
            market_no_price_cents=int(market_data.get("no_bid", 50)),   # ← CORRECTION
            minutes_remaining=minutes_remaining,
            min_edge=getattr(args, "btc_min_edge", 0.04),
        )
        analysis = {"phase10": result, "phase8": {}, "phase9": {}, "phase5": {}}

        log.info(
            f"[BTC] {ticker} | strike=${float(strike):,.2f} | "
            f"t_restant={minutes_remaining:.1f}min | {result['raison_principale']}"
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
            if manager.mode == "soccer":
                market_data = {
                    "ticker":     args.market,
                    "title":      f"Match {args.market}",
                    "yes_bid":    50,
                    "no_bid":     50,
                    "volume":     0,
                    "close_time": "N/A",
                    "category":   "soccer",
                    "subtitle":   "Donnees indisponibles",
                }
            else:
                market_data = {
                    "ticker":     args.market,
                    "title":      f"Marche {args.market}",
                    "yes_bid":    84,
                    "no_bid":     16,
                    "volume":     8000,
                    "close_time": "2026-07-14T09:30:00Z",
                    "category":   "economic",
                    "subtitle":   "CPI June 2026",
                }

        full_context = ""
        if manager.mode == "soccer":
            if args.context:
                full_context = "CONTEXTE ADDITIONNEL: " + args.context
        else:
            fred_ctx = ""
            if FRED_AVAILABLE and os.getenv("FRED_API_KEY"):
                log.info("Recuperation contexte macro FRED...")
                fred_ctx = get_macro_context(target="CPI")
            if fred_ctx:
                full_context += fred_ctx + "\n\n"
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
                log.info(
                    f"Trade execute: {trade.get('verdict')} | "
                    f"Edge: {trade.get('edge', 0):.1%} | "
                    f"Position: {trade.get('taille_position', '')}"
                )
        else:
            log.error("Analyse vide -- aucun trade ce cycle.")

    # ── Mode scan ─────────────────────────────────────────────────────────────
    elif args.scan:
        default_series = "KXWCGAME" if manager.mode == "soccer" else "economic"
        scan_category = getattr(args, "series", "") or default_series
        log.info(f"Scan des marches actifs (serie: {scan_category})...")
        markets = kalshi.get_active_markets(scan_category)
        if not markets:
            log.warning(
                f"Aucun marche trouve pour la serie '{scan_category}' -- "
                f"verifie KALSHI_KEY_ID dans .env, ou essaie --series KXWC, --series KXWCWIN, etc."
            )
            return 0
        log.info(f"{len(markets)} marches trouves.")
        max_m = getattr(args, "max_markets", 0)
        if max_m > 0:
            markets = markets[:max_m]
            log.info(f"Limite appliquee : {max_m} marches analyses ce cycle.")
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
    parser.add_argument("--market",           type=str,   help="Ticker Kalshi (ex: KXEPLGAME-... ou KXCPI-26JUN-T0.1)")
    parser.add_argument("--scan",             action="store_true", help="Scan tous les marches economiques")
    parser.add_argument("--soccer",           action="store_true",
                        help="Mode soccer : analyse resultat de match (1X2) au lieu du mode macro")
    parser.add_argument("--series",           type=str,   default="",
                        help="Series Kalshi a scanner (ex: KXWC, KXEPLGAME). Defaut: KXWCGAME en mode soccer")
    parser.add_argument("--max-markets",     type=int,   default=0,
                        help="Nombre max de marches analyses par scan (0 = tous, defaut: 0)")
    parser.add_argument("--demo",             action="store_true", default=False,
                        help="Paper trading -- aucun ordre reel (defaut: LIVE)")
    parser.add_argument("--capital",          type=float, default=500.0)
    parser.add_argument("--context",          type=str,   default="")
    parser.add_argument("--loop",             action="store_true", help="Boucle automatique")
    parser.add_argument("--interval",         type=int,   default=300,
                        help="Secondes entre cycles (defaut: 300)")
    parser.add_argument("--max-daily-loss",   type=float, default=MAX_DAILY_LOSS,
                        help=f"Stop loss journalier en $ (defaut: {MAX_DAILY_LOSS})")
    parser.add_argument("--max-trades-cycle", type=int,   default=MAX_TRADES_CYCLE,
                        help=f"Trades max par cycle (defaut: {MAX_TRADES_CYCLE})")
    parser.add_argument("--btc",              action="store_true", help="Mode BTC 15min Kalshi")
    parser.add_argument("--btc-target",       type=float, default=0.0, help="Prix target BTC (fallback si strike API absent)")
    parser.add_argument("--btc-minutes",      type=int,   default=15,  help="Duree contrat BTC en minutes")
    parser.add_argument("--btc-ticker",       type=str,   default="",
                        help="Ticker exact du marche BTC (ex: KXBTC15M-26JUN1814:00).")
    parser.add_argument("--btc-min-edge",     type=float, default=0.04,
                        help="Edge minimum pour trader en mode BTC (defaut: 4%%)")
    args = parser.parse_args()

    if args.btc and args.interval == 300:
        args.interval = 60
        log.info("Mode BTC detecte -- intervalle ajuste automatiquement a 60s "
                 "(utilise --interval pour forcer une autre valeur).")

    mode_label = "DEMO (paper trading)" if args.demo else "LIVE -- ORDRES REELS"
    market_mode = "soccer" if args.soccer else ("btc" if args.btc else "macro")
    sep = "=" * 62

    print("\n" + sep)
    print("   KALSHI MACRO ALPHA ENGINE V3")
    print(sep)
    log.info(f"Mode          : {mode_label}")
    log.info(f"Type marche   : {'BTC 15min (modele math, 0 cout Claude)' if args.btc else ('SOCCER (resultat de match)' if args.soccer else 'MACRO (economique)')}")
    log.info(f"Capital       : ${args.capital:,.2f}")
    log.info(f"Stop loss/jour: ${args.max_daily_loss:,.2f}")
    log.info(f"Max trades/cyc: {args.max_trades_cycle}")
    interval_label = f"{args.interval}s" if args.interval < 90 else f"{args.interval // 60} min"
    log.info(f"Boucle        : {'OUI -- toutes les ' + interval_label if args.loop else 'NON'}")
    log.info(f"Kalshi        : {'CLE CHARGEE' if KALSHI_KEY_ID else 'PAS DE CLE -- donnees fictives'}")
    log.info(f"Claude        : {'CLE OK' if ANTHROPIC_API_KEY else 'MANQUANTE'}")
    print(sep + "\n")

    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY manquant dans .env -- impossible de continuer.")
        sys.exit(1)

    if not args.demo and not KALSHI_KEY_ID:
        log.error("KALSHI_KEY_ID manquant dans .env -- impossible de trader en LIVE.")
        log.error("Ajoutez --demo pour tester sans cle Kalshi.")
        sys.exit(1)

    if not args.demo and not KALSHI_PRIV_KEY.strip():
        log.error("KALSHI_PRIVATE_KEY manquant dans .env -- impossible de signer les ordres.")
        sys.exit(1)

    if not args.market and not args.scan and not args.btc:
        parser.print_help()
        print("\nExemples :")
        print("  python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --loop                  # LIVE macro")
        print("  python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --demo --loop           # DEMO macro")
        print("  python kalshi_alpha_bot.py --market KXEPLGAME-XXX --soccer --demo --loop     # DEMO soccer")
        print("  python kalshi_alpha_bot.py --scan --loop --max-daily-loss 100                # LIVE scan")
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
                log.info("Mode unique -- ajoute --loop pour tourner en continu.")
                break

            cycle += 1
            countdown(args.interval, cycle)

    except KeyboardInterrupt:
        print(f"\n\n  Arret -- {cycle} cycle(s) -- {total_trades} trade(s) total\n")
        log.info(f"Bot arrete proprement apres {cycle} cycles et {total_trades} trades.")


if __name__ == "__main__":
    main()
