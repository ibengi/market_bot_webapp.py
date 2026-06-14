"""
KALSHI MACRO ALPHA ENGINE V3
Bot de trading institutionnel — version corrigee

INSTALLATION:
    pip install anthropic requests python-dotenv flask flask-cors cryptography

FICHIER .env :
    ANTHROPIC_API_KEY=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
    KALSHI_KEY_ID=b6fb1530-999b-481a-862f-5babdf528c6f
    KALSHI_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEA2aJyyFw/O6bV2cmN7YW8quNT2oYb8DDpdmeoRgz6k2h6LQMk
d1+KLtQVzEwI97vXeuJpDLCnzXQhgXVhM9xqhhUXXm3CS9qC6LctXf8QMFHuc5lE
WPbzuNbwYv+PJFFfZ9icplmx3qQ+fViSbU8W6uIFByLIxO7vUOnny2qPkD6u34IF
3KIEueC2QvSiPBsKCBZODq12SbV5IHwcbtHlEe7wEXMSxUPSiBEnF/6bjpA9ysOn
RRFRw9WTA9QzME48XSLyO2Rk1ktW1NEnhry1LBk4owudMacgBtYn0fseB78I6c0l
rXbCI1/ifdb1vcylsTW1HPlB5nNttM/D0m587QIDAQABAoIBAD126K33h5BETQ9G
IkRbye4FZ/BGgetzFOxw2BB4p+gr0J2Xzpu8Kt2Q3lslej7lGTVGbl68IZgf3Tqf
uQUZkiguGrx7iS09GE27Nh/e4maTLSIOvkPV8v1YDuoWvQmHxcchYRSGLnrvrgpe
knQ2qwVJMhxS0Zr01Dfo56MGhGc7FzjsyRXbyVdAKISn/EKFxARSN+2e1Xlv1ub2
IP0lXG6PJRKyp9QMevv6m3QlCUhxMgrSlq66gLl0A/YXc8FJ1Aug1TAS9gOYl4N/
2b60Gxox3p0bT02W+fsFrsQwlmxtzrhtFUxArlA5+m0Ca8BqcMA4OlrYDRWQPZ46
BJh1p80CgYEA3dPxx4Zvx8Y4Int79YWdz9usnBnQrsa6/PWNpMfbE4IJitSCqC+l
5TqVDKVaXHQOQ7XyWJJqMU51yJeMLQCMkT18htNQHyrDw8aAjCKIKfcf5rGOgU0g
eU4DxAoxq9KPH66Dfcpcah6XBhiu6P+T4JzzLXUE/gUWnfPzDzCbF2cCgYEA+yki
XnimvBnFJ9Dsxi7HKTAMFS7flfN2aDZQZ/pOOST1tRnd//bqG7aphMjpzBdch4Sb
EZLk4F2hOE9FlAOrMDNDp3Zn+g+9zgC6SYDrsrDKVqVuuAwbU+pF0DRJfG3A0Ull
7i+d7Ne7j2lO0lmx0UFV9GUvj2PB+8yKZizr+IsCgYEAqoZfew5xp5VNWdSrehZ5
NmycyHNf5LuIyyTI+j9Yvipcw4iR2FCoqwrZwZrrV98viAW6gzQqxjSWftlhPMcL
SCuIKu0OzqbFTx6tewyguAHtWB5ueh+cMyQx6OYdQQTgWW6CTpTdbgaN+nXHH/44
45ZDqlImwE+RD2OVGbw3vUkCgYEAwk/Frk8ruBU76h0CQiWIof3xKyZThsCQF/oF
ZRxLDnzgt5bmoRRXdM9yATAraWGdjZ7zFbqO5mKpy1XRH71i4OyYZ+P8d4NcNhds
CFf8ggey0yw0J6H+NoLmNjltrR2AcqqVeJxQUx9olYBEogsQvjMNHAJ4tDfaqUNK
w9f3TBcCgYEAhARiPrqkCjYzwwb3XXwbfSCGwX9u5LIvRBw1KdswwyM+Do6KQZnG
IkE0Zr+sSlGddphKbL4e4FGQT3gKAOXrz1Pr57JWxSz1nD3D/Qp9wV0JyeQXse/J
O3nsPWuJwf0jV/bDFhIYJ0ZgS3VLlEw5SoPaXah0e1I6JhRcj78kVjM=
-----END RSA PRIVATE KEY-----


USAGE:
    python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --demo --loop
    python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --capital 500 --loop
"""

import os, json, time, sys, argparse, logging, base64
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
import requests
import anthropic

load_dotenv()

# ── Logging UTF-8 (fix Windows) ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("kalshi_alpha.log", encoding="utf-8"),
        logging.StreamHandler(stream=open(sys.stdout.fileno(), mode="w",
                                          encoding="utf-8", closefd=False)),
    ],
)
log = logging.getLogger("KalshiAlpha")

# ── Variables d'environnement ─────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
KALSHI_KEY_ID     = os.getenv("KALSHI_KEY_ID", "")
KALSHI_PRIV_KEY   = os.getenv("KALSHI_PRIVATE_KEY", "").replace("\\n", "\n")

KALSHI_BASE_URL   = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL   = "https://demo-api.kalshi.co/trade-api/v2"

KALSHI_FEE_RATE   = 0.0245
MIN_EDGE          = 0.08
MIN_CONFIDENCE    = 7

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es KALSHI MACRO ALPHA ENGINE V2.
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
- Si edge < 8% OU confiance < 7 : verdict = AUCUN TRADE
- verdict uniquement parmi : ACHETER YES / ACHETER NO / ATTENDRE / AUCUN TRADE
- taille_position uniquement parmi : 0.5% / 1% / 2% / 5% / 10%
"""

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
                log.warning("KALSHI_PRIVATE_KEY invalide — format PEM attendu")
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
            sig = self._pk.sign(msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256())
            return {
                "Content-Type": "application/json",
                "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            }
        except Exception as e:
            log.warning(f"Erreur signature: {e}")
            return {"Content-Type": "application/json"}

    def _req(self, method: str, path: str, **kw):
        headers = self._sign_headers(method, self.base_url + path)
        return self.session.request(method.upper(), self.base_url + path,
                                    headers=headers, timeout=15, **kw)

    def get_market(self, ticker: str) -> dict:
        if not KALSHI_KEY_ID:
            return {}
        try:
            r = self._req("GET", f"/markets/{ticker}")
            r.raise_for_status()
            return r.json().get("market", {})
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
            return r.json().get("markets", [])
        except Exception as e:
            log.error(f"Erreur get_active_markets: {e}")
            return []

    def place_order(self, ticker: str, side: str, count: int,
                    price: int, dry_run: bool = True) -> dict:
        if dry_run or self.demo:
            log.info(f"[DRY RUN] {side.upper()} {count}x {ticker} @ {price}c")
            return {"status": "dry_run", "ticker": ticker,
                    "side": side, "count": count, "price": price}
        try:
            payload = {
                "ticker": ticker,
                "client_order_id": f"alpha_{int(time.time())}",
                "type": "limit", "action": "buy", "side": side, "count": count,
                "yes_price": price if side == "yes" else 100 - price,
                "no_price":  price if side == "no"  else 100 - price,
            }
            r = self._req("POST", "/portfolio/orders", json=payload)
            r.raise_for_status()
            result = r.json()
            log.info(f"ORDER PLACED: {result}")
            return result
        except Exception as e:
            log.error(f"Erreur place_order: {e}")
            return {"error": str(e)}


# ── Moteur Claude avec retry ──────────────────────────────────────────────────
class AlphaEngine:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def analyse(self, market_data: dict, context: str = "",
                retries: int = 3, delay: int = 5) -> dict:
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
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
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
    def __init__(self, kalshi: KalshiClient, capital: float, demo: bool = True):
        self.kalshi  = kalshi
        self.capital = capital
        self.demo    = demo
        self.trades  = []

    def compute_contracts(self, taille_pct: str, price_cents: int) -> int:
        pct = {"0.5%":.005,"1%":.01,"2%":.02,"5%":.05,"10%":.10}.get(taille_pct, .01)
        if price_cents <= 0:
            return 0
        return max(1, int(self.capital * pct / (price_cents / 100)))

    def execute(self, market_data: dict, analysis: dict) -> Optional[dict]:
        p10      = analysis.get("phase10", {})
        verdict  = p10.get("verdict", "AUCUN TRADE")
        edge     = p10.get("edge", 0)
        confiance= p10.get("confiance", 0)
        taille   = p10.get("taille_position", "0%")
        ticker   = market_data.get("ticker", "?")

        if verdict == "AUCUN TRADE":
            log.info(f"[{ticker}] AUCUN TRADE — edge={edge:.1%} conf={confiance}/10")
            return None

        if edge < MIN_EDGE or confiance < MIN_CONFIDENCE:
            log.warning(f"[{ticker}] BLOQUE — edge={edge:.1%} conf={confiance}/10")
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
            log.warning("Nombre de contrats = 0 — trade annule.")
            return None

        result = self.kalshi.place_order(ticker, side, count, price, dry_run=self.demo)

        trade_log = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker, "verdict": verdict,
            "side": side, "count": count, "price": price,
            "edge": edge, "ev_nette": p10.get("ev_nette", 0),
            "grade": p10.get("grade", ""), "confiance": confiance,
            "taille_position": taille, "order": result,
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
            "running":      True,
            "cycle":        cycle,
            "last_ticker":  ticker,
            "last_verdict": p10.get("verdict", "AUCUN TRADE"),
            "last_edge":    p10.get("edge", 0),
            "last_ev":      p10.get("ev_nette") or p8.get("ev_nette", 0),
            "last_grade":   p10.get("grade", "D"),
            "last_reason":  p10.get("raison_principale", ""),
            "last_risk":    p10.get("risque_principal", ""),
            "last_update":  datetime.now().isoformat(),
            "last_conf":    p10.get("confiance", 0),
            "last_risque":  p10.get("risque", 0),
            "last_size":    p10.get("taille_position", ""),
            "prob_reelle":  p10.get("prob_reelle", 0),
            "prob_marche":  p10.get("prob_marche", 0),
            "ev_brute":     p10.get("ev_brute") or p8.get("ev_brute", 0),
            "risque_exogene": p10.get("risque_exogene", ""),
            "score_qualite":    p9.get("qualite_donnees", 0),
            "score_confiance":  p9.get("confiance_statistique", 0),
            "score_risque":     p9.get("risque", 0),
            "score_volatilite": p9.get("volatilite_score", 0),
            "score_edge":       p9.get("edge_score", 0),
            "scenarios":        p5.get("scenarios", []),
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
def print_report(ticker: str, analysis: dict):
    p10 = analysis.get("phase10", {})
    p8  = analysis.get("phase8",  {})
    verdict = p10.get("verdict", "AUCUN TRADE")
    grade   = p10.get("grade", "")
    icon    = {"ACHETER YES":"[YES]","ACHETER NO":"[NO]",
               "ATTENDRE":"[WAIT]","AUCUN TRADE":"[SKIP]"}.get(verdict, verdict)
    print(f"""
{'='*62}
  KALSHI ALPHA ENGINE V2 -- {ticker}
  GRADE {grade:<4}  {icon}  {verdict}
{'='*62}
  Prob reelle   : {p10.get('prob_reelle',0):.1%}
  Prob marche   : {p10.get('prob_marche',0):.1%}
  Edge          : {p10.get('edge',0):.1%}
  EV brute      : {p8.get('ev_brute',0):.1%}
  EV nette      : {p8.get('ev_nette',0):.1%}
  Confiance     : {p10.get('confiance',0)}/10
  Risque        : {p10.get('risque',0)}/10
  Position      : {p10.get('taille_position','N/A')}
{'='*62}
  Raison        : {p10.get('raison_principale','')[:58]}
  Risque princ. : {p10.get('risque_principal','')[:58]}
  Risque exog.  : {p10.get('risque_exogene','')[:58]}
{'='*62}
""")


# ── Compte a rebours ──────────────────────────────────────────────────────────
def countdown(seconds: int, next_cycle: int):
    total = seconds
    for remaining in range(seconds, 0, -1):
        m, s = divmod(remaining, 60)
        filled = int((total - remaining) / total * 28)
        bar = "#" * filled + "-" * (28 - filled)
        sys.stdout.write(
            f"\r  Cycle #{next_cycle} dans {m:02d}:{s:02d}  [{bar}]  (Ctrl+C pour arreter)")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 75 + "\r")
    sys.stdout.flush()


# ── Cycle d'analyse ───────────────────────────────────────────────────────────
def run_cycle(args, kalshi: KalshiClient, engine: AlphaEngine,
              manager: TradeManager, cycle: int) -> int:
    print(f"\n{'='*62}")
    print(f"  CYCLE #{cycle}  --  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{'='*62}")

    trades_this_cycle = 0

    if args.market:
        log.info(f"Analyse du marche : {args.market}")
        market_data = kalshi.get_market(args.market)

        if not market_data:
            log.warning("Donnees Kalshi non disponibles — donnees fictives utilisees")
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

        log.info("Lancement analyse Claude (10 phases)...")
        analysis = engine.analyse(market_data, context=args.context)

        if analysis:
            print_report(args.market, analysis)
            manager.save_state(analysis, args.market, cycle)
            trade = manager.execute(market_data, analysis)
            if trade:
                trades_this_cycle += 1
                log.info(f"Trade execute: {trade.get('verdict')} | "
                         f"Edge: {trade.get('edge',0):.1%} | "
                         f"Position: {trade.get('taille_position','')}")
        else:
            log.error("Analyse vide — aucun trade ce cycle.")

    elif args.scan:
        log.info("Scan des marches economiques actifs...")
        markets = kalshi.get_active_markets("economic")
        if not markets:
            log.warning("Aucun marche trouve — verifie KALSHI_KEY_ID dans .env")
            return 0
        log.info(f"{len(markets)} marches trouves.")
        for market in markets:
            ticker = market.get("ticker", "")
            log.info(f"--- Analyse: {ticker} ---")
            analysis = engine.analyse(market, context=args.context)
            if analysis:
                print_report(ticker, analysis)
                manager.save_state(analysis, ticker, cycle)
                trade = manager.execute(market, analysis)
                if trade:
                    trades_this_cycle += 1
            time.sleep(4)

    return trades_this_cycle


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Kalshi Macro Alpha Engine V3")
    parser.add_argument("--market",   type=str,   help="Ticker (ex: KXCPI-26JUN-T0.1)")
    parser.add_argument("--scan",     action="store_true")
    parser.add_argument("--demo",     action="store_true", help="Paper trading sans ordre reel")
    parser.add_argument("--capital",  type=float, default=500.0)
    parser.add_argument("--context",  type=str,   default="")
    parser.add_argument("--loop",     action="store_true", help="Boucle automatique")
    parser.add_argument("--interval", type=int,   default=300, help="Secondes entre cycles (defaut: 300)")
    args = parser.parse_args()

    print("\n" + "="*62)
    print("   KALSHI MACRO ALPHA ENGINE V3")
    print("="*62)
    log.info(f"Mode     : {'DEMO (paper trading)' if args.demo else 'LIVE'}")
    log.info(f"Capital  : ${args.capital:,.2f}")
    log.info(f"Boucle   : {'OUI — toutes les ' + str(args.interval//60) + ' min' if args.loop else 'NON'}")
    log.info(f"Kalshi   : {'CLE CHARGEE' if KALSHI_KEY_ID else 'PAS DE CLE — donnees fictives'}")
    log.info(f"Claude   : {'CLE OK' if ANTHROPIC_API_KEY else 'MANQUANTE'}")
    print("="*62 + "\n")

    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY manquant dans .env — impossible de continuer.")
        sys.exit(1)

    if not args.market and not args.scan:
        parser.print_help()
        print("\nExemples :")
        print("  python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --demo --loop")
        print("  python kalshi_alpha_bot.py --market KXCPI-26JUN-T0.1 --capital 500 --loop")
        print("  python kalshi_alpha_bot.py --scan --demo --loop")
        return

    kalshi  = KalshiClient(demo=args.demo)
    engine  = AlphaEngine()
    manager = TradeManager(kalshi, args.capital, demo=args.demo)

    cycle, total_trades = 1, 0
    try:
        while True:
            n = run_cycle(args, kalshi, engine, manager, cycle)
            total_trades += n
            print(f"\n  Cycle #{cycle} termine — {n} trade(s) — Total: {total_trades}")

            if not args.loop:
                log.info("Mode unique — ajoute --loop pour tourner en continu.")
                break

            cycle += 1
            countdown(args.interval, cycle)

    except KeyboardInterrupt:
        print(f"\n\n  Arret — {cycle} cycle(s) — {total_trades} trade(s) total\n")
        log.info(f"Bot arrete proprement apres {cycle} cycles et {total_trades} trades.")


if __name__ == "__main__":
    main()
