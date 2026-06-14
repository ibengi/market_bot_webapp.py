"""
╔══════════════════════════════════════════════════════════╗
║         KALSHI MACRO ALPHA ENGINE — V2                  ║
║         Bot de trading institutionnel                    ║
║         Basé sur le système d'analyse en 10 phases      ║
╚══════════════════════════════════════════════════════════╝

INSTALLATION :
    pip install anthropic requests python-dotenv

CONFIGURATION :
    Crée un fichier .env avec :
    ANTHROPIC_API_KEY=sk-ant-...
    KALSHI_API_KEY=<ta clé Kalshi>
    KALSHI_API_SECRET=<ton secret RSA Kalshi>

USAGE :
    KALSHI_KEY_ID = os.getenv("b6fb1530-999b-481a-862f-5babdf528c6f", "")
KALSHI_PRIVATE_KEY = os.getenv("-----BEGIN RSA PRIVATE KEY-----
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

", "")

KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
"""

import os
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import requests
import anthropic

# ─── Configuration ───────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("kalshi_alpha.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("KalshiAlpha")

ANTHROPIC_API_KEY  = os.getenv("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", "")
KALSHI_API_KEY     = os.getenv("b6fb1530-999b-481a-862f-5babdf528c6f", "")
KALSHI_API_SECRET  = os.getenv("-----BEGIN RSA PRIVATE KEY-----
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
", "")

KALSHI_BASE_URL    = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL    = "https://demo-api.kalshi.co/trade-api/v2"

KALSHI_FEE_RATE    = 0.0245      # 2.45% sur les profits
MIN_EDGE           = 0.08        # 8% minimum
MIN_CONFIDENCE     = 7           # /10

# ─── Système prompt KALSHI MACRO ALPHA ENGINE V2 ─────────────────────────────

SYSTEM_PROMPT = """
Tu es KALSHI MACRO ALPHA ENGINE V2 — un analyste macroéconomique institutionnel
spécialiste des marchés prédictifs, de la théorie des probabilités, de l'analyse
des données économiques et de la gestion du risque.

Mission unique : Détecter les erreurs de prix du marché et identifier les trades
ayant une espérance mathématique positive.
JAMAIS : Prédire. Avoir raison. Trader par intuition.

Réponds UNIQUEMENT en JSON valide selon ce schéma exact :

{
  "phase1": {
    "evenement": "",
    "date": "",
    "seuil": "",
    "conditions_resolution": "",
    "prix_yes": 0,
    "prix_no": 0,
    "prob_implicite_yes": 0.0,
    "prob_implicite_no": 0.0,
    "somme_controle": 0.0,
    "incoherence_detectee": false
  },
  "phase2": {
    "tendance_recente": "",
    "moyenne": 0.0,
    "mediane": 0.0,
    "volatilite": "",
    "biais_consensus": "",
    "stabilite_statistique": ""
  },
  "phase3": {
    "nowcast": 0.0,
    "intervalle_confiance_80_min": 0.0,
    "intervalle_confiance_80_max": 0.0,
    "indicateurs_determinants": [
      {"nom": "", "poids": 0.0, "signal": ""}
    ]
  },
  "phase4": {
    "consensus_bloomberg": 0.0,
    "dispersion": "",
    "biais_historique": "",
    "ecart_nowcast_consensus": 0.0,
    "sentiment_consensus": ""
  },
  "phase5": {
    "scenarios": [
      {
        "nom": "haussier",
        "description": "",
        "probabilite": 0.0,
        "resolution_kalshi": "",
        "declencheur": ""
      },
      {
        "nom": "neutre",
        "description": "",
        "probabilite": 0.0,
        "resolution_kalshi": "",
        "declencheur": ""
      },
      {
        "nom": "baissier",
        "description": "",
        "probabilite": 0.0,
        "resolution_kalshi": "",
        "declencheur": ""
      }
    ],
    "somme_scenarios": 0.0
  },
  "phase6": {
    "prob_reelle_estimee": 0.0,
    "prob_implicite_marche": 0.0,
    "edge": 0.0,
    "grade": ""
  },
  "phase7": {
    "argument_haussier": "",
    "argument_baissier": "",
    "argument_market_maker": "",
    "argument_risk_manager": "",
    "edge_tient_apres_stress": true,
    "ajustement_prob": 0.0
  },
  "phase8": {
    "gain_potentiel": 0.0,
    "perte_potentielle": 0.0,
    "ev_brute": 0.0,
    "gain_net_apres_frais": 0.0,
    "ev_nette": 0.0,
    "qualification_ev": ""
  },
  "phase9": {
    "qualite_donnees": 0,
    "confiance_statistique": 0,
    "risque": 0,
    "volatilite_score": 0,
    "edge_score": 0,
    "score_composite": 0.0
  },
  "phase10": {
    "verdict": "",
    "prob_reelle": 0.0,
    "prob_marche": 0.0,
    "edge": 0.0,
    "ev_brute": 0.0,
    "ev_nette": 0.0,
    "confiance": 0,
    "risque": 0,
    "grade": "",
    "raison_principale": "",
    "risque_principal": "",
    "risque_exogene": "",
    "taille_position": ""
  }
}

RÈGLES CRITIQUES :
- La somme YES + NO doit être ≤ 100
- La somme des scénarios doit être exactement 100%
- EDGE = prob_reelle - prob_implicite_marche
- EV nette = (P_gain × gain × 0.9755) - (P_perte × perte)
- Verdict uniquement parmi : ACHETER YES / ACHETER NO / ATTENDRE / AUCUN TRADE
- Si edge < 8% OU confiance < 7 → verdict = AUCUN TRADE obligatoire
- Taille position uniquement : 0.5% / 1% / 2% / 5% / 10%
"""

# ─── Client Kalshi API ────────────────────────────────────────────────────────
import base64
from urllib.parse import urlparse
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    def __init__(self, demo: bool = False):
        self.base_url = KALSHI_DEMO_URL if demo else KALSHI_BASE_URL
        self.demo = demo
        self.session = requests.Session()
        self.private_key = self._load_private_key()

    def _load_private_key(self):
        if not KALSHI_KEY_ID:
            raise ValueError("KALSHI_KEY_ID manquant dans les variables Railway")

        if not KALSHI_PRIVATE_KEY:
            raise ValueError("KALSHI_PRIVATE_KEY manquant dans les variables Railway")

        private_key_text = KALSHI_PRIVATE_KEY.replace("\\n", "\n").strip()

        return serialization.load_pem_private_key(
            private_key_text.encode("utf-8"),
            password=None,
        )

    def _sign(self, method: str, path: str) -> dict:
        timestamp = str(int(time.time() * 1000))

        full_path = urlparse(self.base_url + path).path
        path_without_query = full_path.split("?")[0]

        message = f"{timestamp}{method.upper()}{path_without_query}".encode("utf-8")

        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        signature_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
        }

    def _request(self, method: str, path: str, **kwargs):
        headers = self._sign(method, path)
        url = self.base_url + path

        return self.session.request(
            method=method.upper(),
            url=url,
            headers=headers,
            timeout=10,
            **kwargs,
        )

    def get_market(self, ticker: str) -> dict:
        try:
            r = self._request("GET", f"/markets/{ticker}")
            r.raise_for_status()
            return r.json().get("market", {})
        except Exception as e:
            log.error(f"Erreur get_market({ticker}): {e}")
            return {}

    def get_active_markets(self, category: str = "economic") -> list:
        try:
            params = {"status": "open", "series_ticker": category, "limit": 50}
            r = self._request("GET", "/markets", params=params)
            r.raise_for_status()
            return r.json().get("markets", [])
        except Exception as e:
            log.error(f"Erreur get_active_markets: {e}")
            return []

    def get_balance(self) -> float:
        try:
            r = self._request("GET", "/portfolio/balance")
            r.raise_for_status()
            return r.json().get("balance", 0) / 100
        except Exception as e:
            log.error(f"Erreur get_balance: {e}")
            return 0.0

    def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        price: int,
        dry_run: bool = True,
    ) -> dict:
        if dry_run or self.demo:
            log.info(f"[DRY RUN] ORDER {side.upper()} {count}x {ticker} @ {price}¢")
            return {
                "status": "dry_run",
                "ticker": ticker,
                "side": side,
                "count": count,
                "price": price,
            }

        try:
            payload = {
                "ticker": ticker,
                "client_order_id": f"alpha_{int(time.time())}",
                "type": "limit",
                "action": "buy",
                "side": side,
                "count": count,
                "yes_price": price if side == "yes" else 100 - price,
                "no_price": price if side == "no" else 100 - price,
            }

            r = self._request("POST", "/portfolio/orders", json=payload)
            r.raise_for_status()
            result = r.json()
            log.info(f"ORDER PLACED: {result}")
            return result

        except Exception as e:
            log.error(f"Erreur place_order: {e}")
            return {"error": str(e)}


# ─── Moteur d'analyse Claude ──────────────────────────────────────────────────

class AlphaEngine:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def analyse(self, market_data: dict, context: str = "") -> dict:
        """
        Lance l'analyse complète en 10 phases via Claude.
        Retourne le JSON structuré de la décision.
        """
        market_info = f"""
MARCHÉ À ANALYSER :
- Ticker       : {market_data.get('ticker', 'N/A')}
- Titre        : {market_data.get('title', 'N/A')}
- Prix YES     : {market_data.get('yes_bid', 0)} cents
- Prix NO      : {market_data.get('no_bid', 0)} cents
- Volume       : {market_data.get('volume', 0)}
- Clôture      : {market_data.get('close_time', 'N/A')}
- Catégorie    : {market_data.get('category', 'N/A')}
- Sous-titre   : {market_data.get('subtitle', 'N/A')}

CONTEXTE ÉCONOMIQUE ADDITIONNEL :
{context if context else "Aucun contexte supplémentaire fourni."}

Lance l'analyse complète en 10 phases selon ton système.
Réponds uniquement en JSON valide.
"""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": market_info}],
            )
            raw = response.content[0].text.strip()
            # Nettoie les backticks éventuels
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.error(f"JSON invalide reçu de Claude: {e}")
            return {}
        except Exception as e:
            log.error(f"Erreur analyse Claude: {e}")
            return {}


# ─── Gestionnaire de trades ───────────────────────────────────────────────────

class TradeManager:
    def __init__(self, kalshi: KalshiClient, capital: float, demo: bool = True):
        self.kalshi  = kalshi
        self.capital = capital
        self.demo    = demo
        self.trades  = []

    def compute_contracts(self, verdict: str, taille_pct: str, price_cents: int) -> int:
        """Calcule le nombre de contrats selon la taille de position."""
        pct_map = {"0.5%": 0.005, "1%": 0.01, "2%": 0.02, "5%": 0.05, "10%": 0.10}
        pct = pct_map.get(taille_pct, 0.01)
        budget = self.capital * pct
        if price_cents <= 0:
            return 0
        return max(1, int(budget / (price_cents / 100)))

    def execute(self, market_data: dict, analysis: dict) -> Optional[dict]:
        """Exécute la décision issue de la Phase 10."""
        p10 = analysis.get("phase10", {})
        verdict  = p10.get("verdict", "AUCUN TRADE")
        edge     = p10.get("edge", 0)
        confiance = p10.get("confiance", 0)
        taille   = p10.get("taille_position", "0%")

        # ── Garde-fous ─────────────────────────────────────────────────────────
        if verdict == "AUCUN TRADE":
            log.info(f"[{market_data.get('ticker')}] AUCUN TRADE — edge={edge:.1%} conf={confiance}/10")
            return None

        if edge < MIN_EDGE or confiance < MIN_CONFIDENCE:
            log.warning(
                f"[{market_data.get('ticker')}] BLOQUÉ — edge={edge:.1%} (min {MIN_EDGE:.0%}) "
                f"conf={confiance}/10 (min {MIN_CONFIDENCE}/10)"
            )
            return None

        # ── Détermine le côté et le prix ───────────────────────────────────────
        if verdict == "ACHETER YES":
            side  = "yes"
            price = int(market_data.get("yes_bid", 50))
        elif verdict == "ACHETER NO":
            side  = "no"
            price = int(market_data.get("no_bid", 50))
        else:
            log.info(f"[{market_data.get('ticker')}] ATTENDRE")
            return None

        count = self.compute_contracts(verdict, taille, price)
        if count <= 0:
            log.warning("Nombre de contrats calculé à 0 — trade annulé.")
            return None

        # ── Passe l'ordre ──────────────────────────────────────────────────────
        result = self.kalshi.place_order(
            ticker  = market_data["ticker"],
            side    = side,
            count   = count,
            price   = price,
            dry_run = self.demo,
        )

        trade_log = {
            "timestamp":  datetime.now().isoformat(),
            "ticker":     market_data["ticker"],
            "verdict":    verdict,
            "side":       side,
            "count":      count,
            "price":      price,
            "edge":       edge,
            "ev_nette":   p10.get("ev_nette", 0),
            "grade":      p10.get("grade", ""),
            "confiance":  confiance,
            "taille":     taille,
            "order":      result,
        }
        self.trades.append(trade_log)
        self._save_trade(trade_log)
        return trade_log

    def _save_trade(self, trade: dict):
        """Sauvegarde chaque trade dans un fichier JSON de journalisation."""
        path = "kalshi_trades.json"
        history = []
        if os.path.exists(path):
            with open(path) as f:
                try:
                    history = json.load(f)
                except Exception:
                    history = []
        history.append(trade)
        with open(path, "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        log.info(f"Trade loggé → {path}")


# ─── Affichage terminal ───────────────────────────────────────────────────────

def print_report(ticker: str, analysis: dict):
    p10 = analysis.get("phase10", {})
    p6  = analysis.get("phase6", {})
    p8  = analysis.get("phase8", {})

    grade_colors = {"A+": "🟢", "A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "": "⚪"}
    verdict_icons = {
        "ACHETER YES": "✅ ACHETER YES",
        "ACHETER NO":  "✅ ACHETER NO",
        "ATTENDRE":    "⏳ ATTENDRE",
        "AUCUN TRADE": "🚫 AUCUN TRADE",
    }

    grade   = p10.get("grade", "")
    verdict = p10.get("verdict", "AUCUN TRADE")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  KALSHI MACRO ALPHA ENGINE V2  —  {ticker:<20} ║
╠══════════════════════════════════════════════════════════╣
║  {grade_colors.get(grade, '⚪')} GRADE {grade:<5}   {verdict_icons.get(verdict, verdict):<32}  ║
╠══════════════════════════════════════════════════════════╣
  Probabilité réelle   : {p10.get('prob_reelle', 0):.1%}
  Probabilité marché   : {p10.get('prob_marche', 0):.1%}
  Edge                 : {p10.get('edge', 0):.1%}
  EV brute             : {p8.get('ev_brute', 0):.1%}
  EV nette (frais)     : {p8.get('ev_nette', 0):.1%}  [{p8.get('qualification_ev', '')}]
  Confiance            : {p10.get('confiance', 0)}/10
  Risque               : {p10.get('risque', 0)}/10
  Taille position      : {p10.get('taille_position', 'N/A')}
╠══════════════════════════════════════════════════════════╣
  Raison principale    : {p10.get('raison_principale', '')[:55]}
  Risque principal     : {p10.get('risque_principal', '')[:55]}
  Risque exogène       : {p10.get('risque_exogene', '')[:55]}
╚══════════════════════════════════════════════════════════╝
""")


# ─── Compte à rebours ─────────────────────────────────────────────────────────

def countdown(seconds: int, label: str = "Prochain scan"):
    """Affiche un compte à rebours en temps réel dans le terminal."""
    import sys
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        bar_total  = 30
        bar_filled = int((seconds - remaining) / seconds * bar_total)
        bar        = "#" * bar_filled + "-" * (bar_total - bar_filled)
        line = f"\r  {label} dans {mins:02d}:{secs:02d}  [{bar}]  (Ctrl+C pour arreter)"
        sys.stdout.write(line)
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


# ─── Cycle d'analyse unique ───────────────────────────────────────────────────

def run_cycle(args, kalshi: "KalshiClient", engine: "AlphaEngine", manager: "TradeManager", cycle: int):
    """Exécute un cycle complet d'analyse (marché unique ou scan)."""
    print(f"\n{'='*60}")
    print(f"  CYCLE #{cycle}  —  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{'='*60}")

    trades_this_cycle = 0

    # ── Mode marché unique ────────────────────────────────────────────────────
    if args.market:
        log.info(f"Analyse du marche : {args.market}")
        market_data = kalshi.get_market(args.market)

        if not market_data:
            log.warning("Donnees Kalshi non disponibles — utilisation de donnees fictives pour demo")
            market_data = {
                "ticker":     args.market,
                "title":      f"Marche {args.market}",
                "yes_bid":    55,
                "no_bid":     45,
                "volume":     12500,
                "close_time": "2026-07-11T12:30:00Z",
                "category":   "economic",
                "subtitle":   "Donnees de demonstration",
            }

        log.info("Lancement de l'analyse Claude (10 phases)...")
        analysis = engine.analyse(market_data, context=args.context)

        if analysis:
            print_report(args.market, analysis)
            trade = manager.execute(market_data, analysis)
            if trade:
                trades_this_cycle += 1
                log.info(f"Trade execute : {json.dumps(trade, indent=2, ensure_ascii=False)}")
        else:
            log.error("Analyse vide — aucun trade.")

    # ── Mode scan multi-marchés ───────────────────────────────────────────────
    elif args.scan:
        log.info("Scan des marches economiques actifs...")
        markets = kalshi.get_active_markets("economic")

        if not markets:
            log.warning("Aucun marche trouve — verifie ta cle API Kalshi.")
            return 0

        log.info(f"{len(markets)} marches trouves. Analyse en cours...")

        for market in markets:
            ticker = market.get("ticker", "")
            log.info(f"--- Analyse : {ticker} ---")
            analysis = engine.analyse(market, context=args.context)

            if analysis:
                print_report(ticker, analysis)
                trade = manager.execute(market, analysis)
                if trade:
                    trades_this_cycle += 1
            else:
                log.warning(f"Analyse vide pour {ticker}")

            time.sleep(3)  # Respect rate limits

    return trades_this_cycle


# ─── Point d'entrée ───────────────────────────────────────────────────────────

SCAN_INTERVAL  = 5 * 60   # 5 minutes en secondes

def main():
    parser = argparse.ArgumentParser(description="Kalshi Macro Alpha Engine V2")
    parser.add_argument("--market",   type=str,   help="Ticker du marche a analyser (ex: KXCPI-25JUN-B3)")
    parser.add_argument("--scan",     action="store_true", help="Scanne les marches economiques actifs")
    parser.add_argument("--demo",     action="store_true", help="Mode paper trading (aucun trade reel)")
    parser.add_argument("--capital",  type=float, default=500.0, help="Capital total en USD (defaut: 500)")
    parser.add_argument("--context",  type=str,   default="", help="Contexte economique supplementaire")
    parser.add_argument("--loop",     action="store_true", help="Mode boucle automatique toutes les 5 minutes")
    parser.add_argument("--interval", type=int,   default=SCAN_INTERVAL, help="Intervalle en secondes (defaut: 300)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("   KALSHI MACRO ALPHA ENGINE V2")
    print("="*60)
    log.info(f"Mode      : {'DEMO (paper trading)' if args.demo else 'LIVE'}")
    log.info(f"Capital   : ${args.capital:,.2f}")
    log.info(f"Boucle    : {'OUI — toutes les ' + str(args.interval//60) + ' min' if args.loop else 'NON — analyse unique'}")
    print("="*60 + "\n")

    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY manquant dans .env")
        return

    if not args.market and not args.scan:
        parser.print_help()
        print("\nExemples :")
        print("  python kalshi_alpha_bot.py --market KXCPI-25JUL-B3 --demo")
        print("  python kalshi_alpha_bot.py --market KXCPI-25JUL-B3 --demo --loop")
        print("  python kalshi_alpha_bot.py --scan --demo --capital 1000 --loop")
        return

    kalshi  = KalshiClient(demo=args.demo)
    engine  = AlphaEngine()
    manager = TradeManager(kalshi, args.capital, demo=args.demo)

    cycle       = 1
    total_trades = 0

    try:
        while True:
            trades = run_cycle(args, kalshi, engine, manager, cycle)
            total_trades += trades

            print(f"\n  Cycle #{cycle} termine — {trades} trade(s) ce cycle — {total_trades} trade(s) total")

            # ── Mode unique : on s'arrête après 1 cycle ───────────────────────
            if not args.loop:
                log.info("Mode unique — fin du programme. Ajoute --loop pour tourner en continu.")
                break

            # ── Mode boucle : compte à rebours avant prochain cycle ───────────
            cycle += 1
            print()
            countdown(args.interval, label=f"Cycle #{cycle}")

    except KeyboardInterrupt:
        print("\n\n  Arret demande par l'utilisateur (Ctrl+C)")
        log.info(f"Bot arrete — {cycle} cycle(s) — {total_trades} trade(s) au total")
        print(f"  Total : {cycle} cycle(s) effectue(s) — {total_trades} trade(s) passes")
        print()


if __name__ == "__main__":
    main()
