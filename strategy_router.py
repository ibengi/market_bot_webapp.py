"""
strategy_router.py — v1 (2026-07-11)
Routeur de strategies + evaluateur de signal avec portes edge/EV STRICTES.

PRINCIPES NON NEGOCIABLES :
- Un marche sans strategie compatible est rejete: no_compatible_strategy.
- Une strategie ne peut etre routee QUE vers ses categories declarees
  (jamais la strategie BTC vers un autre type de marche).
- Aucune probabilite n'est inventee : si la strategie ne fournit pas
  model_probability, rejet no_model_probability.
- edge <= 0 ou EV net < seuil => AUCUN trade. Le score de tradabilite du
  ranker n'est JAMAIS une preuve de rentabilite.

Interface strategie (duck typing) :
    strategy.categories : list[str]           categories servies
    strategy.name       : str
    strategy.evaluate(snapshot, fresh_market, book) -> dict | None
        dict = {"side": "yes"|"no", "model_prob": float 0..1 (prob que le
                cote choisi gagne), "confidence": int 0..10,
                "reason": str, "taille": "0.5%"|"1%"|"2%" (indicatif)}
        None = pas d'avis => rejet no_model_probability.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("ROUTER")

REJECTION_REASONS = (
    "no_model_probability", "no_compatible_strategy", "insufficient_confidence",
    "no_positive_edge", "insufficient_net_edge", "negative_net_ev",
    "spread_too_wide", "no_executable_ask", "market_quality_too_low",
    "stale_book", "already_open", "risk_blocked",
    "insufficient_data_quality", "model_not_calibrated",
)

@dataclass
class GateConfig:
    MIN_MODEL_CONFIDENCE: int = 6
    MIN_GROSS_EDGE: float = 0.05
    MIN_NET_EDGE: float = 0.03
    MIN_NET_EV: float = 0.02
    MAX_ACCEPTABLE_SPREAD: int = 4
    MIN_MARKET_SCORE: float = 50.0
    MIN_FILL_PROXY: float = 40.0
    SLIPPAGE_BUFFER_CENTS: int = 1
    UNCERTAINTY_BUFFER: float = 0.01     # marge d'incertitude modele (edge)
    FEE_RATE: float = 0.07               # fallback local documente (a verifier)
    REQUIRE_CALIBRATED: bool = False     # True en LIVE : modele calibre exige

@dataclass
class SignalDecision:
    ticker: str
    accepted: bool
    rejection_reason: Optional[str] = None
    side: Optional[str] = None
    entry_price_cents: Optional[int] = None
    market_probability: Optional[float] = None
    model_probability: Optional[float] = None
    gross_edge: Optional[float] = None
    expected_gross_value: Optional[float] = None
    estimated_fees: Optional[float] = None       # $ / contrat
    expected_slippage: Optional[float] = None    # $ / contrat
    net_edge: Optional[float] = None
    net_ev: Optional[float] = None
    confidence: int = 0
    strategy: Optional[str] = None
    taille: str = "0.5%"
    reason: str = ""
    model_output: Optional[dict] = None

def estimated_fee_per_contract(price_cents: int, fee_rate: float) -> float:
    """Fallback LOCAL documente: ceil(rate*P*(1-P)) au cent, par contrat.
    Utilise uniquement avant l'ordre; apres fill, les frais API priment."""
    p = price_cents / 100.0
    return math.ceil(fee_rate * p * (1 - p) * 100) / 100.0

class StrategyRouter:
    """category -> strategie. Categories sans strategie => rejet explicite."""
    def __init__(self):
        self._by_category = {}

    def register(self, strategy):
        for cat in getattr(strategy, "categories", []):
            self._by_category[cat] = strategy

    def strategy_for(self, category: str):
        return self._by_category.get(category)

    def supported_categories(self) -> list:
        return sorted(self._by_category)

def evaluate_candidate(snapshot, fresh_market: dict, book: dict,
                       router: StrategyRouter,
                       gates: GateConfig) -> SignalDecision:
    """Toutes les portes, dans l'ordre, avec raison explicite.
    book = carnet FRAIS normalise {yes_bid,yes_ask,no_bid,no_ask,spread}."""
    tk = snapshot.ticker
    d = SignalDecision(ticker=tk, accepted=False)

    # 0) qualite de marche (le ranker mesure la TRADABILITE, pas le profit)
    q = getattr(snapshot, "quality", None)
    if q is not None:
        if q.total_score < gates.MIN_MARKET_SCORE:
            d.rejection_reason = "market_quality_too_low"; return d
        if q.fill_probability_score < gates.MIN_FILL_PROXY:
            d.rejection_reason = "market_quality_too_low"; return d

    # 1) strategie compatible ? (categorie PUIS compatibilite fine du marche)
    strat = router.strategy_for(snapshot.category)
    if strat is None:
        d.rejection_reason = "no_compatible_strategy"; return d
    if hasattr(strat, "supports") and not strat.supports(snapshot):
        d.rejection_reason = "no_compatible_strategy"; return d
    d.strategy = getattr(strat, "name", strat.__class__.__name__)

    # 2) carnet frais exploitable ?
    if not book:
        d.rejection_reason = "stale_book"; return d
    spread = book.get("spread")
    if spread is None or spread > gates.MAX_ACCEPTABLE_SPREAD:
        d.rejection_reason = "spread_too_wide"; return d

    # 3) probabilite modele INDEPENDANTE
    out = strat.evaluate(snapshot, fresh_market, book)
    if isinstance(out, dict) and out.get("rejection_reason"):
        d.rejection_reason = out["rejection_reason"]        # ex: qualite data
        d.reason = str(out.get("reason", ""))[:160]
        return d
    if not out or out.get("model_prob") is None:
        d.rejection_reason = "no_model_probability"; return d
    side = out.get("side")
    if side not in ("yes", "no"):
        d.rejection_reason = "no_model_probability"; return d
    model_p = float(out["model_prob"])
    if not (0.0 < model_p < 1.0) or model_p != model_p:
        d.rejection_reason = "no_model_probability"; return d
    d.side, d.model_probability = side, model_p
    conf = out.get("confidence", 0)
    # confiance acceptee en 0..1 (modele) ou 0..10 (legacy)
    d.confidence = int(round(conf * 10)) if isinstance(conf, float) \
        and conf <= 1.0 else int(conf)
    d.taille = out.get("taille", "0.5%")
    d.reason = str(out.get("reason", ""))[:160]
    d.model_output = out.get("model_output")

    if gates.REQUIRE_CALIBRATED and not out.get("calibrated", False):
        d.rejection_reason = "model_not_calibrated"; return d
    if d.confidence < gates.MIN_MODEL_CONFIDENCE:
        d.rejection_reason = "insufficient_confidence"; return d

    # 4) ask executable du cote achete (jamais inventer un prix)
    ask = book.get("yes_ask") if side == "yes" else book.get("no_ask")
    if ask is None or not (1 <= int(ask) <= 99):
        d.rejection_reason = "no_executable_ask"; return d
    d.entry_price_cents = int(ask)

    # 5) edge & EV apres frais + slippage (formules par cote)
    d.market_probability = d.entry_price_cents / 100.0
    d.gross_edge = d.model_probability - d.market_probability
    if d.gross_edge <= 0:
        d.rejection_reason = "no_positive_edge"; return d
    if d.gross_edge < gates.MIN_GROSS_EDGE:
        d.rejection_reason = "insufficient_net_edge"; return d

    p = d.market_probability
    d.expected_gross_value = d.model_probability * (1 - p) \
        - (1 - d.model_probability) * p              # EV brute / contrat ($)
    d.estimated_fees = estimated_fee_per_contract(d.entry_price_cents,
                                                  gates.FEE_RATE)
    d.expected_slippage = gates.SLIPPAGE_BUFFER_CENTS / 100.0
    d.net_edge = d.gross_edge - d.estimated_fees - d.expected_slippage \
        - gates.UNCERTAINTY_BUFFER
    d.net_ev = d.expected_gross_value - d.estimated_fees - d.expected_slippage \
        - gates.UNCERTAINTY_BUFFER
    if d.net_edge < gates.MIN_NET_EDGE:
        d.rejection_reason = "insufficient_net_edge"; return d
    if d.net_ev < gates.MIN_NET_EV:
        d.rejection_reason = "negative_net_ev"; return d

    d.accepted = True
    return d

# ── Strategie crypto court terme (adaptateur, ne s'applique QU'A Crypto) ────

class CryptoShortTermStrategy:
    """Adapte l'analyse BTC existante SI et SEULEMENT SI elle fournit une
    probabilite modele independante (cle 'model_prob' ou 'prob_reelle'
    distincte de la prob marche). Sinon retourne None -> rejet honnete
    no_model_probability. AUCUNE probabilite n'est fabriquee ici."""
    name = "crypto_short_term_v1"
    categories = ["Crypto"]

    def __init__(self, decision_fn=None):
        # decision_fn(snapshot, market, book) -> dict de l'analyse existante
        self.decision_fn = decision_fn

    def evaluate(self, snapshot, fresh_market, book):
        if self.decision_fn is None:
            return None
        try:
            dec = self.decision_fn(snapshot, fresh_market, book) or {}
        except Exception as e:
            log.warning(f"{self.name}: decision_fn erreur: {e}")
            return None
        verdict = str(dec.get("verdict", "")).upper()
        side = "yes" if "YES" in verdict else "no" if "NO" in verdict else None
        model_p = dec.get("model_prob", dec.get("prob_reelle"))
        market_p = dec.get("market_prob")
        if side is None or model_p is None:
            return None
        # Une prob strictement egale a la prob marche n'est PAS un modele.
        if market_p is not None and abs(float(model_p) - float(market_p)) < 1e-9:
            return None
        return {"side": side, "model_prob": float(model_p),
                "confidence": int(dec.get("confidence", 0)),
                "taille": dec.get("taille", "0.5%"),
                "reason": dec.get("reason", "")}


# ── Strategie BTC 15 minutes basee sur le modele de probabilite ──────────────

class BtcModelStrategy:
    """Strategie BTC15M : contexte via btc_context, probabilite via
    btc_probability_model. NE S'APPLIQUE QU'AUX marches de la serie BTC 15
    minutes (supports()) : ETH, sports, politique, meteo etc. restent
    no_compatible_strategy. Si le contexte echoue -> no_model_probability ;
    si la qualite des donnees est insuffisante -> insufficient_data_quality.
    AUCUNE probabilite n'est inventee."""
    name = "btc15m_model_v1"
    categories = ["Crypto"]
    SERIES_PREFIX = "KXBTC15M"

    def __init__(self, context_provider=None, model_predict=None):
        # injectables pour tests hors-ligne ; par defaut, les vrais modules
        if context_provider is None or model_predict is None:
            import btc_context as _ctx
            import btc_probability_model as _mdl
            context_provider = context_provider or _ctx.get_btc_context
            model_predict = model_predict or _mdl.predict_or_reason
        self.context_provider = context_provider
        self.model_predict = model_predict

    def supports(self, snapshot) -> bool:
        for attr in ("series_ticker", "ticker"):
            v = getattr(snapshot, attr, None) or ""
            if str(v).upper().startswith(self.SERIES_PREFIX):
                return True
        return False

    @staticmethod
    def _strike(market: dict):
        for k in ("floor_strike", "cap_strike", "strike_price"):
            v = market.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    def evaluate(self, snapshot, fresh_market, book):
        strike = self._strike(fresh_market or {})
        mins = getattr(snapshot, "minutes_remaining", None)
        if strike is None or mins is None:
            return {"rejection_reason": "no_model_probability",
                    "reason": "strike ou temps restant absent"}
        try:
            ctx = self.context_provider(strike=strike, minutes_remaining=mins)
        except Exception as e:                    # echec contexte = pas de prob
            return {"rejection_reason": "no_model_probability",
                    "reason": f"btc_context: {e}"}
        out, why = self.model_predict(ctx)
        if out is None:
            reason = ("insufficient_data_quality"
                      if why and "insufficient_data_quality" in why
                      else "no_model_probability")
            return {"rejection_reason": reason, "reason": why or ""}
        p_yes = out["probability_yes"]
        side = "yes" if p_yes >= 0.5 else "no"
        return {"side": side,
                "model_prob": p_yes if side == "yes"
                              else out["probability_no"],
                "confidence": out["confidence"],          # 0..1 -> mappee
                "calibrated": out.get("calibrated", False),
                "taille": "0.5%",
                "reason": f"{out['model_version']}: {out['reason'][:90]}",
                "model_output": out}
