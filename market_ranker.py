"""
market_ranker.py — v1 (2026-07-11)  |  score_version = "rq-1.0"
Phase 2 : classement des marches par QUALITE D'EXECUTION / TRADABILITE.

Ce module mesure si un marche est TECHNIQUEMENT tradable. Il ne mesure PAS
la rentabilite future, ne contient AUCUNE probabilite issue d'une IA, et
n'envoie AUCUN ordre (garanti par test statique : aucune reference aux
fonctions d'envoi d'ordre).

Depend uniquement de market_scanner (MarketSnapshot) et de la stdlib.

════════════════════════════════════════════════════════════════════════════
MATHEMATIQUE DU SCORE (deterministe, documente, 0..100)
════════════════════════════════════════════════════════════════════════════
Chaque composante est bornee 0..100. Le total est une moyenne ponderee :

    total = 0.20*spread + 0.15*liquidity + 0.15*volume + 0.10*open_interest
          + 0.10*time  + 0.10*book_quality + 0.10*fill_proxy + 0.10*stability

Somme des poids = 1.00. Constantes de reference en tete de module.
PORTE DURE : si book_quality_score == 0 (carnet vide ou incoherent), le
total est PLAFONNE a 25 quelles que soient les autres composantes — un
marche sans carnet n'est pas tradable, peu importe son volume passe.

1) spread_score  = 100 * (1 - S_eff / MAX_SPREAD)      borne a [0,100]
   ou S_eff = max(spread_cents, 100 * spread_rel) et
   spread_rel = spread_cents / min(yes_mid, no_mid).
   -> combine spread ABSOLU et RELATIF : 2c sur un mid a 4c est enorme
   (50 % relatif) meme si 2c parait petit en absolu.

2) liquidity_score = 100 * ln(1+L) / ln(1+L_REF)       borne, L_REF=10 000$
   Echelle logarithmique : passer de 0 a 1 000$ compte plus que de
   100 000 a 101 000.

3) volume_score = 100 * ln(1+V) / ln(1+V_REF)          V_REF=10 000 contrats
   Si volume_24h existe : V = 0.5*volume + 0.5*volume_24h (activite recente).

4) open_interest_score = 100 * ln(1+OI) / ln(1+OI_REF) OI_REF=5 000

5) time_score : trapeze sur le temps restant t (minutes)
   t < MIN_MINUTES              -> 0    (rejete de toute facon)
   MIN..T_OK (30 min)           -> rampe lineaire 40..100
   T_OK..T_LONG (14 jours)      -> 100
   au-dela                      -> decroit lineairement vers 30 a 90 jours
   -> penalise l'expiration imminente ET l'immobilisation tres longue.

6) book_quality_score = 100 si bid ET ask presents des DEUX cotes,
   0 si carnet vide/incoherent, puis penalites :
   -40 si le meilleur ask est extreme (<=3c ou >=97c) : quasi aucun gain
   possible cote acheteur, remplissage improbable cote vendeur — un marche
   a 99c ne peut PAS etre favorise par son seul volume (regle imposee).
   -20 si un seul cote etait reellement cote (l'autre derive).

7) fill_probability_score — PROXY DETERMINISTE, PAS UNE PROBABILITE MESUREE.
   Mesurer une vraie probabilite de fill exige d'envoyer des ordres, ce que
   cette phase s'interdit. Proxy documente :
       fill = 100 * (0.45*T_spread + 0.35*T_activite + 0.20*T_profondeur)
   T_spread    = 1 - min(1, spread_cents/8)        (serre = remplissable)
   T_activite  = ln(1+V)/ln(1+V_REF)               (ca traite = ca remplit)
   T_profondeur= ln(1+L)/ln(1+L_REF)
   Limite reconnue : ce proxy sera CALIBRE en phase d'execution contre les
   fills reels ; d'ici la il ordonne, il ne predit pas.

8) stability_score : mesure sur l'HISTORIQUE (market_snapshots_history.json)
   avec au moins RANKER_MIN_BOOK_OBSERVATIONS observations :
       penalites = 3*sigma(mid) + 4*sigma(spread) + 60*freq_carnet_vide
                 + 8*nb_flaps_liquidite
       stability = 100 - penalites, borne [0,100]
   sigma en cents (ecart-type population). freq = part des observations a
   carnet vide. flap = transition vide<->cote entre deux observations.
   Moins de RANKER_MIN_BOOK_OBSERVATIONS observations -> score NEUTRE 50 et
   drapeau insufficient_history=True (jamais un motif d'exclusion : un
   marche nouveau n'est pas un mauvais marche).

ELIGIBILITE (eligible=True) : snapshot inclus par le scanner
   ET spread_cents <= RANKER_MAX_SPREAD
   ET fill_probability_score >= 100*RANKER_MIN_FILL_PROBABILITY
   ET total_score >= RANKER_MIN_SCORE
Sinon exclusion_reason parmi : (raison du scanner) | spread_too_wide |
low_fill_probability | below_min_score.
════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import math
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from statistics import pstdev
from typing import Optional

from market_scanner import MarketSnapshot, ScanConfig, run_scan, _save_json

log = logging.getLogger("RANKER")

SCORE_VERSION = "rq-1.0"

# Constantes de reference (documentees ci-dessus)
L_REF, V_REF, OI_REF = 10_000.0, 10_000.0, 5_000.0
T_OK_MIN, T_LONG_MIN, T_MAX_MIN = 30.0, 14 * 1440.0, 90 * 1440.0
EXTREME_LOW_C, EXTREME_HIGH_C = 3, 97

WEIGHTS = {"spread": 0.20, "liquidity": 0.15, "volume": 0.15,
           "open_interest": 0.10, "time": 0.10, "book_quality": 0.10,
           "fill_proxy": 0.10, "stability": 0.10}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

def _env_f(name, default):
    try: return float(os.getenv(name, str(default)))
    except ValueError: return default

def _env_i(name, default):
    try: return int(os.getenv(name, str(default)))
    except ValueError: return default

class RankConfig:
    MIN_SCORE             = _env_f("RANKER_MIN_SCORE", 40.0)
    TOP_N                 = _env_i("RANKER_TOP_N", 50)
    HISTORY_WINDOW        = _env_i("RANKER_HISTORY_WINDOW", 30)
    MIN_BOOK_OBSERVATIONS = _env_i("RANKER_MIN_BOOK_OBSERVATIONS", 5)
    MAX_SPREAD            = _env_i("RANKER_MAX_SPREAD", 6)
    MIN_FILL_PROBABILITY  = _env_f("RANKER_MIN_FILL_PROBABILITY", 0.30)
    HISTORY_FILE          = os.getenv("RANKER_HISTORY_FILE",
                                      "market_snapshots_history.json")
    RANKINGS_FILE         = os.getenv("RANKER_RANKINGS_FILE",
                                      "market_rankings.json")
    REPORT_FILE           = os.getenv("RANKER_REPORT_FILE",
                                      "market_ranker_report.json")

RCFG = RankConfig()

# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class MarketQualityScore:
    ticker: str
    total_score: float
    liquidity_score: float
    spread_score: float
    volume_score: float
    open_interest_score: float
    time_score: float
    book_quality_score: float
    fill_probability_score: float
    stability_score: float
    exclusion_reason: Optional[str]
    eligible: bool
    score_version: str = SCORE_VERSION
    category: str = "Other"
    title: Optional[str] = None
    insufficient_history: bool = False
    observations: int = 0

# ── Composantes (chacune 0..100, deterministe) ───────────────────────────────

def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))

def _log_scale(v: Optional[float], ref: float) -> float:
    v = max(0.0, float(v or 0.0))
    return _clamp(100.0 * math.log1p(v) / math.log1p(ref))

def spread_score(s: MarketSnapshot, cfg: RankConfig = RCFG) -> float:
    if s.spread_yes is None or s.yes_mid is None or s.no_mid is None:
        return 0.0
    rel = s.spread_yes / max(1, min(s.yes_mid, s.no_mid))    # spread relatif
    s_eff = max(float(s.spread_yes), 100.0 * rel)
    return _clamp(100.0 * (1.0 - s_eff / max(1.0, float(cfg.MAX_SPREAD) * 2)))

def volume_score(s: MarketSnapshot) -> float:
    v = s.volume or 0.0
    if s.volume_24h is not None:
        v = 0.5 * v + 0.5 * s.volume_24h        # activite recente ponderee
    return _log_scale(v, V_REF)

def liquidity_score(s: MarketSnapshot) -> float:
    return _log_scale(s.liquidity, L_REF)

def open_interest_score(s: MarketSnapshot) -> float:
    return _log_scale(s.open_interest, OI_REF)

def time_score(s: MarketSnapshot, cfg_scan_min: float = 5.0) -> float:
    t = s.minutes_remaining
    if t is None:
        return 50.0                              # inconnu = neutre, documente
    if t < cfg_scan_min:
        return 0.0
    if t <= T_OK_MIN:
        return _clamp(40.0 + 60.0 * (t - cfg_scan_min) / (T_OK_MIN - cfg_scan_min))
    if t <= T_LONG_MIN:
        return 100.0
    if t >= T_MAX_MIN:
        return 30.0
    return _clamp(100.0 - 70.0 * (t - T_LONG_MIN) / (T_MAX_MIN - T_LONG_MIN))

def book_quality_score(s: MarketSnapshot) -> float:
    if s.yes_bid is None or s.yes_ask is None \
            or s.no_bid is None or s.no_ask is None:
        return 0.0                               # carnet vide/incoherent
    score = 100.0
    if s.yes_ask >= EXTREME_HIGH_C or s.yes_ask <= EXTREME_LOW_C \
            or s.no_ask >= EXTREME_HIGH_C or s.no_ask <= EXTREME_LOW_C:
        score -= 40.0                            # prix extreme : regle imposee
    raw = s.raw_market or {}
    quoted_sides = sum(1 for k in ("yes_bid", "yes_ask", "no_bid", "no_ask")
                       if raw.get(k) not in (None, 0, "0", ""))
    if quoted_sides <= 2:
        score -= 20.0                            # un seul cote reellement cote
    return _clamp(score)

def fill_probability_score(s: MarketSnapshot) -> float:
    """PROXY deterministe (voir en-tete). PAS une probabilite mesuree."""
    if s.spread_yes is None:
        return 0.0
    t_spread = 1.0 - min(1.0, s.spread_yes / 8.0)
    t_act    = math.log1p(max(0.0, s.volume or 0.0)) / math.log1p(V_REF)
    t_depth  = math.log1p(max(0.0, s.liquidity or 0.0)) / math.log1p(L_REF)
    return _clamp(100.0 * (0.45 * t_spread + 0.35 * min(1, t_act)
                           + 0.20 * min(1, t_depth)))

def stability_score(obs: list, cfg: RankConfig = RCFG) -> (float, bool):
    """obs = historique du ticker (liste chronologique de dicts).
       Retourne (score, insufficient_history)."""
    if len(obs) < cfg.MIN_BOOK_OBSERVATIONS:
        return 50.0, True                        # neutre, jamais exclusif
    mids    = [o["yes_mid"] for o in obs if o.get("yes_mid") is not None]
    spreads = [o["spread_yes"] for o in obs if o.get("spread_yes") is not None]
    empties = [1 if o.get("empty") else 0 for o in obs]
    flaps = sum(1 for a, b in zip(empties, empties[1:]) if a != b)
    pen = 0.0
    if len(mids) >= 2:    pen += 3.0 * pstdev(mids)
    if len(spreads) >= 2: pen += 4.0 * pstdev(spreads)
    pen += 60.0 * (sum(empties) / len(empties))
    pen += 8.0 * flaps
    return _clamp(100.0 - pen), False

# ── Historique multi-scans ───────────────────────────────────────────────────

def load_history(path: str = None) -> dict:
    path = path or RCFG.HISTORY_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Historique illisible ({e}) -- repart de zero.")
        return {}

def update_history(history: dict, snaps: list,
                   cfg: RankConfig = RCFG,
                   now: Optional[datetime] = None) -> dict:
    """Ajoute une observation par ticker, fenetre glissante HISTORY_WINDOW.
       Les tickers expires/disparus sont purges quand leur derniere
       observation sort de la fenetre temporelle implicite."""
    ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    seen = set()
    for s in snaps:
        if not s.ticker:
            continue
        seen.add(s.ticker)
        rec = {"ts": ts, "yes_bid": s.yes_bid, "yes_ask": s.yes_ask,
               "yes_mid": s.yes_mid, "spread_yes": s.spread_yes,
               "volume": s.volume, "liquidity": s.liquidity,
               "empty": s.exclusion_reason == "no_liquidity"}
        lst = history.setdefault(s.ticker, [])
        lst.append(rec)
        if len(lst) > cfg.HISTORY_WINDOW:
            del lst[:len(lst) - cfg.HISTORY_WINDOW]
    # purge des tickers absents depuis toute la fenetre (marches regles)
    stale = [t for t, lst in history.items()
             if t not in seen and len(lst) >= cfg.HISTORY_WINDOW]
    for t in stale:
        history.pop(t, None)
    return history

# ── Scoring d'un snapshot ────────────────────────────────────────────────────

def score_market(s: MarketSnapshot, history: dict,
                 cfg: RankConfig = RCFG) -> MarketQualityScore:
    obs = history.get(s.ticker, [])
    st, insuff = stability_score(obs, cfg)
    comp = {
        "spread":        spread_score(s, cfg),
        "liquidity":     liquidity_score(s),
        "volume":        volume_score(s),
        "open_interest": open_interest_score(s),
        "time":          time_score(s),
        "book_quality":  book_quality_score(s),
        "fill_proxy":    fill_probability_score(s),
        "stability":     st,
    }
    total = round(sum(WEIGHTS[k] * v for k, v in comp.items()), 2)
    # PORTE DURE (documentee en-tete) : carnet intradable = total plafonne.
    # Sans carnet, volume/OI/temps ne rendent PAS le marche tradable.
    if comp["book_quality"] == 0.0:
        total = min(total, 25.0)

    reason = s.exclusion_reason                 # herite du scanner d'abord
    if reason is None:
        if s.spread_yes is not None and s.spread_yes > cfg.MAX_SPREAD:
            reason = "spread_too_wide"
        elif comp["fill_proxy"] < 100.0 * cfg.MIN_FILL_PROBABILITY:
            reason = "low_fill_probability"
        elif total < cfg.MIN_SCORE:
            reason = "below_min_score"

    return MarketQualityScore(
        ticker=s.ticker, total_score=total,
        liquidity_score=round(comp["liquidity"], 2),
        spread_score=round(comp["spread"], 2),
        volume_score=round(comp["volume"], 2),
        open_interest_score=round(comp["open_interest"], 2),
        time_score=round(comp["time"], 2),
        book_quality_score=round(comp["book_quality"], 2),
        fill_probability_score=round(comp["fill_proxy"], 2),
        stability_score=round(st, 2),
        exclusion_reason=reason, eligible=reason is None,
        category=s.category, title=s.title,
        insufficient_history=insuff, observations=len(obs),
    )

# ── Classement complet + rapports ────────────────────────────────────────────

def run_ranking(client, scan_cfg: ScanConfig = None,
                cfg: RankConfig = RCFG, save: bool = True,
                now: Optional[datetime] = None,
                snapshots: Optional[list] = None) -> dict:
    """Scanne (ou reutilise des snapshots fournis), met a jour l'historique,
       score, classe. AUCUN ordre : ce module n'expose ni n'appelle aucune
       fonction d'envoi d'ordre (garanti par test statique)."""
    if snapshots is None:
        scan = run_scan(client, cfg=scan_cfg or ScanConfig(),
                        save=save, now=now)
        snapshots = scan["snapshots"]
    history = load_history(cfg.HISTORY_FILE) if save else {}
    history = update_history(history, snapshots, cfg, now=now)
    if save:
        _save_json(cfg.HISTORY_FILE, history)

    scores = [score_market(s, history, cfg) for s in snapshots]
    scores.sort(key=lambda q: q.total_score, reverse=True)
    eligible = [q for q in scores if q.eligible]

    dist = {}
    for q in scores:
        bucket = f"{int(q.total_score // 10) * 10}-{int(q.total_score // 10) * 10 + 9}"
        dist[bucket] = dist.get(bucket, 0) + 1
    excl = {}
    for q in scores:
        if q.exclusion_reason:
            excl[q.exclusion_reason] = excl.get(q.exclusion_reason, 0) + 1
    top_by_cat = {}
    for q in eligible:
        top_by_cat.setdefault(q.category, []).append(
            {"ticker": q.ticker, "total_score": q.total_score})
    top_by_cat = {c: v[:5] for c, v in top_by_cat.items()}

    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "score_version": SCORE_VERSION,
        "weights": WEIGHTS,
        "params": {k: getattr(cfg, k) for k in
                   ("MIN_SCORE", "TOP_N", "HISTORY_WINDOW",
                    "MIN_BOOK_OBSERVATIONS", "MAX_SPREAD",
                    "MIN_FILL_PROBABILITY")},
        "markets_scored": len(scores),
        "eligible": len(eligible),
        "excluded_by_reason": excl,
        "score_distribution": dict(sorted(dist.items())),
        "insufficient_history": sum(1 for q in scores if q.insufficient_history),
        "top_tradable": [asdict(q) for q in eligible[:cfg.TOP_N]],
        "top_by_category": top_by_cat,
    }
    if save:
        _save_json(cfg.RANKINGS_FILE, [asdict(q) for q in scores])
        _save_json(cfg.REPORT_FILE, report)
        log.info(f"Classement: {len(scores)} scores ({len(eligible)} eligibles) "
                 f"-> {cfg.RANKINGS_FILE} | rapport -> {cfg.REPORT_FILE}")
    return {"scores": scores, "report": report, "history": history}
