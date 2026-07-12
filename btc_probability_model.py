"""
btc_probability_model.py — v1 (2026-07-12)
Modele de probabilite BTC 15 minutes.

    P(BTC termine AU-DESSUS du strike a l'expiration)

════════════════════════════════════════════════════════════════════════════
STATUT : « btc15m-baseline-0.1 » — BASELINE ANALYTIQUE, **NON VALIDEE**.
Ce modele n'a demontre AUCUNE rentabilite. Il sert de point de depart
mesurable pour le shadow mode, la calibration et le backtest. Il ne doit
JAMAIS etre utilise en live sans passer model_gatekeeper.
════════════════════════════════════════════════════════════════════════════

MATHEMATIQUE (transparente) :
Sous une marche brownienne sans derive du log-prix, a volatilite par minute
sigma_1m et t minutes restantes :

    P(S_T > K) = Phi( ln(S/K) / (sigma_1m * sqrt(t)) + mu_adj )

- Phi = CDF normale standard (via erf, aucune dependance).
- mu_adj = derive de momentum court terme, BORNEE a +/-0.25 ecart-type :
      mu_adj = clamp(momentum_norm, -0.25, 0.25)
      momentum_norm = (rendement log 5m / 5) * t / (sigma_1m * sqrt(t))
  Documentee comme heuristique ; desactivable (use_momentum=False).

CONFIANCE (distincte de la probabilite) : produit borne 0..1 de
    q_data  = data_quality_score / 100
    q_time  = clamp(t / 15, 0.2, 1.0)          (tres peu de temps = bruit)
    q_vol   = 1 si vol mesuree sur >= MIN_KLINES bougies

REGLES DURES :
- probability_yes + probability_no == 1 (exactement, par construction) ;
- jamais de NaN/inf (verifie) ;
- AUCUNE probabilite si le contexte est invalide, la qualite insuffisante,
  la volatilite nulle, ou le temps restant <= 0 : retour None + raison ;
- aucune borne artificielle destinee a provoquer des trades ;
- version + features enregistrees dans chaque sortie.

CALIBRATION : si un fichier model_calibration.json (produit par
model_calibration.fit sur donnees train) est present et correspond a
MODEL_VERSION, la probabilite brute est ajustee et "calibrated"=True.
"""

import json
import math
import os
from typing import Optional

MODEL_VERSION = "btc15m-baseline-0.1"
MIN_DATA_QUALITY = 60.0
MOMENTUM_CAP = 0.25
CALIBRATION_FILE = os.getenv("MODEL_CALIBRATION_FILE",
                             "model_calibration.json")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _load_calibration(path: str = None) -> Optional[dict]:
    path = path or CALIBRATION_FILE
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cal = json.load(f)
        if cal.get("model_version") != MODEL_VERSION:
            return None
        return cal
    except Exception:
        return None


def _apply_calibration(p: float, cal: dict) -> float:
    """Calibration par bins (voir model_calibration.py)."""
    edges, rates = cal["bin_edges"], cal["bin_rates"]
    for i in range(len(edges) - 1):
        if edges[i] <= p < edges[i + 1] or (i == len(edges) - 2
                                            and p == edges[-1]):
            r = rates[i]
            return min(0.999, max(0.001, r)) if r is not None else p
    return p


def predict(ctx, use_momentum: bool = True,
            calibration_path: str = None) -> Optional[dict]:
    """ctx : BtcMarketContext (ou objet equivalent). Retourne le dict de
    sortie specifie, ou None + jamais d'invention si donnees insuffisantes.
    La raison du refus est disponible via predict_or_reason()."""
    out, _ = predict_or_reason(ctx, use_momentum, calibration_path)
    return out


def predict_or_reason(ctx, use_momentum: bool = True,
                      calibration_path: str = None):
    """Retourne (sortie|None, raison_du_refus|None)."""
    if ctx is None or not getattr(ctx, "valid", False):
        return None, f"contexte_invalide:{getattr(ctx, 'reason', 'absent')}"
    if getattr(ctx, "data_quality_score", 0) < MIN_DATA_QUALITY:
        return None, (f"insufficient_data_quality "
                      f"({ctx.data_quality_score}<{MIN_DATA_QUALITY})")
    t = ctx.minutes_remaining
    sig = ctx.realized_vol_1m
    if t is None or t <= 0:
        return None, "temps_restant_invalide"
    if sig is None or sig <= 0 or math.isnan(sig) or math.isinf(sig):
        return None, "volatilite_invalide"
    if ctx.distance_norm is None:
        return None, "strike_ou_spot_absent"

    denom = sig * math.sqrt(t)
    d = ctx.distance_norm / denom

    mu_adj = 0.0
    if use_momentum and ctx.returns.get("5m") is not None:
        drift_per_min = ctx.returns["5m"] / 5.0
        mu_adj = max(-MOMENTUM_CAP,
                     min(MOMENTUM_CAP, drift_per_min * t / denom))

    p_yes_raw = norm_cdf(d + mu_adj)
    if math.isnan(p_yes_raw) or math.isinf(p_yes_raw):
        return None, "probabilite_non_finie"
    p_yes_raw = min(0.9999, max(0.0001, p_yes_raw))

    cal = _load_calibration(calibration_path)
    p_yes = _apply_calibration(p_yes_raw, cal) if cal else p_yes_raw
    calibrated = cal is not None

    q_data = ctx.data_quality_score / 100.0
    q_time = max(0.2, min(1.0, t / 15.0))
    q_vol = 1.0 if ctx.klines_count >= 11 else 0.5
    confidence = round(max(0.0, min(1.0, q_data * q_time * q_vol)), 3)

    features = {
        "spot": ctx.spot, "strike": ctx.strike,
        "distance_norm": round(ctx.distance_norm, 8),
        "sigma_1m": round(sig, 8), "minutes_remaining": round(t, 3),
        "d_score": round(d, 5), "mu_adj": round(mu_adj, 5),
        "dispersion_pct": ctx.dispersion_pct,
        "data_quality_score": ctx.data_quality_score,
        "n_sources": ctx.n_valid_sources,
        "p_raw": round(p_yes_raw, 6),
    }
    return {
        "probability_yes": round(p_yes, 6),
        "probability_no": round(1.0 - p_yes, 6),
        "confidence": confidence,
        "model_version": MODEL_VERSION + ("+cal" if calibrated else ""),
        "calibrated": calibrated,
        "features": features,
        "reason": ("baseline analytique NON VALIDEE: Phi(ln(S/K)/(sigma*sqrt(t))"
                   " + momentum borne)" + (" ; calibree par bins" if calibrated
                                           else ""))
    }, None
