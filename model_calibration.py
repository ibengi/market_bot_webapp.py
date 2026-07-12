"""
model_calibration.py — v1 (2026-07-12)
Calibration des probabilites a partir de PREDICTIONS HISTORIQUES REGLEES.

Methode : CALIBRATION PAR BINS (documentee ci-dessous). scipy/sklearn ne
sont pas des dependances du depot ; si sklearn est disponible, une
calibration isotonique est utilisee a la place (meme interface).

Calibration par bins :
- decouper [0,1] en N bins egaux (defaut 10) ;
- pour chaque bin, taux reel = (gains + 1) / (n + 2)   [lissage de Laplace,
  evite 0 et 1 sur petits echantillons] ;
- bin vide -> None (la probabilite brute est conservee a l'application).

REGLE ABSOLUE : fit() ne recoit QUE des donnees train (la separation
chronologique est la responsabilite de l'appelant, verifiee par le backtest
et par test 13/14). evaluate() est independant de fit().

Format d'une observation : {"p": float 0..1, "outcome": 0|1}
(p = probabilite YES predite ; outcome = 1 si YES s'est realise)
"""

import json
import math
import os
from datetime import datetime, timezone
from typing import Optional

from btc_probability_model import MODEL_VERSION

DEFAULT_BINS = 10


def _bin_edges(n_bins: int) -> list:
    return [i / n_bins for i in range(n_bins + 1)]


def fit(train_obs: list, n_bins: int = DEFAULT_BINS,
        model_version: str = MODEL_VERSION) -> dict:
    """Ajuste la calibration sur les observations TRAIN uniquement."""
    edges = _bin_edges(n_bins)
    counts = [0] * n_bins
    wins = [0] * n_bins
    for o in train_obs:
        p = float(o["p"]); y = int(o["outcome"])
        if not (0.0 <= p <= 1.0):
            continue
        i = min(n_bins - 1, int(p * n_bins))
        counts[i] += 1
        wins[i] += y
    try:
        from sklearn.isotonic import IsotonicRegression  # optionnel
        xs = [float(o["p"]) for o in train_obs]
        ys = [int(o["outcome"]) for o in train_obs]
        iso = IsotonicRegression(y_min=0.001, y_max=0.999,
                                 out_of_bounds="clip").fit(xs, ys)
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
        rates = [float(iso.predict([c])[0]) for c in centers]
        method = "isotonic (sklearn)"
    except ImportError:
        rates = [((wins[i] + 1) / (counts[i] + 2)) if counts[i] > 0 else None
                 for i in range(n_bins)]
        method = "bins + lissage de Laplace"
    return {
        "model_version": model_version,
        "method": method,
        "n_bins": n_bins,
        "bin_edges": edges,
        "bin_counts": counts,
        "bin_rates": rates,
        "n_train": len(train_obs),
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save(calibration: dict, path: str = "model_calibration.json"):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=1)
    os.replace(tmp, path)


# ── Metriques (independantes du fit) ─────────────────────────────────────────

def brier(obs: list) -> Optional[float]:
    if not obs:
        return None
    return sum((o["p"] - o["outcome"]) ** 2 for o in obs) / len(obs)


def log_loss(obs: list) -> Optional[float]:
    if not obs:
        return None
    eps = 1e-9
    tot = 0.0
    for o in obs:
        p = min(1 - eps, max(eps, o["p"]))
        tot += -(o["outcome"] * math.log(p)
                 + (1 - o["outcome"]) * math.log(1 - p))
    return tot / len(obs)


def calibration_curve(obs: list, n_bins: int = DEFAULT_BINS) -> list:
    """[{bin, p_moyenne, taux_reel, n}] — courbe de calibration."""
    edges = _bin_edges(n_bins)
    rows = []
    for i in range(n_bins):
        sel = [o for o in obs
               if edges[i] <= o["p"] < edges[i + 1]
               or (i == n_bins - 1 and o["p"] == 1.0)]
        rows.append({
            "bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
            "p_moyenne": round(sum(o["p"] for o in sel) / len(sel), 4)
                         if sel else None,
            "taux_reel": round(sum(o["outcome"] for o in sel) / len(sel), 4)
                         if sel else None,
            "n": len(sel),
        })
    return rows


def expected_calibration_error(obs: list,
                               n_bins: int = DEFAULT_BINS) -> Optional[float]:
    curve = calibration_curve(obs, n_bins)
    n = sum(r["n"] for r in curve)
    if n == 0:
        return None
    ece = sum(r["n"] / n * abs(r["p_moyenne"] - r["taux_reel"])
              for r in curve if r["n"] > 0)
    return round(ece, 6)


def evaluate(obs: list, n_bins: int = DEFAULT_BINS,
             label: str = "eval") -> dict:
    """Rapport complet de calibration sur un jeu d'observations (test)."""
    return {
        "label": label,
        "n": len(obs),
        "brier": round(brier(obs), 6) if obs else None,
        "log_loss": round(log_loss(obs), 6) if obs else None,
        "ece": expected_calibration_error(obs, n_bins),
        "calibration_curve": calibration_curve(obs, n_bins),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def apply(calibration: dict, p: float) -> float:
    """Applique une calibration ajustee (memes bins que fit)."""
    edges, rates = calibration["bin_edges"], calibration["bin_rates"]
    n = len(rates)
    i = min(n - 1, int(p * n))
    r = rates[i]
    return p if r is None else min(0.999, max(0.001, r))
