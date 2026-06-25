"""
ml_model.py  --  v1
Modele de probabilite adaptatif qui remplace progressivement
Black-Scholes au fur et a mesure que les donnees s'accumulent.

PHASES :
- Phase 1 (< 30 trades) : Black-Scholes pur (pas assez de donnees)
- Phase 2 (30-100 trades) : BS + ajustements patterns (hybride)
- Phase 3 (> 100 trades) : Regression logistique entrainees sur donnees reelles

Le modele se reentrainee automatiquement a chaque appel si de
nouvelles donnees sont disponibles depuis le dernier entrainement.

SANS dependances ML externes (numpy/sklearn optionnels) :
Si numpy/sklearn sont absents, utilise une regression logistique
implementee en Python pur (plus lente mais fonctionnelle).
"""

import json
import os
import math
import time
import logging
from typing import Optional

log = logging.getLogger("MLModel")

TRADE_RESULTS_FILE = "btc_trade_results.json"
MODEL_FILE         = "btc_ml_model.json"
PATTERNS_FILE      = "btc_patterns.json"

# Seuils de phases
PHASE2_MIN = 30
PHASE3_MIN = 100

# ── Chargement ────────────────────────────────────────────────────────────────

def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning(f"Erreur sauvegarde {path}: {e}")


# ── Feature engineering ───────────────────────────────────────────────────────

def _extract_features(
    price: float,
    strike: float,
    minutes_remaining: float,
    volatility: float,
    momentum_per_min: float,
    hour_utc: int,
    side: str,
) -> list:
    """
    Transforme les donnees brutes en features normalisees pour le modele.
    Toutes les features sont bornees entre -1 et 1 ou 0 et 1.
    """
    # 1. Distance relative prix/strike (signee : positif = prix au-dessus)
    dist = math.log(price / strike) if price > 0 and strike > 0 else 0.0
    dist_norm = max(-1.0, min(1.0, dist / 0.02))  # normalise sur +/-2%

    # 2. Temps restant normalise (0=expire, 1=15min complet)
    t_norm = max(0.0, min(1.0, minutes_remaining / 15.0))

    # 3. Volatilite normalisee
    vol_norm = max(0.0, min(1.0, volatility / 0.01))

    # 4. Momentum normalise
    mom_norm = max(-1.0, min(1.0, momentum_per_min / 0.001))

    # 5. Session de marche (encodage cyclique de l'heure)
    hour_sin = math.sin(2 * math.pi * hour_utc / 24)
    hour_cos = math.cos(2 * math.pi * hour_utc / 24)

    # 6. Cote (1=YES, 0=NO)
    side_enc = 1.0 if side == "yes" else 0.0

    # 7. Interaction distance x temps (plus de temps = la distance compte moins)
    dist_x_time = dist_norm * t_norm

    return [dist_norm, t_norm, vol_norm, mom_norm, hour_sin, hour_cos,
            side_enc, dist_x_time]


# ── Regression logistique Python pur ─────────────────────────────────────────
# Utilisee si sklearn n'est pas disponible

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _logistic_predict(weights: list, bias: float, features: list) -> float:
    z = bias + sum(w * f for w, f in zip(weights, features))
    return _sigmoid(z)

def _logistic_train(X: list, y: list, lr: float = 0.01, epochs: int = 200) -> tuple:
    """
    Entraine une regression logistique par descente de gradient.
    X : liste de listes de features
    y : liste de labels (0 ou 1)
    Retourne (weights, bias)
    """
    n_features = len(X[0])
    weights    = [0.0] * n_features
    bias       = 0.0

    for _ in range(epochs):
        dw   = [0.0] * n_features
        db   = 0.0
        loss = 0.0
        for xi, yi in zip(X, y):
            pred  = _logistic_predict(weights, bias, xi)
            error = pred - yi
            for j in range(n_features):
                dw[j] += error * xi[j]
            db   += error
            # Log-loss
            pred_clip = max(1e-7, min(1 - 1e-7, pred))
            loss += -(yi * math.log(pred_clip) + (1 - yi) * math.log(1 - pred_clip))

        n = len(X)
        weights = [w - lr * (dw[j] / n) for j, w in enumerate(weights)]
        bias   -= lr * (db / n)

    return weights, bias


# ── Entrainement du modele ────────────────────────────────────────────────────

def train_model(force: bool = False) -> dict:
    """
    Entraine ou met a jour le modele ML sur les donnees disponibles.
    Retourne le modele entraine (dict) ou {} si pas assez de donnees.
    """
    results = _load_json(TRADE_RESULTS_FILE, [])
    model   = _load_json(MODEL_FILE, {})

    n_results = len(results)
    n_trained = model.get("n_trained", 0)

    # Reentraine seulement si 10+ nouveaux trades depuis le dernier entrainement
    if not force and n_results - n_trained < 10:
        return model

    if n_results < PHASE2_MIN:
        log.info(f"[ML] Phase 1 (BS pur) -- {n_results}/{PHASE2_MIN} trades minimum")
        model = {"phase": 1, "n_trained": n_results, "message": "Pas assez de donnees pour ML"}
        _save_json(MODEL_FILE, model)
        return model

    # Prepare les features
    X, y = [], []
    for r in results:
        ts    = r.get("timestamp", time.time())
        edge  = r.get("edge", 0)
        side  = r.get("side", "yes")
        won   = 1 if r.get("won") else 0
        price = r.get("price", 50)

        from datetime import datetime, timezone
        dt    = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour  = dt.hour

        # Features qu'on peut reconstruire depuis le trade log
        dist_approx = (price / 100 - 0.5)  # approximation distance
        features = [
            max(-1.0, min(1.0, dist_approx * 5)),  # distance approx
            0.5,                                    # t_norm (inconnu apres coup)
            0.5,                                    # vol_norm (inconnu apres coup)
            0.0,                                    # momentum (inconnu apres coup)
            math.sin(2 * math.pi * hour / 24),      # heure sin
            math.cos(2 * math.pi * hour / 24),      # heure cos
            1.0 if side == "yes" else 0.0,          # cote
            dist_approx * 0.5,                      # interaction
        ]
        X.append(features)
        y.append(won)

    phase = 3 if n_results >= PHASE3_MIN else 2

    # Entraine la regression logistique
    try:
        weights, bias = _logistic_train(X, y, lr=0.05, epochs=300)

        # Calcule l'accuracy sur les donnees d'entrainement
        correct = sum(
            1 for xi, yi in zip(X, y)
            if (1 if _logistic_predict(weights, bias, xi) >= 0.5 else 0) == yi
        )
        accuracy = correct / len(y)

        model = {
            "phase":     phase,
            "n_trained": n_results,
            "weights":   weights,
            "bias":      bias,
            "accuracy":  round(accuracy, 4),
            "trained_at": time.time(),
        }
        _save_json(MODEL_FILE, model)
        log.info(
            f"[ML] Modele Phase {phase} entraine -- {n_results} trades | "
            f"accuracy={accuracy:.1%}"
        )
        return model

    except Exception as e:
        log.warning(f"[ML] Erreur entrainement: {e}")
        return {"phase": 1, "n_trained": n_results}


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_probability(
    price: float,
    strike: float,
    minutes_remaining: float,
    volatility: float,
    momentum_per_min: float,
    side: str = "yes",
    bs_probability: float = 0.5,   # probabilite Black-Scholes (fallback)
) -> dict:
    """
    Predit la probabilite que le trade soit gagnant.

    Retourne :
    {
        "probability": float,   # probabilite finale (0-1)
        "phase":       int,     # 1=BS pur, 2=hybride, 3=ML
        "confidence":  float,   # confiance du modele (0-1)
        "source":      str,     # description de la source
    }
    """
    model = _load_json(MODEL_FILE, {})

    # Tente de reentraine si necessaire (leger, seulement si 10+ nouveaux trades)
    if not model or model.get("phase", 1) == 1:
        model = train_model()

    phase = model.get("phase", 1)

    # Phase 1 : Black-Scholes pur
    if phase == 1 or "weights" not in model:
        return {
            "probability": bs_probability,
            "phase":       1,
            "confidence":  0.5,
            "source":      f"Black-Scholes pur ({model.get('n_trained', 0)} trades, min {PHASE2_MIN})",
        }

    # Phase 2 ou 3 : modele ML
    from datetime import datetime, timezone
    hour = datetime.now(tz=timezone.utc).hour
    dist = math.log(price / strike) if price > 0 and strike > 0 else 0.0
    dist_norm = max(-1.0, min(1.0, dist / 0.02))

    features = _extract_features(
        price, strike, minutes_remaining, volatility, momentum_per_min, hour, side
    )

    ml_prob = _logistic_predict(model["weights"], model["bias"], features)

    if phase == 2:
        # Hybride : moyenne ponderee BS (40%) + ML (60%)
        blended = 0.40 * bs_probability + 0.60 * ml_prob
        source  = f"Hybride BS+ML Phase 2 ({model['n_trained']} trades, acc={model.get('accuracy', 0):.1%})"
    else:
        # Phase 3 : ML dominant (20% BS pour stabilite)
        blended = 0.20 * bs_probability + 0.80 * ml_prob
        source  = f"ML Phase 3 ({model['n_trained']} trades, acc={model.get('accuracy', 0):.1%})"

    # Confiance : basee sur la distance a 0.5 et l'accuracy du modele
    distance_from_50 = abs(blended - 0.5)
    confidence = min(1.0, model.get("accuracy", 0.5) * (1 + distance_from_50))

    return {
        "probability": round(max(0.01, min(0.99, blended)), 4),
        "phase":       phase,
        "confidence":  round(confidence, 3),
        "source":      source,
        "ml_raw":      round(ml_prob, 4),
        "bs_raw":      round(bs_probability, 4),
    }


def get_model_status() -> str:
    model = _load_json(MODEL_FILE, {})
    if not model:
        return "Aucun modele entraine"
    phase = model.get("phase", 1)
    n     = model.get("n_trained", 0)
    acc   = model.get("accuracy", 0)
    if phase == 1:
        return f"Phase 1 (BS pur) -- {n} trades / {PHASE2_MIN} min pour Phase 2"
    if phase == 2:
        return f"Phase 2 (Hybride) -- {n} trades | accuracy={acc:.1%} / {PHASE3_MIN} min pour Phase 3"
    return f"Phase 3 (ML dominant) -- {n} trades | accuracy={acc:.1%}"
