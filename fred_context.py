"""
FRED CONTEXT MODULE — Kalshi Alpha Engine V3
Recupere automatiquement les donnees macro reelles avant chaque analyse.

INSTALLATION:
    pip install requests python-dotenv

CONFIGURATION .env:
    FRED_API_KEY=votre_cle_fred  (gratuit sur fred.stlouisfed.org/docs/api/api_key.html)

USAGE:
    from fred_context import get_macro_context
    context = get_macro_context()
    # Passe context au bot
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("FREDContext")

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE    = "https://api.stlouisfed.org/fred"

# ── Indicateurs macro a surveiller ───────────────────────────────────────────
INDICATORS = {
    # Inflation
    "CPIAUCSL":   {"nom": "CPI All Items (YoY)",        "poids": "TRES ELEVE", "categorie": "inflation"},
    "CPILFESL":   {"nom": "Core CPI (YoY)",             "poids": "TRES ELEVE", "categorie": "inflation"},
    "PPIACO":     {"nom": "PPI All Commodities",        "poids": "ELEVE",      "categorie": "inflation"},
    "PCEPILFE":   {"nom": "Core PCE (Fed prefere)",     "poids": "TRES ELEVE", "categorie": "inflation"},
    # Emploi
    "PAYEMS":     {"nom": "NFP Total Emploi",           "poids": "ELEVE",      "categorie": "emploi"},
    "UNRATE":     {"nom": "Taux chomage",               "poids": "ELEVE",      "categorie": "emploi"},
    "AWHAETP":    {"nom": "Heures travaillees moy",     "poids": "MOYEN",      "categorie": "emploi"},
    "CES0500000003": {"nom": "Salaires horaires moy",   "poids": "ELEVE",      "categorie": "emploi"},
    # Activite
    "INDPRO":     {"nom": "Production industrielle",    "poids": "MOYEN",      "categorie": "activite"},
    "RSAFS":      {"nom": "Ventes detail",              "poids": "ELEVE",      "categorie": "activite"},
    "UMCSENT":    {"nom": "Sentiment consommateur",     "poids": "MOYEN",      "categorie": "activite"},
    "ICSA":       {"nom": "Claims hebdo chomage",       "poids": "MOYEN",      "categorie": "emploi"},
    # Marche
    "DGS10":      {"nom": "Treasury 10 ans",            "poids": "MOYEN",      "categorie": "marche"},
    "DGS2":       {"nom": "Treasury 2 ans",             "poids": "MOYEN",      "categorie": "marche"},
    "DCOILWTICO": {"nom": "Petrole WTI",                "poids": "MOYEN",      "categorie": "energie"},
    "DTWEXBGS":   {"nom": "Dollar Index",               "poids": "FAIBLE",     "categorie": "marche"},
}

# ── Cache pour eviter les appels repetitifs ───────────────────────────────────
_cache = {}
CACHE_TTL = 3600  # 1 heure

def _get_series(series_id: str, limit: int = 3) -> list:
    """Recupere les dernieres valeurs d'une serie FRED."""
    cache_key = f"{series_id}_{limit}"
    now = time.time()

    if cache_key in _cache:
        data, ts = _cache[cache_key]
        if now - ts < CACHE_TTL:
            return data

    if not FRED_API_KEY:
        return []

    try:
        url = f"{FRED_BASE}/series/observations"
        params = {
            "series_id":       series_id,
            "api_key":         FRED_API_KEY,
            "file_type":       "json",
            "sort_order":      "desc",
            "limit":           limit,
            "observation_start": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        result = [
            {"date": o["date"], "value": float(o["value"])}
            for o in obs
            if o["value"] not in (".", "")
        ]
        _cache[cache_key] = (result, now)
        return result
    except Exception as e:
        log.warning(f"FRED {series_id}: {e}")
        return []

def _calc_variation(values: list) -> Optional[float]:
    """Calcule la variation entre les 2 dernieres valeurs."""
    if len(values) < 2:
        return None
    return round(values[0]["value"] - values[1]["value"], 4)

def _calc_yoy(values: list) -> Optional[float]:
    """Calcule la variation annuelle (YoY)."""
    if len(values) < 2:
        return None
    current = values[0]["value"]
    previous = values[-1]["value"]
    if previous == 0:
        return None
    return round((current - previous) / previous * 100, 2)

def get_macro_context(target: str = "CPI") -> str:
    """
    Recupere et formate le contexte macro complet pour l'analyse.
    
    Args:
        target: Indicateur cible ("CPI", "NFP", "PCE")
    
    Returns:
        Contexte textuel formate pour le system prompt Claude
    """
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY manquant — contexte macro non disponible")
        return "Contexte macro non disponible (FRED_API_KEY manquant)."

    log.info("Recuperation des donnees macro FRED...")
    sections = []
    data_collected = {}

    for series_id, meta in INDICATORS.items():
        values = _get_series(series_id, limit=13)  # 13 mois pour YoY
        if not values:
            continue

        last_val   = values[0]["value"]
        last_date  = values[0]["date"]
        variation  = _calc_variation(values)
        yoy        = _calc_yoy(values) if len(values) >= 12 else None

        data_collected[series_id] = {
            "nom":       meta["nom"],
            "poids":     meta["poids"],
            "categorie": meta["categorie"],
            "valeur":    last_val,
            "date":      last_date,
            "variation": variation,
            "yoy":       yoy,
        }

    if not data_collected:
        return "Donnees FRED non disponibles — verifie la cle API."

    # ── Formate par categorie ─────────────────────────────────────────────────
    categories = {}
    for sid, d in data_collected.items():
        cat = d["categorie"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(d)

    lines = [
        f"=== CONTEXTE MACRO REEL — {datetime.now().strftime('%d/%m/%Y %H:%M')} ===",
        f"Source: Federal Reserve Economic Data (FRED)",
        f"Cible d'analyse: {target}",
        "",
    ]

    cat_labels = {
        "inflation": "INFLATION",
        "emploi":    "EMPLOI & MARCHE DU TRAVAIL",
        "activite":  "ACTIVITE ECONOMIQUE",
        "marche":    "MARCHES FINANCIERS",
        "energie":   "ENERGIE",
    }

    for cat, label in cat_labels.items():
        if cat not in categories:
            continue
        lines.append(f"--- {label} ---")
        for d in categories[cat]:
            val_str = f"{d['valeur']:.2f}"
            var_str = f" (var: {d['variation']:+.2f})" if d['variation'] is not None else ""
            yoy_str = f" | YoY: {d['yoy']:+.1f}%" if d['yoy'] is not None else ""
            signal  = _get_signal(d['valeur'], d['variation'], cat, d['nom'])
            lines.append(
                f"  {d['nom']:<35} {val_str}{var_str}{yoy_str} [{d['date']}] {signal}"
            )
        lines.append("")

    # ── Resume et signaux cles ────────────────────────────────────────────────
    lines.append("--- SIGNAUX CLES POUR L'ANALYSE ---")
    signals = _generate_signals(data_collected, target)
    for s in signals:
        lines.append(f"  {s}")

    context = "\n".join(lines)
    log.info(f"Contexte macro recupere — {len(data_collected)} indicateurs")
    return context

def _get_signal(value: float, variation: Optional[float], cat: str, nom: str) -> str:
    """Genere un signal simple base sur la valeur et la variation."""
    if variation is None:
        return ""
    if cat == "inflation":
        if variation > 0.1:
            return "⬆ HAUSSE"
        elif variation < -0.1:
            return "⬇ BAISSE"
    elif cat == "emploi" and "chomage" in nom.lower():
        if variation > 0.1:
            return "⬆ DETERIORATION"
        elif variation < -0.1:
            return "⬇ AMELIORATION"
    return ""

def _generate_signals(data: dict, target: str) -> list:
    """Genere les signaux macro les plus importants."""
    signals = []

    cpi = data.get("CPIAUCSL", {})
    core_cpi = data.get("CPILFESL", {})
    ppi = data.get("PPIACO", {})
    unrate = data.get("UNRATE", {})
    wages = data.get("CES0500000003", {})
    oil = data.get("DCOILWTICO", {})
    t10 = data.get("DGS10", {})

    if cpi and core_cpi:
        cpi_v = cpi.get("variation", 0) or 0
        core_v = core_cpi.get("variation", 0) or 0
        if cpi_v > 0 and core_v > 0:
            signals.append(f"INFLATION HAUSSE: CPI {cpi_v:+.2f} / Core {core_v:+.2f} — signal BAISSIER pour YES si seuil bas")
        elif cpi_v < 0 and core_v < 0:
            signals.append(f"DESINFLATION: CPI {cpi_v:+.2f} / Core {core_v:+.2f} — signal HAUSSIER pour YES si seuil bas")

    if ppi:
        ppi_v = ppi.get("variation", 0) or 0
        if abs(ppi_v) > 0.5:
            signals.append(f"PPI fort ({ppi_v:+.2f}) — signal avance inflation dans 1-2 mois")

    if wages:
        w_v = wages.get("variation", 0) or 0
        if w_v > 0.1:
            signals.append(f"Salaires en hausse ({w_v:+.2f}) — pression inflationniste salariale")

    if oil:
        oil_v = oil.get("variation", 0) or 0
        if abs(oil_v) > 2:
            signals.append(f"Petrole WTI {oil_v:+.1f}$ — impact composante energie CPI")

    if t10:
        t10_v = t10.get("valeur", 0) or 0
        signals.append(f"Treasury 10 ans: {t10_v:.2f}% — marche anticipe {'hausse' if t10_v > 4.5 else 'baisse'} inflation")

    if not signals:
        signals.append("Donnees insuffisantes pour generer des signaux automatiques")

    return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\nTest du module FRED Context...")
    context = get_macro_context("CPI")
    print(context)
