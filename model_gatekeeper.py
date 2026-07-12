"""
model_gatekeeper.py — v1 (2026-07-12)
Porte AVANT LIVE : le trading reel reste BLOQUE tant qu'un rapport de
validation recent ne satisfait pas TOUS les criteres. Retourne la liste
precise des criteres echoues. N'active JAMAIS le live automatiquement.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field


@dataclass
class GatekeeperConfig:
    MIN_PREDICTIONS: int = int(os.getenv("GATE_MIN_PREDICTIONS", "300"))
    MIN_THEORETICAL_TRADES: int = int(os.getenv("GATE_MIN_TRADES", "100"))
    MAX_REPORT_AGE_H: float = float(os.getenv("GATE_MAX_REPORT_AGE_H", "168"))
    MAX_DRAWDOWN: float = float(os.getenv("GATE_MAX_DRAWDOWN", "10.0"))
    MAX_ECE: float = float(os.getenv("GATE_MAX_ECE", "0.10"))
    MODEL_FILE: str = "btc_probability_model.py"


def model_hash(path: str = None) -> str:
    path = path or GatekeeperConfig.MODEL_FILE
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def evaluate(validation_report: dict,
             cfg: GatekeeperConfig = None,
             loaded_model_hash: str = None,
             now: float = None) -> (bool, list):
    """(approuve, criteres_echoues). Un rapport absent/malforme echoue tout."""
    cfg = cfg or GatekeeperConfig()
    now = now or time.time()
    failed = []
    r = validation_report or {}

    def g(key, default=None):
        return r.get(key, default)

    if g("n_predictions_test", 0) < cfg.MIN_PREDICTIONS:
        failed.append(f"predictions_insuffisantes "
                      f"({g('n_predictions_test', 0)}<{cfg.MIN_PREDICTIONS})")
    if g("n_theoretical_trades", 0) < cfg.MIN_THEORETICAL_TRADES:
        failed.append(f"trades_theoriques_insuffisants "
                      f"({g('n_theoretical_trades', 0)}"
                      f"<{cfg.MIN_THEORETICAL_TRADES})")
    if not isinstance(g("net_pnl"), (int, float)) or g("net_pnl") <= 0:
        failed.append(f"pnl_net_hors_echantillon_non_positif "
                      f"({g('net_pnl')})")
    b, bm = g("brier_test"), g("brier_market_baseline")
    if not isinstance(b, (int, float)) or not isinstance(bm, (int, float)) \
            or b >= bm:
        failed.append(f"brier_pas_meilleur_que_le_marche "
                      f"(modele={b} vs marche={bm})")
    ece = (g("calibration_test") or {}).get("ece")
    if not isinstance(ece, (int, float)) or ece > cfg.MAX_ECE:
        failed.append(f"calibration_insuffisante (ECE={ece}>{cfg.MAX_ECE})")
    if not isinstance(g("max_drawdown"), (int, float)) \
            or g("max_drawdown") > cfg.MAX_DRAWDOWN:
        failed.append(f"drawdown_excessif ({g('max_drawdown')}"
                      f">{cfg.MAX_DRAWDOWN})")
    if not (g("split") or {}).get("chronological"):
        failed.append("fuite_possible: split non chronologique declare")
    gen = g("generated")
    try:
        from datetime import datetime
        age_h = (now - datetime.fromisoformat(gen).timestamp()) / 3600
        if age_h > cfg.MAX_REPORT_AGE_H:
            failed.append(f"rapport_trop_ancien ({age_h:.0f}h"
                          f">{cfg.MAX_REPORT_AGE_H}h)")
    except (TypeError, ValueError):
        failed.append("rapport_sans_date_valide")
    expected = g("model_hash")
    actual = loaded_model_hash or model_hash()
    if not expected or expected != actual:
        failed.append("hash_modele_différent_du_modele_charge")

    return (len(failed) == 0), failed


def check_live_allowed(report_path: str = "model_validation_report.json",
                       cfg: GatekeeperConfig = None) -> (bool, list):
    """Point d'entree du bot : env + rapport + criteres."""
    failed_env = []
    if os.getenv("LIVE_TRADING_CONFIRMED", "") != "YES":
        failed_env.append("LIVE_TRADING_CONFIRMED != YES")
    if os.getenv("MODEL_APPROVED_FOR_LIVE", "") != "YES":
        failed_env.append("MODEL_APPROVED_FOR_LIVE != YES")
    if not os.path.exists(report_path):
        return False, failed_env + [f"rapport_absent: {report_path}"]
    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    except Exception as e:
        return False, failed_env + [f"rapport_illisible: {e}"]
    ok, failed = evaluate(report, cfg)
    return (ok and not failed_env), failed_env + failed
