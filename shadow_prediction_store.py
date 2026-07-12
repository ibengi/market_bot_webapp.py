"""
shadow_prediction_store.py — v1 (2026-07-12)
Journal SHADOW des predictions : chaque candidat BTC evalue est enregistre
avec toutes ses caracteristiques, puis complete UNE SEULE FOIS au reglement.

- Ecriture ATOMIQUE (fichier temporaire + os.replace).
- IDEMPOTENTE : prediction_id deterministe = "{ticker}|{ts_cycle_iso}" ;
  re-enregistrer le meme id ne cree pas de doublon.
- Reglement unique : une prediction reglee ne peut pas l'etre a nouveau.
"""

import json
import math
import os
import logging
from datetime import datetime, timezone
from typing import Optional, Callable

log = logging.getLogger("SHADOW")

STORE_FILE = os.getenv("SHADOW_STORE_FILE", "shadow_predictions.json")


class ShadowPredictionStore:
    def __init__(self, path: str = None):
        self.path = path or STORE_FILE
        self.rows = self._load()
        self._index = {r["prediction_id"]: r for r in self.rows}

    # ── persistance ──────────────────────────────────────────────────────
    def _load(self) -> list:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"store illisible ({e}) -- demarre vide, l'ancien "
                        f"fichier n'est PAS ecrase avant premiere ecriture ok.")
            return []

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ── enregistrement (idempotent) ──────────────────────────────────────
    @staticmethod
    def prediction_id(ticker: str, cycle_ts_iso: str) -> str:
        return f"{ticker}|{cycle_ts_iso}"

    def record(self, *, ticker: str, cycle_ts_iso: str, market: dict,
               strike, spot, minutes_remaining,
               yes_bid, yes_ask, no_bid, no_ask, spread,
               ranker_score, features: dict,
               probability_yes, probability_no, confidence,
               estimated_fee, estimated_slippage,
               gross_edge, net_edge, net_ev,
               shadow_decision: str, decision_reason: str) -> str:
        pid = self.prediction_id(ticker, cycle_ts_iso)
        if pid in self._index:
            return pid                              # idempotent : pas de doublon
        row = {
            "prediction_id": pid, "ticker": ticker,
            "timestamp": cycle_ts_iso,
            "market": {k: market.get(k) for k in
                       ("title", "close_time", "status")},
            "strike": strike, "spot": spot,
            "minutes_remaining": minutes_remaining,
            "yes_bid": yes_bid, "yes_ask": yes_ask,
            "no_bid": no_bid, "no_ask": no_ask, "spread": spread,
            "ranker_score": ranker_score,
            "features": features,
            "probability_yes": probability_yes,
            "probability_no": probability_no,
            "confidence": confidence,
            "estimated_fee": estimated_fee,
            "estimated_slippage": estimated_slippage,
            "gross_edge": gross_edge, "net_edge": net_edge, "net_ev": net_ev,
            "shadow_decision": shadow_decision,     # "yes"|"no"|"none"
            "decision_reason": decision_reason,
            "settled": False, "result": None,
            "theoretical_gross_pnl": None, "theoretical_fees": None,
            "theoretical_net_pnl": None, "prediction_error": None,
            "settled_at": None,
        }
        self.rows.append(row)
        self._index[pid] = row
        self._flush()
        return pid

    # ── reglement (une seule fois) ───────────────────────────────────────
    def settle(self, prediction_id: str, result: str) -> Optional[dict]:
        """result: 'yes'|'no'. PnL theorique : 1 contrat au ask du cote de
        la decision shadow, frais estimes + slippage inclus. Erreur de
        prediction = Brier de l'observation."""
        row = self._index.get(prediction_id)
        if row is None or row["settled"]:
            return None                              # jamais deux fois
        result = result.lower()
        if result not in ("yes", "no"):
            return None
        row["settled"] = True
        row["result"] = result
        outcome_yes = 1 if result == "yes" else 0
        row["prediction_error"] = round(
            (row["probability_yes"] - outcome_yes) ** 2, 6)

        side = row["shadow_decision"]
        if side in ("yes", "no"):
            ask = row["yes_ask"] if side == "yes" else row["no_ask"]
            if ask is not None:
                p = ask / 100.0
                won = (result == side)
                gross = (1.0 - p) if won else -p
                fees = row["estimated_fee"] or 0.0
                slip = row["estimated_slippage"] or 0.0
                row["theoretical_gross_pnl"] = round(gross, 4)
                row["theoretical_fees"] = round(fees, 4)
                row["theoretical_net_pnl"] = round(gross - fees - slip, 4)
        row["settled_at"] = datetime.now(timezone.utc)\
            .isoformat(timespec="seconds")
        self._flush()
        return row

    def settle_pending(self, fetch_market_fn: Callable,
                       max_lookups: int = 25) -> int:
        """Regle les predictions en attente via fetch_market_fn(ticker)."""
        n = 0
        looked = 0
        cache = {}
        for row in self.rows:
            if row["settled"] or looked >= max_lookups:
                continue
            tk = row["ticker"]
            if tk not in cache:
                looked += 1
                try:
                    m = fetch_market_fn(tk) or {}
                except Exception:
                    m = {}
                r = str(m.get("result", "") or "").lower()
                cache[tk] = r if r in ("yes", "no") else None
            if cache[tk] and self.settle(row["prediction_id"], cache[tk]):
                n += 1
        return n

    # ── acces ────────────────────────────────────────────────────────────
    def pending(self) -> list:
        return [r for r in self.rows if not r["settled"]]

    def settled(self) -> list:
        return [r for r in self.rows if r["settled"]]

    def as_calibration_obs(self) -> list:
        """Observations {'p','outcome'} pour model_calibration."""
        return [{"p": r["probability_yes"],
                 "outcome": 1 if r["result"] == "yes" else 0,
                 "ts": r["timestamp"]}
                for r in self.settled()
                if isinstance(r.get("probability_yes"), (int, float))]
