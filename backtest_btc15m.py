"""
backtest_btc15m.py — v1 (2026-07-12)
Backtest CHRONOLOGIQUE du modele BTC 15 minutes.

GARANTIES ANTI-FUITE :
- les observations sont triees par timestamp et decoupees CHRONOLOGIQUEMENT
  en train / validation / test (defaut 60/20/20) ;
- la calibration est ajustee sur TRAIN uniquement (la validation sert a la
  lecture, jamais a l'ajustement dans cette version) ;
- chaque decision n'utilise que les champs de l'observation courante
  (aucun acces aux observations futures : verifie par test 13) ;
- aucun fill n'est suppose si l'ask du cote choisi est absent/invalide.

Format d'une observation (fixture ou export du shadow store) :
{
 "ts": iso ou epoch, "ticker": str,
 "features": {...} du modele  OU  champs bruts:
   "spot", "strike", "sigma_1m", "minutes_remaining", "ret_5m",
 "yes_bid","yes_ask","no_bid","no_ask",
 "result": "yes"|"no"
}
"""

import math
import json
from datetime import datetime, timezone

import model_calibration as mc
from btc_probability_model import norm_cdf, MODEL_VERSION, MOMENTUM_CAP
from strategy_router import estimated_fee_per_contract

DEFAULT_SPLIT = (0.6, 0.2, 0.2)


def _ts(o):
    v = o["ts"]
    if isinstance(v, (int, float)):
        return float(v)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


def predict_row(o: dict, use_momentum: bool = True) -> float:
    """Probabilite YES depuis les seuls champs de la ligne (pas de futur)."""
    sig, t = float(o["sigma_1m"]), float(o["minutes_remaining"])
    denom = sig * math.sqrt(t)
    d = math.log(o["spot"] / o["strike"]) / denom
    mu = 0.0
    if use_momentum and o.get("ret_5m") is not None:
        mu = max(-MOMENTUM_CAP, min(MOMENTUM_CAP,
                                    (o["ret_5m"] / 5.0) * t / denom))
    return min(0.9999, max(0.0001, norm_cdf(d + mu)))


def split_chronological(obs: list, split=DEFAULT_SPLIT):
    rows = sorted(obs, key=_ts)
    n = len(rows)
    a = int(n * split[0]); b = a + int(n * split[1])
    return rows[:a], rows[a:b], rows[b:]


def run_backtest(obs: list, gates: dict = None,
                 split=DEFAULT_SPLIT, fee_rate: float = 0.07,
                 slippage_cents: int = 1,
                 uncertainty_buffer: float = 0.01,
                 calibrate: bool = True) -> dict:
    """Retourne le rapport complet. Deterministe : memes donnees, memes
    parametres => memes resultats (aucun aleatoire)."""
    g = {"MIN_GROSS_EDGE": 0.05, "MIN_NET_EDGE": 0.03, "MIN_NET_EV": 0.02,
         "MAX_SPREAD": 4}
    g.update(gates or {})

    train, val, test = split_chronological(obs, split)

    # predictions brutes par split (chaque ligne = ses propres champs)
    def preds(rows):
        return [{"p": predict_row(o), "outcome": 1 if o["result"] == "yes"
                 else 0, "row": o} for o in rows]
    p_train, p_val, p_test = preds(train), preds(val), preds(test)

    cal = mc.fit(p_train) if (calibrate and p_train) else None
    def adj(p):
        return mc.apply(cal, p) if cal else p

    # ── simulation de trading sur TEST uniquement ──
    trades = []
    for pr in p_test:
        o = pr["row"]
        p_yes = adj(pr["p"])
        for side, p_model in (("yes", p_yes), ("no", 1 - p_yes)):
            ask = o.get(f"{side}_ask")
            bid = o.get(f"{side}_bid")
            if ask is None or not (1 <= int(ask) <= 99):
                continue                       # PAS de fill suppose
            if bid is not None and (int(ask) - int(bid)) > g["MAX_SPREAD"]:
                continue
            mkt_p = int(ask) / 100.0
            gross_edge = p_model - mkt_p
            if gross_edge <= 0 or gross_edge < g["MIN_GROSS_EDGE"]:
                continue
            fee = estimated_fee_per_contract(int(ask), fee_rate)
            slip = slippage_cents / 100.0
            net_edge = gross_edge - fee - slip - uncertainty_buffer
            ev = p_model * (1 - mkt_p) - (1 - p_model) * mkt_p - fee - slip
            if net_edge < g["MIN_NET_EDGE"] or ev <= g["MIN_NET_EV"]:
                continue
            won = (o["result"] == side)
            gross = (1 - mkt_p) if won else -mkt_p
            net = gross - fee - slip
            trades.append({"ts": o["ts"], "ticker": o.get("ticker"),
                           "side": side, "ask": int(ask),
                           "p_model": round(p_model, 4),
                           "gross_edge": round(gross_edge, 4),
                           "net_edge": round(net_edge, 4),
                           "won": won, "gross": round(gross, 4),
                           "fee": fee, "net": round(net, 4),
                           "minutes_remaining": o["minutes_remaining"],
                           "sigma_1m": o["sigma_1m"]})
            break                              # un seul cote par observation

    # ── agregats ──
    def bucketize(rows, key_fn, buckets):
        out = {}
        for name, lo, hi in buckets:
            sel = [r for r in rows if lo <= key_fn(r) < hi]
            out[name] = {"n": len(sel),
                         "win_rate": round(sum(r["won"] for r in sel)
                                           / len(sel), 4) if sel else None,
                         "net_pnl": round(sum(r["net"] for r in sel), 4)}
        return out

    def hour(r):
        return datetime.fromtimestamp(_ts(r), tz=timezone.utc).hour

    equity, peak, max_dd = 0.0, 0.0, 0.0
    for tr in trades:
        equity += tr["net"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    n_tr = len(trades)
    gross_pnl = sum(t["gross"] for t in trades)
    fees = sum(t["fee"] for t in trades)
    net_pnl = sum(t["net"] for t in trades)
    invested = sum(t["ask"] / 100.0 for t in trades)

    test_obs = [{"p": adj(pr["p"]), "outcome": pr["outcome"]}
                for pr in p_test]
    market_obs = [{"p": (pr["row"].get("yes_ask") or 50) / 100.0,
                   "outcome": pr["outcome"]} for pr in p_test
                  if pr["row"].get("yes_ask")]

    report = {
        "model_version": MODEL_VERSION + ("+cal" if cal else ""),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": {"train": len(train), "validation": len(val),
                  "test": len(test), "chronological": True},
        "gates": g,
        "n_predictions_test": len(p_test),
        "n_theoretical_trades": n_tr,
        "win_rate": round(sum(t["won"] for t in trades) / n_tr, 4)
                    if n_tr else None,
        "gross_pnl": round(gross_pnl, 4),
        "fees": round(fees, 4),
        "net_pnl": round(net_pnl, 4),
        "roi": round(net_pnl / invested, 4) if invested else None,
        "max_drawdown": round(max_dd, 4),
        "brier_test": round(mc.brier(test_obs), 6) if test_obs else None,
        "brier_market_baseline": round(mc.brier(market_obs), 6)
                                 if market_obs else None,
        "log_loss_test": round(mc.log_loss(test_obs), 6) if test_obs else None,
        "calibration_test": mc.evaluate(test_obs, label="test"),
        "calibration_fitted_on": "train uniquement" if cal else "aucune",
        "by_hour": bucketize(trades, hour,
                             [("00-07", 0, 8), ("08-15", 8, 16),
                              ("16-23", 16, 24)]),
        "by_volatility": bucketize(trades, lambda r: r["sigma_1m"],
                                   [("basse<5e-4", 0, 5e-4),
                                    ("moyenne", 5e-4, 15e-4),
                                    ("haute>=15e-4", 15e-4, 1)]),
        "by_time_remaining": bucketize(trades,
                                       lambda r: r["minutes_remaining"],
                                       [("0-5", 0, 5), ("5-10", 5, 10),
                                        ("10-15+", 10, 999)]),
        "by_net_edge": bucketize(trades, lambda r: r["net_edge"],
                                 [("0-3%", 0, .03), ("3-5%", .03, .05),
                                  ("5-8%", .05, .08), ("8%+", .08, 1)]),
        "trades": trades,
    }
    return report


def save_report(report: dict, path: str = "backtest_report.json"):
    import os
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
