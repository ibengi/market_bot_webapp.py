"""
trade_resolver.py  --  v1
Verifie les marches BTC 15min fermes sur Kalshi et enregistre
le resultat reel (won/lost + pnl) dans btc_trade_results.json.

Ce module est la brique fondamentale de l'apprentissage :
sans resultats reels, le modele ne peut pas apprendre.

USAGE autonome :
    python trade_resolver.py          # resout les trades en attente
    python trade_resolver.py --stats  # affiche les stats de performance

USAGE depuis kalshi_alpha_bot.py :
    from trade_resolver import resolve_pending_trades
    resolve_pending_trades(kalshi_client)  # appele au debut de chaque cycle
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("TradeResolver")

TRADES_FILE         = "kalshi_trades.json"       # trades executes par le bot
TRADE_RESULTS_FILE  = "btc_trade_results.json"   # resultats enregistres pour ML
RESOLVED_IDS_FILE   = "resolved_trade_ids.json"  # ids deja resolus (evite doublons)


# ── Chargement / sauvegarde ───────────────────────────────────────────────────

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Erreur lecture {path}: {e}")
        return default

def _save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Erreur ecriture {path}: {e}")


# ── Verifier si un marche est ferme et recuperer son resultat ─────────────────

def _get_market_result(kalshi_client, ticker: str) -> Optional[dict]:
    """
    Interroge l'API Kalshi pour savoir si le marche est ferme
    et quel cote a gagne (yes ou no).

    Retourne :
        {"closed": True,  "winner": "yes"|"no", "yes_price_final": int}
        {"closed": False}
        None si erreur reseau
    """
    try:
        market = kalshi_client.get_market(ticker)
        if not market:
            return None

        status = market.get("status", "")

        # Marche encore ouvert
        if status not in ("finalized", "resolved", "settled", "closed"):
            return {"closed": False}

        # Resultat final
        result        = market.get("result", "")
        yes_price_end = market.get("yes_ask", market.get("yes_bid", 50))

        winner = None
        if result in ("yes", "YES", "Yes"):
            winner = "yes"
        elif result in ("no", "NO", "No"):
            winner = "no"
        else:
            # Essaie de deduire depuis le prix final
            try:
                price = int(yes_price_end)
                winner = "yes" if price >= 90 else "no" if price <= 10 else None
            except Exception:
                pass

        return {
            "closed":          True,
            "winner":          winner,
            "yes_price_final": yes_price_end,
            "status":          status,
        }

    except Exception as e:
        log.debug(f"Erreur get_market_result({ticker}): {e}")
        return None


# ── Calcul du PnL reel ────────────────────────────────────────────────────────

def _compute_pnl(trade: dict, winner: str) -> float:
    """
    Calcule le PnL reel en dollars d'un trade execute.

    trade["side"]  : "yes" ou "no"  (cote achete)
    trade["price"] : prix en cents paye par contrat
    trade["count"] : nombre de contrats
    winner         : "yes" ou "no"  (cote qui a gagne)

    Kalshi : chaque contrat vaut $1 si gagne, $0 si perdu.
    Frais : 2.45% sur les gains uniquement.
    """
    side  = trade.get("side",  "yes")
    price = trade.get("price", 50)    # cents
    count = trade.get("count", 1)

    cost     = (price / 100) * count  # montant investi en dollars
    won_trade = (side == winner)

    if won_trade:
        gross_gain = 1.0 * count               # $1 par contrat gagne
        fee        = gross_gain * 0.0245       # frais Kalshi 2.45%
        pnl        = gross_gain - fee - cost   # gain net
    else:
        pnl = -cost  # perte totale de la mise

    return round(pnl, 4)


# ── Resolution des trades en attente ─────────────────────────────────────────

def resolve_pending_trades(kalshi_client, max_resolve: int = 20) -> int:
    """
    Parcourt les trades executes non encore resolus, interroge Kalshi,
    et enregistre le resultat dans btc_trade_results.json.

    Retourne le nombre de trades resolus ce cycle.
    """
    trades      = _load_json(TRADES_FILE, [])
    resolved    = set(_load_json(RESOLVED_IDS_FILE, []))
    results     = _load_json(TRADE_RESULTS_FILE, [])

    # Filtre : trades BTC non encore resolus
    pending = [
        t for t in trades
        if t.get("market_type") == "btc"
        and t.get("ticker")
        and _trade_id(t) not in resolved
    ]

    if not pending:
        return 0

    log.info(f"[Resolver] {len(pending)} trades BTC en attente de resolution.")
    newly_resolved = 0

    for trade in pending[:max_resolve]:
        tid    = _trade_id(trade)
        ticker = trade["ticker"]

        market_result = _get_market_result(kalshi_client, ticker)
        if market_result is None:
            log.debug(f"[Resolver] {ticker}: erreur reseau, reessai plus tard.")
            continue

        if not market_result.get("closed"):
            log.debug(f"[Resolver] {ticker}: encore ouvert.")
            continue

        winner = market_result.get("winner")
        if winner is None:
            log.warning(f"[Resolver] {ticker}: ferme mais winner indetermine -- skip.")
            resolved.add(tid)  # evite de reessayer indefiniment
            continue

        side     = trade.get("side", "yes")
        won      = (side == winner)
        pnl      = _compute_pnl(trade, winner)
        edge     = trade.get("edge", 0)
        verdict  = trade.get("verdict", "")

        # Enregistre dans btc_trade_results.json pour la calibration ML
        results.append({
            "timestamp":  time.time(),
            "trade_time": trade.get("timestamp", ""),
            "ticker":     ticker,
            "verdict":    verdict,
            "side":       side,
            "winner":     winner,
            "won":        won,
            "pnl":        pnl,
            "edge":       edge,
            "price":      trade.get("price", 50),
            "count":      trade.get("count", 1),
        })

        resolved.add(tid)
        newly_resolved += 1

        status_str = "GAGNE" if won else "PERDU"
        log.info(
            f"[Resolver] {ticker} -> {status_str} | "
            f"side={side} winner={winner} | pnl=${pnl:+.2f} | edge={edge:.1%}"
        )

    # Sauvegarde
    if newly_resolved > 0:
        if len(results) > 500:
            results = results[-500:]
        _save_json(TRADE_RESULTS_FILE, results)
        _save_json(RESOLVED_IDS_FILE, list(resolved))
        log.info(f"[Resolver] {newly_resolved} trades resolus et enregistres.")

    return newly_resolved


def _trade_id(trade: dict) -> str:
    """Identifiant unique d'un trade (ticker + timestamp)."""
    return f"{trade.get('ticker','')}_{trade.get('timestamp','')}"


# ── Statistiques de performance ───────────────────────────────────────────────

def get_full_stats() -> dict:
    """Retourne des statistiques detaillees sur les trades resolus."""
    results = _load_json(TRADE_RESULTS_FILE, [])
    if not results:
        return {"total": 0, "message": "Aucun trade resolu."}

    total     = len(results)
    wins      = [r for r in results if r.get("won")]
    losses    = [r for r in results if not r.get("won")]
    win_rate  = len(wins) / total
    total_pnl = sum(r.get("pnl", 0) for r in results)

    # Stats par tranche d'edge
    edge_buckets = {"0-5%": [], "5-10%": [], "10%+": []}
    for r in results:
        e = abs(r.get("edge", 0))
        if e < 0.05:
            edge_buckets["0-5%"].append(r)
        elif e < 0.10:
            edge_buckets["5-10%"].append(r)
        else:
            edge_buckets["10%+"].append(r)

    edge_stats = {}
    for bucket, trades in edge_buckets.items():
        if trades:
            wr = sum(1 for t in trades if t.get("won")) / len(trades)
            pnl = sum(t.get("pnl", 0) for t in trades)
            edge_stats[bucket] = {"trades": len(trades), "win_rate": wr, "pnl": pnl}

    # Stats par cote (YES vs NO)
    yes_trades = [r for r in results if r.get("side") == "yes"]
    no_trades  = [r for r in results if r.get("side") == "no"]
    yes_wr = sum(1 for t in yes_trades if t.get("won")) / len(yes_trades) if yes_trades else 0
    no_wr  = sum(1 for t in no_trades  if t.get("won")) / len(no_trades)  if no_trades  else 0

    # 7 derniers jours
    now_ts = time.time()
    recent_7d = [r for r in results if now_ts - r.get("timestamp", 0) < 7 * 86400]
    wr_7d     = sum(1 for r in recent_7d if r.get("won")) / len(recent_7d) if recent_7d else 0
    pnl_7d    = sum(r.get("pnl", 0) for r in recent_7d)

    return {
        "total":       total,
        "win_rate":    win_rate,
        "total_pnl":   total_pnl,
        "wins":        len(wins),
        "losses":      len(losses),
        "edge_stats":  edge_stats,
        "yes_win_rate": yes_wr,
        "no_win_rate":  no_wr,
        "yes_trades":   len(yes_trades),
        "no_trades":    len(no_trades),
        "win_rate_7d":  wr_7d,
        "pnl_7d":       pnl_7d,
        "trades_7d":    len(recent_7d),
    }


def print_stats():
    s = get_full_stats()
    if s.get("total", 0) == 0:
        print("Aucun trade resolu pour le moment.")
        return
    sep = "=" * 55
    print(f"""
{sep}
  PERFORMANCE BTC KALSHI ALPHA BOT
{sep}
  Total trades resolus : {s['total']}
  Win rate global      : {s['win_rate']:.1%}
  PnL total            : ${s['total_pnl']:+.2f}
{sep}
  YES : {s['yes_trades']} trades | WR {s['yes_win_rate']:.1%}
  NO  : {s['no_trades']} trades | WR {s['no_win_rate']:.1%}
{sep}
  7 derniers jours : {s['trades_7d']} trades | WR {s['win_rate_7d']:.1%} | PnL ${s['pnl_7d']:+.2f}
{sep}
  Par tranche d'edge :""")
    for bucket, st in s.get("edge_stats", {}).items():
        print(f"    {bucket:8s} : {st['trades']:3d} trades | WR {st['win_rate']:.1%} | PnL ${st['pnl']:+.2f}")
    print(sep)


# ── Main (usage standalone) ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description="Trade Resolver -- BTC Kalshi")
    parser.add_argument("--stats", action="store_true", help="Affiche les stats de performance")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        # Resolution standalone -- necessite un KalshiClient
        print("Usage: python trade_resolver.py --stats")
        print("Pour resoudre les trades, il est appele automatiquement par kalshi_alpha_bot.py")
