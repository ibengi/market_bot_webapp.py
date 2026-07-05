"""
trade_resolver.py  --  v2 (2026-07-05)
Adapte a l'architecture v10 de kalshi_alpha_bot.

Role : resoudre les trades simules (DRY RUN) enregistres dans
kalshi_trades.json en interrogeant l'API Kalshi de production
(donnees publiques via kalshi.get_market), puis calculer le PnL
apres frais et alimenter les statistiques de btc_context.

Interface attendue par kalshi_alpha_bot :
    resolve_pending_trades(kalshi, **kwargs) -> int   (nb de trades resolus)
    print_stats() -> None
"""

import json, os, math, time, logging
from datetime import datetime, timezone

log = logging.getLogger("TradeResolver")

TRADES_FILE = "kalshi_trades.json"

# Frais de trading Kalshi.
# Formule usuelle publiee : frais = 0.07 x contrats x P x (1-P), arrondis
# au cent superieur, preleves a l'execution (gagnant ou perdant).
# ATTENTION : bareme a VERIFIER dans le document officiel de Kalshi
# ("fee schedule") -- il peut changer et varier selon les series.
FEE_RATE = float(os.getenv("KALSHI_FEE_RATE_TRADING", "0.07"))

# Ne tenter de resoudre un trade qu'apres ce delai (le marche 15min doit
# avoir eu le temps de fermer ET d'etre regle par Kalshi).
MIN_AGE_MINUTES = float(os.getenv("RESOLVER_MIN_AGE_MIN", "20"))

# Nombre max de requetes API par passage (le resolveur tourne tous les
# 5 cycles ; on evite de marteler l'API si beaucoup de trades s'accumulent).
MAX_LOOKUPS_PER_PASS = int(os.getenv("RESOLVER_MAX_LOOKUPS", "25"))


def _fees(count: int, price_cents: int) -> float:
    p = price_cents / 100.0
    return math.ceil(FEE_RATE * count * p * (1 - p) * 100) / 100.0


def _age_minutes(iso_ts: str) -> float:
    """Age du trade en minutes. En cas de timestamp illisible, retourne
    une valeur enorme pour ne pas bloquer la resolution indefiniment."""
    try:
        t = datetime.fromisoformat(iso_ts)
        if t.tzinfo is None:
            # Les timestamps du bot sont naifs ; le conteneur tourne en UTC.
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 60.0
    except Exception:
        return 1e9


def _load_trades() -> list:
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"[Resolver] Lecture {TRADES_FILE} impossible: {e}")
        return []


def _save_trades(trades: list):
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning(f"[Resolver] Ecriture {TRADES_FILE} impossible: {e}")


def resolve_pending_trades(kalshi, **kwargs) -> int:
    """Resout les trades en attente. Retourne le nombre resolu ce passage."""
    trades = _load_trades()
    if not trades:
        return 0

    # Import tardif pour eviter tout import circulaire.
    try:
        from btc_context import record_trade_result
    except ImportError:
        def record_trade_result(*a, **k):  # noqa
            pass

    n_resolved  = 0
    n_lookups   = 0
    n_unsettled = 0
    n_noresult  = 0
    changed     = False

    for t in trades:
        if t.get("resolution"):
            continue                      # deja resolu
        ticker = t.get("ticker")
        if not ticker:
            continue
        if _age_minutes(t.get("timestamp", "")) < MIN_AGE_MINUTES:
            continue                      # marche pas encore ferme/regle
        if n_lookups >= MAX_LOOKUPS_PER_PASS:
            break

        n_lookups += 1
        m = kalshi.get_market(ticker)
        if not m:
            continue

        # Champs attendus du marche regle : status ("settled"/"finalized")
        # et result ("yes"/"no"). NOMS ET VALEURS A VERIFIER dans la doc
        # Kalshi actuelle -- si "result" n'apparait jamais, le compteur
        # n_noresult ci-dessous le rendra visible dans les logs.
        result = str(m.get("result", "") or "").lower()
        status = str(m.get("status", "") or "").lower()

        if result not in ("yes", "no"):
            if status in ("settled", "finalized", "closed"):
                n_noresult += 1
            else:
                n_unsettled += 1
            continue

        side  = str(t.get("side", "")).lower()
        count = int(t.get("count", 0))
        price = int(t.get("price", 0))
        if side not in ("yes", "no") or count <= 0 or not (1 <= price <= 99):
            t["resolution"] = {"error": "donnees de trade invalides"}
            changed = True
            continue

        won  = (result == side)
        cost = count * price / 100.0
        fees = _fees(count, price)
        # Gagnant : paiement 1$/contrat - cout - frais ; Perdant : -cout - frais.
        pnl  = round((count * (100 - price) / 100.0) - fees, 2) if won \
               else round(-(cost + fees), 2)

        t["resolution"] = {
            "result":      result,
            "won":         won,
            "pnl":         pnl,
            "fees":        round(fees, 2),
            "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        changed = True
        n_resolved += 1

        try:
            record_trade_result(verdict=t.get("verdict", ""),
                                edge=float(t.get("edge", 0) or 0),
                                won=won, pnl=pnl)
        except Exception as e:
            log.warning(f"[Resolver] record_trade_result: {e}")

        log.info(f"[Resolver] {ticker} -> {result.upper()} | "
                 f"{'GAGNE' if won else 'PERDU'} | PnL={pnl:+.2f}$ "
                 f"(frais={fees:.2f}$)")

    if n_noresult:
        log.warning(f"[Resolver] {n_noresult} marche(s) fermes sans champ "
                    f"'result' exploitable -- verifier le nom du champ dans "
                    f"la reponse API Kalshi.")
    if changed:
        _save_trades(trades)
    return n_resolved


def print_stats():
    """Affiche un bilan des trades resolus (appele par le bot ou a la main)."""
    trades   = _load_trades()
    resolved = [t for t in trades if t.get("resolution", {}).get("result")]
    pending  = [t for t in trades if not t.get("resolution")]
    if not trades:
        print("[Resolver] Aucun trade enregistre.")
        return
    wins  = [t for t in resolved if t["resolution"].get("won")]
    pnl   = sum(t["resolution"].get("pnl", 0) for t in resolved)
    fees  = sum(t["resolution"].get("fees", 0) for t in resolved)
    wr    = len(wins) / len(resolved) if resolved else 0.0
    print(f"""
==================== BILAN TRADES ====================
  Enregistres : {len(trades)}   Resolus : {len(resolved)}   En attente : {len(pending)}
  Win rate    : {wr:.1%}  ({len(wins)} gagnes / {len(resolved) - len(wins)} perdus)
  PnL net     : {pnl:+.2f}$   (dont frais : -{fees:.2f}$)
======================================================""")
