"""
opportunity_pipeline.py — v1 (2026-07-11)
MarketOpportunityPipeline : SCANNER -> RANKER -> ROUTEUR -> PORTES EDGE/EV.

Utilise a CHAQUE cycle normal du bot (plus seulement --scan-only/--rank-only).
Ce module ne place AUCUN ordre : il produit des SignalDecision acceptees,
classees par qualite, que le moteur d'execution consomme ensuite (avec sa
propre relecture du carnet et ses portes de risque).

Rapport de cycle ecrit par l'appelant (cycle_report.json) :
    scanned, scanner_included, ranker_eligible, strategy_supported,
    positive_edge, orders_submitted, fills_confirmed, rejections{...}
"""

import logging
from typing import Optional, Callable

from market_scanner import run_scan, ScanConfig, MarketSnapshot
from market_ranker import run_ranking, RankConfig
from strategy_router import (StrategyRouter, GateConfig, SignalDecision,
                             evaluate_candidate)

log = logging.getLogger("PIPELINE")


class MarketOpportunityPipeline:
    def __init__(self, client, router: StrategyRouter,
                 gates: GateConfig = None,
                 scan_cfg: ScanConfig = None,
                 rank_cfg: RankConfig = None,
                 fresh_book_fn: Optional[Callable] = None,
                 save_artifacts: bool = True,
                 observer: Optional[Callable] = None):
        """observer(snapshot, book, decision) : appele pour CHAQUE candidat
        evalue (accepte ou rejete) — utilise par le journal shadow."""
        """fresh_book_fn(ticker) -> (fresh_market_dict, normalized_book|None)
        Fournie par le moteur : relecture du carnet JUSTE avant decision.
        Si None, le carnet du snapshot (age du scan) est utilise — reserve
        aux tests."""
        self.client = client
        self.router = router
        self.gates = gates or GateConfig()
        self.scan_cfg = scan_cfg or ScanConfig()
        self.rank_cfg = rank_cfg or RankConfig()
        self.fresh_book_fn = fresh_book_fn
        self.save_artifacts = save_artifacts
        self.observer = observer

    # ── un cycle complet ─────────────────────────────────────────────────
    def run_cycle(self, max_accepted: int = 1,
                  skip_ticker_fn: Optional[Callable] = None) -> dict:
        """Retourne {"accepted": [SignalDecision...], "report": {...}}.
        Parcourt les candidats du meilleur au moins bon ; n'abandonne pas
        au premier rejet ; s'arrete a max_accepted ou epuisement."""
        ranking = run_ranking(self.client, scan_cfg=self.scan_cfg,
                              cfg=self.rank_cfg, save=self.save_artifacts)
        scores = ranking["scores"]
        rep_rank = ranking["report"]
        snaps_by_ticker = {}
        # run_ranking re-scanne : retrouver les snapshots via history n'est
        # pas fiable ; on rescanne les snapshots depuis le rapport interne.
        # run_ranking garde l'ordre des scores ; les snapshots sont fournis
        # par run_scan a l'interieur — on les reconstruit via raw si absent.
        for s in ranking.get("snapshots", []):
            snaps_by_ticker[s.ticker] = s

        rejections = {}
        accepted = []
        examined = 0
        supported = 0
        positive_edge = 0

        eligible = [q for q in scores if q.eligible]
        for q in eligible:
            if len(accepted) >= max_accepted:
                break
            snap = snaps_by_ticker.get(q.ticker)
            if snap is None:
                continue
            snap.quality = q                       # visible pour les portes
            examined += 1

            if skip_ticker_fn and skip_ticker_fn(q.ticker):
                rejections["already_open"] = rejections.get("already_open", 0) + 1
                continue

            # carnet FRAIS si le moteur fournit la fonction, sinon snapshot
            if self.fresh_book_fn:
                fresh_market, book = self.fresh_book_fn(q.ticker)
            else:
                fresh_market = snap.raw_market
                book = self._book_from_snapshot(snap)

            dec = evaluate_candidate(snap, fresh_market or {}, book,
                                     self.router, self.gates)
            if self.observer:
                try:
                    self.observer(snap, book, dec)
                except Exception as e:
                    log.warning(f"observer: {e}")
            if dec.strategy:
                supported += 1
            if dec.gross_edge is not None and dec.gross_edge > 0:
                positive_edge += 1
            if dec.accepted:
                accepted.append(dec)
                log.info(f"[PIPELINE] CANDIDAT VALIDE {dec.ticker} "
                         f"{dec.side.upper()} @ {dec.entry_price_cents}c | "
                         f"edge_net={dec.net_edge:+.3f} ev_net={dec.net_ev:+.3f} "
                         f"strat={dec.strategy}")
            else:
                rejections[dec.rejection_reason] = \
                    rejections.get(dec.rejection_reason, 0) + 1
                log.info(f"[PIPELINE] rejet {dec.ticker}: {dec.rejection_reason}")

        # les non-eligibles du ranker comptent aussi dans les rejets
        for q in scores:
            if not q.eligible and q.exclusion_reason:
                rejections[q.exclusion_reason] = \
                    rejections.get(q.exclusion_reason, 0) + 1

        report = {
            "scanned": rep_rank["markets_scored"],
            "scanner_included": rep_rank["markets_scored"]
                                - sum(v for k, v in
                                      rep_rank["excluded_by_reason"].items()
                                      if k in ("no_liquidity", "expired",
                                               "closes_too_soon", "invalid_book",
                                               "spread_too_wide", "low_volume",
                                               "unsupported")),
            "ranker_eligible": rep_rank["eligible"],
            "candidates_examined": examined,
            "strategy_supported": supported,
            "positive_edge": positive_edge,
            "accepted": len(accepted),
            "orders_submitted": 0,       # complete par le moteur
            "fills_confirmed": 0,        # complete par le moteur
            "rejections": rejections,
        }
        return {"accepted": accepted, "report": report}

    @staticmethod
    def _book_from_snapshot(s: MarketSnapshot) -> Optional[dict]:
        if s.yes_bid is None or s.yes_ask is None:
            return None
        return {"yes_bid": s.yes_bid, "yes_ask": s.yes_ask,
                "no_bid": s.no_bid, "no_ask": s.no_ask,
                "spread": s.spread_yes}
