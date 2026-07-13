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
from market_taxonomy import classification_report
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
        # ── CORRECTION DESYNCHRONISATION (2026-07-12) ──
        # Avant : le pipeline dependait du retour "snapshots" de
        # market_ranker.run_ranking ; toute version anterieure du ranker
        # (sans cette cle) faisait tomber 100% des eligibles en
        # snapshot_missing. Desormais le pipeline est PROPRIETAIRE du scan :
        # il scanne lui-meme, garde ses snapshots, et les FOURNIT au ranker
        # (parametre `snapshots` present dans toutes les versions du ranker).
        # Le couplage par valeur de retour est supprime ; un melange de
        # versions de fichiers ne peut plus provoquer cette panne.
        scan = run_scan(self.client, cfg=self.scan_cfg,
                        save=self.save_artifacts)
        own_snapshots = scan["snapshots"]
        try:
            ranking = run_ranking(self.client, scan_cfg=self.scan_cfg,
                                  cfg=self.rank_cfg, save=self.save_artifacts,
                                  snapshots=own_snapshots)
        except TypeError:
            # market_ranker ANCIEN (sans parametre `snapshots`) deploye a
            # cote d'un pipeline recent : mode degrade tolere -- le ranker
            # re-scanne en interne (cout: un scan de plus), mais la
            # correspondance se fait sur NOS snapshots, par ticker. Plus
            # aucun snapshot_missing possible du fait d'un melange de
            # versions.
            log.warning("[PIPELINE] market_ranker.py ANCIEN detecte (pas de "
                        "parametre 'snapshots') -- mode degrade actif. "
                        "Mettre a jour market_ranker.py pour eviter le "
                        "double scan.")
            ranking = run_ranking(self.client, scan_cfg=self.scan_cfg,
                                  cfg=self.rank_cfg, save=self.save_artifacts)
        scores = ranking["scores"]
        rep_rank = ranking["report"]
        snaps_by_ticker = {s.ticker: s for s in own_snapshots}
        log.info(f"[PIPELINE] snapshots (scan pipeline): "
                 f"{len(snaps_by_ticker)} | scores: {len(scores)} "
                 f"| eligibles: {rep_rank['eligible']}")
        if rep_rank["eligible"] > 0 and not snaps_by_ticker:
            log.error("[PIPELINE] INCOHERENCE INTERNE: scores eligibles sans "
                      "snapshots issus du scan du pipeline -- a signaler.")

        rejections = {}
        accepted = []
        examined = 0
        supported = 0
        positive_edge = 0
        with_model_prob = 0
        positive_net_ev = 0
        n_classified = 0
        n_unknown_type = 0
        n_ambiguous = 0
        n_prefiltered = 0
        self.last_traces = []

        eligible = [q for q in scores if q.eligible]
        for q in eligible:
            if len(accepted) >= max_accepted:
                break
            snap = snaps_by_ticker.get(q.ticker)
            if snap is None:
                rejections["snapshot_missing"] = \
                    rejections.get("snapshot_missing", 0) + 1
                log.info(f"[REJECT] {q.ticker}: snapshot_missing "
                         f"(desynchronisation ranker/pipeline)")
                continue
            snap.quality = q                       # visible pour les portes
            examined += 1
            log.info(f"[RANK] {q.ticker} | cat={q.category} "
                     f"| score={q.total_score}")
            mt = getattr(snap, "market_type", "unknown")
            log.info(f"[CLASSIFY] {q.ticker} category={q.category} "
                     f"market_type={mt}")
            if mt == "unknown":
                n_unknown_type += 1
            else:
                n_classified += 1

            if skip_ticker_fn and skip_ticker_fn(q.ticker):
                rejections["already_open"] = rejections.get("already_open", 0) + 1
                continue

            # ── PRE-FILTRAGE STRATEGIE (optimisation imposee) ──
            # Resolution AVANT fresh_book_fn et AVANT tout appel externe du
            # modele : un marche sans strategie compatible ne coute plus
            # aucun appel API.
            pre_strat, pre_why = self.router.resolve(snap)
            if pre_strat is None:
                n_prefiltered += 1
                if pre_why == "ambiguous_strategy_match":
                    n_ambiguous += 1
                rejections[pre_why] = rejections.get(pre_why, 0) + 1
                log.info(f"[REJECT] {q.ticker} {pre_why}")
                self.last_traces.append({
                    "ticker": q.ticker, "ranker_score": q.total_score,
                    "category": q.category, "market_type": mt,
                    "strategy": None, "model_probability": None,
                    "confidence": None, "market_probability": None,
                    "gross_edge": None, "estimated_fees": None,
                    "slippage": None, "net_edge": None, "net_ev": None,
                    "decision": "REJECT", "reject_reason": pre_why})
                continue

            # carnet FRAIS si le moteur fournit la fonction, sinon snapshot
            if self.fresh_book_fn:
                fresh_market, book = self.fresh_book_fn(q.ticker)
            else:
                fresh_market = snap.raw_market
                book = self._book_from_snapshot(snap)

            dec = evaluate_candidate(snap, fresh_market or {}, book,
                                     self.router, self.gates)
            log.info(f"[ROUTER] {q.ticker} -> strategie="
                     f"{dec.strategy or 'AUCUNE'}")
            if dec.model_probability is not None:
                log.info(f"[MODEL] {q.ticker} p_modele="
                         f"{dec.model_probability:.4f} conf={dec.confidence}")
            if dec.gross_edge is not None:
                def _f(v):     # tolerant: net/ev absents si rejet en amont
                    return f"{v:+.4f}" if isinstance(v, (int, float)) else "n/a"
                log.info(f"[EDGE] {q.ticker} p_marche="
                         f"{dec.market_probability:.4f} "
                         f"brut={_f(dec.gross_edge)} "
                         f"frais={dec.estimated_fees} slip={dec.expected_slippage} "
                         f"net={_f(dec.net_edge)} ev={_f(dec.net_ev)}")
            log.info(f"[{'EXECUTION-CANDIDAT' if dec.accepted else 'REJECT'}] "
                     f"{q.ticker}"
                     f"{'' if dec.accepted else ': ' + str(dec.rejection_reason)}")
            self.last_traces.append({
                "ticker": q.ticker, "ranker_score": q.total_score,
                "category": q.category,
                "strategy": dec.strategy,
                "model_probability": dec.model_probability,
                "confidence": dec.confidence,
                "market_probability": dec.market_probability,
                "gross_edge": dec.gross_edge,
                "estimated_fees": dec.estimated_fees,
                "slippage": dec.expected_slippage,
                "net_edge": dec.net_edge, "net_ev": dec.net_ev,
                "decision": "ACCEPT" if dec.accepted else "REJECT",
                "reject_reason": dec.rejection_reason})
            if self.observer:
                try:
                    self.observer(snap, book, dec)
                except Exception as e:
                    log.warning(f"observer: {e}")
            if dec.strategy:
                supported += 1
            if dec.model_probability is not None:
                with_model_prob += 1
            if dec.gross_edge is not None and dec.gross_edge > 0:
                positive_edge += 1
            if dec.net_ev is not None and dec.net_ev > 0:
                positive_net_ev += 1
            if dec.accepted:
                accepted.append(dec)
                log.info(f"[PIPELINE] CANDIDAT VALIDE {dec.ticker} "
                         f"{dec.side.upper()} @ {dec.entry_price_cents}c | "
                         f"edge_net={dec.net_edge:+.3f} ev_net={dec.net_ev:+.3f} "
                         f"strat={dec.strategy}")
            else:
                rejections[dec.rejection_reason] = \
                    rejections.get(dec.rejection_reason, 0) + 1

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
            "valid": None,   # renseigne ci-dessous (= inclus par le scanner)
            "ranker_eligible": rep_rank["eligible"],
            "eligible": rep_rank["eligible"],
            "candidates_examined": examined,
            "classified": n_classified,
            "unknown_market_type": n_unknown_type,
            "prefiltered_no_strategy": n_prefiltered,
            "ambiguous_strategy_match": n_ambiguous,
            "strategy_supported": supported,
            "model_probability": with_model_prob,
            "positive_edge": positive_edge,
            "positive_net_ev": positive_net_ev,
            "risk_passed": 0,            # complete par le moteur
            "accepted": len(accepted),
            "orders_submitted": 0,       # complete par le moteur
            "orders": 0,                 # alias, complete par le moteur
            "fills_confirmed": 0,        # complete par le moteur
            "rejections": rejections,
        }
        report["valid"] = report["scanner_included"]
        if self.save_artifacts:
            from market_scanner import _save_json as _sj
            _sj("classification_report.json",
                classification_report(own_snapshots))
            top = sorted(self.last_traces,
                         key=lambda r: r["ranker_score"], reverse=True)[:20]
            from market_scanner import _save_json
            _save_json("candidate_trace.json",
                       {"generated": report.get("scanned"), "top20": top,
                        "all_examined": len(self.last_traces)})
        return {"accepted": accepted, "report": report}

    @staticmethod
    def _book_from_snapshot(s: MarketSnapshot) -> Optional[dict]:
        if s.yes_bid is None or s.yes_ask is None:
            return None
        return {"yes_bid": s.yes_bid, "yes_ask": s.yes_ask,
                "no_bid": s.no_bid, "no_ask": s.no_ask,
                "spread": s.spread_yes}
