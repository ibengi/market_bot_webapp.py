"""
Tests d'integration A-O (unittest, hors-ligne, deterministes, sans reseau).
Chaque test correspond a une exigence de la mission d'integration.
"""
import os, sys, json, unittest, tempfile, shutil
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_alpha_bot as bot
from kalshi_alpha_bot import (ExecutionEngine, FeeModel, OrderManager,
                              PositionManager, TradeLogger, Config)
from strategy_router import (StrategyRouter, GateConfig, evaluate_candidate,
                             SignalDecision)
from opportunity_pipeline import MarketOpportunityPipeline
from market_scanner import build_snapshot, ScanConfig
from market_ranker import RankConfig

NOW = datetime.now(timezone.utc)

def mkt(ticker, mins=60, yb=50, ya=51, vol=5000, oi=2000, liq=8000,
        serie="KXETH", empty=False, **kw):
    m = {"ticker": ticker, "status": "open", "series_ticker": serie,
         "close_time": (NOW + timedelta(minutes=mins)).isoformat(),
         "volume": vol, "open_interest": oi, "liquidity": liq}
    if empty:
        m.update({"yes_bid": 0, "yes_ask": 0, "no_bid": 0, "no_ask": 0})
    else:
        m.update({"yes_bid": yb, "yes_ask": ya,
                  "no_bid": 100 - ya, "no_ask": 100 - yb})
    m.update(kw)
    return m


class StubStrategy:
    """Strategie de test : probabilite FOURNIE PAR LE TEST (jamais inventee
    par le code de prod)."""
    def __init__(self, categories, prob_by_ticker, confidence=8):
        self.categories = categories
        self.name = "stub"
        self.prob_by_ticker = prob_by_ticker
        self.confidence = confidence
    def evaluate(self, snapshot, market, book):
        p = self.prob_by_ticker.get(snapshot.ticker)
        if p is None:
            return None
        return {"side": "yes", "model_prob": p,
                "confidence": self.confidence, "taille": "1%",
                "reason": "stub"}


class FakeBroker:
    """Client hors-ligne complet : /markets pagines, get_market frais,
    ordres, fills, solde, positions. Espionne create_order."""
    def __init__(self, markets, balance=1000.0, fills_fee=None,
                 fresh_override=None, order_counts=True,
                 broker_positions=None):
        self.markets = {m["ticker"]: m for m in markets}
        self.pages = [markets]
        self.balance = balance
        self.fills_fee = fills_fee          # ex: {"taker_fees": "0.05"}
        self.fresh_override = fresh_override or {}
        self.order_counts = order_counts
        self.broker_positions = broker_positions or []
        self.created = []                    # espion create_order
        self.env = "demo"

    def _req(self, method, path, params=None, **kw):
        if path == "/markets":
            if (params or {}).get("cursor"):
                return {"markets": [], "cursor": None}
            return {"markets": self.pages[0], "cursor": None}
        raise AssertionError(f"_req inattendu: {path}")
    def _log_raw_once(self, *a): pass
    def get_market(self, tk):
        return self.fresh_override.get(tk, self.markets.get(tk, {}))
    def get_balance(self): return self.balance
    def get_positions(self): return self.broker_positions
    def create_order(self, ticker, side, count, price):
        self.created.append((ticker, side, count, price))
        base = {"order_id": f"o{len(self.created)}", "status": "executed"}
        if self.order_counts:
            base["fill_count"] = count
        return base
    def get_order(self, oid): return {"order_id": oid, "status": "executed"}
    def cancel_order(self, oid): return {}
    def get_fills(self, oid):
        f = {"fill_id": f"f-{oid}", "count": self.created[-1][2],
             "yes_price": self.created[-1][3]}
        if self.fills_fee:
            f.update(self.fills_fee)
        return [f]


class EngineHarness(unittest.TestCase):
    """Prepare un ExecutionEngine complet sur repertoire temporaire."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = bot.CFG.DATA_DIR
        bot.CFG.DATA_DIR = self.tmp
        self._saved = {k: getattr(bot.CFG, k) for k in
                       ("SHADOW_MODE", "KILL_SWITCH", "MAX_TRADES_CYCLE",
                        "MIN_MODEL_CONFIDENCE", "MIN_GROSS_EDGE",
                        "MIN_NET_EDGE", "MIN_NET_EV")}
        os.makedirs(os.path.join(self.tmp, "reports"), exist_ok=True)
    def tearDown(self):
        bot.CFG.DATA_DIR = self._old
        for k, v in self._saved.items(): setattr(bot.CFG, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def engine(self, client, capital=500.0, probs=None, cats=("Crypto",)):
        eng = ExecutionEngine(client, capital)
        # remplace le routeur par un stub controle par le test
        eng.router = StrategyRouter()
        if probs is not None:
            eng.router.register(StubStrategy(list(cats), probs))
        eng.pipeline = MarketOpportunityPipeline(
            client, eng.router, gates=eng.gates,
            fresh_book_fn=eng.fresh_book, save_artifacts=False)
        # ranker: pas d'historique requis pour ces tests
        eng.pipeline.rank_cfg = RankConfig()
        eng.pipeline.rank_cfg.MIN_BOOK_OBSERVATIONS = 999  # stabilite neutre
        eng.pipeline.rank_cfg.HISTORY_FILE = os.path.join(self.tmp, "h.json")
        eng.pipeline.scan_cfg = ScanConfig()
        eng.pipeline.scan_cfg.UNIVERSE_FILE = os.path.join(self.tmp, "u.json")
        eng.pipeline.scan_cfg.REPORT_FILE = os.path.join(self.tmp, "r.json")
        eng.pipeline.rank_cfg.RANKINGS_FILE = os.path.join(self.tmp, "rk.json")
        eng.pipeline.rank_cfg.REPORT_FILE = os.path.join(self.tmp, "rr.json")
        eng.pipeline.save_artifacts = True
        return eng


class TestA_NormalModeUsesPipeline(EngineHarness):
    def test_cycle_calls_scanner_and_ranker(self):
        cli = FakeBroker([mkt("KXETH-A")])
        eng = self.engine(cli, probs={})
        eng.cycle(1)
        # preuve : les artefacts scanner+ranker du cycle existent
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "u.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "rk.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "cycle_report.json")))


class TestB_NotStuckOnEmptyBTC(EngineHarness):
    def test_empty_btc_other_candidate_selected(self):
        markets = [mkt("KXBTC15M-X", serie="KXBTC15M", empty=True,
                       vol=999999),
                   mkt("KXETH-GOOD", yb=50, ya=51)]
        cli = FakeBroker(markets)
        eng = self.engine(cli, probs={"KXETH-GOOD": 0.62})
        placed = eng.cycle(1)
        self.assertEqual(placed, 1)
        self.assertEqual(cli.created[0][0], "KXETH-GOOD")   # pas le BTC vide


class TestC_AllBooksEmpty_NoOrder(EngineHarness):
    def test_zero_create_order(self):
        markets = [mkt(f"E{i}", empty=True, vol=10**6) for i in range(4)]
        cli = FakeBroker(markets)
        eng = self.engine(cli, probs={f"E{i}": 0.9 for i in range(4)})
        placed = eng.cycle(1)
        self.assertEqual(placed, 0)
        self.assertEqual(cli.created, [])                   # INVARIANT


class TestD_NoStrategy_Rejected(EngineHarness):
    def test_liquid_market_without_strategy(self):
        cli = FakeBroker([mkt("FED-X", serie="FED", vol=90000, oi=40000,
                              liq=200000, yb=71, ya=72)])
        eng = self.engine(cli, probs={})     # routeur vide: aucune strategie
        placed = eng.cycle(1)
        self.assertEqual(placed, 0)
        rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertGreaterEqual(
            rep["rejections"].get("no_compatible_strategy", 0), 1)
        self.assertEqual(cli.created, [])


class TestE_EdgeZero_NoOrder(EngineHarness):
    def test_model_equals_market(self):
        cli = FakeBroker([mkt("KXETH-A", yb=50, ya=51)])
        eng = self.engine(cli, probs={"KXETH-A": 0.51})     # = ask => edge 0
        placed = eng.cycle(1)
        self.assertEqual(placed, 0)
        rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertGreaterEqual(rep["rejections"].get("no_positive_edge", 0), 1)
        self.assertEqual(cli.created, [])


class TestF_NegativeNetEV_NoOrder(EngineHarness):
    def test_gross_edge_eaten_by_fees_and_slippage(self):
        bot.CFG.MIN_GROSS_EDGE = 0.01
        bot.CFG.MIN_NET_EDGE = -1.0        # isole la porte EV
        cli = FakeBroker([mkt("KXETH-A", yb=50, ya=51)])
        eng = self.engine(cli, probs={"KXETH-A": 0.53})
        # gross_edge=0.02 ; frais~0.02 + slippage 0.01 => net_ev < 0
        eng.gates.MIN_GROSS_EDGE = 0.01
        eng.gates.MIN_NET_EDGE = -1.0
        placed = eng.cycle(1)
        self.assertEqual(placed, 0)
        rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertGreaterEqual(rep["rejections"].get("negative_net_ev", 0), 1)
        self.assertEqual(cli.created, [])


class TestG_FirstRejected_SecondTraded(EngineHarness):
    def test_two_candidates(self):
        markets = [mkt("BIG-NOEDGE", vol=90000, oi=50000, liq=90000,
                       yb=50, ya=51),
                   mkt("SMALL-EDGE", vol=4000, oi=1500, liq=6000,
                       yb=50, ya=51)]
        cli = FakeBroker(markets)
        eng = self.engine(cli, probs={"BIG-NOEDGE": 0.51,   # edge 0 -> rejet
                                      "SMALL-EDGE": 0.65})  # edge net > seuils
        placed = eng.cycle(1)
        self.assertEqual(placed, 1)
        self.assertEqual(cli.created[0][0], "SMALL-EDGE")
        rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertGreaterEqual(rep["rejections"].get("no_positive_edge", 0), 1)


class TestH_PartialFill(EngineHarness):
    def test_only_filled_quantity_becomes_position(self):
        class PartialBroker(FakeBroker):
            def create_order(self, ticker, side, count, price):
                self.created.append((ticker, side, count, price))
                return {"order_id": "op", "status": "executed",
                        "fill_count": max(1, count // 2)}     # partiel
            def get_fills(self, oid):
                c = max(1, self.created[-1][2] // 2)
                return [{"fill_id": "fp1", "count": c,
                         "yes_price": self.created[-1][3]}]
        cli = PartialBroker([mkt("KXETH-A", yb=50, ya=51)])
        eng = self.engine(cli, probs={"KXETH-A": 0.65})
        placed = eng.cycle(1)
        self.assertEqual(placed, 1)
        req = cli.created[0][2]
        pos = list(eng.posmgr.positions.values())[0]
        self.assertEqual(pos["count"], max(1, req // 2))     # quantite REELLE
        self.assertLess(pos["count"], req)


class TestI_ExecutedWithoutCounts_ChecksFills(EngineHarness):
    def test_fills_endpoint_is_source_of_truth(self):
        cli = FakeBroker([mkt("KXETH-A", yb=50, ya=51)], order_counts=False)
        eng = self.engine(cli, probs={"KXETH-A": 0.65})
        placed = eng.cycle(1)
        self.assertEqual(placed, 1)                          # confirme via /fills
        pos = list(eng.posmgr.positions.values())[0]
        self.assertGreater(pos["count"], 0)


class TestJ_RestartRebuildNoDuplicates(EngineHarness):
    def test_broker_reconciliation_idempotent(self):
        cli = FakeBroker([mkt("KXETH-A")],
                         broker_positions=[{"ticker": "KXETH-A",
                                            "position": 3, "avg_price": 48}])
        tl = TradeLogger()
        pm = PositionManager(cli, tl)
        pm.reconcile_with_broker()
        self.assertEqual(pm.open_count(), 1)                 # reconstruite
        pm.reconcile_with_broker()                           # redemarrage
        pm2 = PositionManager(cli, TradeLogger())
        pm2.reconcile_with_broker()
        self.assertEqual(pm2.open_count(), 1)                # PAS de doublon
        self.assertIn("brk-KXETH-A-yes", pm2.positions)


class TestK_RealBalanceCapsSizing(EngineHarness):
    def test_sizing_uses_min_of_balance_and_config(self):
        cli = FakeBroker([mkt("KXETH-A", yb=50, ya=51)], balance=200.0)
        eng = self.engine(cli, capital=500.0, probs={"KXETH-A": 0.65})
        placed = eng.cycle(1)
        self.assertEqual(placed, 1)
        self.assertEqual(eng.capital, 200.0)                 # solde reel prime
        count = cli.created[0][2]
        # 1% de 200$ = 2$ a ~51c => 3 contrats max (jamais base sur 500$)
        self.assertLessEqual(count, int(200 * 0.01 / 0.51) + 1)
        self.assertLess(count, int(500 * 0.01 / 0.51))


class TestL_BookVanishesBeforeOrder(EngineHarness):
    def test_fresh_reread_blocks_order(self):
        good = mkt("KXETH-A", yb=50, ya=51)
        vanished = mkt("KXETH-A", empty=True)
        cli = FakeBroker([good], fresh_override={"KXETH-A": vanished})
        eng = self.engine(cli, probs={"KXETH-A": 0.65})
        placed = eng.cycle(1)
        self.assertEqual(placed, 0)
        self.assertEqual(cli.created, [])                    # bloque AVANT ordre
        rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertGreaterEqual(rep["rejections"].get("stale_book", 0), 1)


class TestM_APIFeesPreferred(EngineHarness):
    def test_api_fees_override_local_model(self):
        cli = FakeBroker([mkt("KXETH-A", yb=50, ya=51)],
                         fills_fee={"taker_fees": "0.05"})
        eng = self.engine(cli, probs={"KXETH-A": 0.65})
        placed = eng.cycle(1)
        self.assertEqual(placed, 1)
        t = eng.tlog.trades[-1]
        self.assertEqual(t["analysis"]["fee_source"], "api")
        self.assertAlmostEqual(t["fees"], 0.05, places=4)
        self.assertNotEqual(t["fees"],
                            FeeModel.trading_fee(cli.created[0][2], 51))


class TestN_PaginationAllPages(unittest.TestCase):
    def test_pipeline_scans_every_page(self):
        from market_scanner import fetch_all_markets
        class Paged(FakeBroker):
            def __init__(self):
                super().__init__([])
                self.pages = [[mkt(f"P0-{i}") for i in range(3)],
                              [mkt(f"P1-{i}") for i in range(3)],
                              [mkt("P2-0")]]
            def _req(self, method, path, params=None, **kw):
                cur = (params or {}).get("cursor")
                i = 0 if not cur else int(cur)
                batch = self.pages[i] if i < len(self.pages) else []
                nxt = str(i + 1) if i + 1 < len(self.pages) else None
                return {"markets": batch, "cursor": nxt}
        markets, pages, errs = fetch_all_markets(Paged())
        self.assertEqual((len(markets), pages, errs), (7, 3, 0))


class TestO_CycleReportCountsRejections(EngineHarness):
    def test_all_reasons_counted(self):
        markets = [mkt("EMPTY", empty=True, vol=10**6),
                   mkt("NOSTRAT", serie="FED", yb=70, ya=71),
                   mkt("NOEDGE", yb=50, ya=51),
                   mkt("GOOD", yb=50, ya=51)]
        cli = FakeBroker(markets)
        eng = self.engine(cli, probs={"NOEDGE": 0.51, "GOOD": 0.66})
        placed = eng.cycle(7)
        rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertEqual(rep["cycle"], 7)
        self.assertEqual(placed, 1)
        self.assertEqual(rep["fills_confirmed"], 1)
        self.assertEqual(rep["orders_submitted"], 1)
        r = rep["rejections"]
        self.assertGreaterEqual(r.get("no_liquidity", 0), 1)          # EMPTY
        self.assertGreaterEqual(r.get("no_compatible_strategy", 0), 1)  # NOSTRAT
        self.assertGreaterEqual(r.get("no_positive_edge", 0), 1)      # NOEDGE
        for k in ("scanned", "ranker_eligible", "strategy_supported",
                  "positive_edge"):
            self.assertIn(k, rep)


class TestShadowAndKillSwitch(EngineHarness):
    def test_shadow_full_decision_zero_orders(self):
        bot.CFG.SHADOW_MODE = True
        cli = FakeBroker([mkt("KXETH-A", yb=50, ya=51)])
        eng = self.engine(cli, probs={"KXETH-A": 0.65})
        placed = eng.cycle(1)
        self.assertEqual((placed, cli.created), (0, []))
        rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertGreaterEqual(rep["rejections"].get("shadow_mode", 0), 1)

    def test_kill_switch_blocks_everything(self):
        bot.CFG.KILL_SWITCH = True
        cli = FakeBroker([mkt("KXETH-A", yb=50, ya=51)])
        eng = self.engine(cli, probs={"KXETH-A": 0.65})
        self.assertEqual(eng.cycle(1), 0)
        self.assertEqual(cli.created, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestDesyncRankerPipeline(EngineHarness):
    """Regression de la panne du 12/07 : 100% des eligibles en
    snapshot_missing quand un market_ranker ancien est deploye a cote d'un
    pipeline recent. Le pipeline doit etre PROPRIETAIRE de ses snapshots."""

    def test_new_ranker_zero_snapshot_missing(self):
        cli = FakeBroker([mkt("KXETH-GOOD", yb=50, ya=51)])
        eng = self.engine(cli, probs={"KXETH-GOOD": 0.65})
        placed = eng.cycle(1)
        rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertEqual(rep["rejections"].get("snapshot_missing", 0), 0)
        self.assertEqual(placed, 1)

    def test_legacy_ranker_without_snapshots_param(self):
        """Simule un market_ranker.py ANCIEN : refuse le kwarg `snapshots`
        et ne renvoie PAS de cle 'snapshots'. Le pipeline doit quand meme
        examiner les candidats (mode degrade), zero snapshot_missing."""
        import opportunity_pipeline as op
        import market_ranker as mr
        real = op.run_ranking
        def legacy_run_ranking(client, scan_cfg=None, cfg=None, save=True,
                               now=None):                 # PAS de snapshots=
            res = real(client, scan_cfg=scan_cfg, cfg=cfg, save=save, now=now)
            res.pop("snapshots", None)                    # ancien retour
            return res
        op.run_ranking = legacy_run_ranking
        try:
            cli = FakeBroker([mkt("KXETH-GOOD", yb=50, ya=51)])
            eng = self.engine(cli, probs={"KXETH-GOOD": 0.65})
            placed = eng.cycle(1)
            rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
            self.assertEqual(rep["rejections"].get("snapshot_missing", 0), 0,
                             "melange de versions => snapshot_missing")
            self.assertEqual(placed, 1)                   # candidat traite
        finally:
            op.run_ranking = real
