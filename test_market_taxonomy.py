"""
test_market_taxonomy.py — taxonomie + routeur strict (hors-ligne).
Couvre les 12 cas imposes par le cahier des charges.
"""
import os
import sys
import unittest
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_taxonomy import classify_market_type, MARKET_TYPES
from market_scanner import build_snapshot
from strategy_router import (StrategyRouter, BtcModelStrategy,
                             build_default_router, available_strategies)
from opportunity_pipeline import MarketOpportunityPipeline
from market_ranker import RankConfig
from market_scanner import ScanConfig

NOW = datetime.now(timezone.utc)


def snap(ticker, serie=None, title="", cat_extra=None, mins=60,
         yb=50, ya=51):
    m = {"ticker": ticker, "series_ticker": serie or ticker.split("-")[0],
         "status": "open", "title": title,
         "close_time": (NOW + timedelta(minutes=mins)).isoformat(),
         "yes_bid": yb, "yes_ask": ya, "no_bid": 100 - ya, "no_ask": 100 - yb,
         "volume": 5000, "open_interest": 2000, "liquidity": 8000}
    if cat_extra:
        m["category"] = cat_extra
    return build_snapshot(m, now=NOW)


class TestClassification(unittest.TestCase):
    def test_btc15m(self):
        s = snap("KXBTC15M-26JUL131500-00")
        self.assertEqual(s.market_type, "btc_above_strike_15m")
    def test_btc_hourly_is_not_15m(self):
        s = snap("KXBTC1H-26JUL13-T64000", serie="KXBTC1H")
        self.assertEqual(s.market_type, "btc_above_strike_1h")
    def test_eth_15m(self):
        s = snap("KXETH15M-26JUL131500-00", serie="KXETH15M")
        self.assertEqual(s.market_type, "eth_above_strike_15m")
    def test_sports_moneyline(self):
        s = snap("KXMLBGAME-26JUL13HOUTEX-HOU", serie="KXMLBGAME")
        self.assertEqual(s.market_type, "sports_moneyline")
    def test_sports_total_over_under(self):
        s = snap("KXMLBTOTAL-26JUL13-O8.5", serie="KXMLBTOTAL",
                 title="Astros vs Rangers: total runs over/under 8.5")
        self.assertEqual(s.market_type, "sports_total")
    def test_sports_spread(self):
        s = snap("KXMLBSPREAD-26JUL13AZLAD-LAD3", serie="KXMLBSPREAD")
        self.assertEqual(s.market_type, "sports_spread")
    def test_sports_player_prop(self):
        s = snap("KXMLBHIT-26JUL13-CINEDELACRUZ44-1", serie="KXMLBHIT")
        self.assertEqual(s.market_type, "sports_player_prop")
    def test_economics_weather_elections(self):
        self.assertEqual(classify_market_type({"ticker": "KXCPI-26AUG-T3.1"}),
                         "cpi_above_threshold")
        self.assertEqual(classify_market_type({"ticker": "KXFED-26SEP-CUT"}),
                         "fed_rate_decision")
        self.assertEqual(classify_market_type({"title": "July jobs report"
                                               " above 150k payrolls?"}),
                         "jobs_report")
        self.assertEqual(classify_market_type({"ticker": "KXHIGHTOKC-26JUL13-T90"}),
                         "weather_high_temperature")
        self.assertEqual(classify_market_type({"ticker": "PRES-2028-DEM"}),
                         "election_winner")
    def test_unknown(self):
        s = snap("ZZZUNKNOWN-1", serie="ZZZ", title="???")
        self.assertEqual(s.market_type, "unknown")
    def test_deterministic_same_input_same_output(self):
        m = {"ticker": "KXMLBSPREAD-26JUL13AZLAD-LAD3",
             "series_ticker": "KXMLBSPREAD", "title": "x"}
        outs = {classify_market_type(dict(m)) for _ in range(50)}
        self.assertEqual(outs, {"sports_spread"})
    def test_priority_series_over_title(self):
        # serie BTC15M mais titre sportif : la serie (etape 1) l'emporte
        mt = classify_market_type({"series_ticker": "KXBTC15M",
                                   "title": "MLB game winner"})
        self.assertEqual(mt, "btc_above_strike_15m")
    def test_all_outputs_are_known_types(self):
        for t in ("KXBTC15M-X", "KXNBA-GAME", "KXCPI-1", "junk"):
            self.assertIn(classify_market_type({"ticker": t}), MARKET_TYPES)


class TestBtc15mSupports(unittest.TestCase):
    def setUp(self):
        self.strat = BtcModelStrategy(context_provider=lambda **k: None,
                                      model_predict=lambda c: (None, "x"))
    def test_accepts_only_btc15m(self):
        self.assertTrue(self.strat.supports(snap("KXBTC15M-26JUL131500-00")))
    def test_rejects_btc_hourly(self):
        self.assertFalse(self.strat.supports(
            snap("KXBTC1H-26JUL13-T64000", serie="KXBTC1H")))
    def test_rejects_eth(self):
        self.assertFalse(self.strat.supports(
            snap("KXETH15M-26JUL131500-00", serie="KXETH15M")))
    def test_rejects_sports(self):
        self.assertFalse(self.strat.supports(
            snap("KXMLBGAME-26JUL13HOUTEX-HOU", serie="KXMLBGAME")))


class TestStrictRouter(unittest.TestCase):
    class S:
        def __init__(self, name, ok):
            self.name, self._ok = name, ok
            self.categories = ["Test"]
        def supports(self, s): return self._ok
        def evaluate(self, *a): return None

    def test_unknown_market_no_compatible_strategy(self):
        r = build_default_router()
        strat, why = r.resolve(snap("ZZZUNKNOWN-1", serie="ZZZ"))
        self.assertIsNone(strat)
        self.assertEqual(why, "no_compatible_strategy")

    def test_ambiguous_when_two_match(self):
        r = StrategyRouter()
        r.register(self.S("a", True)); r.register(self.S("b", True))
        strat, why = r.resolve(snap("KXBTC15M-X"))
        self.assertIsNone(strat)
        self.assertEqual(why, "ambiguous_strategy_match")

    def test_single_match_resolves(self):
        r = StrategyRouter()
        r.register(self.S("a", True)); r.register(self.S("b", False))
        strat, why = r.resolve(snap("X-1"))
        self.assertEqual((strat.name, why), ("a", None))

    def test_strategy_without_supports_is_refused(self):
        class NoSupports:
            name = "bad"
        r = StrategyRouter()
        r.register(NoSupports())
        self.assertEqual(r.strategies(), [])   # jamais de routage implicite

    def test_registry_contains_only_real_strategies(self):
        self.assertEqual([c.__name__ for c in available_strategies()],
                         ["BtcModelStrategy"])


class TestPipelinePrefilter(unittest.TestCase):
    def test_no_fresh_book_call_without_strategy_and_no_order(self):
        """Un marche sans strategie ne declenche NI fresh_book_fn NI ordre."""
        calls = []
        class Cli:
            def _req(self, m, p, params=None, **k):
                if (params or {}).get("cursor"):
                    return {"markets": [], "cursor": None}
                mm = snap("KXMLBGAME-26JUL13HOUTEX-HOU",
                          serie="KXMLBGAME").raw_market
                return {"markets": [mm], "cursor": None}
            def _log_raw_once(self, *a): pass
            def create_order(self, *a, **k):
                raise AssertionError("ordre cree par classifier/router !")
        pipe = MarketOpportunityPipeline(
            Cli(), build_default_router(),
            fresh_book_fn=lambda tk: calls.append(tk) or ({}, None),
            save_artifacts=False)
        pipe.rank_cfg = RankConfig(); pipe.rank_cfg.MIN_BOOK_OBSERVATIONS = 999
        import tempfile, os as _os
        tmp = tempfile.mkdtemp()
        pipe.rank_cfg.HISTORY_FILE = _os.path.join(tmp, "h.json")
        pipe.scan_cfg = ScanConfig()
        res = pipe.run_cycle(max_accepted=1)
        self.assertEqual(calls, [], "fresh_book_fn appele sans strategie")
        self.assertEqual(res["report"]["rejections"]
                         .get("no_compatible_strategy", 0), 1)
        self.assertGreaterEqual(res["report"]["classified"], 1)

    def test_counters_present(self):
        class Cli:
            def _req(self, m, p, params=None, **k):
                return {"markets": [], "cursor": None}
            def _log_raw_once(self, *a): pass
        pipe = MarketOpportunityPipeline(Cli(), build_default_router(),
                                         save_artifacts=False)
        import tempfile, os as _os
        pipe.rank_cfg = RankConfig()
        pipe.rank_cfg.HISTORY_FILE = _os.path.join(tempfile.mkdtemp(), "h.json")
        rep = pipe.run_cycle()["report"]
        for k in ("classified", "unknown_market_type", "strategy_supported",
                  "prefiltered_no_strategy", "ambiguous_strategy_match"):
            self.assertIn(k, rep)

    def test_taxonomy_module_has_no_order_path(self):
        src = open("market_taxonomy.py", encoding="utf-8").read()
        for banned in ("create_order", "place_and_track", "portfolio/orders",
                       "requests"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
