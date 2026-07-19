"""
test_scanner_priority_series.py — reproduit la panne observee en prod
(troncature MAX_PAGES => KXBTC15M jamais dans l'univers) et prouve la
correction (fetch cible par series_ticker + fusion dedupliquee).
Hors-ligne, deterministe, aucun reseau.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_scanner import (ScanConfig, fetch_universe, fetch_all_markets,
                            run_scan)

NOW = datetime.now(timezone.utc)


def _mkt(ticker, serie, mins=10):
    return {"ticker": ticker, "series_ticker": serie, "status": "open",
            "close_time": (NOW + timedelta(minutes=mins)).isoformat(),
            "yes_bid": 50, "yes_ask": 51, "no_bid": 49, "no_ask": 50,
            "volume": 5000, "open_interest": 2000, "liquidity": 8000}


class HugeUniverseClient:
    """Simule l'API Kalshi : le crawl general renvoie une masse de marches
    sportifs page apres page (curseur sans fin) ; les marches KXBTC15M ne
    sont renvoyes QUE si le parametre serveur series_ticker est fourni."""
    def __init__(self):
        self.calls = []

    def _req(self, method, path, params=None, **kw):
        assert path == "/markets"
        p = dict(params or {})
        self.calls.append(p)
        limit = int(p.get("limit", 200))
        if p.get("series_ticker") == "KXBTC15M":
            if p.get("cursor"):
                return {"markets": [], "cursor": None}
            return {"markets": [_mkt("KXBTC15M-26JUL191500-00", "KXBTC15M"),
                                _mkt("KXBTC15M-26JUL191515-00", "KXBTC15M",
                                     mins=25)],
                    "cursor": None}
        if p.get("series_ticker"):                 # serie inconnue -> vide
            return {"markets": [], "cursor": None}
        # crawl general : pages infinies de marches sportifs distincts
        page = int(p.get("cursor") or 0)
        batch = [_mkt(f"KXWNBATOTAL-P{page}-N{i}", "KXWNBATOTAL")
                 for i in range(limit)]
        return {"markets": batch, "cursor": str(page + 1)}

    def _log_raw_once(self, *a):
        pass


def _cfg(priority):
    cfg = ScanConfig()
    cfg.PAGE_LIMIT = 5
    cfg.MAX_PAGES = 3                  # force la troncature comme en prod
    cfg.PRIORITY_MAX_PAGES = 3
    cfg.PRIORITY_SERIES = priority
    cfg.UNIVERSE_FILE = os.path.join("/tmp", "u_prio.json")
    cfg.REPORT_FILE = os.path.join("/tmp", "r_prio.json")
    return cfg


class Test1_BugReproduced_WithoutPrioritySeries(unittest.TestCase):
    def test_general_crawl_truncated_misses_btc15m(self):
        cli = HugeUniverseClient()
        markets, pages, errors = fetch_all_markets(cli, cfg=_cfg([]))
        tickers = {m["ticker"] for m in markets}
        self.assertEqual(pages, 3)                       # garde-fou atteint
        self.assertEqual(len(markets), 15)               # 3 pages x 5
        self.assertFalse(any(t.startswith("KXBTC15M") for t in tickers),
                         "le crawl tronque ne doit PAS contenir KXBTC15M "
                         "(reproduction de la panne observee)")


class Test2_Fix_PrioritySeriesAlwaysInUniverse(unittest.TestCase):
    def test_btc15m_present_and_deduplicated(self):
        cli = HugeUniverseClient()
        uni = fetch_universe(cli, cfg=_cfg(["KXBTC15M"]))
        tickers = [m["ticker"] for m in uni["markets"]]
        self.assertIn("KXBTC15M-26JUL191500-00", tickers)
        self.assertIn("KXBTC15M-26JUL191515-00", tickers)
        self.assertEqual(len(tickers), len(set(tickers)), "doublons")
        self.assertEqual(uni["priority_markets_received"]["KXBTC15M"], 2)
        self.assertEqual(uni["priority_series_empty"], [])
        # le fetch prioritaire a bien utilise le filtre serveur
        self.assertTrue(any(c.get("series_ticker") == "KXBTC15M"
                            for c in cli.calls))

    def test_run_scan_snapshots_include_btc15m_with_correct_type(self):
        cli = HugeUniverseClient()
        res = run_scan(cli, cfg=_cfg(["KXBTC15M"]), save=False)
        snaps = {s.ticker: s for s in res["snapshots"]}
        self.assertIn("KXBTC15M-26JUL191500-00", snaps)
        s = snaps["KXBTC15M-26JUL191500-00"]
        self.assertEqual(s.market_type, "btc_above_strike_15m")
        self.assertTrue(s.included, f"exclu: {s.exclusion_reason}")
        rep = res["report"]
        self.assertEqual(rep["priority_series"], ["KXBTC15M"])
        self.assertEqual(rep["priority_markets_received"]["KXBTC15M"], 2)


class Test3_EmptyPrioritySeriesIsReportedNotFatal(unittest.TestCase):
    def test_empty_series_flagged(self):
        cli = HugeUniverseClient()
        uni = fetch_universe(cli, cfg=_cfg(["KXNOSUCHSERIES"]))
        self.assertEqual(uni["priority_series_empty"], ["KXNOSUCHSERIES"])
        self.assertEqual(len(uni["markets"]), 15)        # crawl general seul


class Test4_Determinism(unittest.TestCase):
    def test_same_input_same_universe(self):
        t1 = [m["ticker"] for m in
              fetch_universe(HugeUniverseClient(),
                             cfg=_cfg(["KXBTC15M"]))["markets"]]
        t2 = [m["ticker"] for m in
              fetch_universe(HugeUniverseClient(),
                             cfg=_cfg(["KXBTC15M"]))["markets"]]
        self.assertEqual(t1, t2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
