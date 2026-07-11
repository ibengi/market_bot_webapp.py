"""
Tests du market_scanner (unittest — pytest indisponible sans reseau).
Fixtures hors-ligne : faux client pagine, marches aux schemas varies.
"""
import os, sys, json, unittest, tempfile, shutil
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import market_scanner as ms
from market_scanner import (parse_cents, parse_number, normalize_book,
                            classify, build_snapshot, fetch_all_markets,
                            run_scan, ScanConfig, MarketSnapshot)

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)

def mk(ticker="T1", mins=30, **kw):
    m = {"ticker": ticker, "status": "open",
         "close_time": (NOW + timedelta(minutes=mins)).isoformat(),
         "yes_bid": 59, "yes_ask": 60, "no_bid": 40, "no_ask": 41,
         "volume": 100, "open_interest": 50, "liquidity": 1000}
    m.update(kw); return m


class FakePagedClient:
    """3 pages de marches, curseur simule ; espionne tout appel d'ordre."""
    def __init__(self, pages):
        self.pages = pages
        self.order_calls = 0
        self.raw_logged = []
    def _req(self, method, path, params=None, **kw):
        assert method == "GET" and path == "/markets"
        cur = (params or {}).get("cursor")
        idx = 0 if not cur else int(cur)
        batch = self.pages[idx] if idx < len(self.pages) else []
        nxt = str(idx + 1) if idx + 1 < len(self.pages) else None
        return {"markets": batch, "cursor": nxt}
    def _log_raw_once(self, kind, payload):
        self.raw_logged.append(kind)
    def create_order(self, *a, **k):          # ne doit JAMAIS etre appele
        self.order_calls += 1
        raise AssertionError("create_order appele par le scanner !")


class TestNumberNormalization(unittest.TestCase):
    def test_int_cents(self):
        self.assertEqual(parse_cents({"yes_bid": 59}, "yes_bid"), 59)
    def test_fp_string(self):
        self.assertEqual(parse_cents({"yes_bid_fp": "59.00"}, "yes_bid"), 59)
    def test_dollars(self):
        self.assertEqual(parse_cents({"yes_bid_dollars": 0.12}, "yes_bid"), 12)
    def test_dollars_string(self):
        self.assertEqual(parse_cents({"yes_bid_dollars": "0.1200"}, "yes_bid"), 12)
    def test_zero_is_empty_not_price(self):
        self.assertIsNone(parse_cents({"yes_bid": 0}, "yes_bid"))
    def test_null_and_garbage(self):
        self.assertIsNone(parse_cents({"yes_bid": None}, "yes_bid"))
        self.assertIsNone(parse_cents({"yes_bid": "abc"}, "yes_bid"))
        self.assertIsNone(parse_cents({}, "yes_bid"))
    def test_out_of_range_rejected(self):
        self.assertIsNone(parse_cents({"yes_bid": 150}, "yes_bid"))
    def test_generic_number_variants(self):
        self.assertEqual(parse_number({"volume": "123"}, "volume"), 123.0)
        self.assertEqual(parse_number({"volume_fp": "7.00"}, "volume"), 7.0)
        self.assertIsNone(parse_number({}, "volume"))


class TestBookDerivation(unittest.TestCase):
    def test_no_side_derived_from_yes(self):
        b = normalize_book({"yes_bid": 59, "yes_ask": 60})
        self.assertTrue(b["ok"])
        self.assertEqual((b["no_bid"], b["no_ask"]), (40, 41))
        self.assertEqual(b["spread_yes"], 1)
    def test_yes_derived_from_no(self):
        b = normalize_book({"no_bid": 40, "no_ask": 41})
        self.assertTrue(b["ok"])
        self.assertEqual((b["yes_bid"], b["yes_ask"]), (59, 60))
    def test_empty_book_not_invented(self):
        b = normalize_book({"yes_bid": 0, "yes_ask": 0, "no_bid": 0, "no_ask": 0})
        self.assertFalse(b["ok"]); self.assertEqual(b["reason"], "no_liquidity")
        self.assertIsNone(b["yes_mid"])
    def test_single_leg_not_invented(self):
        b = normalize_book({"yes_bid": 59})   # aucune contrepartie derivable
        self.assertFalse(b["ok"]); self.assertEqual(b["reason"], "no_liquidity")
    def test_crossed_book_invalid(self):
        b = normalize_book({"yes_bid": 61, "yes_ask": 60,
                            "no_bid": 39, "no_ask": 40})
        self.assertFalse(b["ok"]); self.assertEqual(b["reason"], "invalid_book")


class TestClassification(unittest.TestCase):
    def test_native_category_wins(self):
        self.assertEqual(classify({"category": "Climate and Weather",
                                   "ticker": "NFLGAME"}), "Climate")
    def test_series_crypto(self):
        self.assertEqual(classify({"series_ticker": "KXBTC15M"}), "Crypto")
    def test_series_sports(self):
        self.assertEqual(classify({"event_ticker": "NBA-FINALS-26"}), "Sports")
    def test_series_economics(self):
        self.assertEqual(classify({"title": "Will CPI exceed 3% in July?"}),
                         "Economics")
    def test_fallback_other(self):
        self.assertEqual(classify({"ticker": "ZZZ", "title": "???"}), "Other")


class TestFiltersAndExclusions(unittest.TestCase):
    def cfg(self, **kw):
        c = ScanConfig()
        for k, v in kw.items(): setattr(c, k, v)
        return c

    def test_valid_market_included(self):
        s = build_snapshot(mk(), now=NOW)
        self.assertTrue(s.included); self.assertIsNone(s.exclusion_reason)
    def test_expired(self):
        s = build_snapshot(mk(mins=-5), now=NOW)
        self.assertEqual(s.exclusion_reason, "expired")
        s = build_snapshot(mk(status="settled"), now=NOW)
        self.assertEqual(s.exclusion_reason, "expired")
    def test_closes_too_soon(self):
        s = build_snapshot(mk(mins=2), now=NOW)
        self.assertEqual(s.exclusion_reason, "closes_too_soon")
    def test_empty_book_reason_kept_not_deleted(self):
        s = build_snapshot(mk(yes_bid=0, yes_ask=0, no_bid=0, no_ask=0), now=NOW)
        self.assertFalse(s.included)
        self.assertEqual(s.exclusion_reason, "no_liquidity")
        self.assertEqual(s.ticker, "T1")      # conserve, pas supprime
    def test_invalid_book(self):
        s = build_snapshot(mk(yes_bid=61, yes_ask=60, no_bid=39, no_ask=40),
                           now=NOW)
        self.assertEqual(s.exclusion_reason, "invalid_book")
    def test_spread_too_wide(self):
        s = build_snapshot(mk(yes_bid=40, yes_ask=60, no_bid=40, no_ask=60),
                           now=NOW, cfg=self.cfg(MAX_SPREAD_CENTS=5))
        self.assertEqual(s.exclusion_reason, "spread_too_wide")
    def test_low_volume(self):
        s = build_snapshot(mk(volume=3), now=NOW, cfg=self.cfg(MIN_VOLUME=10))
        self.assertEqual(s.exclusion_reason, "low_volume")
    def test_category_allow_and_exclude(self):
        c = self.cfg(ALLOWED_CATEGORIES=["Crypto"])
        s = build_snapshot(mk(series_ticker="NBA-X"), now=NOW, cfg=c)
        self.assertEqual(s.exclusion_reason, "unsupported")
        c = self.cfg(EXCLUDED_CATEGORIES=["Sports"])
        s = build_snapshot(mk(series_ticker="NBA-X"), now=NOW, cfg=c)
        self.assertEqual(s.exclusion_reason, "unsupported")
    def test_no_liquidity_kept_if_not_required(self):
        c = self.cfg(REQUIRE_LIQUIDITY=False)
        s = build_snapshot(mk(yes_bid=0, yes_ask=0, no_bid=0, no_ask=0),
                           now=NOW, cfg=c)
        self.assertTrue(s.included)


class TestPagination(unittest.TestCase):
    def test_all_pages_fetched(self):
        pages = [[mk(f"P0-{i}") for i in range(3)],
                 [mk(f"P1-{i}") for i in range(3)],
                 [mk(f"P2-{i}") for i in range(2)]]
        cli = FakePagedClient(pages)
        markets, n_pages, errors = fetch_all_markets(cli)
        self.assertEqual(len(markets), 8)
        self.assertEqual(n_pages, 3)
        self.assertEqual(errors, 0)
        self.assertIn("markets_page", cli.raw_logged)   # 1re page loggee
    def test_max_pages_guard(self):
        endless = FakePagedClient([[mk(f"X{i}")] for i in range(999)])
        c = ScanConfig(); c.MAX_PAGES = 5
        markets, n_pages, _ = fetch_all_markets(endless, cfg=c)
        self.assertEqual(n_pages, 5)                    # garde-fou anti-boucle


class TestScanOnlyNeverOrders(unittest.TestCase):
    def test_run_scan_places_zero_orders_and_reports(self):
        tmp = tempfile.mkdtemp()
        try:
            c = ScanConfig()
            c.UNIVERSE_FILE = os.path.join(tmp, "market_universe.json")
            c.REPORT_FILE = os.path.join(tmp, "market_scanner_report.json")
            pages = [[mk("A", series_ticker="KXBTC15M", volume=500),
                      mk("B", series_ticker="NBA-F", volume=900),
                      mk("C", yes_bid=0, yes_ask=0, no_bid=0, no_ask=0),
                      mk("D", mins=1)]]
            cli = FakePagedClient(pages)
            res = run_scan(cli, cfg=c, now=NOW)
            self.assertEqual(cli.order_calls, 0)        # AUCUN ordre
            rep = res["report"]
            self.assertEqual(rep["total_markets_received"], 4)
            self.assertEqual(rep["valid_markets"], 2)
            self.assertEqual(rep["excluded_by_reason"],
                             {"no_liquidity": 1, "closes_too_soon": 1})
            self.assertEqual(rep["by_category"].get("Crypto"), 1)
            self.assertEqual(rep["by_category"].get("Sports"), 1)
            self.assertIn("C", rep["empty_book_markets"])
            self.assertEqual(rep["top20_volume"][0]["ticker"], "B")
            # fichiers ecrits et JSON valides
            u = json.load(open(c.UNIVERSE_FILE))
            self.assertEqual(len(u), 4)
            self.assertNotIn("raw_market", u[0])
            json.load(open(c.REPORT_FILE))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_scanner_module_has_no_order_path(self):
        """Garantie statique : le module scanner ne reference jamais
        create_order / place_and_track / portfolio/orders."""
        src = open("market_scanner.py", encoding="utf-8").read()
        for forbidden in ("create_order", "place_and_track", "portfolio/orders"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
