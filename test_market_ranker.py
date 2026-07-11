"""
Tests du market_ranker (unittest — pytest indisponible sans reseau).
Tous hors-ligne, deterministes, sans IA, sans ordre.
"""
import os, sys, json, unittest, tempfile, shutil
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import market_ranker as mr
from market_ranker import (score_market, stability_score, update_history,
                           run_ranking, RankConfig, WEIGHTS, SCORE_VERSION)
from market_scanner import build_snapshot, ScanConfig

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)

def snap(ticker="T1", mins=60, yb=52, ya=53, vol=2000, oi=800, liq=5000, **kw):
    m = {"ticker": ticker, "status": "open",
         "close_time": (NOW + timedelta(minutes=mins)).isoformat(),
         "volume": vol, "open_interest": oi, "liquidity": liq}
    if yb is not None:
        m.update({"yes_bid": yb, "yes_ask": ya,
                  "no_bid": 100 - ya, "no_ask": 100 - yb})
    else:
        m.update({"yes_bid": 0, "yes_ask": 0, "no_bid": 0, "no_ask": 0})
    m.update(kw)
    return build_snapshot(m, now=NOW)

def cfg(**kw):
    c = RankConfig()
    for k, v in kw.items(): setattr(c, k, v)
    return c


class TestSpreadMonotonicity(unittest.TestCase):
    def test_score_decreases_as_spread_widens(self):
        totals = []
        for spread in (1, 2, 3, 4, 5):
            s = snap(yb=50, ya=50 + spread)
            q = score_market(s, {}, cfg())
            totals.append(q.total_score)
        self.assertEqual(totals, sorted(totals, reverse=True),
                         f"le score doit decroitre avec le spread: {totals}")
    def test_relative_spread_penalizes_cheap_contracts(self):
        # 2c de spread sur mid ~4c (50% relatif) << 2c sur mid ~50c (4%)
        cheap = score_market(snap(yb=3, ya=5), {}, cfg())
        mid   = score_market(snap(yb=49, ya=51), {}, cfg())
        self.assertLess(cheap.spread_score, mid.spread_score)


class TestEmptyBookHeavilyPenalized(unittest.TestCase):
    def test_empty_book_low_score_and_ineligible(self):
        q = score_market(snap(yb=None, vol=500000, oi=90000, liq=0), {}, cfg())
        self.assertEqual(q.book_quality_score, 0.0)
        self.assertEqual(q.fill_probability_score, 0.0)
        self.assertEqual(q.spread_score, 0.0)
        self.assertFalse(q.eligible)
        self.assertEqual(q.exclusion_reason, "no_liquidity")   # herite scanner
        self.assertLess(q.total_score, 40.0)

    def test_volume_alone_is_insufficient(self):
        """Regle imposee : un volume enorme ne sauve ni un carnet vide ni
        un ask a 99c."""
        empty_huge_vol = score_market(
            snap(yb=None, vol=10**7, oi=10**6), {}, cfg())
        self.assertFalse(empty_huge_vol.eligible)
        extreme = snap(yb=98, ya=99, vol=10**7, oi=10**6, liq=10**6)
        q = score_market(extreme, {}, cfg())
        self.assertLessEqual(q.book_quality_score, 60.0)   # -40 prix extreme
        healthy = score_market(snap(vol=300), {}, cfg())   # petit volume sain
        self.assertLess(q.book_quality_score, healthy.book_quality_score)


class TestTimeRules(unittest.TestCase):
    def test_closes_too_soon_rejected(self):
        q = score_market(snap(mins=2), {}, cfg())
        self.assertFalse(q.eligible)
        self.assertEqual(q.exclusion_reason, "closes_too_soon")
        self.assertEqual(q.time_score, 0.0)
    def test_comfortable_window_scores_high(self):
        self.assertEqual(score_market(snap(mins=120), {}, cfg()).time_score, 100.0)
    def test_very_far_expiry_penalized(self):
        far = score_market(snap(mins=80 * 1440), {}, cfg())
        near = score_market(snap(mins=2 * 1440), {}, cfg())
        self.assertLess(far.time_score, near.time_score)


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_score_and_version(self):
        s = snap()
        q1, q2 = score_market(s, {}, cfg()), score_market(s, {}, cfg())
        self.assertEqual(q1.total_score, q2.total_score)
        self.assertEqual(q1, q2)
        self.assertEqual(q1.score_version, SCORE_VERSION)
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)


class TestStabilityHistory(unittest.TestCase):
    def obs(self, mids, spreads=None, empties=None):
        n = len(mids)
        return [{"yes_mid": mids[i],
                 "spread_yes": (spreads or [1] * n)[i],
                 "empty": (empties or [False] * n)[i]} for i in range(n)]

    def test_insufficient_history_is_neutral_not_exclusive(self):
        st, insuff = stability_score(self.obs([50, 50]), cfg(MIN_BOOK_OBSERVATIONS=5))
        self.assertEqual((st, insuff), (50.0, True))
        q = score_market(snap(), {}, cfg())
        self.assertTrue(q.insufficient_history)
        self.assertNotEqual(q.exclusion_reason, "insufficient_history")

    def test_volatile_mid_scores_lower_than_stable(self):
        c = cfg(MIN_BOOK_OBSERVATIONS=5)
        stable, _  = stability_score(self.obs([50] * 8), c)
        volatile, _ = stability_score(self.obs([30, 60, 35, 65, 40, 70, 30, 65]), c)
        self.assertLess(volatile, stable)

    def test_empty_book_frequency_and_flaps_penalized(self):
        c = cfg(MIN_BOOK_OBSERVATIONS=5)
        clean, _ = stability_score(self.obs([50] * 8), c)
        flappy, _ = stability_score(
            self.obs([50] * 8, empties=[False, True] * 4), c)
        self.assertLess(flappy, clean)

    def test_history_window_trimmed_and_stale_purged(self):
        c = cfg(HISTORY_WINDOW=3)
        h = {}
        for i in range(6):
            h = update_history(h, [snap("A", yb=50 + i % 2, ya=52 + i % 2)],
                               c, now=NOW + timedelta(minutes=i))
        self.assertEqual(len(h["A"]), 3)                 # fenetre respectee
        for i in range(3):                               # A disparait
            h = update_history(h, [snap("B")], c,
                               now=NOW + timedelta(minutes=10 + i))
        self.assertNotIn("A", h)                         # purge des regles


class TestNoAINoOrders(unittest.TestCase):
    def test_no_ai_dependency(self):
        src = open("market_ranker.py", encoding="utf-8").read().lower()
        for banned in ("anthropic", "openai", "claude", "gpt",
                       "predict_probability", "ml_model"):
            self.assertNotIn(banned, src)
    def test_no_order_path(self):
        src = open("market_ranker.py", encoding="utf-8").read()
        for banned in ("create_order", "place_and_track", "portfolio/orders"):
            self.assertNotIn(banned, src)

    def test_run_ranking_places_zero_orders_and_writes_reports(self):
        tmp = tempfile.mkdtemp()
        try:
            rc = cfg(MIN_BOOK_OBSERVATIONS=2, HISTORY_WINDOW=10)
            rc.HISTORY_FILE  = os.path.join(tmp, "market_snapshots_history.json")
            rc.RANKINGS_FILE = os.path.join(tmp, "market_rankings.json")
            rc.REPORT_FILE   = os.path.join(tmp, "market_ranker_report.json")
            sc = ScanConfig()
            sc.UNIVERSE_FILE = os.path.join(tmp, "u.json")
            sc.REPORT_FILE   = os.path.join(tmp, "r.json")

            class SpyClient:
                order_calls = 0
                def _req(self, m, p, params=None, **k):
                    if (params or {}).get("cursor"):
                        return {"markets": [], "cursor": None}
                    return {"markets": [
                        {"ticker": "GOOD", "status": "open",
                         "close_time": (NOW + timedelta(minutes=90)).isoformat(),
                         "yes_bid": 51, "yes_ask": 52, "no_bid": 48, "no_ask": 49,
                         "volume": 5000, "open_interest": 2000, "liquidity": 8000},
                        {"ticker": "EMPTY", "status": "open",
                         "close_time": (NOW + timedelta(minutes=90)).isoformat(),
                         "yes_bid": 0, "yes_ask": 0, "no_bid": 0, "no_ask": 0,
                         "volume": 999999}], "cursor": None}
                def _log_raw_once(self, *a): pass
                def create_order(self, *a, **k):
                    SpyClient.order_calls += 1
                    raise AssertionError("ordre envoye par le ranker !")

            # deux passes pour construire l'historique
            run_ranking(SpyClient(), scan_cfg=sc, cfg=rc, now=NOW)
            res = run_ranking(SpyClient(), scan_cfg=sc, cfg=rc,
                              now=NOW + timedelta(minutes=1))
            self.assertEqual(SpyClient.order_calls, 0)
            rep = res["report"]
            self.assertEqual(rep["markets_scored"], 2)
            self.assertEqual(rep["eligible"], 1)
            self.assertEqual(rep["excluded_by_reason"], {"no_liquidity": 1})
            self.assertEqual(rep["top_tradable"][0]["ticker"], "GOOD")
            self.assertEqual(rep["score_version"], SCORE_VERSION)
            hist = json.load(open(rc.HISTORY_FILE))
            self.assertEqual(len(hist["GOOD"]), 2)       # 2 observations
            json.load(open(rc.RANKINGS_FILE))
            self.assertIn("score_distribution", json.load(open(rc.REPORT_FILE)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
