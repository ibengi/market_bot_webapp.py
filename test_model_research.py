"""
test_model_research.py — les 20 tests obligatoires de la phase modele.
Tous hors-ligne, deterministes, sans reseau (sources injectees).
Numerotation testNN_ = exigence NN du cahier des charges.
"""
import json
import math
import os
import sys
import time
import unittest
import tempfile
import shutil
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc_context as bc
import btc_probability_model as mdl
import model_calibration as mc
import model_gatekeeper as gk
import backtest_btc15m as bt
from shadow_prediction_store import ShadowPredictionStore
from strategy_router import (StrategyRouter, GateConfig, BtcModelStrategy,
                             evaluate_candidate)

NOW = time.time()

def src(name, price, age=0.0):
    return lambda: {"source": name, "price": price, "ts": NOW - age}

def klines(n=30, start=64000.0, step=2.0, wobble=5.0):
    """Bougies synthetiques croissantes avec variation deterministe."""
    out = []
    px = start
    for i in range(n):
        px = px + step + (wobble if i % 3 == 0 else -wobble / 2)
        out.append({"ts": NOW - (n - i) * 60.0, "open": px, "high": px + 3,
                    "low": px - 3, "close": px, "volume": 10.0})
    return out

def ctx(strike=64000.0, mins=10.0, sources=None, kl=None):
    bc.clear_cache()
    return bc.get_btc_context(
        strike=strike, minutes_remaining=mins,
        spot_sources=sources or (src("a", 64050), src("b", 64060),
                                 src("c", 64055)),
        klines_fn=(lambda: kl if kl is not None else klines()),
        now=NOW, use_cache=False)


class Test01_TwoCoherentSources(unittest.TestCase):
    def test01(self):
        c = ctx(sources=(src("a", 64050), src("b", 64070)))
        self.assertTrue(c.valid)
        self.assertEqual(c.n_valid_sources, 2)
        self.assertAlmostEqual(c.spot, 64060, delta=10.01)


class Test02_OneSourceFailsTwoRemain(unittest.TestCase):
    def test02(self):
        c = ctx(sources=(lambda: None, src("b", 64050), src("c", 64060)))
        self.assertTrue(c.valid)
        self.assertEqual(c.n_valid_sources, 2)


class Test03_LessThanTwoSources_NoProbability(unittest.TestCase):
    def test03(self):
        c = ctx(sources=(src("a", 64050), lambda: None, lambda: None))
        self.assertFalse(c.valid)
        self.assertIn("sources_spot_insuffisantes", c.reason)
        out, why = mdl.predict_or_reason(c)
        self.assertIsNone(out)
        self.assertIn("contexte_invalide", why)


class Test04_StaleData_NoProbability(unittest.TestCase):
    def test04(self):
        c = ctx(sources=(src("a", 64050, age=500), src("b", 64060, age=500),
                         src("c", 64055, age=500)))
        self.assertFalse(c.valid)
        self.assertIsNone(mdl.predict(c))


class Test05_ZeroOrInvalidVol_NoProbability(unittest.TestCase):
    def test05(self):
        flat = [{"ts": NOW - (30 - i) * 60.0, "open": 64000, "high": 64000,
                 "low": 64000, "close": 64000.0, "volume": 1} for i in range(30)]
        c = ctx(kl=flat)                       # vol realisee = 0
        self.assertFalse(c.valid)
        self.assertIn("volatilite", c.reason)
        out, why = mdl.predict_or_reason(c)
        self.assertIsNone(out)


class Test06_ProbabilitiesSumToOne(unittest.TestCase):
    def test06(self):
        out = mdl.predict(ctx())
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["probability_yes"]
                               + out["probability_no"], 1.0, places=9)
        for v in (out["probability_yes"], out["probability_no"],
                  out["confidence"]):
            self.assertFalse(math.isnan(v) or math.isinf(v))


class Test07_ProbabilityRisesWithSpotAboveStrike(unittest.TestCase):
    def test07(self):
        probs = []
        for spot in (63900, 64000, 64100, 64300):
            c = ctx(strike=64000,
                    sources=(src("a", spot), src("b", spot), src("c", spot)),
                    kl=klines(start=spot - 60))
            out = mdl.predict(c, use_momentum=False)
            probs.append(out["probability_yes"])
        self.assertEqual(probs, sorted(probs),
                         f"P(YES) doit croitre avec le spot: {probs}")
        self.assertGreater(probs[-1], 0.9)


class Test08_UncertaintyGrowsWithTimeAndVol(unittest.TestCase):
    def test08(self):
        # meme distance : plus de temps => P plus proche de 0.5
        kl_hi = klines(wobble=40.0)
        p_short = mdl.predict(ctx(strike=64040, mins=2, kl=kl_hi),
                              use_momentum=False)["probability_yes"]
        p_long = mdl.predict(ctx(strike=64040, mins=14, kl=kl_hi),
                             use_momentum=False)["probability_yes"]
        self.assertLess(abs(p_long - 0.5), abs(p_short - 0.5))
        # plus de volatilite => P plus proche de 0.5
        p_calm = mdl.predict(ctx(strike=64040, kl=klines(wobble=15.0)),
                             use_momentum=False)["probability_yes"]
        p_wild = mdl.predict(ctx(strike=64040, kl=klines(wobble=60.0)),
                             use_momentum=False)["probability_yes"]
        self.assertLess(abs(p_wild - 0.5), abs(p_calm - 0.5))
        for p in (p_short, p_long, p_calm, p_wild):
            self.assertLess(p, 0.999)      # hors zone de saturation


# ── 9-12 : comportement moteur (reutilise les harnais d'integration) ────────

def _snap(ticker="KXBTC15M-X", mins=10):
    from market_scanner import build_snapshot
    now = datetime.now(timezone.utc)
    return build_snapshot(
        {"ticker": ticker, "series_ticker": "KXBTC15M", "status": "open",
         "close_time": (now + timedelta(minutes=mins)).isoformat(),
         "floor_strike": 64000, "yes_bid": 50, "yes_ask": 51,
         "no_bid": 49, "no_ask": 50, "volume": 5000,
         "open_interest": 2000, "liquidity": 8000}, now=now)

def _router(strategy):
    r = StrategyRouter(); r.register(strategy); return r

def _btc_strategy(ctx_obj):
    return BtcModelStrategy(
        context_provider=lambda **k: ctx_obj,
        model_predict=mdl.predict_or_reason)

BOOK = {"yes_bid": 50, "yes_ask": 51, "no_bid": 49, "no_ask": 50, "spread": 1}
MKT = {"floor_strike": 64000}


class Test09_10_11_NoOrderPaths(unittest.TestCase):
    def test09_shadow_mode_no_order(self):
        import kalshi_alpha_bot as bot
        import test_integration_pipeline as tip
        h = tip.EngineHarness(); h.setUp()
        try:
            bot.CFG.SHADOW_MODE = True
            cli = tip.FakeBroker([tip.mkt("KXETH-A")])
            eng = h.engine(cli, probs={"KXETH-A": 0.65})
            self.assertEqual((eng.cycle(1), cli.created), (0, []))
        finally:
            h.tearDown()

    def test10_no_model_probability_no_order(self):
        bad_ctx = ctx(sources=(src("a", 64050),))     # 1 source -> invalide
        dec = evaluate_candidate(_snap(), MKT, BOOK,
                                 _router(_btc_strategy(bad_ctx)), GateConfig())
        self.assertFalse(dec.accepted)
        self.assertEqual(dec.rejection_reason, "no_model_probability")

    def test11_net_edge_nonpositive_no_order(self):
        good = ctx(strike=64055)          # strike = spot exact => p = 0.5
        strat = BtcModelStrategy(
            context_provider=lambda **k: good,
            model_predict=lambda c: mdl.predict_or_reason(
                c, use_momentum=False))
        g = GateConfig(MIN_MODEL_CONFIDENCE=0)
        dec = evaluate_candidate(_snap(), MKT, BOOK, _router(strat), g)
        self.assertFalse(dec.accepted)
        self.assertIn(dec.rejection_reason,
                      ("no_positive_edge", "insufficient_net_edge",
                       "negative_net_ev"))


class Test12_LiveBlockedWithoutApproval(unittest.TestCase):
    def test12(self):
        old = {k: os.environ.get(k) for k in
               ("LIVE_TRADING_CONFIRMED", "MODEL_APPROVED_FOR_LIVE")}
        try:
            os.environ.pop("LIVE_TRADING_CONFIRMED", None)
            os.environ.pop("MODEL_APPROVED_FOR_LIVE", None)
            ok, failed = gk.check_live_allowed("__rapport_inexistant__.json")
            self.assertFalse(ok)
            self.assertTrue(any("LIVE_TRADING_CONFIRMED" in f for f in failed))
            self.assertTrue(any("MODEL_APPROVED_FOR_LIVE" in f for f in failed))
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v


# ── 13-16 : backtest / calibration ───────────────────────────────────────────

def _dataset(n=300):
    """Jeu synthetique deterministe : outcome correle a la distance."""
    rows = []
    x = 0
    for i in range(n):
        x = (x * 1103515245 + 12345) % (2 ** 31)      # LCG deterministe
        u = x / (2 ** 31)
        spot = 64000 + (u - 0.5) * 400
        sig = 0.0008 + 0.0004 * ((i % 7) / 7)
        mins = 3 + (i % 12)
        d = math.log(spot / 64000) / (sig * math.sqrt(mins))
        p_true = bt.norm_cdf(d)
        x = (x * 1103515245 + 12345) % (2 ** 31)
        result = "yes" if (x / 2 ** 31) < p_true else "no"
        rows.append({"ts": NOW - (n - i) * 900, "ticker": f"T{i}",
                     "spot": spot, "strike": 64000.0, "sigma_1m": sig,
                     "minutes_remaining": float(mins), "ret_5m": 0.0,
                     "yes_bid": 50, "yes_ask": 52, "no_bid": 48, "no_ask": 50,
                     "result": result})
    return rows


class Test13_NoLeakBetweenSplits(unittest.TestCase):
    def test13(self):
        data = _dataset(200)
        tr, va, te = bt.split_chronological(data)
        self.assertEqual(len(tr) + len(va) + len(te), 200)
        self.assertLess(max(bt._ts(o) for o in tr),
                        min(bt._ts(o) for o in va))
        self.assertLess(max(bt._ts(o) for o in va),
                        min(bt._ts(o) for o in te))
        ids = lambda rows: {o["ticker"] for o in rows}
        self.assertFalse(ids(tr) & ids(te))
        self.assertFalse(ids(va) & ids(te))


class Test14_CalibrationFitOnTrainOnly(unittest.TestCase):
    def test14(self):
        data = _dataset(200)
        tr, va, te = bt.split_chronological(data)
        cal = mc.fit([{"p": bt.predict_row(o),
                       "outcome": 1 if o["result"] == "yes" else 0}
                      for o in tr])
        self.assertEqual(cal["n_train"], len(tr))     # jamais val/test
        rep = bt.run_backtest(data)
        self.assertEqual(rep["calibration_fitted_on"], "train uniquement")


class Test15_16_BacktestReproducible(unittest.TestCase):
    def test15_16(self):
        data = _dataset(300)
        r1 = bt.run_backtest(data)
        r2 = bt.run_backtest(data)
        for k in ("n_predictions_test", "n_theoretical_trades", "net_pnl",
                  "brier_test", "win_rate", "max_drawdown"):
            self.assertEqual(r1[k], r2[k], f"non reproductible: {k}")
        self.assertEqual(r1["trades"], r2["trades"])


# ── 17-19 : shadow store ─────────────────────────────────────────────────────

class TestShadowStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = ShadowPredictionStore(os.path.join(self.tmp, "s.json"))
        self.kw = dict(ticker="KXBTC15M-A", cycle_ts_iso="2026-07-12T10:00:00",
                       market={}, strike=64000, spot=64050,
                       minutes_remaining=10, yes_bid=50, yes_ask=52,
                       no_bid=48, no_ask=50, spread=2, ranker_score=80,
                       features={}, probability_yes=0.62, probability_no=0.38,
                       confidence=0.7, estimated_fee=0.02,
                       estimated_slippage=0.01, gross_edge=0.10,
                       net_edge=0.06, net_ev=0.05,
                       shadow_decision="yes", decision_reason="accepted")
    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test17_idempotent_write(self):
        pid1 = self.store.record(**self.kw)
        pid2 = self.store.record(**self.kw)            # meme id -> pas de dup
        self.assertEqual(pid1, pid2)
        self.assertEqual(len(self.store.rows), 1)
        store2 = ShadowPredictionStore(self.store.path)  # relecture disque
        self.assertEqual(len(store2.rows), 1)

    def test18_settle_once_only(self):
        pid = self.store.record(**self.kw)
        r1 = self.store.settle(pid, "yes")
        self.assertIsNotNone(r1)
        r2 = self.store.settle(pid, "no")              # deuxieme reglement
        self.assertIsNone(r2)
        self.assertEqual(self.store._index[pid]["result"], "yes")

    def test19_fees_and_slippage_in_pnl(self):
        pid = self.store.record(**self.kw)
        r = self.store.settle(pid, "yes")              # gagne a 52c
        self.assertAlmostEqual(r["theoretical_gross_pnl"], 0.48, places=4)
        self.assertAlmostEqual(r["theoretical_net_pnl"],
                               0.48 - 0.02 - 0.01, places=4)
        self.assertAlmostEqual(r["prediction_error"],
                               (0.62 - 1) ** 2, places=6)


class Test20_GatekeeperBlocksOnAnyFailure(unittest.TestCase):
    def test20(self):
        good = {
            "n_predictions_test": 500, "n_theoretical_trades": 150,
            "net_pnl": 12.5, "brier_test": 0.20, "brier_market_baseline": 0.24,
            "calibration_test": {"ece": 0.05}, "max_drawdown": 3.0,
            "split": {"chronological": True},
            "generated": datetime.now(timezone.utc).isoformat(),
            "model_hash": gk.model_hash(),
        }
        ok, failed = gk.evaluate(good)
        self.assertTrue(ok, f"rapport sain refuse: {failed}")
        # CHAQUE critere, casse un par un, doit bloquer
        breakers = [
            ("n_predictions_test", 10), ("n_theoretical_trades", 5),
            ("net_pnl", -1.0), ("brier_test", 0.30),
            ("calibration_test", {"ece": 0.5}), ("max_drawdown", 99.0),
            ("split", {"chronological": False}),
            ("generated", "2020-01-01T00:00:00+00:00"),
            ("model_hash", "deadbeef"),
        ]
        for key, bad in breakers:
            r = dict(good); r[key] = bad
            ok, failed = gk.evaluate(r)
            self.assertFalse(ok, f"critere non bloquant: {key}")
            self.assertGreaterEqual(len(failed), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
