"""
Suite d'audit v11.0-pro — unittest (pytest indisponible : pas de reseau).
Deux familles :
  [OK-attendu]  tests de comportement : verifient ce qui doit marcher.
  [SPEC]        tests de conformite au cahier des charges : un ECHEC = un
                defaut documente (numerote D1..D10 dans le rapport).
Aucun test ne touche au reseau. Tout tourne dans des repertoires temporaires.
"""
import os, sys, json, math, shutil, tempfile, unittest, py_compile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_alpha_bot as bot
from kalshi_alpha_bot import (JsonStore, TradeLogger, PositionManager,
                              OrderManager, RiskManager, PositionSizer,
                              MarketValidator, FeeModel, Config, KalshiClient)
import trade_resolver  # v1 present sur disque


class FakeClient:
    """Client hors-ligne : renvoie des marches regles predefinis."""
    def __init__(self, market_by_ticker=None):
        self.markets = market_by_ticker or {}
    def get_market(self, ticker):
        return self.markets.get(ticker, {})


class TmpDataMixin:
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old_dir = bot.CFG.DATA_DIR
        bot.CFG.DATA_DIR = self.tmp
    def tearDown(self):
        bot.CFG.DATA_DIR = self._old_dir
        shutil.rmtree(self.tmp, ignore_errors=True)


# ═════════════════════════ [OK-attendu] comportement ═════════════════════════

class TestJsonStore(TmpDataMixin, unittest.TestCase):
    def test_atomic_write_and_corruption_recovery(self):
        p = os.path.join(self.tmp, "x.json")
        self.assertTrue(JsonStore.save(p, {"v": 1}))
        self.assertTrue(JsonStore.save(p, {"v": 2}))          # cree .bak1
        with open(p, "w") as f:
            f.write("{corrompu")                               # corruption
        data = JsonStore.load(p, None)
        self.assertEqual(data, {"v": 1})                       # recupere bak1

class TestTradeLoggerLegacy(TmpDataMixin, unittest.TestCase):
    def test_legacy_records_archived_not_mixed(self):
        path = os.path.join(self.tmp, Config.TRADES_FILE)
        legacy = [{"ticker": "T1", "order": {"status": "dry_run"}, "edge": 1.0}]
        JsonStore.save(path, legacy)
        tl = TradeLogger()
        self.assertEqual(tl.trades, [])                        # exclus des stats
        arch = JsonStore.load(os.path.join(self.tmp, "kalshi_trades_legacy.json"), [])
        self.assertEqual(len(arch), 1)                         # archives, pas perdus

class TestSettlementMath(TmpDataMixin, unittest.TestCase):
    def _open(self, tl, side="yes", price=60, count=4):
        return tl.open_trade(ticker="TK", market_title="t", side=side,
                             req_price=price, avg_price=price, req_count=count,
                             filled_count=count, spread=1,
                             fees=FeeModel.trading_fee(count, price),
                             edge=0.0, ev=0.0, confidence=3, grade="C",
                             reason="", analysis={}, order_id="o1",
                             order_status="executed")

    def test_losing_settlement_matches_observed_log(self):
        """4x YES @60c, resultat NO -> net -2.47$ (valeur observee en prod le 10/07)."""
        tl = TradeLogger()
        pm = PositionManager(FakeClient({"TK": {"result": "no", "status": "settled"}}), tl)
        pm.open_position(self._open(tl))
        realized = pm.check_settlements()
        self.assertEqual(len(realized), 1)
        self.assertAlmostEqual(realized[0]["net_pnl"], -2.47, places=2)
        self.assertEqual(realized[0]["state"], "settled")

    def test_winning_settlement(self):
        """4x YES @61c, resultat YES -> gross 1.56$, frais 0.07$, net 1.49$ (observe)."""
        tl = TradeLogger()
        pm = PositionManager(FakeClient({"TK": {"result": "yes", "status": "settled"}}), tl)
        pm.open_position(self._open(tl, price=61))
        realized = pm.check_settlements()
        self.assertAlmostEqual(realized[0]["net_pnl"], 1.49, places=2)

    def test_no_double_settlement(self):
        tl = TradeLogger()
        pm = PositionManager(FakeClient({"TK": {"result": "no", "status": "settled"}}), tl)
        pm.open_position(self._open(tl))
        self.assertEqual(len(pm.check_settlements()), 1)
        self.assertEqual(len(pm.check_settlements()), 0)       # position retiree
        self.assertEqual(len(tl.settled_trades()), 1)

class TestRiskManager(TmpDataMixin, unittest.TestCase):
    def test_daily_stop_uses_realized_pnl(self):
        tl = TradeLogger()
        pm = PositionManager(FakeClient(), tl)
        old = bot.CFG.MAX_DAILY_LOSS
        bot.CFG.MAX_DAILY_LOSS = 2.0
        try:
            rm = RiskManager(tl, pm, capital=500.0)
            ok, _ = rm.can_trade(0)
            self.assertTrue(ok)                                # rien de realise
            t = tl.open_trade(ticker="TK", market_title="t", side="yes",
                              req_price=60, avg_price=60, req_count=4,
                              filled_count=4, spread=1, fees=0.07, edge=0.0,
                              ev=0.0, confidence=3, grade="C", reason="",
                              analysis={}, order_id="o", order_status="executed")
            tl.settle_trade(t["trade_id"], "no", False, -2.40, -2.47)
            ok, why = rm.can_trade(0)
            self.assertFalse(ok)                               # stop declenche
            self.assertIn("STOP JOURNALIER", why)
        finally:
            bot.CFG.MAX_DAILY_LOSS = old

class TestPositionSizer(TmpDataMixin, unittest.TestCase):
    def test_hard_cap_1pct_and_weak_signal_half(self):
        # "2%" demande -> plafonne a 1% ; confiance 3 -> /2 -> 0.5% de 500$ = 2.50$
        n = PositionSizer.contracts(500.0, 50, "2%", confidence=3,
                                    drawdown=0.0, open_risk=0.0)
        self.assertEqual(n, 5)                                 # 2.50$ / 0.50$ = 5
    def test_budget_limit(self):
        # budget 5% de 500$ = 25$ ; deja 24$ ouverts -> reste 1$ -> 2 contrats a 50c
        n = PositionSizer.contracts(500.0, 50, "1%", confidence=8,
                                    drawdown=0.0, open_risk=24.0)
        self.assertEqual(n, 2)
    def test_unknown_size_string_gives_zero(self):
        self.assertEqual(PositionSizer.contracts(500.0, 50, "0%", 8, 0.0, 0.0), 0)

class TestMarketValidator(unittest.TestCase):
    def test_derives_missing_no_side(self):
        b = MarketValidator.normalize_book({"yes_bid": 59, "yes_ask": 60})
        self.assertEqual((b["no_bid"], b["no_ask"]), (40, 41))
        self.assertEqual(b["spread"], 1)
    def test_rejects_crossed_book(self):
        self.assertIsNone(MarketValidator.normalize_book({"yes_bid": 61, "yes_ask": 60}))

class TestWeeklyFilterBehavior(TmpDataMixin, unittest.TestCase):
    def test_weekly_lambda_evaluates_to_date_string(self):
        """L'expression `toordinal()-7 and iso` FONCTIONNE par accident
        (int non nul -> renvoie la chaine) mais reste interdite par le cahier
        des charges — voir test SPEC D10."""
        from datetime import date
        today = date(2026, 7, 6)
        val = (today.toordinal() - 7 and
               date.fromordinal(today.toordinal() - 7).isoformat())
        self.assertEqual(val, "2026-06-29")


# ═════════════════ [SPEC] conformite : ECHEC = defaut documente ═════════════

class SpecFillTracking(unittest.TestCase):
    def test_D1_fp_fields_must_be_supported(self):
        """Cahier des charges §1 : fill_count_fp/remaining_count_fp doivent etre
        reconnus (fixture reelle fournie par l'utilisateur)."""
        order = {"status": "resting", "fill_count_fp": "2.00",
                 "remaining_count_fp": "0.00", "initial_count_fp": "2.00",
                 "average_fill_price": "0.1200"}
        status, filled = OrderManager._extract(order, requested=2)
        self.assertEqual(filled, 2, "champs *_fp ignores -> fill reel manque")

    def test_D2_status_executed_alone_must_not_mean_filled(self):
        """§1 : interdiction de considerer rempli sur le seul statut."""
        status, filled = OrderManager._extract({"status": "executed"}, requested=4)
        self.assertEqual(filled, 0, "fill suppose depuis status='executed' seul")

class SpecDemoCredentials(unittest.TestCase):
    def test_D3_demo_without_demo_keys_must_raise(self):
        """REGLE ABSOLUE : cles demo absentes -> RuntimeError, jamais de repli
        silencieux sur les cles PROD."""
        old_id, old_pk = Config.DEMO_KEY_ID, Config.DEMO_PRIV_KEY
        Config.DEMO_KEY_ID, Config.DEMO_PRIV_KEY = "", ""
        try:
            with self.assertRaises(RuntimeError):
                KalshiClient("demo")
        finally:
            Config.DEMO_KEY_ID, Config.DEMO_PRIV_KEY = old_id, old_pk

class SpecEdgeGates(unittest.TestCase):
    def test_D4_min_edge_and_ev_gates_must_exist(self):
        """§5 : MIN_REAL_EDGE / MIN_NET_EV / MIN_MODEL_CONFIDENCE /
        MAX_ACCEPTABLE_SPREAD doivent exister et bloquer le trade."""
        for name in ("MIN_REAL_EDGE", "MIN_NET_EV",
                     "MIN_MODEL_CONFIDENCE", "MAX_ACCEPTABLE_SPREAD"):
            self.assertTrue(hasattr(Config, name), f"Config.{name} absent")

class SpecFees(unittest.TestCase):
    def test_D5_fee_source_api_must_be_supported(self):
        """§2 : extraction des frais API (maker/taker_fees...) + fee_source."""
        self.assertTrue(hasattr(FeeModel, "from_api") or
                        hasattr(FeeModel, "extract_api_fees"),
                        "aucune extraction des frais API, formule locale seule")

class SpecResolverCoherence(unittest.TestCase):
    def test_D6_two_fee_models_must_agree(self):
        """Le repo contient DEUX modeles de frais contradictoires :
        v11: 0.07*C*P*(1-P) arrondi sup ; resolver v1: 2.45% des gains bruts,
        gagnant uniquement. Meme trade -> PnL differents = stats incoherentes."""
        count, price = 4, 60
        # v11 (gagnant): gross = 4*(1-0.60)=1.60 ; net = 1.60 - fee_v11
        fee_v11 = FeeModel.trading_fee(count, price)
        net_v11 = round(count * (100 - price) / 100.0 - fee_v11, 4)
        net_v1 = trade_resolver._compute_pnl(
            {"side": "yes", "price": price, "count": count}, winner="yes")
        # v1: gross_gain=4.00 ; fee=0.098 ; pnl = 4.00-0.098-2.40 = 1.502
        self.assertAlmostEqual(net_v11, net_v1, places=2,
                               msg=f"PnL v11={net_v11}$ vs resolver v1={net_v1}$")

class SpecModules(unittest.TestCase):
    def test_D7_pattern_engine_must_import(self):
        try:
            import pattern_engine  # noqa
        except ImportError as e:
            self.fail(f"pattern_engine inimportable: {e}")

    def test_D8_debug_kalshi_must_compile(self):
        try:
            py_compile.compile("debug_kalshi.py", doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"debug_kalshi.py invalide: {str(e)[:80]}")

    def test_D9_edge_measure_cli_client_signature(self):
        """kalshi_edge_measure ne doit plus appeler KalshiClient(demo=...)
        (signature v10) mais KalshiClient(env) (signature v11)."""
        src = open("kalshi_edge_measure.py", encoding="utf-8").read()
        self.assertNotIn("KalshiClient(demo=", src,
                         "appel v10 KalshiClient(demo=...) toujours present")

class SpecStatsCode(unittest.TestCase):
    def test_D10_banned_toordinal_and_expression_removed(self):
        """§13 : l'expression `today.toordinal() - 7 and ...` doit etre
        supprimee du code (dates explicites exigees)."""
        src = open("kalshi_alpha_bot.py", encoding="utf-8").read()
        self.assertNotIn("toordinal() - 7 and", src,
                         "expression interdite toujours presente (ligne ~820)")




# ═══════════════ tests des correctifs du 2026-07-11 (carnet + fills) ═════════

class TestBookFixes(unittest.TestCase):
    def test_zero_means_empty_side_reconstructed(self):
        """yes_ask=0 (cote vide) : reconstruit depuis no_bid au lieu de
        produire un carnet croise."""
        b = MarketValidator.normalize_book(
            {"yes_bid": 59, "yes_ask": 0, "no_bid": 40, "no_ask": 0})
        self.assertIsNotNone(b)
        self.assertEqual((b["yes_bid"], b["yes_ask"]), (59, 60))

    def test_fully_empty_book_returns_none(self):
        self.assertIsNone(MarketValidator.normalize_book(
            {"yes_bid": 0, "yes_ask": 0, "no_bid": 0, "no_ask": 0}))

class TestFillConfirmation(TmpDataMixin, unittest.TestCase):
    def test_executed_status_confirmed_via_fills_endpoint(self):
        """statut 'executed' sans compteurs -> confirmation par /fills,
        y compris prix des fills au format dollars ("0.6000")."""
        class C:
            def __init__(s): s.calls = 0
            def create_order(s, *a, **k):
                return {"order_id": "o9", "status": "executed"}  # aucun compteur
            def get_order(s, oid):
                return {"order_id": "o9", "status": "executed"}
            def cancel_order(s, oid): return {}
            def get_fills(s, oid):
                return [{"count": 4, "yes_price": "0.6000"}]     # dollars-chaine
        om = OrderManager(C())
        res = om.place_and_track("TK", "yes", 4, 60)
        self.assertEqual(res.filled, 4)          # confirme via fills, pas statut
        self.assertEqual(res.avg_price, 60)      # "0.6000" -> 60c
        self.assertEqual(res.state, "filled")

    def test_executed_status_with_no_fills_records_nothing(self):
        class C:
            def create_order(s, *a, **k):
                return {"order_id": "o0", "status": "executed"}
            def get_order(s, oid): return {"order_id": "o0", "status": "executed"}
            def cancel_order(s, oid): return {}
            def get_fills(s, oid): return []      # /fills ne confirme rien
        om = OrderManager(C())
        res = om.place_and_track("TK", "yes", 4, 60)
        self.assertEqual(res.filled, 0)           # AUCUN trade fantome
        self.assertEqual(res.state, "cancelled")


if __name__ == "__main__":
    unittest.main(verbosity=2)
