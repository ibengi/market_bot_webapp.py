"""
test_repo_integrity.py — reproductibilite du depot dans un repertoire vierge.
Couvre les 8 exigences : imports propres, aucun module local manquant,
decouverte complete, rapport exact, generation automatique du rapport,
demarrage shadow sans btc_context, aucun ordre sans modele, scanner
reellement utilise en mode normal.
"""
import ast
import importlib
import json
import os
import subprocess
import sys
import unittest
import tempfile
import shutil
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CORE_MODULES = ["market_taxonomy", "market_scanner", "market_ranker", "strategy_router",
                "opportunity_pipeline", "pattern_engine",
                "kalshi_edge_measure", "btc_context",
                "btc_probability_model", "model_calibration",
                "shadow_prediction_store", "backtest_btc15m",
                "model_gatekeeper", "kalshi_alpha_bot"]
TEST_MODULES = ["test_market_scanner", "test_market_ranker",
                "test_audit_v11", "test_integration_pipeline",
                "test_model_research", "test_market_taxonomy",
                "test_repo_integrity"]
DECLARED_THIRD_PARTY = {"requests", "cryptography", "dotenv",
                        "pytest", "pytest_cov",
                        "sklearn"}   # optionnel: import protege (calibration)
OPTIONAL_LOCAL = {"btc_context", "trade_resolver", "debug_kalshi"}


class Test1_2_AllModulesImportCleanly(unittest.TestCase):
    def test_1_every_core_module_imports(self):
        for mod in CORE_MODULES:
            with self.subTest(module=mod):
                importlib.import_module(mod)

    def test_2_no_missing_local_import(self):
        """Analyse AST de tous les .py : chaque import local doit etre
        present dans le depot, stdlib, ou dependance declaree."""
        stdlib = set(sys.stdlib_module_names)
        local = {f[:-3] for f in os.listdir(HERE) if f.endswith(".py")}
        missing = []
        for f in sorted(local):
            tree = ast.parse(open(os.path.join(HERE, f + ".py"),
                                  encoding="utf-8").read())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module \
                        and node.level == 0:
                    names = [node.module.split(".")[0]]
                for n in names:
                    if n in stdlib or n in local \
                            or n in DECLARED_THIRD_PARTY \
                            or n in OPTIONAL_LOCAL:
                        continue
                    missing.append(f"{f}.py importe '{n}' (introuvable)")
        self.assertEqual(missing, [])


class Test3_DiscoveryFindsAllSuites(unittest.TestCase):
    def test_3_discover_loads_every_test_module(self):
        loader = unittest.TestLoader()
        suite = loader.discover(HERE, pattern="test_*.py")
        found = set()
        def walk(s):
            for item in s:
                if isinstance(item, unittest.TestSuite):
                    walk(item)
                else:
                    found.add(type(item).__module__)
        walk(suite)
        for mod in TEST_MODULES:
            self.assertIn(mod, found, f"suite {mod} non decouverte")
        self.assertEqual(loader.errors, [],
                         f"erreurs de chargement: {loader.errors}")


class Test4_5_ReportMatchesReality(unittest.TestCase):
    def test_4_5_run_tests_writes_accurate_report(self):
        """Execute run_tests.py en sous-processus (garde anti-recursion via
        REPO_INTEGRITY_GUARD) et verifie que test_report.json est ecrit
        automatiquement avec EXACTEMENT les comptes reels."""
        if os.environ.get("REPO_INTEGRITY_GUARD") == "1":
            self.skipTest("execution imbriquee (garde anti-recursion)")
        env = {**os.environ, "REPO_INTEGRITY_GUARD": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            for f in os.listdir(HERE):
                if f.endswith((".py", ".txt")):
                    shutil.copy(os.path.join(HERE, f), tmp)
            r = subprocess.run([sys.executable, "run_tests.py"], cwd=tmp,
                               env=env, capture_output=True, text=True,
                               timeout=600)
            self.assertTrue(os.path.exists(os.path.join(tmp, "test_report.json")),
                            "test_report.json non genere automatiquement")
            rep = json.load(open(os.path.join(tmp, "test_report.json")))
            import re
            m = re.search(r"Ran (\d+) tests", r.stderr)
            self.assertIsNotNone(m, f"sortie unittest illisible:\n{r.stderr[-800:]}")
            self.assertEqual(rep["ran"], int(m.group(1)),
                             "le rapport n'annonce pas le nombre reel de tests")
            self.assertEqual(rep["ran"], rep["passed"] + rep["failed"]
                             + rep["errors"] + rep["skipped"])
            self.assertTrue(rep["success"], f"suite en echec dans le depot "
                            f"autonome: {rep['failed_detail'] + rep['error_detail']}")


def _mkt(ticker, mins=60, empty=False, **extra):
    now = datetime.now(timezone.utc)
    serie = "KXBTC15M" if ticker.upper().startswith("KXBTC15M") else "KXETH"
    m = {"ticker": ticker, "status": "open", "series_ticker": serie,
         "close_time": (now + timedelta(minutes=mins)).isoformat(),
         "volume": 5000, "open_interest": 2000, "liquidity": 8000}
    if empty:
        m.update({"yes_bid": 0, "yes_ask": 0, "no_bid": 0, "no_ask": 0})
    else:
        m.update({"yes_bid": 50, "yes_ask": 51, "no_bid": 49, "no_ask": 50})
    m.update(extra)
    return m


class _Broker:
    env = "demo"
    def __init__(self, markets):
        self.markets = {m["ticker"]: m for m in markets}
        self.created = []
        self.markets_endpoint_calls = 0
    def _req(self, method, path, params=None, **kw):
        if path == "/markets":
            self.markets_endpoint_calls += 1
            if (params or {}).get("cursor"):
                return {"markets": [], "cursor": None}
            return {"markets": list(self.markets.values()), "cursor": None}
        raise AssertionError(path)
    def _log_raw_once(self, *a): pass
    def get_market(self, tk): return self.markets.get(tk, {})
    def get_balance(self): return 1000.0
    def get_positions(self): return []
    def create_order(self, *a, **k):
        self.created.append(a); return {"order_id": "x", "status": "executed",
                                        "fill_count": a[2]}
    def get_order(self, oid): return {"order_id": oid, "status": "executed"}
    def cancel_order(self, oid): return {}
    def get_fills(self, oid): return []


class Test6_7_8_EngineWithoutModel(unittest.TestCase):
    def setUp(self):
        import kalshi_alpha_bot as bot
        self.bot = bot
        self.tmp = tempfile.mkdtemp()
        self._old_dir = bot.CFG.DATA_DIR
        self._old_shadow = bot.CFG.SHADOW_MODE
        bot.CFG.DATA_DIR = self.tmp
        os.makedirs(os.path.join(self.tmp, "reports"), exist_ok=True)
    def tearDown(self):
        self.bot.CFG.DATA_DIR = self._old_dir
        self.bot.CFG.SHADOW_MODE = self._old_shadow
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _engine(self, cli):
        from market_scanner import ScanConfig
        from market_ranker import RankConfig
        eng = self.bot.ExecutionEngine(cli, 500.0)
        eng.pipeline.scan_cfg = ScanConfig()
        eng.pipeline.scan_cfg.UNIVERSE_FILE = os.path.join(self.tmp, "u.json")
        eng.pipeline.scan_cfg.REPORT_FILE = os.path.join(self.tmp, "r.json")
        eng.pipeline.rank_cfg = RankConfig()
        eng.pipeline.rank_cfg.MIN_BOOK_OBSERVATIONS = 999
        eng.pipeline.rank_cfg.HISTORY_FILE = os.path.join(self.tmp, "h.json")
        eng.pipeline.rank_cfg.RANKINGS_FILE = os.path.join(self.tmp, "rk.json")
        eng.pipeline.rank_cfg.REPORT_FILE = os.path.join(self.tmp, "rr.json")
        return eng

    def test_6_shadow_starts_without_btc_context_and_survives_cycle(self):
        """Le moteur demarre en shadow et boucle sans crash ni ordre, meme
        quand aucune strategie ne s'applique (le depot embarque desormais
        btc_context v2 ; la strategie BTC ne s'applique qu'a KXBTC15M)."""
        self.bot.CFG.SHADOW_MODE = True
        cli = _Broker([_mkt("KXETH-A")])
        eng = self._engine(cli)
        placed = eng.cycle(1)                        # ne doit pas lever
        self.assertEqual((placed, cli.created), (0, []))

    def test_7_no_model_means_zero_create_order(self):
        """Sans modele applicable, create_order n'est JAMAIS appele.
        (a) marche hors perimetre BTC15M -> no_compatible_strategy ;
        (b) BTC15M mais contexte invalide -> no_model_probability."""
        cli = _Broker([_mkt("KXETH-A")])
        eng = self._engine(cli)
        placed = eng.cycle(1)
        self.assertEqual((placed, cli.created), (0, []))
        rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertGreaterEqual(
            rep["rejections"].get("no_compatible_strategy", 0), 1)
        # (b) contexte invalide injecte (hors-ligne, instantane)
        import btc_context as bc
        from strategy_router import StrategyRouter, BtcModelStrategy
        import btc_probability_model as mdl
        bad_ctx = bc.get_btc_context(strike=64000, minutes_remaining=10,
                                     spot_sources=(lambda: None,),
                                     klines_fn=lambda: [], use_cache=False)
        cli2 = _Broker([_mkt("KXBTC15M-Z", floor_strike=64000)])
        eng2 = self._engine(cli2)
        eng2.router = StrategyRouter()
        eng2.router.register(BtcModelStrategy(
            context_provider=lambda **k: bad_ctx,
            model_predict=mdl.predict_or_reason))
        eng2.pipeline.router = eng2.router
        placed2 = eng2.cycle(1)
        self.assertEqual((placed2, cli2.created), (0, []))
        rep2 = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
        self.assertGreaterEqual(
            rep2["rejections"].get("no_model_probability", 0), 1)

    def test_8_normal_mode_really_uses_scanner(self):
        """Preuve directe : le cycle normal interroge /markets (pagination
        du scanner), pas une selection mono-serie."""
        cli = _Broker([_mkt("KXETH-A"), _mkt("KXBTC15M-B", empty=True)])
        eng = self._engine(cli)
        eng.cycle(1)
        self.assertGreaterEqual(cli.markets_endpoint_calls, 1)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "u.json")),
                        "artefact scanner absent: le pipeline n'a pas tourne")


if __name__ == "__main__":
    unittest.main(verbosity=2)
