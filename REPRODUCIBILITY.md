# REPRODUCTIBILITE — sortie BRUTE et complete

Protocole : repertoire VIERGE, decompression du seul kalshi_alpha_engine.zip,
puis execution exacte des commandes ci-dessous (Python 3.12, deps:
pip install -r requirements.txt => requests, cryptography).

```
$ python -m py_compile *.py
(succes, code retour 0, aucune sortie)

$ python -m unittest discover -v
2026-07-12 03:34:59  WARNING [BOT] btc_context absent -- strategie crypto DESACTIVEE explicitement: aucun modele de probabilite => aucun trade (rejets 'no_model_probability'). Le pipeline, --scan-only, --rank-only et --shadow restent operationnels.
test_D3_demo_without_demo_keys_must_raise (test_audit_v11.SpecDemoCredentials.test_D3_demo_without_demo_keys_must_raise)
REGLE ABSOLUE : cles demo absentes -> RuntimeError, jamais de repli ... ok
test_D4_min_edge_and_ev_gates_must_exist (test_audit_v11.SpecEdgeGates.test_D4_min_edge_and_ev_gates_must_exist)
§5 : portes edge/EV obligatoires. Noms acceptes : ancienne spec ... ok
test_D5_fee_source_api_must_be_supported (test_audit_v11.SpecFees.test_D5_fee_source_api_must_be_supported)
§2 : extraction des frais API (maker/taker_fees...) + fee_source. ... ok
test_D1_fp_fields_must_be_supported (test_audit_v11.SpecFillTracking.test_D1_fp_fields_must_be_supported)
Cahier des charges §1 : fill_count_fp/remaining_count_fp doivent etre ... ok
test_D2_status_executed_alone_must_not_mean_filled (test_audit_v11.SpecFillTracking.test_D2_status_executed_alone_must_not_mean_filled)
§1 : interdiction de considerer rempli sur le seul statut. ... ok
test_D7_pattern_engine_must_import (test_audit_v11.SpecModules.test_D7_pattern_engine_must_import) ... ok
test_D8_debug_kalshi_must_compile (test_audit_v11.SpecModules.test_D8_debug_kalshi_must_compile) ... skipped 'debug_kalshi.py (herite) absent du depot -- recommande supprime; test temoin non applicable'
test_D9_edge_measure_cli_client_signature (test_audit_v11.SpecModules.test_D9_edge_measure_cli_client_signature)
kalshi_edge_measure ne doit plus appeler KalshiClient(demo=...) ... /home/claude/freshtest/test_audit_v11.py:239: ResourceWarning: unclosed file <_io.TextIOWrapper name='kalshi_edge_measure.py' mode='r' encoding='utf-8'>
  src = open("kalshi_edge_measure.py", encoding="utf-8").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_D6_two_fee_models_must_agree (test_audit_v11.SpecResolverCoherence.test_D6_two_fee_models_must_agree) ... skipped 'trade_resolver.py (herite) absent du depot -- recommande supprime; test temoin non applicable'
test_D10_banned_toordinal_and_expression_removed (test_audit_v11.SpecStatsCode.test_D10_banned_toordinal_and_expression_removed)
§13 : l'expression `today.toordinal() - 7 and ...` doit etre ... /home/claude/freshtest/test_audit_v11.py:247: ResourceWarning: unclosed file <_io.TextIOWrapper name='kalshi_alpha_bot.py' mode='r' encoding='utf-8'>
  src = open("kalshi_alpha_bot.py", encoding="utf-8").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_fully_empty_book_returns_none (test_audit_v11.TestBookFixes.test_fully_empty_book_returns_none) ... ok
test_zero_means_empty_side_reconstructed (test_audit_v11.TestBookFixes.test_zero_means_empty_side_reconstructed)
yes_ask=0 (cote vide) : reconstruit depuis no_bid au lieu de ... ok
test_executed_status_confirmed_via_fills_endpoint (test_audit_v11.TestFillConfirmation.test_executed_status_confirmed_via_fills_endpoint)
statut 'executed' sans compteurs -> confirmation par /fills, ... 2026-07-12 03:34:59  INFO    [API] Ordre o9: statut 'executed' sans compteur -- 4 contrat(s) confirme(s) via /fills.
ok
test_executed_status_with_no_fills_records_nothing (test_audit_v11.TestFillConfirmation.test_executed_status_with_no_fills_records_nothing) ... ok
test_atomic_write_and_corruption_recovery (test_audit_v11.TestJsonStore.test_atomic_write_and_corruption_recovery) ... /home/claude/freshtest/kalshi_alpha_bot.py:177: ResourceWarning: unclosed file <_io.BufferedReader name='/tmp/tmp_ss_hcne/x.json'>
  raw = open(cand, "rb").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/kalshi_alpha_bot.py:177: ResourceWarning: unclosed file <_io.BufferedReader name='/tmp/tmp_ss_hcne/x.json.bak1'>
  raw = open(cand, "rb").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
2026-07-12 03:34:59  WARNING [BOT] JsonStore: /tmp/tmp_ss_hcne/x.json corrompu/absent -- recupere depuis /tmp/tmp_ss_hcne/x.json.bak1.
ok
test_derives_missing_no_side (test_audit_v11.TestMarketValidator.test_derives_missing_no_side) ... ok
test_rejects_crossed_book (test_audit_v11.TestMarketValidator.test_rejects_crossed_book) ... ok
test_budget_limit (test_audit_v11.TestPositionSizer.test_budget_limit) ... ok
test_hard_cap_1pct_and_weak_signal_half (test_audit_v11.TestPositionSizer.test_hard_cap_1pct_and_weak_signal_half) ... ok
test_unknown_size_string_gives_zero (test_audit_v11.TestPositionSizer.test_unknown_size_string_gives_zero) ... ok
test_daily_stop_uses_realized_pnl (test_audit_v11.TestRiskManager.test_daily_stop_uses_realized_pnl) ... 2026-07-12 03:34:59  INFO    [TRADE] OUVERT TK YES 4/4 @ 60c (frais 0.07$) ordre=o
2026-07-12 03:34:59  INFO    [TRADE] REGLE  TK -> NO | PERDU | net -2.47$
ok
test_losing_settlement_matches_observed_log (test_audit_v11.TestSettlementMath.test_losing_settlement_matches_observed_log)
4x YES @60c, resultat NO -> net -2.47$ (valeur observee en prod le 10/07). ... 2026-07-12 03:34:59  INFO    [TRADE] OUVERT TK YES 4/4 @ 60c (frais 0.07$) ordre=o1
2026-07-12 03:34:59  INFO    [POSITION] TK: YES x4 @ 60c
2026-07-12 03:34:59  INFO    [TRADE] REGLE  TK -> NO | PERDU | net -2.47$
ok
test_no_double_settlement (test_audit_v11.TestSettlementMath.test_no_double_settlement) ... 2026-07-12 03:34:59  INFO    [TRADE] OUVERT TK YES 4/4 @ 60c (frais 0.07$) ordre=o1
2026-07-12 03:34:59  INFO    [POSITION] TK: YES x4 @ 60c
2026-07-12 03:34:59  INFO    [TRADE] REGLE  TK -> NO | PERDU | net -2.47$
ok
test_winning_settlement (test_audit_v11.TestSettlementMath.test_winning_settlement)
4x YES @61c, resultat YES -> gross 1.56$, frais 0.07$, net 1.49$ (observe). ... 2026-07-12 03:34:59  INFO    [TRADE] OUVERT TK YES 4/4 @ 61c (frais 0.07$) ordre=o1
2026-07-12 03:34:59  INFO    [POSITION] TK: YES x4 @ 61c
2026-07-12 03:34:59  INFO    [TRADE] REGLE  TK -> YES | GAGNE | net +1.49$
ok
test_legacy_records_archived_not_mixed (test_audit_v11.TestTradeLoggerLegacy.test_legacy_records_archived_not_mixed) ... /home/claude/freshtest/kalshi_alpha_bot.py:177: ResourceWarning: unclosed file <_io.BufferedReader name='/tmp/tmpn0mfj0rn/kalshi_trades.json'>
  raw = open(cand, "rb").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/kalshi_alpha_bot.py:180: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmpn0mfj0rn/kalshi_trades.json.sha256' mode='r' encoding='utf-8'>
  want = open(path + ".sha256").read().strip()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
2026-07-12 03:34:59  WARNING [TRADE] 1 enregistrement(s) heritee(s) (dry-run/ancien schema) archives dans kalshi_trades_legacy.json -- exclus des statistiques.
/home/claude/freshtest/kalshi_alpha_bot.py:177: ResourceWarning: unclosed file <_io.BufferedReader name='/tmp/tmpn0mfj0rn/kalshi_trades_legacy.json'>
  raw = open(cand, "rb").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/kalshi_alpha_bot.py:180: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmpn0mfj0rn/kalshi_trades_legacy.json.sha256' mode='r' encoding='utf-8'>
  want = open(path + ".sha256").read().strip()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_weekly_lambda_evaluates_to_date_string (test_audit_v11.TestWeeklyFilterBehavior.test_weekly_lambda_evaluates_to_date_string)
L'expression `toordinal()-7 and iso` FONCTIONNE par accident ... ok
test_cycle_calls_scanner_and_ranker (test_integration_pipeline.TestA_NormalModeUsesPipeline.test_cycle_calls_scanner_and_ranker) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmp582zhnup/u.json | rapport -> /tmp/tmp582zhnup/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmp582zhnup/rk.json | rapport -> /tmp/tmp582zhnup/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] rejet KXETH-A: no_model_probability
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=0 ordres=0 fills=0 rejets={'no_model_probability': 1}
ok
test_empty_btc_other_candidate_selected (test_integration_pipeline.TestB_NotStuckOnEmptyBTC.test_empty_btc_other_candidate_selected) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 2 marches (1 valides) -> /tmp/tmpuw1owjkw/u.json | rapport -> /tmp/tmpuw1owjkw/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 2 scores (1 eligibles) -> /tmp/tmpuw1owjkw/rk.json | rapport -> /tmp/tmpuw1owjkw/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] CANDIDAT VALIDE KXETH-GOOD YES @ 51c | edge_net=+0.080 ev_net=+0.080 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] [SIGNAL VALIDE] KXETH-GOOD YES x9 @ 51c | modele=62.0% marche=51.0% edge_net=+0.080 ev_net=+0.080 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] OUVERT KXETH-GOOD YES 9/9 @ 51c (frais 0.16$) ordre=o1
2026-07-12 03:34:59  INFO    [POSITION] KXETH-GOOD: YES x9 @ 51c
2026-07-12 03:34:59  INFO    [RISK] risque_ouvert=4.59$ pnl_jour=0$ frais_cumules=0.16$ (source=estimated)
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=2 eligibles=1 acceptes=1 ordres=1 fills=1 rejets={'no_liquidity': 1}
ok
test_zero_create_order (test_integration_pipeline.TestC_AllBooksEmpty_NoOrder.test_zero_create_order) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 4 marches (0 valides) -> /tmp/tmpmwuiip01/u.json | rapport -> /tmp/tmpmwuiip01/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 4 scores (0 eligibles) -> /tmp/tmpmwuiip01/rk.json | rapport -> /tmp/tmpmwuiip01/rr.json
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=4 eligibles=0 acceptes=0 ordres=0 fills=0 rejets={'no_liquidity': 4}
ok
test_liquid_market_without_strategy (test_integration_pipeline.TestD_NoStrategy_Rejected.test_liquid_market_without_strategy) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmpcc_4tk74/u.json | rapport -> /tmp/tmpcc_4tk74/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmpcc_4tk74/rk.json | rapport -> /tmp/tmpcc_4tk74/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] rejet FED-X: no_compatible_strategy
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=0 ordres=0 fills=0 rejets={'no_compatible_strategy': 1}
/home/claude/freshtest/test_integration_pipeline.py:172: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmpcc_4tk74/cycle_report.json' mode='r' encoding='utf-8'>
  rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_model_equals_market (test_integration_pipeline.TestE_EdgeZero_NoOrder.test_model_equals_market) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmptpf_tw5n/u.json | rapport -> /tmp/tmptpf_tw5n/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmptpf_tw5n/rk.json | rapport -> /tmp/tmptpf_tw5n/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] rejet KXETH-A: no_positive_edge
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=0 ordres=0 fills=0 rejets={'no_positive_edge': 1}
/home/claude/freshtest/test_integration_pipeline.py:184: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmptpf_tw5n/cycle_report.json' mode='r' encoding='utf-8'>
  rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_gross_edge_eaten_by_fees_and_slippage (test_integration_pipeline.TestF_NegativeNetEV_NoOrder.test_gross_edge_eaten_by_fees_and_slippage) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmp9ns_p6mv/u.json | rapport -> /tmp/tmp9ns_p6mv/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmp9ns_p6mv/rk.json | rapport -> /tmp/tmp9ns_p6mv/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] rejet KXETH-A: negative_net_ev
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=0 ordres=0 fills=0 rejets={'negative_net_ev': 1}
/home/claude/freshtest/test_integration_pipeline.py:200: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmp9ns_p6mv/cycle_report.json' mode='r' encoding='utf-8'>
  rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_two_candidates (test_integration_pipeline.TestG_FirstRejected_SecondTraded.test_two_candidates) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 2 marches (2 valides) -> /tmp/tmphnbklnt3/u.json | rapport -> /tmp/tmphnbklnt3/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 2 scores (2 eligibles) -> /tmp/tmphnbklnt3/rk.json | rapport -> /tmp/tmphnbklnt3/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] rejet BIG-NOEDGE: no_positive_edge
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] CANDIDAT VALIDE SMALL-EDGE YES @ 51c | edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] [SIGNAL VALIDE] SMALL-EDGE YES x9 @ 51c | modele=65.0% marche=51.0% edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] OUVERT SMALL-EDGE YES 9/9 @ 51c (frais 0.16$) ordre=o1
2026-07-12 03:34:59  INFO    [POSITION] SMALL-EDGE: YES x9 @ 51c
2026-07-12 03:34:59  INFO    [RISK] risque_ouvert=4.59$ pnl_jour=0$ frais_cumules=0.16$ (source=estimated)
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=2 eligibles=2 acceptes=1 ordres=1 fills=1 rejets={'no_positive_edge': 1}
/home/claude/freshtest/test_integration_pipeline.py:217: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmphnbklnt3/cycle_report.json' mode='r' encoding='utf-8'>
  rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_only_filled_quantity_becomes_position (test_integration_pipeline.TestH_PartialFill.test_only_filled_quantity_becomes_position) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmpkmlix6m8/u.json | rapport -> /tmp/tmpkmlix6m8/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmpkmlix6m8/rk.json | rapport -> /tmp/tmpkmlix6m8/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] CANDIDAT VALIDE KXETH-A YES @ 51c | edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] [SIGNAL VALIDE] KXETH-A YES x9 @ 51c | modele=65.0% marche=51.0% edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] OUVERT KXETH-A YES 4/9 @ 51c (frais 0.07$) ordre=op
2026-07-12 03:34:59  INFO    [POSITION] KXETH-A: YES x4 @ 51c
2026-07-12 03:34:59  INFO    [RISK] risque_ouvert=2.04$ pnl_jour=0$ frais_cumules=0.07$ (source=estimated)
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=1 ordres=1 fills=1 rejets={}
ok
test_fills_endpoint_is_source_of_truth (test_integration_pipeline.TestI_ExecutedWithoutCounts_ChecksFills.test_fills_endpoint_is_source_of_truth) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmpjp8dk6y0/u.json | rapport -> /tmp/tmpjp8dk6y0/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmpjp8dk6y0/rk.json | rapport -> /tmp/tmpjp8dk6y0/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] CANDIDAT VALIDE KXETH-A YES @ 51c | edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] [SIGNAL VALIDE] KXETH-A YES x9 @ 51c | modele=65.0% marche=51.0% edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [API] Ordre o1: statut 'executed' sans compteur -- 9 contrat(s) confirme(s) via /fills.
2026-07-12 03:34:59  INFO    [TRADE] OUVERT KXETH-A YES 9/9 @ 51c (frais 0.16$) ordre=o1
2026-07-12 03:34:59  INFO    [POSITION] KXETH-A: YES x9 @ 51c
2026-07-12 03:34:59  INFO    [RISK] risque_ouvert=4.59$ pnl_jour=0$ frais_cumules=0.16$ (source=estimated)
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=1 ordres=1 fills=1 rejets={}
ok
test_broker_reconciliation_idempotent (test_integration_pipeline.TestJ_RestartRebuildNoDuplicates.test_broker_reconciliation_idempotent) ... 2026-07-12 03:34:59  WARNING [POSITION] Reconciliation broker: reconstruites=['KXETH-A'] fantomes=[]
/home/claude/freshtest/kalshi_alpha_bot.py:177: ResourceWarning: unclosed file <_io.BufferedReader name='/tmp/tmpw4itkvhr/positions_state.json'>
  raw = open(cand, "rb").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/kalshi_alpha_bot.py:180: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmpw4itkvhr/positions_state.json.sha256' mode='r' encoding='utf-8'>
  want = open(path + ".sha256").read().strip()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/kalshi_alpha_bot.py:177: ResourceWarning: unclosed file <_io.BufferedReader name='/tmp/tmpw4itkvhr/seen_fill_ids.json'>
  raw = open(cand, "rb").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/kalshi_alpha_bot.py:180: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmpw4itkvhr/seen_fill_ids.json.sha256' mode='r' encoding='utf-8'>
  want = open(path + ".sha256").read().strip()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_sizing_uses_min_of_balance_and_config (test_integration_pipeline.TestK_RealBalanceCapsSizing.test_sizing_uses_min_of_balance_and_config) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=200.00$ capital_effectif=200.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmpn_y1wu1f/u.json | rapport -> /tmp/tmpn_y1wu1f/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmpn_y1wu1f/rk.json | rapport -> /tmp/tmpn_y1wu1f/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] CANDIDAT VALIDE KXETH-A YES @ 51c | edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] [SIGNAL VALIDE] KXETH-A YES x3 @ 51c | modele=65.0% marche=51.0% edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] OUVERT KXETH-A YES 3/3 @ 51c (frais 0.06$) ordre=o1
2026-07-12 03:34:59  INFO    [POSITION] KXETH-A: YES x3 @ 51c
2026-07-12 03:34:59  INFO    [RISK] risque_ouvert=1.53$ pnl_jour=0$ frais_cumules=0.06$ (source=estimated)
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=1 ordres=1 fills=1 rejets={}
ok
test_fresh_reread_blocks_order (test_integration_pipeline.TestL_BookVanishesBeforeOrder.test_fresh_reread_blocks_order) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmp6yv7oblx/u.json | rapport -> /tmp/tmp6yv7oblx/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmp6yv7oblx/rk.json | rapport -> /tmp/tmp6yv7oblx/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] rejet KXETH-A: stale_book
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=0 ordres=0 fills=0 rejets={'stale_book': 1}
/home/claude/freshtest/test_integration_pipeline.py:290: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmp6yv7oblx/cycle_report.json' mode='r' encoding='utf-8'>
  rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_api_fees_override_local_model (test_integration_pipeline.TestM_APIFeesPreferred.test_api_fees_override_local_model) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmpf6i9icqd/u.json | rapport -> /tmp/tmpf6i9icqd/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmpf6i9icqd/rk.json | rapport -> /tmp/tmpf6i9icqd/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] CANDIDAT VALIDE KXETH-A YES @ 51c | edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] [SIGNAL VALIDE] KXETH-A YES x9 @ 51c | modele=65.0% marche=51.0% edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] OUVERT KXETH-A YES 9/9 @ 51c (frais 0.05$) ordre=o1
2026-07-12 03:34:59  INFO    [POSITION] KXETH-A: YES x9 @ 51c
2026-07-12 03:34:59  INFO    [RISK] risque_ouvert=4.59$ pnl_jour=0$ frais_cumules=0.05$ (source=api)
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=1 ordres=1 fills=1 rejets={}
ok
test_pipeline_scans_every_page (test_integration_pipeline.TestN_PaginationAllPages.test_pipeline_scans_every_page) ... ok
test_all_reasons_counted (test_integration_pipeline.TestO_CycleReportCountsRejections.test_all_reasons_counted) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #7 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 4 marches (3 valides) -> /tmp/tmp34zmq9yx/u.json | rapport -> /tmp/tmp34zmq9yx/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 4 scores (3 eligibles) -> /tmp/tmp34zmq9yx/rk.json | rapport -> /tmp/tmp34zmq9yx/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] rejet NOEDGE: no_positive_edge
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] CANDIDAT VALIDE GOOD YES @ 51c | edge_net=+0.120 ev_net=+0.120 strat=stub
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] rejet NOSTRAT: no_compatible_strategy
2026-07-12 03:34:59  INFO    [TRADE] [SIGNAL VALIDE] GOOD YES x9 @ 51c | modele=66.0% marche=51.0% edge_net=+0.120 ev_net=+0.120 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] OUVERT GOOD YES 9/9 @ 51c (frais 0.16$) ordre=o1
2026-07-12 03:34:59  INFO    [POSITION] GOOD: YES x9 @ 51c
2026-07-12 03:34:59  INFO    [RISK] risque_ouvert=4.59$ pnl_jour=0$ frais_cumules=0.16$ (source=estimated)
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=4 eligibles=3 acceptes=1 ordres=1 fills=1 rejets={'no_positive_edge': 1, 'no_compatible_strategy': 1, 'no_liquidity': 1}
/home/claude/freshtest/test_integration_pipeline.py:336: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmp34zmq9yx/cycle_report.json' mode='r' encoding='utf-8'>
  rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_kill_switch_blocks_everything (test_integration_pipeline.TestShadowAndKillSwitch.test_kill_switch_blocks_everything) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  WARNING [RISK] KILL_SWITCH actif -- aucun ordre ce cycle.
ok
test_shadow_full_decision_zero_orders (test_integration_pipeline.TestShadowAndKillSwitch.test_shadow_full_decision_zero_orders) ... 2026-07-12 03:34:59  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:34:59  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmp8009mq_e/u.json | rapport -> /tmp/tmp8009mq_e/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmp8009mq_e/rk.json | rapport -> /tmp/tmp8009mq_e/rr.json
2026-07-12 03:34:59  INFO    [PIPELINE] [PIPELINE] CANDIDAT VALIDE KXETH-A YES @ 51c | edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] [SIGNAL VALIDE] KXETH-A YES x9 @ 51c | modele=65.0% marche=51.0% edge_net=+0.110 ev_net=+0.110 strat=stub
2026-07-12 03:34:59  INFO    [TRADE] [SHADOW] ordre NON envoye (mode shadow).
2026-07-12 03:34:59  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=1 ordres=0 fills=0 rejets={'shadow_mode': 1}
/home/claude/freshtest/test_integration_pipeline.py:357: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmp8009mq_e/cycle_report.json' mode='r' encoding='utf-8'>
  rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_same_input_same_score_and_version (test_market_ranker.TestDeterminism.test_same_input_same_score_and_version) ... ok
test_weights_sum_to_one (test_market_ranker.TestDeterminism.test_weights_sum_to_one) ... ok
test_empty_book_low_score_and_ineligible (test_market_ranker.TestEmptyBookHeavilyPenalized.test_empty_book_low_score_and_ineligible) ... ok
test_volume_alone_is_insufficient (test_market_ranker.TestEmptyBookHeavilyPenalized.test_volume_alone_is_insufficient)
Regle imposee : un volume enorme ne sauve ni un carnet vide ni ... ok
test_no_ai_dependency (test_market_ranker.TestNoAINoOrders.test_no_ai_dependency) ... /home/claude/freshtest/test_market_ranker.py:140: ResourceWarning: unclosed file <_io.TextIOWrapper name='market_ranker.py' mode='r' encoding='utf-8'>
  src = open("market_ranker.py", encoding="utf-8").read().lower()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_no_order_path (test_market_ranker.TestNoAINoOrders.test_no_order_path) ... /home/claude/freshtest/test_market_ranker.py:145: ResourceWarning: unclosed file <_io.TextIOWrapper name='market_ranker.py' mode='r' encoding='utf-8'>
  src = open("market_ranker.py", encoding="utf-8").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_run_ranking_places_zero_orders_and_writes_reports (test_market_ranker.TestNoAINoOrders.test_run_ranking_places_zero_orders_and_writes_reports) ... 2026-07-12 03:34:59  INFO    [SCANNER] Univers: 2 marches (1 valides) -> /tmp/tmp2wwnk3nh/u.json | rapport -> /tmp/tmp2wwnk3nh/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 2 scores (1 eligibles) -> /tmp/tmp2wwnk3nh/market_rankings.json | rapport -> /tmp/tmp2wwnk3nh/market_ranker_report.json
2026-07-12 03:34:59  INFO    [SCANNER] Univers: 2 marches (1 valides) -> /tmp/tmp2wwnk3nh/u.json | rapport -> /tmp/tmp2wwnk3nh/r.json
2026-07-12 03:34:59  INFO    [RANKER] Classement: 2 scores (1 eligibles) -> /tmp/tmp2wwnk3nh/market_rankings.json | rapport -> /tmp/tmp2wwnk3nh/market_ranker_report.json
/home/claude/freshtest/test_market_ranker.py:190: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmp2wwnk3nh/market_snapshots_history.json' mode='r' encoding='utf-8'>
  hist = json.load(open(rc.HISTORY_FILE))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_market_ranker.py:192: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmp2wwnk3nh/market_rankings.json' mode='r' encoding='utf-8'>
  json.load(open(rc.RANKINGS_FILE))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_market_ranker.py:193: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmp2wwnk3nh/market_ranker_report.json' mode='r' encoding='utf-8'>
  self.assertIn("score_distribution", json.load(open(rc.REPORT_FILE)))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_relative_spread_penalizes_cheap_contracts (test_market_ranker.TestSpreadMonotonicity.test_relative_spread_penalizes_cheap_contracts) ... ok
test_score_decreases_as_spread_widens (test_market_ranker.TestSpreadMonotonicity.test_score_decreases_as_spread_widens) ... ok
test_empty_book_frequency_and_flaps_penalized (test_market_ranker.TestStabilityHistory.test_empty_book_frequency_and_flaps_penalized) ... ok
test_history_window_trimmed_and_stale_purged (test_market_ranker.TestStabilityHistory.test_history_window_trimmed_and_stale_purged) ... ok
test_insufficient_history_is_neutral_not_exclusive (test_market_ranker.TestStabilityHistory.test_insufficient_history_is_neutral_not_exclusive) ... ok
test_volatile_mid_scores_lower_than_stable (test_market_ranker.TestStabilityHistory.test_volatile_mid_scores_lower_than_stable) ... ok
test_closes_too_soon_rejected (test_market_ranker.TestTimeRules.test_closes_too_soon_rejected) ... ok
test_comfortable_window_scores_high (test_market_ranker.TestTimeRules.test_comfortable_window_scores_high) ... ok
test_very_far_expiry_penalized (test_market_ranker.TestTimeRules.test_very_far_expiry_penalized) ... ok
test_crossed_book_invalid (test_market_scanner.TestBookDerivation.test_crossed_book_invalid) ... ok
test_empty_book_not_invented (test_market_scanner.TestBookDerivation.test_empty_book_not_invented) ... ok
test_no_side_derived_from_yes (test_market_scanner.TestBookDerivation.test_no_side_derived_from_yes) ... ok
test_single_leg_not_invented (test_market_scanner.TestBookDerivation.test_single_leg_not_invented) ... ok
test_yes_derived_from_no (test_market_scanner.TestBookDerivation.test_yes_derived_from_no) ... ok
test_fallback_other (test_market_scanner.TestClassification.test_fallback_other) ... ok
test_native_category_wins (test_market_scanner.TestClassification.test_native_category_wins) ... ok
test_series_crypto (test_market_scanner.TestClassification.test_series_crypto) ... ok
test_series_economics (test_market_scanner.TestClassification.test_series_economics) ... ok
test_series_sports (test_market_scanner.TestClassification.test_series_sports) ... ok
test_category_allow_and_exclude (test_market_scanner.TestFiltersAndExclusions.test_category_allow_and_exclude) ... ok
test_closes_too_soon (test_market_scanner.TestFiltersAndExclusions.test_closes_too_soon) ... ok
test_empty_book_reason_kept_not_deleted (test_market_scanner.TestFiltersAndExclusions.test_empty_book_reason_kept_not_deleted) ... ok
test_expired (test_market_scanner.TestFiltersAndExclusions.test_expired) ... ok
test_invalid_book (test_market_scanner.TestFiltersAndExclusions.test_invalid_book) ... ok
test_low_volume (test_market_scanner.TestFiltersAndExclusions.test_low_volume) ... ok
test_no_liquidity_kept_if_not_required (test_market_scanner.TestFiltersAndExclusions.test_no_liquidity_kept_if_not_required) ... ok
test_spread_too_wide (test_market_scanner.TestFiltersAndExclusions.test_spread_too_wide) ... ok
test_valid_market_included (test_market_scanner.TestFiltersAndExclusions.test_valid_market_included) ... ok
test_dollars (test_market_scanner.TestNumberNormalization.test_dollars) ... ok
test_dollars_string (test_market_scanner.TestNumberNormalization.test_dollars_string) ... ok
test_fp_string (test_market_scanner.TestNumberNormalization.test_fp_string) ... ok
test_generic_number_variants (test_market_scanner.TestNumberNormalization.test_generic_number_variants) ... ok
test_int_cents (test_market_scanner.TestNumberNormalization.test_int_cents) ... ok
test_null_and_garbage (test_market_scanner.TestNumberNormalization.test_null_and_garbage) ... ok
test_out_of_range_rejected (test_market_scanner.TestNumberNormalization.test_out_of_range_rejected) ... ok
test_zero_is_empty_not_price (test_market_scanner.TestNumberNormalization.test_zero_is_empty_not_price) ... ok
test_all_pages_fetched (test_market_scanner.TestPagination.test_all_pages_fetched) ... ok
test_max_pages_guard (test_market_scanner.TestPagination.test_max_pages_guard) ... 2026-07-12 03:34:59  WARNING [SCANNER] Garde-fou MAX_PAGES=5 atteint -- pagination interrompue (univers peut-etre incomplet).
ok
test_run_scan_places_zero_orders_and_reports (test_market_scanner.TestScanOnlyNeverOrders.test_run_scan_places_zero_orders_and_reports) ... 2026-07-12 03:34:59  INFO    [SCANNER] Univers: 4 marches (2 valides) -> /tmp/tmpupc1yxel/market_universe.json | rapport -> /tmp/tmpupc1yxel/market_scanner_report.json
/home/claude/freshtest/test_market_scanner.py:194: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmpupc1yxel/market_universe.json' mode='r' encoding='utf-8'>
  u = json.load(open(c.UNIVERSE_FILE))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_market_scanner.py:197: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmpupc1yxel/market_scanner_report.json' mode='r' encoding='utf-8'>
  json.load(open(c.REPORT_FILE))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_scanner_module_has_no_order_path (test_market_scanner.TestScanOnlyNeverOrders.test_scanner_module_has_no_order_path)
Garantie statique : le module scanner ne reference jamais ... /home/claude/freshtest/test_market_scanner.py:204: ResourceWarning: unclosed file <_io.TextIOWrapper name='market_scanner.py' mode='r' encoding='utf-8'>
  src = open("market_scanner.py", encoding="utf-8").read()
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_1_every_core_module_imports (test_repo_integrity.Test1_2_AllModulesImportCleanly.test_1_every_core_module_imports) ... ok
test_2_no_missing_local_import (test_repo_integrity.Test1_2_AllModulesImportCleanly.test_2_no_missing_local_import)
Analyse AST de tous les .py : chaque import local doit etre ... /home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/kalshi_alpha_bot.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/kalshi_edge_measure.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/market_ranker.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/market_scanner.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/opportunity_pipeline.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/pattern_engine.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/run_tests.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/strategy_router.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/test_audit_v11.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/test_integration_pipeline.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/test_market_ranker.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/test_market_scanner.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/home/claude/freshtest/test_repo_integrity.py:47: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/claude/freshtest/test_repo_integrity.py' mode='r' encoding='utf-8'>
  encoding="utf-8").read())
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_3_discover_loads_every_test_module (test_repo_integrity.Test3_DiscoveryFindsAllSuites.test_3_discover_loads_every_test_module) ... ok
test_4_5_run_tests_writes_accurate_report (test_repo_integrity.Test4_5_ReportMatchesReality.test_4_5_run_tests_writes_accurate_report)
Execute run_tests.py en sous-processus (garde anti-recursion via ... /home/claude/freshtest/test_repo_integrity.py:99: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmphtzbp17s/test_report.json' mode='r' encoding='utf-8'>
  rep = json.load(open(os.path.join(tmp, "test_report.json")))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_6_shadow_starts_without_btc_context_and_survives_cycle (test_repo_integrity.Test6_7_8_EngineWithoutModel.test_6_shadow_starts_without_btc_context_and_survives_cycle)
Le depot est livre SANS btc_context : le moteur doit demarrer en ... 2026-07-12 03:35:00  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:35:00  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:35:00  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmpyqn9ck77/u.json | rapport -> /tmp/tmpyqn9ck77/r.json
2026-07-12 03:35:00  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmpyqn9ck77/rk.json | rapport -> /tmp/tmpyqn9ck77/rr.json
2026-07-12 03:35:00  INFO    [PIPELINE] [PIPELINE] rejet KXETH-A: no_model_probability
2026-07-12 03:35:00  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=0 ordres=0 fills=0 rejets={'no_model_probability': 1}
ok
test_7_no_model_means_zero_create_order (test_repo_integrity.Test6_7_8_EngineWithoutModel.test_7_no_model_means_zero_create_order)
Routeur PAR DEFAUT (aucun stub) : sans fournisseur de probabilite, ... 2026-07-12 03:35:00  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:35:00  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:35:00  INFO    [SCANNER] Univers: 1 marches (1 valides) -> /tmp/tmpebmbpf7m/u.json | rapport -> /tmp/tmpebmbpf7m/r.json
2026-07-12 03:35:00  INFO    [RANKER] Classement: 1 scores (1 eligibles) -> /tmp/tmpebmbpf7m/rk.json | rapport -> /tmp/tmpebmbpf7m/rr.json
2026-07-12 03:35:00  INFO    [PIPELINE] [PIPELINE] rejet KXETH-A: no_model_probability
2026-07-12 03:35:00  INFO    [BOT] [CYCLE-REPORT] scanned=1 eligibles=1 acceptes=0 ordres=0 fills=0 rejets={'no_model_probability': 1}
/home/claude/freshtest/test_repo_integrity.py:193: ResourceWarning: unclosed file <_io.TextIOWrapper name='/tmp/tmpebmbpf7m/cycle_report.json' mode='r' encoding='utf-8'>
  rep = json.load(open(os.path.join(self.tmp, "cycle_report.json")))
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_8_normal_mode_really_uses_scanner (test_repo_integrity.Test6_7_8_EngineWithoutModel.test_8_normal_mode_really_uses_scanner)
Preuve directe : le cycle normal interroge /markets (pagination ... 2026-07-12 03:35:00  INFO    [BOT] ── CYCLE #1 ─────────────────────────────────────────────
2026-07-12 03:35:00  INFO    [RISK] [CAPITAL] solde=1000.00$ capital_effectif=500.00$
2026-07-12 03:35:00  INFO    [SCANNER] Univers: 2 marches (1 valides) -> /tmp/tmplyjp1glt/u.json | rapport -> /tmp/tmplyjp1glt/r.json
2026-07-12 03:35:00  INFO    [RANKER] Classement: 2 scores (1 eligibles) -> /tmp/tmplyjp1glt/rk.json | rapport -> /tmp/tmplyjp1glt/rr.json
2026-07-12 03:35:00  INFO    [PIPELINE] [PIPELINE] rejet KXETH-A: no_model_probability
2026-07-12 03:35:00  INFO    [BOT] [CYCLE-REPORT] scanned=2 eligibles=1 acceptes=0 ordres=0 fills=0 rejets={'no_model_probability': 1, 'no_liquidity': 1}
ok

----------------------------------------------------------------------
Ran 97 tests in 0.920s

OK (skipped=2)
code retour: 0
```

## Commande supplementaire (celle qui echouait chez l'utilisateur)
```
$ python -m unittest -v test_integration_pipeline.py
test_pipeline_scans_every_page (test_integration_pipeline.TestN_PaginationAllPages.test_pipeline_scans_every_page) ... ok
Ran 17 tests in 0.162s
OK
```
