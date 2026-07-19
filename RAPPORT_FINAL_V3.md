# RAPPORT FINAL — v3 (2026-07-19)

## AVERTISSEMENT DE PERIMETRE (a lire en premier)
Le message demandait « le depot complet reellement deploye » et « les logs
Railway les plus recents ». AUCUN nouveau fichier n'a ete uploade dans cette
conversation. Fichiers reels disponibles et corriges : market_taxonomy.py,
market_scanner.py, strategy_router.py, opportunity_pipeline.py,
kalshi_alpha_bot.py (entrypoint reel, patch verifie a l'import avec les
vraies dependances requests/cryptography/dotenv installees).
Fichiers ABSENTS, donc NON audites/testes : market_ranker.py, btc_context.py,
btc_probability_model.py, shadow_prediction_store.py, model_gatekeeper.py,
run_tests.py, Dockerfile, Procfile, railway.json, variables Railway.
Tout ce qui depend de ces fichiers est marque comme tel ci-dessous.

## TABLE DE SUPPORT REEL (section 6 exigee)

| Market type            | Strategie presente | Modele present        | Activee | Support reel |
| ---------------------- | ------------------ | --------------------- | ------- | ------------ |
| sports_moneyline       | non                | non                   | n/a     | NON          |
| sports_total           | non                | non                   | n/a     | NON          |
| sports_spread          | non                | non                   | n/a     | NON          |
| sports_player_prop     | non                | non                   | n/a     | NON          |
| btc_above_strike_15m   | oui (btc15m_model_v1) | oui (btc_probability_model — fichier non fourni ici, present au depot selon MODEL_VALIDATION_GUIDE ; NON VALIDE) | oui (BTC_STRATEGY_ENABLED) | OUI (shadow uniquement) |
| btc_above_strike_daily | non                | non                   | n/a     | NON          |

Consequence appliquee (votre propre exigence) : les marches sports et BTC
daily sont EXCLUS du scan (univers cible derive du registre), au lieu d'etre
classes/rankes puis rejetes des centaines de fois.

## VERDICT AU FORMAT EXIGE

ROOT CAUSE:
  Les rejets no_compatible_strategy des marches sports/BTCD sont le
  comportement CORRECT du routeur strict : aucune strategie reelle n'existe
  pour ces types (table ci-dessus) et vos interdictions proscrivent d'en
  fabriquer. Le vrai defaut, prouve par vos logs des cycles #7018-7019 :
  (1) le scanner crawlait /markets sans filtre serveur, tronque a
  MAX_PAGES=200 (40 000 marches, warning « univers peut-etre incomplet » a
  chaque cycle), si bien que KXBTC15M — la SEULE serie supportee —
  n'apparaissait dans AUCUNE ligne de log (0 occurrence, contre 43 pour
  KXBTCD) ; (2) l'ordre du pipeline etait inverse (ranking de centaines de
  marches PUIS classification PUIS rejet).

SECONDARY CAUSES:
  - Ordre du pipeline corrige : classify -> precheck registre -> ranking des
    seuls supportes -> analyse profonde (avant : ranking d'abord).
  - NON TRANCHE faute de fichier : l'eligibilite du vrai market_ranker
    (MIN_BOOK_OBSERVATIONS) peut exclure des marches de 15 minutes.
  - Environnement : si le demo Kalshi ne liste pas KXBTC15M, le nouveau
    warning priority_series_empty le montrera au premier cycle.

FILES MODIFIED:
  market_taxonomy.py (MarketType str-Enum canonique, normalize_market_type,
    get_env_bool canonique) ; strategy_router.py (registre unique :
    enabled/disabled, doublons refuses, validate() bloquant,
    [STRATEGY_REGISTRY_BOOT], selection deterministe par priorite,
    resolve_detailed -> StrategyResolutionResult avec codes normalises,
    supports_detailed -> SupportCheckResult) ; opportunity_pipeline.py
    (precheck registre AVANT ranking, [STRATEGY_RESOLUTION]/
    [STRATEGY_SELECTED]/[MODEL_PROBABILITY]/[SHADOW_DECISION], cache TTL
    rejets, plafond analyses profondes, metriques exigees,
    signal_accepted/risk_passed/order_authorized/order_submitted +
    would_submit/actual_submission) ; market_scanner.py (univers cible par
    defaut, crawl general opt-in, plafonds, lookahead, dedup, metriques) ;
    kalshi_alpha_bot.py ([BUILD_INFO] + log_boot() + validate() dans le
    VRAI ExecutionEngine — import du module patche verifie).

TESTS EXECUTED:
  53 au total, sortie complete jointe (test_output_full_v3.txt) :
  - test_registry_v3 (13) — liste 8A complete
  - test_real_tickers_from_logs (2, 5 sous-cas) — tickers reels 8C
  - test_scanner_v2 (8, dont benchmark)
  - test_router_declarations (7)
  - test_integration_btc15m_path (3)
  - sous-ensemble executable des tests REELS du depot
    (test_market_taxonomy: 20) — non-regression sur v3

TESTS PASSED: 53/53 (OK)

REAL STRATEGIES LOADED: 1 (btc15m_model_v1) — [STRATEGY_REGISTRY_BOOT]
  reel dans deliverables/example_startup.log

SUPPORTED MARKET TYPES: btc_above_strike_15m
UNSUPPORTED MARKET TYPES: tous les autres (voir table) — exclus du scan

SCAN REDUCTION: benchmark SYNTHETIQUE (mock deterministe reproduisant la
  config prod observee) : 40 000 -> 2 marches recus/cycle, 200 -> 1 appels
  API (>= 99 %). deep_analysis_count borne a 50 (verifie). Le chiffre reel
  doit etre releve sur votre deploiement.

SHADOW STATUS: [SHADOW_DECISION] JSON conforme (would_submit=true,
  actual_submission=false, risk_passed=null renseigne par le moteur) ;
  ambiguite accepted/risk_passed levee (signal_accepted distinct de
  order_submitted, alias historique conserve). Exemple reel joint.

LIVE STATUS: BLOQUE. Verrous existants du bot inchanges (double
  confirmation + LIVE_TRADING=1 + gatekeeper) ; rien ici ne les affaiblit.

REMAINING RISKS:
  1. Chaine modele reelle NON testee ici (btc_context/btc_probability_model
     absents) : l'exigence 8E « vraie strategie ET vrai modele » est
     partiellement remplie (vraie strategie, vrai routage ; modele stub
     etiquete). 2. market_ranker reel non teste (stub etiquete, jamais
     ecrit sous ce nom de fichier). 3. Deploiement Railway invérifiable
     sans Dockerfile/Procfile/railway.json/variables : [BUILD_INFO] est
     ajoute pour le prouver APRES redeploiement. 4. Les 142 tests de
     run_tests.py doivent etre relances dans le depot complet.
  5. KXBTC15M peut etre absent de l'environnement demo (warning dedie).

GO/NO-GO: NO-GO LIVE. Et meme le « GO shadow » definitif exige : (a) les
  fichiers manquants pour finir l'audit, (b) un cycle reel montrant
  [STRATEGY_REGISTRY_BOOT], [STRATEGY_SELECTED], [MODEL_PROBABILITY] et
  [SHADOW_DECISION] dans les logs Railway.

## CHECKLIST DE REDEPLOIEMENT
1. Remplacer les 5 fichiers (diff_v3_complete.patch) sur la branche deployee.
2. Copier .env.example -> variables Railway (SHADOW_MODE=1, KILL_SWITCH a
   votre convenance, BTC_STRATEGY_ENABLED=true, crawl general=false).
3. Redeployer ; verifier au demarrage : [BUILD_INFO] (commit attendu) puis
   [STRATEGY_REGISTRY_BOOT] count=1. Si « No active trading strategies
   registered » : flag BTC_STRATEGY_ENABLED mal defini.
4. Premier cycle : verifier priority_markets_received["KXBTC15M"]>0 dans
   market_scanner_report.json (sinon warning priority_series_empty =>
   probleme d'environnement Kalshi, pas de code).
5. Attendre une fenetre 15 min avec donnees valides : logs
   [STRATEGY_SELECTED] puis [MODEL_PROBABILITY] puis [SHADOW_DECISION]
   (ou rejets honnetes no_model_probability/insufficient_data_quality).
6. Lancer python run_tests.py dans le depot complet ; m'envoyer
   test_report.json + market_ranker.py si un ecart apparait.
7. Ne toucher a AUCUN verrou live.

## COMMANDES DE TEST EXACTES
    python -m unittest -v test_registry_v3 test_real_tickers_from_logs \
        test_scanner_v2 test_router_declarations test_integration_btc15m_path
