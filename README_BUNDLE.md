# Bundle scanner v2 + routeur declaratif — 2026-07-19

## Contenu et destination

Fichiers a REMPLACER dans le depot principal (versions modifiees des votres) :
- `market_scanner.py`   — univers CIBLE par defaut, crawl general opt-in,
  plafond par cycle, fenetre LOOKAHEAD, dedup, metriques API.
- `strategy_router.py`  — strategies declarent supported_series /
  supported_market_types / required_features ; router.supported_series() ;
  resolve_detailed() (diagnostic structure). resolve() inchange.
- `opportunity_pipeline.py` — PRIORITY_SERIES derive du routeur (source
  unique, rien de code en dur), plafond d'analyses profondes, cache TTL des
  rejets, metriques, [ROUTER-DIAG] dans les traces.
- `market_taxonomy.py`  — INCHANGE (copie fournie uniquement pour que les
  tests du bundle tournent de maniere autonome).

Fichiers NOUVEAUX (a ajouter au depot) :
- `test_scanner_v2.py`, `test_router_declarations.py`,
  `test_integration_btc15m_path.py`

Artefacts fournis :
- `test_output_full.txt` — sortie complete de
  `python -m unittest -v` (18 tests, OK).
- `examples/example_market_scanner_report.json`
- `examples/example_pipeline_report.json` — montre eligible=1,
  strategy_supported=1, model_probability=1, accepted=1, orders=0.
- `diff_v2.patch` — diff complet contre vos fichiers uploades.

## Execution autonome

```bash
cd bundle/
python -m unittest -v test_scanner_v2 test_router_declarations test_integration_btc15m_path
```
Aucune dependance externe, aucun reseau.

## Nouvelles variables d'environnement

```env
SCANNER_GENERAL_CRAWL_ENABLED=0      # crawl /markets complet: diagnostic only
SCANNER_PRIORITY_SERIES=             # vide => derive de router.supported_series()
SCANNER_PRIORITY_MAX_PAGES=5
SCANNER_MAX_MARKETS_PER_CYCLE=300
SCANNER_LOOKAHEAD_HOURS=24
SCAN_MAX_DEEP_ANALYSES_PER_CYCLE=50
PIPELINE_REJECT_TTL_PERMANENT_S=3600
PIPELINE_REJECT_TTL_TEMPORARY_S=120
```

## Benchmark (SYNTHETIQUE — mock API deterministe, pas des donnees reelles)

Reproduit la config prod observee dans vos logs (crawl general, 200 pages
x 200) contre le mode cible :

    marches recus / cycle : 40 000 -> 2      (>= 99,9 % de reduction)
    appels API / cycle    : 200    -> 1      (99,5 % de reduction)

Vos logs prod montraient en plus ~548 analyses candidates par cycle, toutes
rejetees ; en mode cible, deep_analysis_count est borne a 50 et ne porte que
sur des marches ayant une strategie. Le benchmark reel avant/apres doit etre
refait sur votre environnement (les 40 000 etaient un plancher : la
pagination etait tronquee).

## LIMITES — a lire avant de considerer quoi que ce soit comme "resolu"

1. `market_ranker.py`, `btc_context.py`, `btc_probability_model.py`,
   `run_tests.py` et les autres modules du depot n'ont PAS ete fournis.
   Le test d'integration utilise un STUB de ranker (installe via
   sys.modules, jamais ecrit sous le nom market_ranker.py) et un STUB de
   modele deterministe. Il prouve le chemin
   scan cible -> normalisation -> classification -> routage -> portes,
   PAS le vrai ranker ni le vrai modele BTC.
2. Apres remplacement des fichiers dans le depot complet, relancer
   `python run_tests.py` (les 142 tests existants) — je n'ai pas pu le
   faire ici. Points de vigilance verifies par lecture, pas par execution :
   les fakes de test_repo_integrity et test_market_taxonomy tolerent le
   parametre series_ticker supplementaire ; la dedup absorbe les doublons.
3. Si l'environnement demo Kalshi ne liste pas KXBTC15M, le warning
   `priority_series_empty` le dira au premier cycle — c'est une limite
   d'environnement, pas de code.
4. Cause secondaire non auditee : l'eligibilite du vrai ranker
   (MIN_BOOK_OBSERVATIONS) peut exclure des marches de 15 minutes.
   Fournir market_ranker.py ou market_ranker_report.json pour trancher.
5. Modes disabled/paper/shadow/live (point 7 de vos exigences) : le bot
   possede deja SHADOW_MODE + verrous live multiples + gatekeeper
   (constate dans kalshi_alpha_bot.py). La refonte en 4 modes formels
   n'est PAS incluse ici : je refuse de modifier un fichier de 85 Ko dont
   les dependances (requests, cryptography, btc_context...) manquent pour
   tester. A traiter dans une passe dediee, avec le depot complet.

## Rentabilite

Rien ici ne demontre ni ne pretend une rentabilite. Le gatekeeper et le
shadow trading restent les seules voies vers le live.
