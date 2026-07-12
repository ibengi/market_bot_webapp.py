# AUDIT_INTEGRATION_FINAL — Pipeline intégré Kalshi Alpha Engine
*2026-07-11. Périmètre : intégration scanner→ranker→routeur→edge/EV→risque→exécution.*

## 1. Architecture avant / après

**Avant** : `ExecutionEngine.cycle()` appelait exclusivement `BtcStrategy.signal()`
(sélection mono-série KXBTC15M), testait UN marché puis abandonnait le cycle,
n'avait aucune porte edge/EV (edge=0 enregistré), un capital statique (--capital),
des frais locaux uniquement, `positions[ticker]`, aucune relecture du carnet avant
l'ordre, aucun mode shadow, kill switch ou double confirmation prod.

**Après** : chaque cycle normal exécute
`SCANNER (pagination complète) → RANKER (tradabilité) → StrategyRouter (par
catégorie) → portes edge/EV nettes → budgets de risque → relecture carnet frais →
exécution → confirmation fills (/fills) → frais API → réconciliation broker`.
Le pipeline parcourt les candidats classés du meilleur au moins bon, journalise
chaque rejet avec sa raison, écrit `cycle_report.json`, et n'est jamais bloqué
sur un ticker vide. `--scan-only`/`--rank-only` restent des modes de diagnostic.

Fichiers nouveaux : `strategy_router.py`, `opportunity_pipeline.py`,
`test_integration_pipeline.py`, `.env.example`, `test_report.json`.
Fichiers modifiés : `kalshi_alpha_bot.py` (Config, FeeModel.from_api,
PositionManager par trade_id + réconciliation broker, invariant place_and_track,
ExecutionEngine réécrit, main double-confirmation + --shadow),
`market_ranker.py` (expose les snapshots), `test_audit_v11.py` (nommage D4).

## 2. Bugs corrigés (vérifiés par tests)

1. Dépendance exclusive à KXBTC15M : le moteur sélectionne désormais tout marché
   éligible de toute catégorie servie par une stratégie (TEST B).
2. Abandon au premier rejet : parcours multi-candidats jusqu'à MAX_TRADES_CYCLE
   ou épuisement (TEST G).
3. Trade avec edge=0/EV=0 : impossible — portes no_positive_edge /
   insufficient_net_edge / negative_net_ev (TESTS E, F).
4. Carnet vide : bloqué à 4 niveaux — scanner (no_liquidity), ranker (plafond 25
   + inéligible), relecture fraîche avant ordre (stale_book, TEST L), et
   invariant dur dans place_and_track : `create_order` inatteignable sans prix
   1..99 valide (TEST C : zéro appel espionné).
5. Marché sans stratégie : rejet no_compatible_strategy ; la stratégie BTC n'est
   routée QUE vers Crypto, et seulement si `btc_context` fournit une probabilité
   indépendante (sinon no_model_probability — rien n'est inventé) (TEST D).
6. Capital statique : `effective_capital = min(plafond configuré, solde broker)`
   à chaque cycle ; prod sans solde = blocage ; demo = secours uniquement via
   ALLOW_FALLBACK_CAPITAL=1, journalisé (TEST K).
7. Frais : priorité réponse d'ordre → fills → formule locale ; `fee_source`,
   `estimated_fee_before_order`, `actual_fee_after_fill` journalisés (TEST M).
8. `positions[ticker]` : désormais `positions[trade_id]` avec ticker, lots,
   order_ids, fill_ids, stratégie, edge/EV d'entrée, état ; migration
   automatique de l'ancien format ; règlement écrit AVANT retrait de la
   position (plus de trade zombie possible).
9. Réconciliation broker : positions présentes chez Kalshi et absentes
   localement reconstruites avec un ID STABLE (`brk-{ticker}-{side}`) —
   idempotent, zéro doublon après redémarrage (TEST J) ; positions locales
   absentes du broker marquées ghost ; `reconciliation_report.json` écrit ;
   dédup des fill_ids persistée.
10. Fill présumé sur statut : /fills reste la source de vérité (TEST I),
    fills partiels = quantité réelle uniquement (TEST H).
11. Sécurité : KILL_SWITCH, SHADOW_MODE (--shadow), prod = double confirmation
    KALSHI_ENV_CONFIRM=LIVE **et** LIVE_TRADING_CONFIRMED=YES.

## 3. Résultats réels des tests (Étapes 5-6)

```
python -m py_compile kalshi_alpha_bot.py market_scanner.py market_ranker.py
                     strategy_router.py opportunity_pipeline.py   -> OK (5/5)

test_integration_pipeline  : Ran 17, OK        (TESTS A-O + shadow + kill switch)
test_market_ranker         : Ran 16, OK
test_market_scanner        : Ran 31, OK
test_audit_v11             : Ran 26, 24 OK, 2 FAILED (connus, voir ci-dessous)
TOTAL                      : 90 tests, 88 passés, 2 échecs connus
```

Échecs connus, volontairement conservés comme témoins :
- `test_D6_two_fee_models_must_agree` : `trade_resolver.py` v1 (hérité) utilise
  un barème 2,45 % incompatible → **supprimer ce fichier du dépôt** (la v11
  règle elle-même les positions).
- `test_D8_debug_kalshi_must_compile` : `debug_kalshi.py` est un fragment
  invalide → **supprimer du dépôt**.
Rapport machine : `test_report.json`. Couverture : NON MESURABLE ici (pas de
réseau, module coverage non installable) — exécuter
`pip install pytest pytest-cov && pytest --cov` sur ta machine.

## 4. Commandes exactes

```
# Diagnostic (aucun ordre)
python kalshi_alpha_bot.py --demo --scan-only
python kalshi_alpha_bot.py --demo --rank-only

# Shadow (pipeline + décisions complètes, AUCUN ordre)
python kalshi_alpha_bot.py --demo --btc --loop --shadow

# Demo réel (ordres réels sur demo-api ; clés demo OBLIGATOIRES)
python kalshi_alpha_bot.py --demo --btc --loop

# Production (argent réel) : exige les DEUX variables
KALSHI_ENV_CONFIRM=LIVE LIVE_TRADING_CONFIRMED=YES \
python kalshi_alpha_bot.py --btc --loop
```

**Procédure demo** : (1) clés demo dans Railway ; (2) 24 h en `--shadow` et
vérifier `cycle_report.json` (rejets cohérents, aucun ordre) ; (3) passer en
demo réel, valider les 10 premiers ordres via `[RAW:create_order]`/`[RAW:fills]`;
(4) comparer `fee_source=api` vs estimé.

**Critères avant production** : ≥100-300 trades demo réglés avec IC95 bas du
taux de réussite > prix moyen payé + frais (kalshi_edge_measure) ; zéro écart de
réconciliation sur 1 semaine ; couverture mesurée ≥90 % sur OrderManager /
PositionManager / FeeModel ; kill switch testé en conditions réelles ; volume
Railway persistant monté (DATA_DIR).

## 5. Risques résiduels — honnêtement

1. **Aucune stratégie ne produit encore de probabilité indépendante** : tant que
   `btc_context` ne renvoie pas `prob_reelle` fiable, TOUS les candidats crypto
   seront rejetés no_model_probability et le bot ne tradera pas. C'est voulu
   (aucune prob inventée), mais il faut le savoir : le pipeline est prêt, le
   modèle ne l'est pas. L'ancien module ML est démontré invalide (audit
   précédent) et ne doit PAS être rebranché tel quel.
2. Endpoints /markets (cursor), /portfolio/positions et champs de frais suivent
   la doc v2 telle que connue mais n'ont jamais été confrontés à l'API réelle
   depuis cet environnement (pas de réseau) — les [RAW:*] du premier run réel
   sont à vérifier humainement.
3. La réconciliation broker reconstruit les quantités mais le prix moyen broker
   (`avg_price`) peut différer du champ réellement renvoyé — à valider sur le
   premier [RAW:positions].
4. Pas de verrou distribué multi-instances (exigence notée, non implémentée) :
   ne lancer qu'UNE instance (Railway replicas=1). Un verrou fichier ne
   protégerait pas deux conteneurs distincts.
5. L'heuristique d'unités de frais (cents vs dollars) est défensive mais devra
   être figée après observation du premier fill réel.
6. Slippage = tampon constant configurable, pas un modèle de profondeur.

## 6. Notes (sur 10), justifiées

| Catégorie | Note | Pourquoi pas 10 |
|---|---|---|
| Architecture | 8 | Pipeline propre, modules découplés, routeur extensible ; reste : stratégies = duck typing sans ABC formelle, PnL latent conservateur non intégré aux portes. |
| Intégration | 8 | Le mode normal utilise réellement scanner+ranker (TEST A), multi-candidats, rapport de cycle complet ; reste : catégorie non propagée dans le budget par catégorie (approximée via stratégie), artefacts rescannés à chaque cycle (coût API à optimiser avec un cache). |
| Exécution | 7 | Invariant dur anti-carnet-vide, relecture fraîche, fills partiels exacts, TTL ; reste : pas de repricing intra-TTL, prix moyen de repli = limite si /fills muet. |
| Risque | 7 | Solde réel, plafonds position/budget/positions ouvertes/drawdown/marché ; reste : MAX_CATEGORY_RISK_PCT approximé (catégorie non stockée dans la position à l'entrée du budget), PnL latent non compté dans les portes. |
| Récupération après crash | 7 | Réconciliation ordres au démarrage, positions par trade_id, règlement avant retrait (plus de zombie), réconciliation broker idempotente testée sur fake ; reste : jamais validée contre le vrai broker, client_order_id toujours non persisté AVANT l'envoi. |
| Observabilité | 7 | cycle_report.json, raisons de rejet exhaustives, fee_source, logs par canal, [RAW:*] ; reste : pas de health_state.json ni de métriques de latence structurées. |
| Tests | 8 | 90 tests, 88 verts, A-O couverts, espions create_order, hors-ligne et déterministes ; reste : couverture non mesurable ici, 2 témoins rouges sur fichiers hérités, pas de test de charge/pagination longue réelle. |
| Sécurité production | 8 | Double confirmation, kill switch, shadow, clés demo obligatoires, aucun secret en dur, aucun fallback silencieux ; reste : pas de verrou distribué, pas d'alerte externe (webhook) sur erreur critique broker. |

**Le système n'est PAS déclaré production-ready** : les tests unitaires passent,
mais aucune validation contre l'API réelle n'a encore eu lieu depuis cet
environnement, et aucune stratégie ne fournit encore de probabilité modèle
exploitable. Prochaine étape logique : le modèle de probabilité crypto (Phase 4),
mesuré d'abord en shadow avec kalshi_edge_measure.

## 7. Addendum reproductibilité (2026-07-12)

Correction d'une faute de livraison : les résultats précédents avaient été
obtenus dans un répertoire de travail contenant des fichiers hérités non
inclus dans les livrables. Le dépôt est désormais AUTONOME : testé en
répertoire vierge avec `python -m py_compile *.py` puis
`python -m unittest discover -v` → **Ran 97 tests, OK (skipped=2)** ; les 2
skips sont explicites (fichiers hérités volontairement exclus du dépôt).
`test_report.json` est généré automatiquement par `python run_tests.py` à
partir des résultats réels. Sorties brutes complètes : REPRODUCIBILITY.md.
La stratégie crypto est explicitement désactivée sans btc_context (option 3
+ injection de fournisseur possible via StrategyRouter) : démarrage shadow
sans crash et zéro create_order sans modèle, vérifiés par test_repo_integrity.
