# DIAGNOSTIC_FUNNEL — Pourquoi 480 éligibles → 0 accepté
*2026-07-12. Audit de la phase MarketRanker → ExecutionEngine. Aucun seuil,
aucun filtre, aucune logique modifiés — instrumentation et rapports JSON
uniquement (+ correction de 2 doublons de logs introduits par une passe
d'instrumentation antérieure incohérente, dont un double comptage de
`risk_passed` : bug de statistique, pas de logique de trading).*

## 1. Ce que tes logs prouvent déjà (export du 12/07, 04:08-04:09 UTC)

- Demo **opérationnel** : clés DEMO dédiées, solde réel 93,26 $, positions
  broker vides, 40 000 marchés paginés (garde-fou MAX_PAGES=200×200 atteint
  → l'univers demo dépasse 40 000 : pagination interrompue, à savoir).
- Funnel : 40 000 scannés → 587 valides scanner → 480 éligibles ranker →
  0 accepté, 0 ordre.
- La ligne `rejets={...}` est **tronquée par l'export Railway** après
  `{'unsupported': 9053, 'spread_too_wide': 270, 'invalid_book': 1,
  'below_min_score': 44, …`. Ces 9 368 rejets visibles + les 480 éligibles
  laissent ~30 152 rejets tronqués (masse quasi certaine : `no_liquidity`,
  `closes_too_soon`, `expired`, `low_fill_probability`). C'est précisément
  pourquoi l'instrumentation écrit désormais des FICHIERS intronquables.

## 2. Diagnostic structurel PROUVÉ (lecture du code, ordre exact des portes)

Chaque candidat éligible traverse, dans cet ordre
(`strategy_router.evaluate_candidate`) :
qualité marché → **stratégie compatible** → spread → **probabilité modèle**
→ confiance → ask exécutable → edge brut → edge net → EV net.

**Cause n°1 — `no_compatible_strategy` (attendu : l'écrasante majorité des
480).** L'unique stratégie enregistrée est `btc15m_model_v1`, dont
`supports()` n'accepte QUE les tickers/séries commençant par `KXBTC15M`.
Ton propre `[RAW:markets_page]` montre un univers dominé par le sport
(`KXMLBOUTS…`) : tout marché éligible non-BTC15M est rejeté ici, par
conception (règle absolue de ta mission Phase 3). 480 éligibles ne
signifie pas 480 tradables : le ranker mesure la qualité d'exécution,
pas la compatibilité stratégique.

**Cause n°2 — plafond structurel de confiance sous 9 minutes (prouvé
arithmétiquement).** Pour les rares KXBTC15M : confiance =
q_data×q_time×q_vol avec q_time = clamp(t/15, 0.2, 1), mappée sur 0-10,
seuil MIN_MODEL_CONFIDENCE=6. Même avec des données PARFAITES :
t=10 min → 7/10 (passe) ; **t=8 min → 5/10 (rejet) ; t=5 min → 3/10**.
Avec SCANNER_MIN_MINUTES=5, une fenêtre 15 min n'est donc candidate
crédible qu'entre ~9 et ~15 min restantes — et le cycle complet
(scan 40 000 marchés ≈ 12 s + ranking + 480 relectures de carnet) consomme
de ce budget. `insufficient_confidence` est attendu dominant sur les BTC15M,
suivi de `no_positive_edge`/`insufficient_net_edge` (la baseline non
calibrée s'écarte rarement de ≥5 points du prix marché, seuil
MIN_GROSS_EDGE=0.05).

**Aggravant identifié (non modifié, à ta décision)** : chaque candidat
éligible déclenche `fresh_book_fn` = un `get_market` par ticker → jusqu'à
480 appels API par cycle AVANT le routeur. Un tri « stratégie compatible
d'abord » éviterait ~99 % de ces appels — c'est une modification de
logique, donc HORS périmètre de cette mission ; je la note seulement.

## 3. Instrumentation ajoutée (logs + fichiers, zéro logique)

Chaîne par candidat : `[RANK] ticker|cat|score` → `[ROUTER] stratégie|AUCUNE`
→ `[MODEL] p|confiance` → `[EDGE] p_marché|brut|frais|slippage|net|ev` →
`[RISK] portes passées|taille` → `[EXECUTION] envoi` ou `[REJECT] raison`.
À chaque cycle, trois fichiers intronquables :
- `pipeline_stats.json` : {scanned, valid, eligible, strategy_supported,
  model_probability, positive_edge, positive_net_ev, risk_passed,
  accepted, orders} — exactement le bloc demandé, aussi loggé `[STATS]`.
- `reject_reasons.json` : {reject_reasons: {raison: n}}.
- `candidate_trace.json` : les 20 meilleurs par score avec les 13 colonnes
  demandées (ticker, score, stratégie, p modèle, confiance, p marché,
  edge brut, frais, slippage, edge net, EV net, décision, raison).
Vérifié hors-ligne sur mini-univers (sortie ci-dessus dans la conversation)
et par la suite complète : **116 tests, 114 OK, 0 échec, 2 skips**, protocole
répertoire vierge re-passé.

## 4. Ce qu'il te reste à faire pour les chiffres EXACTS (2 minutes)

L'export de logs étant tronqué, les comptes exacts par raison sur TES 480
sortiront du prochain cycle : déploie le ZIP, laisse tourner UN cycle, puis
récupère `pipeline_stats.json`, `reject_reasons.json` et
`candidate_trace.json` (CWD du service) et envoie-les-moi. Prédiction
falsifiable, à vérifier contre ces fichiers : `no_compatible_strategy`
≥ ~95 % des 480 ; le reste réparti entre `insufficient_confidence`,
`no_model_probability`/`insufficient_data_quality` et
`no_positive_edge`/`insufficient_net_edge` sur les seuls KXBTC15M.

## 5. Conclusion — le système fait ce qui a été exigé

0 accepté n'est pas une panne : c'est la conséquence directe de tes règles
absolues (« un marché sans stratégie n'est jamais tradé », « aucune
probabilité inventée », « baseline non validée = pas de trade facile »).
Le chemin vers des acceptations passe par la validation du modèle BTC en
shadow (MODEL_VALIDATION_GUIDE.md), pas par l'assouplissement des portes.
