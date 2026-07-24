# Changelog

## [12.0.0] — 2026-07-24
### Corrige
- Registre de strategies canonique indexe par `market_type` ; refus de
  demarrage si registre vide/incomplet (etait : 1 seule strategie limitee a
  une serie absente de l'univers → `strategy_supported=0`).
- Classification deterministe par prefixe de serie (etait : sous-chaines —
  « Las Vegas » → Commodities, « POL » → Politics, etc.).
- Scanner cible + cache univers ≥ 30 min avec rafraichissement incremental
  (etait : 40 001 marches re-telecharges par minute).
- Limites de risque en % du capital effectif = min(solde, plafond) ;
  stop jour 5 %, 1 %/trade, 3 positions max, arret apres 3 pertes
  consecutives (etait : stop fixe −50 $ sur un compte de 93,26 $).
### Ajoute
- 68 tests (classification, routage, scanner, sizing petit compte,
  cycle de vie des ordres, reconciliation, separation DEMO/LIVE).
- `model_gatekeeper` : LIVE impossible sans tests verts, validation modele
  et leve explicite de `NO_LIVE_PROMOTION`.
- Resume de cycle structure unique `[CYCLE-SUMMARY]` + decisions JSONL rotatif.

## [11.x] — historique interne (pre-open-source)
