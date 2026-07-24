# Contribuer

Merci de votre intérêt ! Règles du projet :

## Règles non négociables
1. **Aucun assouplissement des seuils de risque ou d'edge** pour « faire
   passer » un trade ou un test. Une PR qui baisse `MIN_NET_EDGE`,
   `MIN_NET_EV`, `MIN_MODEL_CONFIDENCE` ou les limites de risque sans
   justification statistique documentée sera refusée.
2. **Aucune probabilité inventée** : une stratégie sans modèle calibré doit
   rejeter `no_model_probability`, pas estimer au jugé.
3. **Prix exécutables uniquement** (ask/bid), jamais `last_price`.
4. **Jamais de clé API dans le code, les tests ou les fixtures.**
5. Le mode LIVE ne doit jamais pouvoir s'activer sans les verrous du
   `model_gatekeeper`.

## Flux de travail
- Fork → branche `feat/...` ou `fix/...` → PR vers `main`.
- `python run_tests.py` doit être vert (68+ tests, 0 échec) ; toute
  correction de bug ajoute un test de non-régression.
- Style : PEP 8, pas de dépendance ajoutée sans discussion préalable
  (issue), messages de commit à l'impératif.
- Les tests doivent rester **hors-ligne et déterministes** (clients
  injectés, pas de réseau, pas d'horloge réelle non contrôlée).

## Signaler un bug
Ouvrez une issue avec : version, extrait de `[CYCLE-SUMMARY]`, tickers
concernés (anonymisez tout identifiant de compte), comportement attendu.
Pour les failles de sécurité : voir SECURITY.md (pas d'issue publique).
