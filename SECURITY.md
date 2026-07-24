# Politique de sécurité

## Signalement
Ne signalez **jamais** une vulnérabilité via une issue publique.
Contactez le mainteneur en privé (adresse à renseigner dans le profil du
dépôt). Délai de réponse visé : 72 h.

## Périmètre sensible
- Gestion des clés API Kalshi (démo et production) et signatures de requêtes.
- Verrous DEMO/LIVE (`model_gatekeeper`, confirmations d'environnement).
- Intégrité des fichiers d'état (`JsonStore` : écriture atomique + sha256).
- Idempotence des ordres (`client_order_id`) et réconciliation au démarrage.

## Règles pour les utilisateurs
- Utilisez des clés **démo** dédiées ; ne réutilisez jamais des clés prod.
- Ne committez jamais `.env` (couvert par `.gitignore`).
- Vérifiez `test_report.json` et le rapport d'audit avant toute idée de
  passage LIVE ; `NO_LIVE_PROMOTION=1` est le défaut et doit le rester tant
  que vous n'avez pas de validation shadow suffisante.

## Divulgation
Correctif publié avant les détails ; crédit au découvreur si souhaité.
