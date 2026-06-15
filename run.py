"""
KALSHI ALPHA ENGINE V3 — Lanceur principal
Lance simultanement :
  - Serveur dashboard (port 8080)
  - Bot CPI toutes les 5 minutes
  - Bot BTC 15min toutes les 60 secondes
"""

import os, sys, threading, subprocess, time, logging, requests
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("kalshi_alpha.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("Launcher")

def get_current_btc_price() -> float:
    """Recupere le prix BTC actuel pour le target dynamique."""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price",
                        params={"symbol": "BTCUSDT"}, timeout=5)
        return float(r.json()["price"])
    except Exception:
        return 65000.0

def run_cpi_bot():
    """Lance le bot CPI en boucle toutes les 5 minutes."""
    log.info("Bot CPI demarre — KXCPI-26JUN-T0.1 toutes les 5 min")
    while True:
        try:
            subprocess.run([
                sys.executable, "kalshi_alpha_bot.py",
                "--market", "KXCPI-26JUN-T0.1",
                "--capital", "500",
                "--loop",
                "--interval", "300",
                "--context", "Analyse CPI juin 2026. Publication le 14 juillet 2026 a 8h30 ET."
            ])
        except Exception as e:
            log.error(f"Erreur CPI bot: {e}")
        time.sleep(10)

def run_btc_bot():
    """Lance le bot BTC 15min en boucle toutes les 60 secondes."""
    log.info("Bot BTC 15min demarre — analyse toutes les 60s")
    while True:
        try:
            # Prix BTC dynamique pour le target
            btc_price = get_current_btc_price()
            # Arrondit au tick le plus proche (Kalshi utilise des seuils precis)
            log.info(f"BTC actuel: ${btc_price:,.2f}")

            subprocess.run([
                sys.executable, "kalshi_alpha_bot.py",
                "--btc",
                "--btc-target", str(round(btc_price, 2)),
                "--btc-minutes", "15",
                "--capital", "500",
            ], timeout=55)  # timeout avant le prochain cycle
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            log.error(f"Erreur bot BTC: {e}")
        time.sleep(60)

def run_fred_updater():
    """Met a jour le contexte FRED toutes les heures."""
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        log.info("FRED_API_KEY absent — updater desactive")
        return
    log.info("FRED updater demarre — mise a jour toutes les heures")
    while True:
        try:
            from fred_context import get_macro_context
            ctx = get_macro_context("CPI")
            with open("fred_cache.txt", "w", encoding="utf-8") as f:
                f.write(ctx)
            log.info("Contexte FRED mis a jour")
        except Exception as e:
            log.error(f"Erreur FRED updater: {e}")
        time.sleep(3600)

if __name__ == "__main__":
    log.info("="*60)
    log.info("  KALSHI ALPHA ENGINE V3 — DEMARRAGE COMPLET")
    log.info("="*60)
    log.info(f"  Heure : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    log.info(f"  BTC   : ${get_current_btc_price():,.2f}")
    log.info("="*60)

    # Lance tous les bots en threads paralleles
    threads = [
        threading.Thread(target=run_cpi_bot,     daemon=True, name="CPI-Bot"),
       # threading.Thread(target=run_btc_bot,     daemon=True, name="BTC-Bot"),
        threading.Thread(target=run_fred_updater,daemon=True, name="FRED-Updater"),
    ]

    for t in threads:
        t.start()
        log.info(f"Thread {t.name} demarre")
        time.sleep(2)  # Decale les demarrages

    # Lance le serveur Flask (bloquant — doit etre en dernier)
    log.info("Demarrage serveur dashboard...")
    from server import app
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
