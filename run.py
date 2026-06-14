import threading, sys, os, subprocess
from server import app

def bot():
    subprocess.run([
        sys.executable, "kalshi_alpha_bot.py",
        "--market", "KXCPI-26JUN-T0.1",
        "--capital", "500",
        "--loop",
        "--interval", "300"
    ])

t = threading.Thread(target=bot, daemon=True)
t.start()

port = int(os.environ.get("PORT", 8080))
print(f"Serveur dashboard sur port {port}")
app.run(host="0.0.0.0", port=port)
