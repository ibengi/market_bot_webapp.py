import threading, sys, subprocess

def bot():
    subprocess.run([sys.executable, "kalshi_alpha_bot.py",
                    "--market", "KXCPI-26JUN-T0.1",
                    "--capital", "500", "--loop", "--interval", "300"])

t = threading.Thread(target=bot, daemon=True)
t.start()

import os
from server import app
port = int(os.environ.get("PORT", 8080))
app.run(host="0.0.0.0", port=port)
