import time, base64, requests
from pathlib import Path
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

API_KEY_ID       = "e80b071b-4d9e-419b-91df-e9abb0bdd462"
PRIVATE_KEY_PATH = "kalshi_private_key.pem"
BASE_URL         = "https://demo-api.kalshi.co/trade-api/v2"

key = load_pem_private_key(Path(PRIVATE_KEY_PATH).read_bytes(), password=None)

def signer(method, path):
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path).encode()
    sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": API_KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(), "Content-Type": "application/json"}

path = "/trade-api/v2/markets?status=open&limit=5"
r = requests.get(BASE_URL + "/markets?status=open&limit=5", headers=signer("GET", path), timeout=10)
marches = r.json().get("markets", [])

print(f"\n{len(marches)} marches recus. Voici le premier :\n")
if marches:
    m = marches[0]
    print(f"Titre      : {m.get('title')}")
    print(f"Ticker     : {m.get('ticker')}")
    print(f"yes_bid    : {m.get('yes_bid')}")
    print(f"yes_ask    : {m.get('yes_ask')}")
    print(f"no_bid     : {m.get('no_bid')}")
    print(f"no_ask     : {m.get('no_ask')}")
    print(f"last_price : {m.get('last_price')}")
    print(f"volume     : {m.get('volume')}")
    print(f"\nTous les champs disponibles :")
    for k, v in m.items():
        print(f"  {k}: {v}")