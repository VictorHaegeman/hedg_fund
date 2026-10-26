"""Connecteur Alpaca — compte de courtage réel en mode paper (argent fictif).

REST pur (urllib, aucune dépendance). Les clés ne sont JAMAIS dans le code :
  • GitHub Actions : Secrets du dépôt (ALPACA_API_KEY, ALPACA_SECRET_KEY, …)
  • En local : fichier .env à la racine du projet (ignoré par git)

Deux comptes possibles :
  • ALPACA_API_KEY / ALPACA_SECRET_KEY       → compte du fonds (miroir de live)
  • ALPACA_AI_API_KEY / ALPACA_AI_SECRET_KEY → compte de la poche IA (module 14)
Si la poche IA n'a pas ses propres clés, elle partage le compte du fonds mais
ne touche jamais aux actifs de l'univers du fonds (univers disjoints).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class AlpacaError(RuntimeError):
    """Erreur renvoyée par l'API Alpaca (statut HTTP + message)."""


def load_env(path: str | None = None) -> bool:
    """Charge un .env local dans os.environ (UTF-8 ou UTF-16, espaces, guillemets).

    Ne remplace jamais une variable déjà définie (les Secrets GitHub priment).
    """
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    elif len(raw) > 1 and raw[1:2] == b"\x00":
        text = raw.decode("utf-16-le")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and "COLLE" not in v.upper() and not os.environ.get(k):
            os.environ[k] = v
    return True


def to_alpaca_symbol(symbol: str) -> str:
    """Ticker Yahoo → symbole d'ordre Alpaca (BTC-USD → BTC/USD)."""
    if symbol.endswith("-USD"):
        return symbol[:-4] + "/USD"
    return symbol


def position_symbol(symbol: str) -> str:
    """Symbole tel qu'il apparaît dans /v2/positions (BTC/USD → BTCUSD)."""
    return to_alpaca_symbol(symbol).replace("/", "")


def is_crypto(symbol: str) -> bool:
    return "/" in symbol or symbol.endswith("-USD")


class Alpaca:
    def __init__(self, key: str, secret: str, paper: bool = True, timeout: int = 20,
                 label: str = "fonds"):
        self.key, self.secret = key, secret
        self.base = PAPER_URL if paper else LIVE_URL
        self.paper = paper
        self.timeout = timeout
        self.label = label

    @classmethod
    def from_env(cls, prefix: str = "ALPACA", label: str = "fonds") -> "Alpaca | None":
        load_env()
        key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        secret = os.environ.get(f"{prefix}_SECRET_KEY", "").strip()
        if not key or not secret:
            return None
        paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() != "false"
        return cls(key, secret, paper=paper, label=label)

    # ── transport ──────────────────────────────────────────────────────────
    def request(self, method: str, path: str, base: str | None = None,
                params: dict | None = None, body: dict | None = None):
        url = (base or self.base) + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                txt = r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise AlpacaError(f"{method} {path} → {e.code}: {detail}") from None
        return json.loads(txt) if txt.strip() else {}

    def data(self, path: str, params: dict | None = None):
        return self.request("GET", path, base=DATA_URL, params=params)

    # ── compte & ordres ────────────────────────────────────────────────────
    def account(self) -> dict:
        return self.request("GET", "/v2/account")

    def clock(self) -> dict:
        return self.request("GET", "/v2/clock")

    def positions(self) -> list[dict]:
        return self.request("GET", "/v2/positions")

    def open_orders(self) -> list[dict]:
        return self.request("GET", "/v2/orders", params={"status": "open", "limit": 500})

    def asset(self, symbol: str) -> dict:
        return self.request("GET", f"/v2/assets/{urllib.parse.quote(symbol, safe='')}")

    def submit_order(self, body: dict) -> dict:
        return self.request("POST", "/v2/orders", body=body)

    def get_order(self, order_id: str) -> dict:
        return self.request("GET", f"/v2/orders/{order_id}")

    def cancel_order(self, order_id: str) -> None:
        self.request("DELETE", f"/v2/orders/{order_id}")

    def close_position(self, symbol: str) -> dict:
        return self.request("DELETE", f"/v2/positions/{position_symbol(symbol)}")

    def wait_fill(self, order_id: str, timeout: int = 15, sleep=time.sleep) -> dict | None:
        """Attend l'exécution ; annule si rien après `timeout` s. Renvoie l'ordre rempli."""
        o: dict = {}
        for _ in range(timeout):
            o = self.get_order(order_id)
            if o.get("status") == "filled":
                return o
            if o.get("status") in ("canceled", "expired", "rejected"):
                return None
            sleep(1)
        try:
            self.cancel_order(order_id)
        except AlpacaError:
            pass
        return o if o.get("status") == "partially_filled" else None


def account_line(client: Alpaca) -> str:
    """Résumé d'une ligne d'un compte Alpaca (fail-safe)."""
    try:
        a = client.account()
        eq, last = float(a["equity"]), float(a.get("last_equity") or a["equity"])
        mode = "paper" if client.paper else "RÉEL"
        return (f"Alpaca [{client.label}, {mode}] équité {eq:,.2f} $ "
                f"({(eq / last - 1) * 100 if last else 0:+.2f} % aujourd'hui) | "
                f"cash {float(a['cash']):,.2f} $")
    except Exception as e:  # réseau, clés invalides…
        return f"Alpaca [{client.label}] indisponible : {str(e)[:120]}"
