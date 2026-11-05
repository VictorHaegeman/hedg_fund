"""Miroir du fonds sur Alpaca — le compte papier reste la source de vérité.

La logique de trading du fonds (live.py, backtestée) ne change pas : elle décide
sur paper_account.json. Ce module recopie ensuite le portefeuille cible sur un
vrai compte de courtage Alpaca (paper), par RÉCONCILIATION :

    cible Alpaca = positions du compte papier × (équité Alpaca / équité papier)

Pour chaque actif de l'univers du fonds : on achète/vend la différence entre la
cible et ce que détient Alpaca. Idempotent : relancer le cycle ne double jamais
un ordre, et un ordre raté est rattrapé au cycle suivant. Seuls les actifs de
l'univers du fonds sont touchés (la poche IA peut partager le compte).

Les ordres actions passés après la clôture sont mis en file par Alpaca et
exécutés à l'ouverture suivante ; la crypto s'exécute immédiatement.

VRAIS STOPS (nov. 2026) : le compte papier vend au prix du stop dès qu'il est touché dans
la journée ; sans ordre chez le courtier, Alpaca ne vendait que le lendemain matin
(−1,4 %/an sur 2007-2026). Chaque soir, on pose donc un ordre stop GTC chez Alpaca pour
chaque position actions du fonds (parts entières), et les nouveaux achats partent avec leur
stop attaché (ordre OTO) pour être protégés dès le premier jour. Pas de stop crypto.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone

from .alpaca import Alpaca, AlpacaError, is_crypto, position_symbol, to_alpaca_symbol
from .broker import Account

MIN_NOTIONAL = 5.0       # $ : en dessous, l'écart est ignoré (bruit d'arrondi)
TOLERANCE = 0.02         # 2 % d'écart relatif toléré avant de rééquilibrer


def plan_mirror(account: Account, prices: dict[str, float], universe: list[str],
                alpaca_positions: dict[str, float], alpaca_equity: float,
                pending: set[str] | None = None, reserve: float = 0.0) -> list[dict]:
    """Calcule les ordres pour aligner Alpaca sur le compte papier.

    alpaca_positions : {symbole de position Alpaca (ex. BTCUSD): quantité}
    pending          : symboles de position ayant déjà un ordre en attente
    """
    pending = pending or set()
    fund_eq = account.equity(prices)
    # reserve : part du compte Alpaca laissée libre (poche IA sur le même compte)
    scale = alpaca_equity * (1 - reserve) / fund_eq if fund_eq > 0 else 1.0
    targets = {s: q * scale for s, q in account.holdings().items()}

    actions = []
    for sym in sorted(set(universe) | set(targets)):
        psym = position_symbol(sym)
        if psym in pending:
            continue
        want = targets.get(sym, 0.0)
        have = float(alpaca_positions.get(psym, 0.0))
        price = (prices.get(sym) or (account.positions.get(sym) or {}).get("entry")
                 or (account.treasury.get(sym) or {}).get("last") or 0.0)
        if want <= 0 and have > 0:
            actions.append({"symbol": sym, "action": "close", "qty": have,
                            "reason": "position fermée par le fonds"})
            continue
        diff = want - have
        if price <= 0 or abs(diff) * price < MIN_NOTIONAL:
            continue
        if want > 0 and abs(diff) / want < TOLERANCE:
            continue
        actions.append({"symbol": sym, "action": "buy" if diff > 0 else "sell",
                        "qty": abs(diff),
                        "reason": "nouvelle position" if have == 0 else "rééquilibrage"})
    return actions


STOP_TAG = "ql-stop"


def _stop_px(p: float) -> float:
    """Prix stop arrondi vers le bas (2 décimales au-dessus de 1 $)."""
    return math.floor(p * 100) / 100 if p >= 1 else math.floor(p * 10000) / 10000


def _valid_stop(account: Account, sym: str, prices: dict[str, float]) -> float | None:
    pos = account.positions.get(sym)
    if not pos or is_crypto(sym):
        return None
    stop, px = pos.get("stop"), prices.get(sym) or pos.get("entry")
    if not stop or not px or stop <= 0 or stop >= px * 0.999:
        return None
    return _stop_px(float(stop))


def _stop_body(sym: str, qty: int, stop: float) -> dict:
    return {"symbol": to_alpaca_symbol(sym), "side": "sell", "type": "stop", "qty": str(qty),
            "stop_price": f"{stop:.2f}" if stop >= 1 else f"{stop:.4f}",
            "time_in_force": "gtc",
            "client_order_id": f"{STOP_TAG}-{sym}-{datetime.now(timezone.utc):%Y%m%d%H%M%S%f}"[:48]}


def cancel_fund_stops(client: Alpaca, managed: set[str], sleep=time.sleep) -> int:
    """Annule les ordres stop de vente du fonds encore ouverts (ils sont reposés après le
    rééquilibrage, à la bonne quantité). Ne touche pas aux ordres de la poche IA."""
    ids = [o["id"] for o in client.open_orders()
           if o.get("type") == "stop" and o.get("side") == "sell"
           and o.get("symbol", "").replace("/", "") in managed]
    for oid in ids:
        try:
            client.cancel_order(oid)
        except AlpacaError:
            pass
    for _ in range(10):                      # attendre que les annulations soient prises en compte
        if not ids or not {o["id"] for o in client.open_orders()} & set(ids):
            break
        sleep(1)
    return len(ids)


def stopped_out(client: Alpaca, account: Account) -> set[str]:
    """Positions vendues par un VRAI stop Alpaca alors que le compte papier les garde
    (données de prix légèrement différentes) : on ne les rachète pas."""
    after = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = set()
    for o in client.order_history("closed", after):
        if o.get("type") != "stop" or o.get("side") != "sell" or o.get("status") != "filled":
            continue
        sym = next((s for s in account.positions
                    if position_symbol(s) == o.get("symbol", "").replace("/", "")), None)
        if sym and str(o.get("filled_at", ""))[:10] >= str(account.positions[sym].get("entry_date", ""))[:10]:
            out.add(position_symbol(sym))
    return out


def place_stops(client: Alpaca, account: Account, prices: dict[str, float],
                held: dict[str, float], actions: list[dict], skip: set[str] = frozenset(),
                scale: float = 1.0) -> list[str]:
    """Pose un stop GTC (parts entières) sur chaque position actions du fonds détenue chez
    Alpaca, moins ce qui part déjà à la vente ce soir."""
    lines = []
    leaving = {}
    for a in actions:
        if a["action"] in ("sell", "close"):
            leaving[a["symbol"]] = float("inf") if a["action"] == "close" else a["qty"]
    for sym in sorted(account.positions):
        stop = _valid_stop(account, sym, prices)
        psym = position_symbol(sym)
        if stop is None or psym in skip:
            continue
        # jamais plus que la part « stratégie » (le socle de trésorerie SPY n'a pas de stop)
        qty = int(min(float(held.get(psym, 0.0)) - leaving.get(sym, 0.0),
                      account.positions[sym]["shares"] * scale))
        if qty < 1:
            continue
        try:
            client.submit_order(_stop_body(sym, qty, stop))
            lines.append(f"  ALPACA stop {sym} : {qty} parts à {stop:,.2f}")
        except AlpacaError as e:
            lines.append(f"  [!] ALPACA stop {sym} refusé — {str(e)[:120]}")
    return lines


def _order_body(sym: str, side: str, qty: float) -> dict:
    crypto = is_crypto(sym)
    return {"symbol": to_alpaca_symbol(sym), "side": side, "type": "market",
            "qty": f"{qty:.6f}" if crypto else f"{qty:.4f}",
            "time_in_force": "gtc" if crypto else "day"}


def execute_mirror(client: Alpaca, actions: list[dict], account: Account | None = None,
                   prices: dict[str, float] | None = None) -> list[str]:
    lines = []
    for a in actions:
        sym = a["symbol"]
        try:
            if a["action"] == "close":
                client.close_position(sym)
                lines.append(f"  ALPACA ferme {sym} ({a['reason']})")
                continue
            stop = (_valid_stop(account, sym, prices or {})
                    if account is not None and a["action"] == "buy" and a["reason"] == "nouvelle position"
                    else None)
            if stop is not None and int(a["qty"]) >= 1:
                # achat + stop attaché (OTO) : protégé dès le premier jour
                body = dict(_order_body(sym, "buy", float(int(a["qty"]))), order_class="oto",
                            stop_loss={"stop_price": f"{stop:.2f}" if stop >= 1 else f"{stop:.4f}"})
                body["qty"] = str(int(a["qty"]))
                try:
                    client.submit_order(body)
                    lines.append(f"  ALPACA achète {int(a['qty'])} {sym} + stop {stop:,.2f} ({a['reason']})")
                    continue
                except AlpacaError as e:
                    lines.append(f"  ALPACA {sym} : achat avec stop refusé ({str(e)[:80]}), achat simple")
            try:
                client.submit_order(_order_body(sym, a["action"], a["qty"]))
            except AlpacaError as e:
                # actif non fractionnable : on arrondit à l'unité inférieure
                if "fractional" not in str(e).lower() or int(a["qty"]) < 1:
                    raise
                client.submit_order(_order_body(sym, a["action"], float(int(a["qty"]))))
            verb = "achète" if a["action"] == "buy" else "vend"
            lines.append(f"  ALPACA {verb} {a['qty']:.4f} {sym} ({a['reason']})")
        except Exception as e:
            lines.append(f"  [!] ALPACA {sym} : ordre refusé — {str(e)[:140]}")
    return lines


def managed_universe(account: Account, universe: list[str],
                     protected: set[str] | None = None) -> list[str]:
    """Actifs gérés par le fonds : univers du cycle + positions + actifs déjà tradés
    (utile avec le screener, dont l'univers change d'un jour à l'autre), moins les
    positions protégées de la poche IA si elle partage le compte."""
    syms = (set(universe) | set(account.positions) | set(account.treasury)
            | {t["symbol"] for t in account.history})
    protected = protected or set()
    return sorted(s for s in syms if position_symbol(s) not in protected)


def mirror_report(client: Alpaca | None, account: Account, prices: dict[str, float],
                  universe: list[str], protected: set[str] | None = None,
                  reserve: float = 0.0, sleep=time.sleep) -> str:
    """Synchronise le compte Alpaca et renvoie le compte rendu (jamais d'exception)."""
    lines = ["", "— MIROIR ALPACA —"]
    if client is None:
        lines.append("  Clés Alpaca absentes (ALPACA_API_KEY / ALPACA_SECRET_KEY) : "
                     "miroir ignoré, le compte papier reste à jour.")
        return "\n".join(lines)
    try:
        acct = client.account()
        positions = {p["symbol"]: float(p["qty"]) for p in client.positions()}
        managed = managed_universe(account, universe, protected)
        try:
            n = cancel_fund_stops(client, {position_symbol(s) for s in managed}, sleep=sleep)
            if n:
                lines.append(f"  {n} stop(s) Alpaca du fonds annulé(s) pour être reposés")
        except AlpacaError as e:
            lines.append(f"  [!] stops Alpaca non annulés : {str(e)[:100]}")
        pending = {o["symbol"].replace("/", "") for o in client.open_orders()}
        try:
            out = stopped_out(client, account)
        except Exception:
            out = set()
        if out:
            lines.append(f"  Vendu(s) par un vrai stop Alpaca, pas de rachat : {', '.join(sorted(out))}")
        actions = plan_mirror(account, prices, managed, positions, float(acct["equity"]),
                              pending | out, reserve)
        lines += execute_mirror(client, actions, account, prices) or ["  Alpaca déjà aligné sur le fonds."]
        fund_eq = account.equity(prices)
        scale = float(acct["equity"]) * (1 - reserve) / fund_eq if fund_eq > 0 else 1.0
        lines += place_stops(client, account, prices, positions, actions, skip=pending | out,
                             scale=scale)
        if pending:
            lines.append(f"  Ordres en attente (exécution à l'ouverture) : "
                         f"{', '.join(sorted(pending))}")
        lines.append(f"  Équité Alpaca : {float(acct['equity']):,.2f} $ | "
                     f"cash {float(acct['cash']):,.2f} $")
    except Exception as e:
        lines.append(f"  [!] Alpaca indisponible ce cycle : {str(e)[:160]} "
                     "(rattrapage automatique au prochain cycle)")
    return "\n".join(lines)
