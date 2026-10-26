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
"""

from __future__ import annotations

from .alpaca import Alpaca, AlpacaError, is_crypto, position_symbol, to_alpaca_symbol
from .broker import Account

MIN_NOTIONAL = 5.0       # $ : en dessous, l'écart est ignoré (bruit d'arrondi)
TOLERANCE = 0.02         # 2 % d'écart relatif toléré avant de rééquilibrer


def plan_mirror(account: Account, prices: dict[str, float], universe: list[str],
                alpaca_positions: dict[str, float], alpaca_equity: float,
                pending: set[str] | None = None) -> list[dict]:
    """Calcule les ordres pour aligner Alpaca sur le compte papier.

    alpaca_positions : {symbole de position Alpaca (ex. BTCUSD): quantité}
    pending          : symboles de position ayant déjà un ordre en attente
    """
    pending = pending or set()
    fund_eq = account.equity(prices)
    scale = alpaca_equity / fund_eq if fund_eq > 0 else 1.0
    targets = {s: p["shares"] * scale for s, p in account.positions.items()}

    actions = []
    for sym in sorted(set(universe) | set(targets)):
        psym = position_symbol(sym)
        if psym in pending:
            continue
        want = targets.get(sym, 0.0)
        have = float(alpaca_positions.get(psym, 0.0))
        price = prices.get(sym) or (account.positions.get(sym) or {}).get("entry") or 0.0
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


def _order_body(sym: str, side: str, qty: float) -> dict:
    crypto = is_crypto(sym)
    return {"symbol": to_alpaca_symbol(sym), "side": side, "type": "market",
            "qty": f"{qty:.6f}" if crypto else f"{qty:.4f}",
            "time_in_force": "gtc" if crypto else "day"}


def execute_mirror(client: Alpaca, actions: list[dict]) -> list[str]:
    lines = []
    for a in actions:
        sym = a["symbol"]
        try:
            if a["action"] == "close":
                client.close_position(sym)
                lines.append(f"  ALPACA ferme {sym} ({a['reason']})")
                continue
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
    syms = set(universe) | set(account.positions) | {t["symbol"] for t in account.history}
    protected = protected or set()
    return sorted(s for s in syms if position_symbol(s) not in protected)


def mirror_report(client: Alpaca | None, account: Account, prices: dict[str, float],
                  universe: list[str], protected: set[str] | None = None) -> str:
    """Synchronise le compte Alpaca et renvoie le compte rendu (jamais d'exception)."""
    lines = ["", "— MIROIR ALPACA —"]
    if client is None:
        lines.append("  Clés Alpaca absentes (ALPACA_API_KEY / ALPACA_SECRET_KEY) : "
                     "miroir ignoré, le compte papier reste à jour.")
        return "\n".join(lines)
    try:
        acct = client.account()
        positions = {p["symbol"]: float(p["qty"]) for p in client.positions()}
        pending = {o["symbol"].replace("/", "") for o in client.open_orders()}
        actions = plan_mirror(account, prices, managed_universe(account, universe, protected),
                              positions, float(acct["equity"]), pending)
        lines += execute_mirror(client, actions) or ["  Alpaca déjà aligné sur le fonds."]
        if pending:
            lines.append(f"  Ordres en attente (exécution à l'ouverture) : "
                         f"{', '.join(sorted(pending))}")
        lines.append(f"  Équité Alpaca : {float(acct['equity']):,.2f} $ | "
                     f"cash {float(acct['cash']):,.2f} $")
    except Exception as e:
        lines.append(f"  [!] Alpaca indisponible ce cycle : {str(e)[:160]} "
                     "(rattrapage automatique au prochain cycle)")
    return "\n".join(lines)
