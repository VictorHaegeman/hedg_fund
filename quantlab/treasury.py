"""Trésorerie du fonds : pas de cash qui dort (gestion de liquidité d'un vrai fonds).

Un fonds d'investissement ne laisse pas l'argent non utilisé sur un compte à 0 % :
  1. Socle « bêta » : jusqu'à CORE_PCT (30 %) de l'actif dans le S&P 500 (SPY)
  2. Le reste du cash non utilisé → bons du Trésor US très courts (BIL, ~T-bills 1-3 mois,
     duration ≈ 0,1 an : quasiment aucun risque de taux — cf. module obligations)
  3. Un petit coussin de cash (BUFFER_PCT) reste disponible pour les frais
  4. Les stratégies restent PRIORITAIRES : quand une entrée a besoin d'argent, on vend
     d'abord BIL, puis le socle SPY (raise_cash). Rien n'est jamais bloqué.

Les positions de trésorerie sont séparées des positions de stratégie (pas de stop, ne
comptent pas dans le nombre max de positions, pas dans les statistiques de trades).
Testé sur 2007→2026 (voir README) : ~+12 %/an contre +10,6 % pour le S&P 500,
avec une pire baisse de −28 % contre −55 %.
"""

from __future__ import annotations

CORE_SYMBOL = "SPY"
CORE_PCT = 0.30
SWEEP_SYMBOL = "BIL"
BUFFER_PCT = 0.01        # cash gardé en vrai (frais, arrondis)
BAND_PCT = 0.03          # on ne rééquilibre que si l'écart dépasse 3 % de l'actif
COMMISSION_BPS = 5.0


def _px(account, prices: dict, sym: str) -> float:
    p = prices.get(sym)
    if p:
        return float(p)
    return float((account.treasury.get(sym) or {}).get("last") or 0.0)


def value(account, prices: dict) -> float:
    return sum(h["shares"] * _px(account, prices, s) for s, h in account.treasury.items())


def _trade(account, sym: str, dollars: float, price: float) -> float:
    """Achète (+) ou vend (−) `dollars` de `sym` au prix donné. Renvoie le montant réel."""
    if price <= 0 or abs(dollars) < 1:
        return 0.0
    fee = COMMISSION_BPS / 10_000.0
    h = account.treasury.setdefault(sym, {"shares": 0.0, "cost": 0.0, "last": price})
    if dollars > 0:
        dollars = min(dollars, account.cash / (1 + fee))
        if dollars < 1:
            return 0.0
        sh = dollars / price
        account.cash -= dollars * (1 + fee)
        h["shares"] += sh
        h["cost"] += dollars
    else:
        sh = min(-dollars / price, h["shares"])
        if h["shares"] > 0:
            h["cost"] *= 1 - sh / h["shares"]
        h["shares"] -= sh
        account.cash += sh * price * (1 - fee)
        dollars = -sh * price
    h["last"] = price
    if h["shares"] <= 1e-9:
        account.treasury.pop(sym, None)
    return dollars


def raise_cash(account, needed: float, prices: dict) -> list[str]:
    """Libère `needed` $ de cash pour une entrée de stratégie : BIL d'abord, puis SPY."""
    lines = []
    short = needed - account.cash
    for sym in (SWEEP_SYMBOL, CORE_SYMBOL):
        if short <= 0:
            break
        h = account.treasury.get(sym)
        px = _px(account, prices, sym)
        if not h or px <= 0:
            continue
        sold = -_trade(account, sym, -min(short * 1.002, h["shares"] * px), px)
        if sold > 0:
            lines.append(f"  Trésorerie : vente {sold:,.0f} $ de {sym} pour financer une entrée")
        short = needed - account.cash
    return lines


def rebalance(account, prices: dict, core_pct: float = CORE_PCT,
              sweep: str | None = SWEEP_SYMBOL, band: float = BAND_PCT,
              buffer: float = BUFFER_PCT, strategy_value: float | None = None) -> list[str]:
    """Place le cash non utilisé : socle SPY (≤ core_pct) puis le reste en BIL."""
    for s in list(account.treasury):        # mémorise le dernier prix connu
        if prices.get(s):
            account.treasury[s]["last"] = float(prices[s])
    tre = value(account, prices)
    if strategy_value is None:
        strategy_value = sum(p["shares"] * prices.get(s, p["entry"])
                             for s, p in account.positions.items())
    equity = account.cash + tre + strategy_value
    if equity <= 0:
        return []
    avail = max(account.cash + tre - buffer * equity, 0.0)
    targets = {CORE_SYMBOL: min(core_pct * equity, avail)}
    if sweep:
        targets[sweep] = max(avail - targets[CORE_SYMBOL], 0.0)
    lines = []
    # ventes d'abord (libère du cash), achats ensuite
    order = sorted(targets, key=lambda s: targets[s] - (account.treasury.get(s, {}).get("shares", 0)
                                                        * _px(account, prices, s)))
    for sym in order:
        px = _px(account, prices, sym)
        if px <= 0:
            continue
        cur = (account.treasury.get(sym) or {}).get("shares", 0.0) * px
        diff = targets[sym] - cur
        tol = band * equity if sym == CORE_SYMBOL else 0.01 * equity
        if abs(diff) < tol and not (targets[sym] == 0 and cur > 0):
            continue
        done = _trade(account, sym, diff, px)
        if abs(done) >= 1:
            what = "socle S&P 500" if sym == CORE_SYMBOL else "bons du Trésor"
            lines.append(f"  Trésorerie : {'achat' if done > 0 else 'vente'} {abs(done):,.0f} $ "
                         f"de {sym} ({what})")
    return lines


def summary_line(account, prices: dict) -> str:
    if not account.treasury:
        return "  Trésorerie placée : rien (tout est investi ou en cash)."
    parts = []
    for s, h in account.treasury.items():
        v = h["shares"] * _px(account, prices, s)
        pnl = v - h.get("cost", v)
        parts.append(f"{s} {v:,.0f} $ ({pnl:+,.0f} $)")
    return "  Trésorerie placée : " + " | ".join(parts)
