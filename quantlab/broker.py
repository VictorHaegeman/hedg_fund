"""Courtier papier (paper trading) — état persistant en JSON.

Simule un compte réel : cash, positions ouvertes (avec stop/target), journal
complet des trades. Aucune clé API requise ; les prix viennent des données de
marché réelles chargées par data.load(fresh=True).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

DEFAULT_STATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_account.json")


@dataclass
class Account:
    cash: float
    initial_capital: float
    positions: dict = field(default_factory=dict)   # symbol -> position dict
    history: list = field(default_factory=list)     # trades clôturés
    created: str = ""
    last_cycle: str = ""

    def equity(self, prices: dict[str, float]) -> float:
        value = self.cash
        for sym, pos in self.positions.items():
            value += pos["shares"] * prices.get(sym, pos["entry"])
        return value


def load_account(path: str = DEFAULT_STATE, capital: float = 10_000.0) -> Account:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return Account(**d)
    return Account(cash=capital, initial_capital=capital,
                   created=datetime.now().isoformat(timespec="seconds"))


def save_account(account: Account, path: str = DEFAULT_STATE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(account.__dict__, f, indent=2, ensure_ascii=False, default=str)


def reset_account(path: str = DEFAULT_STATE, capital: float = 10_000.0) -> Account:
    account = Account(cash=capital, initial_capital=capital,
                      created=datetime.now().isoformat(timespec="seconds"))
    save_account(account, path)
    return account


def buy(account: Account, symbol: str, price: float, stop: float, target: float | None,
        strategy: str, date: str, risk_pct: float = 1.0,
        commission_bps: float = 5.0, equity: float | None = None) -> dict | None:
    """Achat au prix donné, taille = risque % de l'équité / distance au stop."""
    if symbol in account.positions or price <= stop:
        return None
    eq = equity if equity is not None else account.cash
    fee = commission_bps / 10_000.0
    risk_amount = eq * risk_pct / 100.0
    shares = risk_amount / (price - stop)
    max_shares = account.cash / (price * (1 + fee))
    shares = min(shares, max_shares)
    if shares <= 0:
        return None
    cost = shares * price * (1 + fee)
    account.cash -= cost
    pos = {"symbol": symbol, "shares": shares, "entry": price, "stop": stop,
           "target": target, "strategy": strategy, "entry_date": date,
           "risk_amount": round(risk_amount, 2)}
    account.positions[symbol] = pos
    return pos


def sell(account: Account, symbol: str, price: float, date: str, reason: str,
         commission_bps: float = 5.0) -> dict | None:
    pos = account.positions.pop(symbol, None)
    if pos is None:
        return None
    fee = commission_bps / 10_000.0
    proceeds = pos["shares"] * price * (1 - fee)
    account.cash += proceeds
    pnl = proceeds - pos["shares"] * pos["entry"] * (1 + fee)
    trade = {**pos, "exit": price, "exit_date": date, "reason": reason,
             "pnl": round(pnl, 2),
             "pnl_pct": round(100 * (price / pos["entry"] - 1), 2)}
    account.history.append(trade)
    return trade
