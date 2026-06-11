"""Simulation du FONDS complet — multi-actifs, multi-stratégies, capital partagé.

Reproduit en backtest la logique exacte du mode live (live.py) :
  • univers de plusieurs actifs, les 3 stratégies en parallèle
  • une position max par actif, plafond global de positions simultanées
  • sizing : risque % de l'équité TOTALE courante par trade
  • exécution à l'open du lendemain, stops/targets intraday, commissions

C'est l'entraînement demandé avant tout investissement réel : « comment notre
hedge fund s'en serait sorti avec ce capital sur les N dernières années ? »
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult, compute_metrics
from .strategies import ALL_STRATEGIES, get_strategy


def run_fund(prices: dict[str, pd.DataFrame], capital: float = 100_000.0,
             risk_pct: float = 1.0, max_positions: int = 5,
             commission_bps: float = 5.0,
             strategy_keys: tuple[str, ...] = ("trend", "meanrev", "breakout"),
             ) -> BacktestResult:
    fee = commission_bps / 10_000.0

    # Pré-calculs par actif : arrays OHLC + signaux de chaque stratégie
    arr: dict[str, dict] = {}
    for sym, df in prices.items():
        d = {"dates": df.index,
             "o": df["Open"].to_numpy(float), "h": df["High"].to_numpy(float),
             "l": df["Low"].to_numpy(float), "c": df["Close"].to_numpy(float),
             "idx": {ts: i for i, ts in enumerate(df.index)}, "sig": {}}
        for key in strategy_keys:
            s = get_strategy(key).generate_signals(df)
            d["sig"][key] = {
                "entry": s["entry"].to_numpy(bool), "exit": s["exit"].to_numpy(bool),
                "stop": s["stop"].to_numpy(float), "target": s["target"].to_numpy(float),
            }
        arr[sym] = d

    calendar = sorted(set().union(*[set(d["dates"]) for d in arr.values()]))
    cash = capital
    positions: dict[str, dict] = {}
    pending_entry: dict[str, str] = {}   # sym -> clé stratégie (signal à la veille)
    pending_exit: set[str] = set()
    last_close: dict[str, float] = {}
    equity_vals: list[float] = []
    trades: list[dict] = []

    def mark_equity() -> float:
        return cash + sum(p["shares"] * last_close.get(s, p["entry"])
                          for s, p in positions.items())

    def close_position(sym: str, i: int, price: float, date, reason: str):
        nonlocal cash
        p = positions.pop(sym)
        proceeds = p["shares"] * price * (1 - fee)
        cost = p["shares"] * p["entry"] * (1 + fee)
        pnl = proceeds - cost
        risk = p["shares"] * (p["entry"] - p["stop"]) if p["entry"] > p["stop"] else np.nan
        trades.append({
            "symbol": sym, "strategy": p["strategy"],
            "entry_date": p["entry_date"], "exit_date": date,
            "entry": round(p["entry"], 4), "exit": round(price, 4),
            "shares": round(p["shares"], 6), "pnl": round(pnl, 2),
            "pnl_pct": round(100 * (price / p["entry"] - 1), 2),
            "r_multiple": round(pnl / risk, 2) if risk and risk > 0 else np.nan,
            "bars_held": i - p["entry_i"], "reason": reason,
        })
        cash += proceeds

    for t in calendar:
        eq_prev = mark_equity()
        for sym, d in arr.items():
            i = d["idx"].get(t)
            if i is None:
                continue

            # 1) Exécutions à l'open (ordres décidés à la clôture précédente)
            if sym in pending_exit and sym in positions:
                close_position(sym, i, d["o"][i], t, "signal")
                pending_exit.discard(sym)
            if sym in pending_entry and sym not in positions and i > 0:
                key = pending_entry.pop(sym)
                if len(positions) < max_positions:
                    sg = d["sig"][key]
                    j = i - 1
                    stop_dist = d["c"][j] - sg["stop"][j]
                    if np.isfinite(stop_dist) and stop_dist > 0:
                        entry = d["o"][i]
                        stop = entry - stop_dist
                        target = (entry + (sg["target"][j] - d["c"][j])
                                  if np.isfinite(sg["target"][j]) else np.nan)
                        shares = min(eq_prev * risk_pct / 100.0 / stop_dist,
                                     cash / (entry * (1 + fee)))
                        if shares > 0:
                            cash -= shares * entry * (1 + fee)
                            positions[sym] = {
                                "shares": shares, "entry": entry, "stop": stop,
                                "target": target, "strategy": key,
                                "entry_date": t, "entry_i": i,
                            }

            # 2) Stops / targets intraday
            if sym in positions:
                p = positions[sym]
                if d["l"][i] <= p["stop"]:
                    close_position(sym, i, min(d["o"][i], p["stop"]), t, "stop")
                elif np.isfinite(p["target"]) and d["h"][i] >= p["target"]:
                    close_position(sym, i, max(d["o"][i], p["target"]), t, "target")

            # 3) Signaux à la clôture pour le bar suivant
            last_close[sym] = d["c"][i]
            if sym in positions:
                if d["sig"][positions[sym]["strategy"]]["exit"][i]:
                    pending_exit.add(sym)
            elif sym not in pending_entry:
                for key in strategy_keys:
                    if d["sig"][key]["entry"][i]:
                        pending_entry[sym] = key
                        break

        equity_vals.append(mark_equity())

    # Clôture forcée des positions restantes au dernier cours connu
    final_date = calendar[-1]
    for sym in list(positions):
        i = len(arr[sym]["c"]) - 1
        close_position(sym, i, last_close[sym], final_date, "fin_historique")
    equity_vals[-1] = cash

    eq = pd.Series(equity_vals, index=pd.DatetimeIndex(calendar), name="equity")
    trades_df = pd.DataFrame(trades)
    return BacktestResult(
        equity=eq, trades=trades_df,
        metrics=compute_metrics(eq, trades_df, capital),
        symbol=" + ".join(prices), strategy_name="FONDS multi-stratégies",
    )


def report(res: BacktestResult, capital: float) -> str:
    from .backtest import metrics_table, yearly_returns

    m = res.metrics
    lines = [
        "=" * 78,
        "SIMULATION DU FONDS — ENTRAÎNEMENT SUR HISTORIQUE",
        f"Univers : {res.symbol}",
        f"Capital de départ : {capital:,.0f} $ | Stratégies : trend + meanrev + breakout",
        f"Période : {res.equity.index[0].date()} → {res.equity.index[-1].date()} "
        f"({m['years']:.1f} ans)",
        "=" * 78,
        "",
        metrics_table(m).to_string(index=False),
        "",
        "Rendements annuels (%) :",
        yearly_returns(res.equity).to_string(),
    ]
    if len(res.trades):
        by_sym = res.trades.groupby("symbol")["pnl"].agg(["count", "sum"]).round(0)
        by_strat = res.trades.groupby("strategy")["pnl"].agg(["count", "sum"]).round(0)
        by_sym.columns = by_strat.columns = ["trades", "PnL_$"]
        lines += [
            "",
            "Contribution par actif :", by_sym.to_string(),
            "",
            "Contribution par stratégie :", by_strat.to_string(),
        ]
    lines += [
        "",
        f"Verdict chiffré : {capital:,.0f} $ → {m['final_equity']:,.0f} $ "
        f"({m['total_return'] * 100:+.1f}%), pire creux {m['max_drawdown'] * 100:.1f}%.",
        "Rappel : performance simulée, frais de 5 bps inclus, aucune garantie future.",
    ]
    return "\n".join(lines)
