"""Module 2 — Moteur de backtesting + métriques (CAGR, Sharpe, max drawdown, win rate)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series          # courbe d'équité quotidienne
    trades: pd.DataFrame       # un enregistrement par trade clôturé
    metrics: dict
    symbol: str
    strategy_name: str
    source: str = ""

    @property
    def daily_returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def run_backtest(df: pd.DataFrame, strategy: Strategy, capital: float = 10_000.0,
                 risk_pct: float = 1.0, commission_bps: float = 5.0,
                 symbol: str = "", source: str = "") -> BacktestResult:
    """Backtest bar-par-bar, long-only.

    - Signal calculé à la clôture du jour J → exécution à l'open de J+1.
    - Stop/target fixés à l'entrée (distances ATR translatées au prix d'exécution).
    - Taille de position : (capital_courant × risque%) / distance au stop.
    - Commission + slippage : commission_bps points de base par côté.
    """
    sig = strategy.generate_signals(df)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    entry_sig = sig["entry"].to_numpy(bool)
    exit_sig = sig["exit"].to_numpy(bool)
    stop_lvl = sig["stop"].to_numpy(float)
    tgt_lvl = sig["target"].to_numpy(float)
    n = len(df)
    fee = commission_bps / 10_000.0

    cash = capital
    shares = 0.0
    stop = target = entry_price = np.nan
    entry_i = -1
    pending_entry = pending_exit = False
    equity = np.empty(n)
    trades: list[dict] = []

    def close_trade(i: int, price: float, reason: str):
        nonlocal cash, shares, stop, target, entry_price, entry_i
        proceeds = shares * price * (1 - fee)
        cost = shares * entry_price * (1 + fee)
        pnl = proceeds - cost
        risk = shares * (entry_price - stop) if entry_price > stop else np.nan
        trades.append({
            "entry_date": df.index[entry_i], "exit_date": df.index[i],
            "entry": round(entry_price, 4), "exit": round(price, 4),
            "shares": round(shares, 6), "pnl": round(pnl, 2),
            "pnl_pct": round(100 * (price / entry_price - 1), 2),
            "r_multiple": round(pnl / risk, 2) if risk and risk > 0 else np.nan,
            "bars_held": i - entry_i, "reason": reason,
        })
        cash += proceeds
        shares = 0.0
        stop = target = entry_price = np.nan
        entry_i = -1

    for i in range(n):
        # 1) Exécutions à l'open (ordres décidés à la clôture précédente)
        if shares > 0 and pending_exit:
            close_trade(i, o[i], "signal")
            pending_exit = False
        if shares == 0 and pending_entry:
            j = i - 1  # bar du signal
            stop_dist = c[j] - stop_lvl[j]
            if np.isfinite(stop_dist) and stop_dist > 0:
                entry_price = o[i]
                stop = entry_price - stop_dist
                target = entry_price + (tgt_lvl[j] - c[j]) if np.isfinite(tgt_lvl[j]) else np.nan
                risk_amount = cash * risk_pct / 100.0
                size = risk_amount / stop_dist
                max_size = cash / (entry_price * (1 + fee))  # pas de levier
                shares = min(size, max_size)
                if shares > 0:
                    cash -= shares * entry_price * (1 + fee)
                    entry_i = i
                else:
                    entry_price = stop = target = np.nan
            pending_entry = False

        # 2) Stops / targets en intraday
        if shares > 0:
            if l[i] <= stop:
                px = min(o[i], stop)  # gap baissier : exécuté à l'open
                close_trade(i, px, "stop")
            elif np.isfinite(target) and h[i] >= target:
                px = max(o[i], target)
                close_trade(i, px, "target")

        # 3) Signaux à la clôture pour le bar suivant
        if shares > 0 and exit_sig[i]:
            pending_exit = True
        elif shares == 0 and entry_sig[i]:
            pending_entry = True

        equity[i] = cash + shares * c[i]

    # Clôture forcée en fin d'historique (position encore ouverte)
    if shares > 0:
        close_trade(n - 1, c[-1], "fin_historique")
        equity[-1] = cash

    eq = pd.Series(equity, index=df.index, name="equity")
    trades_df = pd.DataFrame(trades)
    return BacktestResult(
        equity=eq, trades=trades_df,
        metrics=compute_metrics(eq, trades_df, capital),
        symbol=symbol, strategy_name=strategy.spec.name, source=source,
    )


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, capital: float) -> dict:
    rets = equity.pct_change().dropna()
    # Années calculées sur les dates réelles : un calendrier mixte actions (5j/7)
    # + crypto (7j/7) fausserait le compte de barres / 252
    if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 1:
        n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    else:
        n_years = max(len(equity) / 252.0, 1e-9)
    total_return = float(equity.iloc[-1] / capital - 1)
    cagr = float((equity.iloc[-1] / capital) ** (1 / n_years) - 1)
    vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else 0.0
    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    if len(trades):
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        win_rate = len(wins) / len(trades)
        gross_win = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
        avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
        expectancy_r = float(trades["r_multiple"].mean()) if trades["r_multiple"].notna().any() else np.nan
        avg_bars = float(trades["bars_held"].mean())
    else:
        win_rate = profit_factor = avg_win = avg_loss = expectancy_r = avg_bars = 0.0

    return {
        "total_return": total_return, "cagr": cagr, "volatility": vol,
        "sharpe": sharpe, "sortino": sortino, "max_drawdown": mdd, "calmar": calmar,
        "win_rate": win_rate, "profit_factor": profit_factor,
        "n_trades": int(len(trades)), "avg_win": avg_win, "avg_loss": avg_loss,
        "expectancy_r": expectancy_r, "avg_bars_held": avg_bars,
        "final_equity": float(equity.iloc[-1]), "years": n_years,
    }


def metrics_table(metrics: dict) -> pd.DataFrame:
    m = metrics
    rows = [
        ("Équité finale", f"${m['final_equity']:,.0f}"),
        ("Rendement total", f"{m['total_return'] * 100:+.1f}%"),
        ("CAGR", f"{m['cagr'] * 100:+.2f}%"),
        ("Volatilité annualisée", f"{m['volatility'] * 100:.1f}%"),
        ("Sharpe", f"{m['sharpe']:.2f}"),
        ("Sortino", f"{m['sortino']:.2f}"),
        ("Max drawdown", f"{m['max_drawdown'] * 100:.1f}%"),
        ("Calmar", f"{m['calmar']:.2f}"),
        ("Trades", f"{m['n_trades']}"),
        ("Win rate", f"{m['win_rate'] * 100:.1f}%"),
        ("Profit factor", f"{m['profit_factor']:.2f}"),
        ("Gain moyen", f"${m['avg_win']:,.0f}"),
        ("Perte moyenne", f"${m['avg_loss']:,.0f}"),
        ("Espérance (R)", f"{m['expectancy_r']:.2f}R" if np.isfinite(m["expectancy_r"]) else "n/a"),
        ("Durée moyenne (jours)", f"{m['avg_bars_held']:.0f}"),
    ]
    return pd.DataFrame(rows, columns=["Métrique", "Valeur"])


def yearly_returns(equity: pd.Series) -> pd.Series:
    yearly = equity.resample("YE").last()
    start = equity.iloc[0]
    prev = yearly.shift(1)
    prev.iloc[0] = start
    out = (yearly / prev - 1) * 100
    out.index = out.index.year
    return out.round(1)


def report(res: BacktestResult) -> str:
    lines = [
        "=" * 78,
        "MODULE 2 — BACKTESTING",
        f"Stratégie : {res.strategy_name}",
        f"Actif : {res.symbol} | Source données : {res.source} | "
        f"Période : {res.equity.index[0].date()} → {res.equity.index[-1].date()} "
        f"({res.metrics['years']:.1f} ans)",
        "=" * 78,
        "",
        metrics_table(res.metrics).to_string(index=False),
        "",
        "Rendements annuels (%) :",
        yearly_returns(res.equity).to_string(),
    ]
    yr = yearly_returns(res.equity)
    if len(yr) >= 2:
        lines += [
            "",
            f"Meilleure année : {yr.idxmax()} ({yr.max():+.1f}%) — la stratégie performe "
            "le mieux quand le marché correspond à ses conditions idéales (voir module 1).",
            f"Pire année : {yr.idxmin()} ({yr.min():+.1f}%) — typiquement un marché en range "
            "ou un retournement rapide qui multiplie les faux signaux.",
        ]
    m = res.metrics
    lines += [
        "",
        "Ce qui casse la stratégie :",
        "  • Changement brutal de régime (ex. krach) avant que les signaux de sortie ne réagissent",
        "  • Longs marchés sans tendance : accumulation de petites pertes (whipsaw)",
        f"  • Drawdown max observé : {m['max_drawdown'] * 100:.1f}% — à comparer à votre tolérance",
    ]
    return "\n".join(lines)
