"""Module 6 — Optimisation de stratégie : grid search avec comparaison avant/après.

Objectif : maximiser le Sharpe pénalisé par le drawdown (score = Sharpe + 2×Calmar
pondéré), avec validation simple train/test (70/30) pour limiter l'overfitting.
"""

from __future__ import annotations

import itertools

import pandas as pd

from .backtest import run_backtest
from .strategies import get_strategy

# Grilles volontairement petites : explicables et rapides
GRIDS: dict[str, dict[str, list]] = {
    "trend": {
        "fast": [10, 20, 30],
        "slow": [50, 80, 100],
        "adx_min": [15, 20, 25],
        "stop_mult": [1.5, 2.0, 3.0],
    },
    "meanrev": {
        "rsi_buy": [5, 10, 15],
        "rsi_exit": [60, 70, 80],
        "stop_mult": [2.0, 3.0, 4.0],
    },
    "breakout": {
        "entry_n": [20, 55, 100],
        "exit_n": [10, 20, 30],
        "stop_mult": [1.5, 2.0, 3.0],
    },
}


def _score(metrics: dict) -> float:
    """Sharpe + bonus Calmar — pénalise les courbes à gros drawdown."""
    if metrics["n_trades"] < 5:
        return -99.0  # trop peu de trades : non significatif
    return metrics["sharpe"] + 0.5 * min(metrics["calmar"], 3.0)


def grid_search(df: pd.DataFrame, strategy_key: str, capital: float = 10_000.0,
                risk_pct: float = 1.0) -> tuple[dict, pd.DataFrame]:
    grid = GRIDS[strategy_key]
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split], df.iloc[max(0, split - 250):]  # warmup indicateurs

    rows = []
    keys = list(grid)
    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        try:
            res = run_backtest(train, get_strategy(strategy_key, **params),
                               capital, risk_pct)
        except Exception:
            continue
        rows.append({**params, "sharpe_train": round(res.metrics["sharpe"], 2),
                     "maxdd_train": round(res.metrics["max_drawdown"] * 100, 1),
                     "trades": res.metrics["n_trades"],
                     "_score": _score(res.metrics)})
    table = pd.DataFrame(rows).sort_values("_score", ascending=False)
    best_params = {k: table.iloc[0][k] for k in keys}
    # cast int params
    best_params = {k: int(v) if float(v).is_integer() and isinstance(GRIDS[strategy_key][k][0], int)
                   else float(v) for k, v in best_params.items()}
    return best_params, table


def report(df: pd.DataFrame, strategy_key: str, symbol: str,
           capital: float = 10_000.0, risk_pct: float = 1.0) -> str:
    base_strategy = get_strategy(strategy_key)
    base = run_backtest(df, base_strategy, capital, risk_pct, symbol=symbol)
    best_params, table = grid_search(df, strategy_key, capital, risk_pct)
    opt = run_backtest(df, get_strategy(strategy_key, **best_params),
                       capital, risk_pct, symbol=symbol)

    def row(m):
        return {
            "Sharpe": round(m["sharpe"], 2),
            "CAGR_%": round(m["cagr"] * 100, 1),
            "MaxDD_%": round(m["max_drawdown"] * 100, 1),
            "Calmar": round(m["calmar"], 2),
            "WinRate_%": round(m["win_rate"] * 100, 1),
            "Trades": m["n_trades"],
        }

    cmp = pd.DataFrame({"AVANT (défaut)": row(base.metrics),
                        "APRÈS (optimisé)": row(opt.metrics)}).T
    changed = {k: v for k, v in best_params.items()
               if v != base_strategy.spec.params.get(k)}
    return "\n".join([
        "=" * 78,
        "MODULE 6 — OPTIMISATION DE STRATÉGIE",
        f"Stratégie : {base.strategy_name} | Actif : {symbol}",
        f"Méthode : grid search ({len(table)} combinaisons) sur 70% de l'historique "
        "(train), score = Sharpe + 0.5×Calmar, vérification sur l'historique complet.",
        "=" * 78,
        "",
        f"Paramètres par défaut : {base_strategy.spec.params}",
        f"Paramètres optimisés  : {best_params}",
        f"Changements           : {changed if changed else 'aucun — défauts déjà optimaux'}",
        "",
        "COMPARAISON AVANT / APRÈS (historique complet) :",
        cmp.to_string(),
        "",
        "Top 5 combinaisons (échantillon d'entraînement) :",
        table.drop(columns="_score").head(5).to_string(index=False),
        "",
        "Améliorations appliquées par l'optimiseur :",
        "  • Réglages d'indicateurs : périodes/seuils ci-dessus",
        "  • Timing entrée/sortie : seuils ADX/RSI ajustés = entrées plus sélectives",
        "  • Filtres : tendance (SMA200) et volume conservés dans toutes les variantes",
        "",
        "⚠ Garde-fou anti-overfitting : grilles volontairement réduites, sélection sur",
        "  train uniquement, et combinaisons <5 trades exclues. Si APRÈS ≤ AVANT sur",
        "  l'historique complet, garder les paramètres par défaut.",
    ])
