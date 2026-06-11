"""Tests fonctionnels des 10 modules sur données synthétiques (déterministes, hors-ligne)."""

import numpy as np
import pandas as pd
import pytest

from quantlab import backtest as bt
from quantlab import drawdown as dd
from quantlab import montecarlo as mc
from quantlab import multifactor as mf
from quantlab import optimize as opt
from quantlab import portfolio as pf
from quantlab import regime as rg
from quantlab import risk as rk
from quantlab import setups as st
from quantlab.data import synthetic_ohlcv
from quantlab.strategies import ALL_STRATEGIES, describe_all, get_strategy


@pytest.fixture(scope="module")
def df():
    return synthetic_ohlcv("TEST", years=10, seed=7)


@pytest.fixture(scope="module")
def prices():
    return {s: synthetic_ohlcv(s, years=8, seed=i) for i, s in
            enumerate(["AAA", "BBB", "CCC", "DDD"])}


@pytest.fixture(scope="module")
def result(df):
    return bt.run_backtest(df, get_strategy("trend"), 10_000, 1.0, symbol="TEST")


def test_synthetic_data(df):
    assert len(df) > 2400
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-9).all()
    assert (df["Close"] > 0).all()


def test_module1_strategies(df):
    text = describe_all()
    assert "STRATÉGIE 3" in text
    for key in ALL_STRATEGIES:
        sig = get_strategy(key).generate_signals(df)
        assert set(sig.columns) == {"entry", "exit", "stop", "target"}
        assert sig["entry"].sum() > 0, f"{key} ne génère aucun signal"


def test_module2_backtest(result):
    m = result.metrics
    assert m["n_trades"] > 5
    assert np.isfinite(m["sharpe"]) and np.isfinite(m["cagr"])
    assert -1.0 <= m["max_drawdown"] <= 0.0
    assert 0.0 <= m["win_rate"] <= 1.0
    assert (result.equity > 0).all()
    assert "MODULE 2" in bt.report(result)


def test_backtest_no_lookahead(df):
    """L'équité au jour J ne doit pas dépendre des données après J."""
    strat = get_strategy("trend")
    full = bt.run_backtest(df, strat, 10_000, 1.0)
    cut = len(df) - 200
    partial = bt.run_backtest(df.iloc[:cut], strat, 10_000, 1.0)
    # Identique sauf si une position était ouverte (clôture forcée en fin d'historique)
    diff = (full.equity.iloc[: cut - 1] - partial.equity.iloc[: cut - 1]).abs().max()
    assert diff < 1e-6


def test_risk_sizing(result):
    """Aucun trade ne doit perdre beaucoup plus que le risque défini (hors gaps)."""
    t = result.trades
    losses_r = t.loc[t["pnl"] < 0, "r_multiple"].dropna()
    if len(losses_r):
        assert losses_r.min() > -2.0  # tolérance gaps : jamais pire que -2R


def test_module3_riskreward(result):
    text = rk.report(result)
    assert "MODULE 3" in text and "réduire le risque" in text


def test_module4_regime(df):
    r = rg.detect(df)
    assert r.trend in ("bull", "bear", "sideways")
    assert r.volatility in ("low", "normal", "high")
    assert "MODULE 4" in rg.report(df, "TEST")


def test_module5_multifactor(prices):
    scores = mf.composite(prices)
    assert len(scores) == 4
    assert abs(scores["score"].mean()) < 1.0  # z-scores ~centrés
    port = mf.example_portfolio(scores)
    assert abs(port["allocation_%"].sum() - 100) < 0.5
    assert "MODULE 5" in mf.report(prices, {s: "synthetic" for s in prices})


def test_module6_optimize(df):
    best, table = opt.grid_search(df, "meanrev")
    assert set(best) == {"rsi_buy", "rsi_exit", "stop_mult"}
    assert len(table) > 10


def test_module7_portfolio(prices):
    p = pf.build(prices, "medium")
    total = p["alloc"]["allocation_%"].sum() + p["cash_pct"]
    assert abs(total - 100) < 1.0
    assert p["expected_vol"] <= 0.125 + 0.02  # vol cible respectée (marge)
    assert "MODULE 7" in pf.report(prices)


def test_module8_setups(prices):
    text = st.report(prices)
    assert "MODULE 8" in text


def test_module9_montecarlo(result):
    s = mc.simulate(result, n_paths=500, horizon_days=126)
    assert 0.0 <= s["p_loss"] <= 1.0
    assert s["worst"] <= s["p5"] <= s["median"] <= s["p95"] <= s["best"]
    assert s["mdd_median"] <= 0.0
    assert "VERDICT" in mc.report(result, n_paths=500)


def test_module10_drawdown(result):
    eps = dd.drawdown_episodes(result.equity)
    assert (eps["profondeur_%"] < 0).all() if len(eps) else True
    text = dd.report(result)
    assert "MODULE 10" in text and "position sizing" in text


def test_cli_smoke(monkeypatch, capsys):
    from quantlab import cli
    monkeypatch.setattr(cli, "load",
                        lambda sym, years=10, **k: (synthetic_ohlcv(sym, years, seed=3),
                                                    "synthetic"))
    assert cli.main(["strategies"]) == 0
    assert cli.main(["backtest", "--strategy", "meanrev", "--symbol", "X"]) == 0
    assert cli.main(["regime", "--symbol", "X"]) == 0
    out = capsys.readouterr().out
    assert "MODULE 4" in out
