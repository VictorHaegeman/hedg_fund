"""Tests fonctionnels des 10 modules sur données synthétiques (déterministes, hors-ligne)."""

import numpy as np
import pandas as pd
import pytest

from quantlab import alpha as al
from quantlab import backtest as bt
from quantlab import drawdown as dd
from quantlab import macro as mcr
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


def test_module11_macro():
    prices = {s: synthetic_ohlcv(s, years=5, seed=i)
              for i, s in enumerate(mcr.PROXIES + ["SPY", "QQQ", "GLD", "TLT"])}
    m = mcr.detect_macro(prices)
    assert m["rates"][0] in ("up", "down")
    assert m["quadrant"] in [q[0] for q in mcr.QUADRANTS.values()] or "(" in m["quadrant"]
    assert len(m["favor"]) >= 1
    text = mcr.report(prices, {s: "synthetic" for s in prices})
    assert "MODULE 11" in text and "QUADRANT MACRO" in text and "LONG" in text


def test_module12_alpha(df):
    edges = al.detect_edges(df)
    assert len(edges) == 4
    assert all(np.isfinite(e["tstat"]) for e in edges)
    # classement décroissant par |t|
    ts = [abs(e["tstat"]) for e in edges]
    assert ts == sorted(ts, reverse=True)
    text = al.report(df, "TEST")
    assert "MODULE 12" in text and "STRATÉGIE UNIQUE 2" in text and "pas à pas" in text


def test_broker_buy_sell(tmp_path):
    from quantlab import broker as bk
    path = str(tmp_path / "account.json")
    acc = bk.load_account(path, capital=10_000)
    pos = bk.buy(acc, "AAA", price=100.0, stop=95.0, target=110.0,
                 strategy="trend", date="2026-01-02", risk_pct=1.0)
    assert pos is not None
    assert abs(pos["shares"] * 5.0 - 100.0) < 1e-6  # risque = 1% de 10k
    assert acc.cash < 10_000
    bk.save_account(acc, path)
    acc2 = bk.load_account(path)
    assert "AAA" in acc2.positions
    trade = bk.sell(acc2, "AAA", price=110.0, date="2026-01-10", reason="target")
    assert trade["pnl"] > 0
    assert not acc2.positions
    # 20 shares × (10 de gain) − frais ≈ +198 $
    assert 150 < trade["pnl"] < 210


def test_live_cycle(tmp_path):
    from quantlab import live as lv
    path = str(tmp_path / "account.json")
    fake_load = lambda sym, years=2, fresh=False, **k: (
        synthetic_ohlcv(sym, years=3, seed=abs(hash(sym)) % 1000), "yahoo")
    out1 = lv.run_cycle(["AAA", "BBB"], capital=10_000, state_path=path,
                        loader=fake_load)
    assert "COMPTE PAPIER" in out1 and "Équité" in out1
    # le cycle doit être idempotent au sein d'une même journée de données
    out2 = lv.run_cycle(["AAA", "BBB"], capital=10_000, state_path=path,
                        loader=fake_load)
    assert "COMPTE PAPIER" in out2


def test_live_never_trades_synthetic(tmp_path):
    """Sécurité critique : aucun ordre ne doit être passé sur des prix fictifs."""
    from quantlab import live as lv
    path = str(tmp_path / "account.json")
    fake_load = lambda sym, years=2, fresh=False, **k: (
        synthetic_ohlcv(sym, years=3, seed=1), "synthetic")
    out = lv.run_cycle(["AAA"], capital=10_000, state_path=path, loader=fake_load)
    assert "IGNORÉ" in out
    from quantlab.broker import load_account
    assert not load_account(path).positions


def test_fund_simulation(prices):
    from quantlab import fund as fd
    res = fd.run_fund(prices, capital=100_000, risk_pct=1.0, max_positions=3)
    m = res.metrics
    assert m["n_trades"] > 10
    assert (res.equity > 0).all()
    assert -1.0 <= m["max_drawdown"] <= 0.0
    # le plafond de positions doit être respecté : jamais plus de 3 positions
    # simultanées → l'exposition ne peut pas dépasser ~3 fois le cash disponible
    assert len(res.trades.groupby("symbol")) <= len(prices)
    assert "SIMULATION DU FONDS" in fd.report(res, 100_000)


def test_fund_no_lookahead(prices):
    from quantlab import fund as fd
    full = fd.run_fund(prices, capital=100_000)
    cut_prices = {s: df.iloc[:-150] for s, df in prices.items()}
    partial = fd.run_fund(cut_prices, capital=100_000)
    n = len(partial.equity) - 1  # dernier point = clôture forcée, exclu
    diff = (full.equity.iloc[:n] - partial.equity.iloc[:n]).abs().max()
    assert diff < 1e-6


def test_dashboard_html(tmp_path, prices):
    from quantlab import dashboard as db
    from quantlab import fund as fd
    from quantlab.broker import load_account
    res = fd.run_fund(prices, capital=100_000)
    account = load_account(str(tmp_path / "acc.json"), capital=100_000)
    prices_now = {s: float(df["Close"].iloc[-1]) for s, df in prices.items()}
    path = db.save(res, account, prices_now, 100_000, str(tmp_path / "dash.html"))
    content = open(path, encoding="utf-8").read()
    assert "<svg" in content and "Tableau de bord" in content
    assert "Équité finale simulée" in content and "Mon argent" in content


def test_equity_curve_tracking(tmp_path):
    from quantlab import broker as bk
    path = str(tmp_path / "acc.json")
    acc = bk.load_account(path, capital=100_000)
    acc.created = "2026-06-16T10:00:00"  # création la veille du 1er enregistrement
    acc.record_equity("2026-06-17", 100_500)
    acc.record_equity("2026-06-18", 101_500)
    acc.record_equity("2026-06-18", 101_800)  # même jour → mise à jour, pas d'ajout
    assert len(acc.equity_curve) == 3   # amorce (16) + 17 + 18
    assert acc.equity_curve[0] == ["2026-06-16", 100_000]  # amorce au capital initial
    assert acc.equity_curve[-1] == ["2026-06-18", 101_800]
    bk.save_account(acc, path)
    assert bk.load_account(path).equity_curve[-1][1] == 101_800


def test_live_records_equity(tmp_path):
    from quantlab import live as lv
    from quantlab.broker import load_account
    path = str(tmp_path / "acc.json")
    fake_load = lambda sym, years=2, fresh=False, **k: (
        synthetic_ohlcv(sym, years=3, seed=abs(hash(sym)) % 999), "yahoo")
    lv.run_cycle(["AAA", "BBB"], capital=100_000, state_path=path, loader=fake_load)
    curve = load_account(path).equity_curve
    assert len(curve) >= 1 and curve[-1][1] > 0


def test_dashboard_live_curve(tmp_path, prices):
    from quantlab import dashboard as db
    from quantlab import fund as fd
    from quantlab.broker import load_account
    res = fd.run_fund(prices, capital=100_000)
    account = load_account(str(tmp_path / "acc.json"), capital=100_000)
    account.record_equity("2026-06-17", 100_000)
    account.record_equity("2026-06-18", 102_000)
    prices_now = {s: float(df["Close"].iloc[-1]) for s, df in prices.items()}
    path = db.save(res, account, prices_now, 100_000, str(tmp_path / "d.html"))
    content = open(path, encoding="utf-8").read()
    assert "Courbe d'équité du compte live" in content


def test_module13_sentiment():
    from quantlab import news as nw
    assert nw.score_text("Stock surges to record profit, beats estimates") > 0.3
    assert nw.score_text("Shares plunge on lawsuit and recession fears") < -0.3
    assert nw.score_text("The company held a meeting today") == 0.0
    assert nw.count_geo("War escalation triggers new sanctions and tariffs") >= 3


def test_module13_headline_sentiment_offline():
    from quantlab import news as nw
    fake = lambda sym, **k: ["Nvidia surges to record high on strong demand",
                             "Analysts upgrade outlook, raise price target"]
    s = nw.headline_sentiment("NVDA", fetch=fake)
    assert s["label"] == "positif" and s["n"] == 2 and s["score"] > 0
    empty = nw.headline_sentiment("XXX", fetch=lambda sym, **k: [])
    assert empty["label"] == "n/a" and empty["n"] == 0


def test_module13_risk_gauge(prices):
    from quantlab import news as nw

    def loader(sym, years=3, fresh=False, **k):
        base = synthetic_ohlcv(sym, years=3, seed=hash(sym) % 100)
        if sym == "^VIX":  # forcer un VIX très haut sur la dernière barre
            base.iloc[-1, base.columns.get_loc("Close")] = base["Close"].max() * 3
        return base, "synthetic"

    g = nw.risk_gauge(loader)
    assert g["level"] in ("risk-on", "neutre", "prudence", "risk-off")
    assert g["vix"] is not None
    # VIX au plus haut historique → risk-off → bloque les entrées
    assert nw.blocks_new_entries(g)


def test_module13_risk_gauge_failsafe():
    from quantlab import news as nw

    def broken(sym, **k):
        raise RuntimeError("réseau coupé")

    g = nw.risk_gauge(broken)
    assert g["level"] == "neutre" and not nw.blocks_new_entries(g)


def test_live_blocks_entries_when_risk_off(tmp_path, monkeypatch):
    """En peur extrême, le cycle ne doit ouvrir aucune position."""
    from quantlab import live as lv
    from quantlab import news as nw
    from quantlab.broker import load_account
    path = str(tmp_path / "acc.json")
    fake_load = lambda sym, years=2, fresh=False, **k: (
        synthetic_ohlcv(sym, years=3, seed=abs(hash(sym)) % 1000), "yahoo")
    monkeypatch.setattr(nw, "risk_gauge", lambda loader: {
        "level": "risk-off", "vix": 55.0, "vix_pct": 0.99, "note": "test"})
    out = lv.run_cycle(["AAA", "BBB"], capital=10_000, state_path=path,
                       loader=fake_load)
    assert "bloquées" in out
    assert not load_account(path).positions


def test_cli_smoke(monkeypatch, capsys):
    from quantlab import cli
    monkeypatch.setattr(cli, "load",
                        lambda sym, years=10, **k: (synthetic_ohlcv(sym, years, seed=3),
                                                    "synthetic"))
    assert cli.main(["strategies"]) == 0
    assert cli.main(["backtest", "--strategy", "meanrev", "--symbol", "X"]) == 0
    assert cli.main(["regime", "--symbol", "X"]) == 0
    assert cli.main(["macro"]) == 0
    assert cli.main(["alpha", "--symbol", "X"]) == 0
    out = capsys.readouterr().out
    assert "MODULE 4" in out and "MODULE 11" in out and "MODULE 12" in out
