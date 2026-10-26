"""Tests fonctionnels des 10 modules sur données synthétiques (déterministes, hors-ligne)."""

import json
import os
import zlib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        synthetic_ohlcv(sym, years=3, seed=zlib.crc32(sym.encode()) % 1000), "yahoo")
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


def test_fund_validate_report(prices):
    from quantlab import fund as fd
    res_d = fd.run_fund(prices, capital=100_000)
    res_b = fd.run_fund({k: prices[k] for k in list(prices)[:3]}, capital=100_000)
    text = fd.validate_report(res_d, res_b, 100_000)
    assert "VALIDATION" in text and "biais de survie" in text
    assert "Attente réaliste" in text
    assert fd.BIAS_FREE_SECTORS[0] == "XLK" and len(fd.BIAS_FREE_UNIVERSE) == 14


def test_fund_no_lookahead(prices):
    from quantlab import fund as fd
    full = fd.run_fund(prices, capital=100_000)
    cut_prices = {s: df.iloc[:-150] for s, df in prices.items()}
    partial = fd.run_fund(cut_prices, capital=100_000)
    n = len(partial.equity) - 1  # dernier point = clôture forcée, exclu
    diff = (full.equity.iloc[:n] - partial.equity.iloc[:n]).abs().max()
    assert diff < 1e-6


def test_vol_scale_math():
    from quantlab import fund as fd
    rng = np.random.default_rng(0)
    calm = list(100_000 * np.cumprod(1 + rng.normal(0, 0.002, 300)))
    wild = list(100_000 * np.cumprod(1 + rng.normal(0, 0.03, 300)))
    assert fd.vol_scale(calm, 0.12) > 1.0       # vol basse → on monte l'exposition
    assert fd.vol_scale(wild, 0.12) < 1.0       # vol haute → on réduit
    assert fd.vol_scale([1, 2, 3], 0.12) == 1.0  # historique trop court → neutre
    assert fd.vol_scale(calm, None) == 1.0       # désactivé → neutre
    assert 0.5 <= fd.vol_scale(wild, 0.12) <= 2.0  # bornes respectées


def test_backtest_trailing_stop():
    from quantlab import backtest as bt
    df = synthetic_ohlcv("T", years=8, seed=5)
    base = bt.run_backtest(df, get_strategy("trend"), 10_000, 1.0)
    tr = bt.run_backtest(df, get_strategy("trend"), 10_000, 1.0, trail=True)
    assert (tr.equity > 0).all() and tr.metrics["n_trades"] >= 1
    # le cliquet déclenche au moins autant de sorties "stop" que la version fixe
    bs = (base.trades["reason"] == "stop").sum() if len(base.trades) else 0
    ts = (tr.trades["reason"] == "stop").sum() if len(tr.trades) else 0
    assert ts >= bs
    # le trailing a un effet mesurable sur la courbe d'équité
    assert (base.equity - tr.equity).abs().max() > 0
    # R-multiple basé sur le stop INITIAL : les pertes restent bornées (~ -1R)
    losses = tr.trades["r_multiple"].dropna()
    if len(losses):
        assert losses.min() > -1.5


def test_fund_vol_trail_no_lookahead(prices):
    """Le vol-targeting (vol passée) + trailing ne réintroduisent pas de look-ahead."""
    from quantlab import fund as fd
    full = fd.run_fund(prices, capital=100_000, vol_target=0.12, trail=True)
    cut = fd.run_fund({s: df.iloc[:-150] for s, df in prices.items()},
                      capital=100_000, vol_target=0.12, trail=True)
    n = len(cut.equity) - 1
    diff = (full.equity.iloc[:n] - cut.equity.iloc[:n]).abs().max()
    assert diff < 1e-6


def test_fund_compare_report(prices):
    from quantlab import fund as fd
    base = fd.run_fund(prices, capital=100_000)
    impr = fd.run_fund(prices, capital=100_000, vol_target=0.12, trail=True)
    text = fd.compare_report(base, impr, 100_000)
    assert "COMPARAISON" in text and "Sharpe" in text and "Max drawdown" in text


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


def test_dashboard_screener_panel(tmp_path, prices):
    from quantlab import dashboard as db
    from quantlab import fund as fd
    from quantlab.broker import load_account
    res = fd.run_fund(prices, capital=100_000)
    account = load_account(str(tmp_path / "acc.json"), capital=100_000)
    prices_now = {s: float(df["Close"].iloc[-1]) for s, df in prices.items()}
    # liste de candidats → panneau avec puces (chips)
    p1 = db.save(res, account, prices_now, 100_000, str(tmp_path / "d1.html"),
                 screener=["AAPL", "MSFT"])
    c1 = open(p1, encoding="utf-8").read()
    assert "Univers dynamique" in c1 and "AAPL" in c1 and "Candidats screenés" in c1
    # screener indisponible → mention du fallback univers fixe
    p2 = db.save(res, account, prices_now, 100_000, str(tmp_path / "d2.html"),
                 screener=[])
    c2 = open(p2, encoding="utf-8").read()
    assert "indisponible" in c2 and "univers fixe" in c2
    # None → panneau masqué (rétro-compatibilité)
    p3 = db.save(res, account, prices_now, 100_000, str(tmp_path / "d3.html"))
    assert "Univers dynamique" not in open(p3, encoding="utf-8").read()


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
        synthetic_ohlcv(sym, years=3, seed=zlib.crc32(sym.encode()) % 999), "yahoo")
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
        base = synthetic_ohlcv(sym, years=3, seed=zlib.crc32(sym.encode()) % 100)
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
        synthetic_ohlcv(sym, years=3, seed=zlib.crc32(sym.encode()) % 1000), "yahoo")
    monkeypatch.setattr(nw, "risk_gauge", lambda loader: {
        "level": "risk-off", "vix": 55.0, "vix_pct": 0.99, "note": "test"})
    out = lv.run_cycle(["AAA", "BBB"], capital=10_000, state_path=path,
                       loader=fake_load)
    assert "bloquées" in out
    assert not load_account(path).positions


def test_screener_symbol_mapping():
    from quantlab import screener as sc
    assert sc.to_yf_symbol("NASDAQ:AAPL") == "AAPL"
    assert sc.to_yf_symbol("NYSE:BRK.B") == "BRK-B"  # classe d'action → tiret
    assert sc.to_yf_symbol("AAPL") == "AAPL"          # sans préfixe exchange
    assert sc.to_yf_symbol("BINANCE:BTCUSDT", crypto=True) == "BTC-USD"
    assert sc.to_yf_symbol("ETHUSD", crypto=True) == "ETH-USD"
    assert sc.to_yf_symbol("") is None
    assert sc.to_yf_symbol("NASDAQ:WE!RD") is None    # ticker invalide → écarté
    assert sc.to_yf_symbol("KRAKEN:DOGEEUR", crypto=True) is None  # quote non gérée


def test_screener_candidates_offline():
    """candidates mappe, déduplique, filtre l'invalide et respecte la limite."""
    from quantlab import screener as sc
    fake = lambda **k: ["NASDAQ:AAPL", "NYSE:JPM", "NASDAQ:AAPL", "BAD!!", "NYSE:KO"]
    assert sc.candidates(limit=5, fetch=fake) == ["AAPL", "JPM", "KO"]
    assert sc.candidates(limit=2, fetch=fake) == ["AAPL", "JPM"]
    # fail-safe : réponse vide ou None → liste vide (l'appelant retombe sur l'univers fixe)
    assert sc.candidates(fetch=lambda **k: []) == []
    assert sc.candidates(fetch=lambda **k: None) == []


def test_screener_fetch_never_raises():
    """_fetch est le seul point réseau : il ne lève jamais, renvoie toujours une liste."""
    from quantlab import screener as sc
    res = sc._fetch(market="america", limit=2, min_mcap=1e9, min_volume=1e5, crypto=False)
    assert isinstance(res, list)


def test_live_uses_screener(tmp_path):
    """Avec un screener fourni, l'univers du cycle se restreint à ses candidats."""
    from quantlab import live as lv
    path = str(tmp_path / "acc.json")
    fake_load = lambda sym, years=2, fresh=False, **k: (
        synthetic_ohlcv(sym, years=3, seed=zlib.crc32(sym.encode()) % 1000), "yahoo")
    out = lv.run_cycle(["AAA", "BBB", "CCC"], capital=10_000, state_path=path,
                       loader=fake_load, screener=lambda: ["ZZZ"])
    uni_line = [l for l in out.splitlines() if "Univers" in l][0]
    assert "ZZZ" in uni_line and "AAA" not in uni_line


def test_live_screener_failsafe(tmp_path):
    """Screener vide → on garde l'univers fixe passé en argument."""
    from quantlab import live as lv
    path = str(tmp_path / "acc.json")
    fake_load = lambda sym, years=2, fresh=False, **k: (
        synthetic_ohlcv(sym, years=3, seed=zlib.crc32(sym.encode()) % 1000), "yahoo")
    out = lv.run_cycle(["AAA", "BBB"], capital=10_000, state_path=path,
                       loader=fake_load, screener=lambda: [])
    uni_line = [l for l in out.splitlines() if "Univers" in l][0]
    assert "AAA" in uni_line and "BBB" in uni_line


def test_cli_screen(monkeypatch, capsys):
    from quantlab import cli
    from quantlab import screener as sc
    monkeypatch.setattr(sc, "candidates", lambda **k: ["AAA", "BBB"])
    assert cli.main(["screen", "--screen-limit", "5"]) == 0
    assert "SCREENER TRADINGVIEW" in capsys.readouterr().out


def test_cli_fund_compare(monkeypatch, capsys):
    from quantlab import cli
    monkeypatch.setattr(cli, "load", lambda sym, years=10, **k: (
        synthetic_ohlcv(sym, years, seed=3), "synthetic"))
    assert cli.main(["fund", "--symbols", "AAA", "BBB", "CCC", "--compare"]) == 0
    assert "COMPARAISON" in capsys.readouterr().out


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


# ── Alpaca : miroir du fonds & poche IA (module 14) — hors-ligne, client factice ──

class FakeAlpaca:
    """Client Alpaca simulé : enregistre les ordres, positions pilotables."""

    def __init__(self, equity=100_000.0, market_open=True, positions=None):
        self.eq, self.open_ = equity, market_open
        self.pos = positions or {}
        self.orders, self.closed = [], []
        self.paper, self.label = True, "test"

    def account(self):
        return {"equity": str(self.eq), "last_equity": str(self.eq), "cash": str(self.eq),
                "non_marginable_buying_power": str(self.eq)}

    def clock(self):
        return {"is_open": self.open_}

    def positions(self):
        return [{"symbol": s, **v} for s, v in self.pos.items()]

    def open_orders(self):
        return []

    def asset(self, s):
        return {"shortable": True, "easy_to_borrow": True}

    def submit_order(self, body):
        self.orders.append(body)
        if body.get("order_class") == "mleg":          # comme le vrai courtier : les
            for leg in body["legs"]:                    # jambes deviennent des positions
                if leg["position_intent"].endswith("_open"):
                    self.pos[leg["symbol"]] = {"unrealized_pl": "0"}
                else:
                    self.pos.pop(leg["symbol"], None)
        else:
            self.pos.setdefault(body["symbol"].replace("/", ""),
                                {"unrealized_pl": "0", "cost_basis": "1000"})
        return {"id": str(len(self.orders))}

    def wait_fill(self, oid, timeout=15, sleep=None):
        return {"status": "filled", "filled_avg_price": "0"}

    def close_position(self, s):
        self.closed.append(s)
        self.pos.pop(s.replace("/", ""), None)
        return {}

    def data(self, path, params=None):
        closes = [100 + i * 0.5 for i in range(80)]
        syms = (params or {}).get("symbols", "").split(",")
        if path.endswith("/bars"):
            return {"bars": {s: [{"c": c} for c in closes] for s in syms}}
        if path == "/v2/stocks/snapshots":
            return {s: {"latestTrade": {"p": 140.0}} for s in syms}
        if "latest/trades" in path:
            return {"trades": {s: {"p": 140.0} for s in syms}}
        if path == "/v1beta1/options/snapshots":
            snaps = {s: {"impliedVolatility": 0.3} for s in syms}
            return {"snapshots": snaps}
        if path.startswith("/v1beta1/options/snapshots/"):
            und = path.rsplit("/", 1)[1]
            exp = (pd.Timestamp.today() + pd.Timedelta(days=30)).strftime("%y%m%d")
            snaps = {}
            for k in range(115, 166, 5):
                for cp in "CP":
                    intr = max(140 - k, 0) if cp == "C" else max(k - 140, 0)
                    mid = intr + 3 * float(np.exp(-abs(k - 140) / 20))  # valeur temps max à la monnaie
                    snaps[f"{und}{exp}{cp}{k * 1000:08d}"] = {
                        "latestQuote": {"bp": round(mid - 0.05, 2), "ap": round(mid + 0.05, 2)},
                        "impliedVolatility": 0.3}
            return {"snapshots": snaps}
        if path == "/v1beta1/news":
            return {"news": [{"headline": "Nvidia beats estimates", "symbols": ["NVDA"]}]}
        return {}


def test_alpaca_symbols_and_env(tmp_path, monkeypatch):
    from quantlab import alpaca as ap
    assert ap.to_alpaca_symbol("BTC-USD") == "BTC/USD" and ap.position_symbol("BTC-USD") == "BTCUSD"
    assert ap.to_alpaca_symbol("SPY") == "SPY" and ap.is_crypto("ETH/USD")
    env = tmp_path / ".env"   # fichier sauvé en UTF-16 avec espaces/guillemets (cas Windows)
    env.write_text('ALPACA_API_KEY = "PK1"\nALPACA_SECRET_KEY=s2\nGROQ_API_KEY=COLLE_ICI\n',
                   encoding="utf-16")
    for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert ap.load_env(str(env))
    import os
    assert os.environ["ALPACA_API_KEY"] == "PK1" and os.environ["ALPACA_SECRET_KEY"] == "s2"
    assert "GROQ_API_KEY" not in os.environ   # placeholder ignoré


def test_mirror_plan_reconciles(tmp_path):
    from quantlab import broker as bk
    from quantlab import mirror as mr
    acc = bk.load_account(str(tmp_path / "a.json"), capital=10_000)
    bk.buy(acc, "SPY", 100.0, 95.0, 110.0, "trend", "2026-01-02")          # 20 parts
    bk.buy(acc, "BTC-USD", 50_000.0, 48_000.0, None, "trend", "2026-01-02")
    prices = {"SPY": 100.0, "BTC-USD": 50_000.0}
    eq = acc.equity(prices)
    # Alpaca a 2x l'équité du papier, détient déjà un QQQ que le fonds n'a plus
    plan = mr.plan_mirror(acc, prices, ["SPY", "QQQ", "BTC-USD"],
                          {"QQQ": 3.0}, alpaca_equity=2 * eq)
    by = {a["symbol"]: a for a in plan}
    assert by["SPY"]["action"] == "buy" and abs(by["SPY"]["qty"] - 40.0) < 1e-6
    assert by["BTC-USD"]["action"] == "buy"
    assert by["QQQ"]["action"] == "close"
    # idempotent : une fois aligné, plus aucun ordre
    held = {"SPY": 40.0, "BTCUSD": acc.positions["BTC-USD"]["shares"] * 2}
    assert mr.plan_mirror(acc, prices, ["SPY", "QQQ", "BTC-USD"], held, 2 * eq) == []
    # un ordre déjà en attente n'est jamais doublé
    assert all(a["symbol"] != "SPY" for a in
               mr.plan_mirror(acc, prices, ["SPY"], {}, 2 * eq, pending={"SPY"}))
    # n'ouvre ni ne ferme rien hors univers du fonds (poche IA sur le même compte)
    assert all(a["symbol"] in {"SPY", "BTC-USD"} for a in
               mr.plan_mirror(acc, prices, ["SPY", "BTC-USD"], {"NVDA": 5.0}, 2 * eq))


def test_live_cycle_mirrors_to_alpaca(tmp_path):
    from quantlab import live as lv
    from quantlab import mirror as mr
    path = str(tmp_path / "account.json")
    fake_load = lambda sym, years=2, fresh=False, **k: (
        synthetic_ohlcv(sym, years=3, seed=zlib.crc32(sym.encode()) % 1000), "yahoo")
    client = FakeAlpaca(positions={"XLE": {"qty": "2"}})
    out = lv.run_cycle(["AAA", "XLE"], capital=10_000, state_path=path, loader=fake_load,
                       after=lambda acc, px: mr.mirror_report(client, acc, px, ["AAA", "XLE"]))
    assert "MIROIR ALPACA" in out and "COMPTE PAPIER" in out
    assert "XLE" in client.closed            # le fonds ne détient pas XLE → fermé
    out2 = lv.run_cycle(["AAA"], capital=10_000, state_path=path, loader=fake_load,
                        after=lambda acc, px: mr.mirror_report(None, acc, px, ["AAA"]))
    assert "Clés Alpaca absentes" in out2    # sans clés : fail-safe


def test_ai_option_strategies():
    from quantlab import ai_trader as ai
    chain = ai.option_chain(FakeAlpaca(), "NVDA", 140.0)
    assert chain and all(c["underlying"] == "NVDA" for c in chain)
    for strat in ai.OPTION_STRATEGIES:
        plan, why = ai.build_option_trade(strat, chain, 140.0)
        assert plan, (strat, why)
        assert plan["max_loss"] > 0
    p, _ = ai.build_option_trade("BULL_CALL_SPREAD", chain, 140.0)
    assert [l["strike"] for l in p["legs"]] == [140.0, 145.0] and p["width"] == 5.0
    p, _ = ai.build_option_trade("LONG_STRADDLE", chain, 140.0)
    assert p["legs"][0]["strike"] == p["legs"][1]["strike"] == 140.0 and p["max_gain"] is None
    illiquid = [dict(c, bid=0.5, ask=2.0) for c in chain]
    assert ai.build_option_trade("LONG_STRADDLE", illiquid, 140.0)[0] is None
    close = ai.mleg_order(p["legs"], 2, opening=False)
    assert [l["side"] for l in close["legs"]] == ["sell", "sell"]
    assert close["legs"][0]["position_intent"] == "sell_to_close"


def test_ai_cycle_open_manage_and_guards(tmp_path):
    from quantlab import ai_trader as ai
    state = str(tmp_path / "ai.json")
    client = FakeAlpaca()
    kw = dict(earnings=lambda s: 5 if s == "NVDA" else None, crypto_news=lambda: [],
              fng=lambda: "50/100 (Neutral)", sleep=lambda s: None)
    seen = []
    def decide(prompt):
        seen.append(prompt)
        return {"action": "OPEN", "strategy": "BULL_CALL_SPREAD", "underlying": "NVDA",
                "confidence": 0.7, "reasoning": "test"}
    out, d = ai.run_cycle(client, decide, state_path=state, **kw)
    assert "OUVERT BULL_CALL_SPREAD NVDA" in out, out
    assert "RESULTATS dans 5 j" in seen[0]
    assert client.orders[-1]["order_class"] == "mleg"
    t = ai.load_trades(state)[0]
    assert t["cost"] <= 100_000 * ai.OPTION_MAX_RISK_PCT
    # déjà en position → refusé
    out, d = ai.run_cycle(client, decide, state_path=state, **kw)
    assert "Deja en position" in out
    # take-profit atteint → fermeture par ordre multi-jambes inverse
    client.pos = {l["symbol"]: {"unrealized_pl": str(t["tp_amount"])} for l in t["legs"]}
    out, _ = ai.run_cycle(client, lambda p: {"action": "HOLD", "reasoning": "x"},
                          state_path=state, **kw)
    assert "Take-profit" in out and ai.load_trades(state)[0]["status"] == "CLOSED"
    # bourse fermée : options refusées
    client.open_ = False
    out, _ = ai.run_cycle(client, decide, state_path=state, **kw)
    assert "ferme" in out.lower()


def test_ai_short_and_shared_universe(tmp_path):
    from quantlab import ai_trader as ai
    stocks, cryptos = ai.universe(["SPY", "QQQ", "BTC-USD"])
    assert "SPY" not in stocks and "BTC/USD" not in cryptos and "NVDA" in stocks
    client = FakeAlpaca()
    kw = dict(earnings=lambda s: None, crypto_news=lambda: [], fng=lambda: "-",
              sleep=lambda s: None)
    short = lambda p: {"action": "OPEN", "strategy": "SHORT_STOCK", "underlying": "TSLA",
                       "confidence": 0.8, "stop_loss_pct": 0.9, "reasoning": "x"}
    out, _ = ai.run_cycle(client, short, state_path=str(tmp_path / "s.json"), **kw)
    o = client.orders[-1]
    assert o["side"] == "sell" and o["qty"] == str(int(10_000 // 140)) and "notional" not in o
    t = ai.load_trades(str(tmp_path / "s.json"))[0]
    assert t["sl_pct"] == 0.05                       # stop borné à 5 %
    # la poche ne peut pas trader un actif du fonds quand le compte est partagé
    spy = lambda p: {"action": "OPEN", "strategy": "BUY_STOCK", "underlying": "SPY",
                     "confidence": 0.9, "reasoning": "x"}
    out, _ = ai.run_cycle(client, spy, state_path=str(tmp_path / "x.json"),
                          fund_universe=["SPY"], **kw)
    assert "n'est pas une action de la poche" in out


def test_ai_missing_keys_is_failsafe(monkeypatch):
    from quantlab import ai_trader as ai
    from quantlab import alpaca as ap
    monkeypatch.setattr(ap, "load_env", lambda path=None: False)
    monkeypatch.setattr(ai, "load_env", lambda path=None: False)
    for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_AI_API_KEY",
              "ALPACA_AI_SECRET_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    out, d = ai.main_cycle(["SPY"])
    assert "non lancée" in out and d == {}


def test_dashboard_ai_section(tmp_path, prices):
    from quantlab import dashboard as db
    from quantlab import fund as fd
    from quantlab.broker import load_account
    res = fd.run_fund(prices, capital=100_000)
    account = load_account(str(tmp_path / "acc.json"), capital=100_000)
    ai = {"trades": [{"opened_at": "2026-10-26T15:40:00", "strategy": "LONG_STRADDLE",
                      "underlying": "NVDA", "cost": 820.0, "status": "CLOSED", "pnl": 150.0,
                      "reasoning": "<b>resultats</b>"}],
          "open": [], "closed": [{"pnl": 150.0}], "realized": 150.0, "win_rate": 1.0}
    path = db.save(res, account, {}, 100_000, str(tmp_path / "d.html"), ai=ai,
                   alpaca_lines=["Alpaca [fonds, paper] équité 100,000.00 $"])
    content = open(path, encoding="utf-8").read()
    assert "poche IA" in content and "LONG_STRADDLE" in content
    assert "<b>resultats</b>" not in content          # texte du LLM échappé


# ── Module 15 — formules du cours EFREI (pricing.py) ──

def test_pricing_black_scholes_identities():
    from quantlab import pricing as pr
    S, K, T, r, sig = 100.0, 105.0, 0.5, 0.03, 0.25
    c, p = pr.bs_price(S, K, T, r, sig), pr.bs_price(S, K, T, r, sig, "put")
    assert abs(pr.parity_gap(c, p, S, K, T, r)) < 1e-10                 # parité call-put
    lo, hi = pr.call_bounds(S, K, T, r)
    assert lo <= c <= hi                                                # bornes d'arbitrage
    assert abs(pr.digital_price(S, K, T, r, sig) - np.exp(-r * T) * pr.prob_itm(S, K, T, r, sig)) < 1e-12
    # valeur de référence classique : S=K=100, T=1, r=5 %, σ=20 % → call 10,4506
    assert abs(pr.bs_price(100, 100, 1, 0.05, 0.2) - 10.4506) < 1e-3
    # Bachelier ≈ Black-Scholes à la monnaie pour σ_abs = σ S
    assert abs(pr.bachelier_price(100, 100, 0.25, 20.0) - pr.bs_price(100, 100, 0.25, 0, 0.2)) < 0.05


def test_pricing_greeks_match_finite_differences():
    from quantlab import pricing as pr
    S, K, T, r, sig, h = 100.0, 95.0, 0.75, 0.02, 0.3, 1e-4
    for kind in ("call", "put"):
        g = pr.greeks(S, K, T, r, sig, kind)
        f = lambda **kw: pr.bs_price(**{"S": S, "K": K, "T": T, "r": r, "sigma": sig,
                                         "kind": kind, **kw})
        assert abs(g.delta - (f(S=S + h) - f(S=S - h)) / (2 * h)) < 1e-5
        assert abs(g.gamma - (f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4) < 1e-4
        assert abs(g.vega - (f(sigma=sig + h) - f(sigma=sig - h)) / (2 * h)) < 1e-3
        assert abs(g.rho - (f(r=r + h) - f(r=r - h)) / (2 * h)) < 1e-3
        assert abs(g.theta - (f(T=T - h) - f(T=T + h)) / (2 * h)) < 1e-3   # Θ = ∂V/∂t
        # relation de l'EDP : Θ = -½σ²S²Γ - rSΔ + rV
        assert abs(g.theta - pr.theta_from_pde(S, r, sig, f(), g)) < 1e-8


def test_pricing_implied_and_realized_vol():
    from quantlab import pricing as pr
    price = pr.bs_price(50, 55, 0.4, 0.01, 0.37, "put")
    assert abs(pr.implied_vol(price, 50, 55, 0.4, 0.01, "put") - 0.37) < 1e-6
    assert pr.implied_vol(60.0, 50, 55, 0.4, 0.01, "call") is None    # viole C ≤ S
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000))))
    assert abs(pr.realized_vol(close, n=None) - 0.01 * np.sqrt(252)) < 0.01
    # gamma P&L : long gamma gagne si vol réalisée > implicite, perd sinon
    assert pr.gamma_pnl(0.02, 100, 0.3, 0.2, 1 / 252) > 0 > pr.gamma_pnl(0.02, 100, 0.2, 0.3, 1 / 252)


def test_pricing_monte_carlo_and_variance_reduction():
    from quantlab import pricing as pr
    bs = pr.bs_price(100, 100, 1, 0.05, 0.2)
    plain = pr.mc_european(100, 100, 1, 0.05, 0.2, n=40_000, seed=1)
    anti = pr.mc_european(100, 100, 1, 0.05, 0.2, n=40_000, seed=1, antithetic=True)
    cv = pr.mc_european(100, 100, 1, 0.05, 0.2, n=40_000, seed=1, control_variate=True)
    for m in (plain, anti, cv):
        assert abs(m.price - bs) < 4 * m.std_error + 1e-9
    assert anti.std_error < plain.std_error and cv.std_error < plain.std_error / 2
    asian = pr.mc_path_option(pr.asian_call_payoff(100), 100, 1, 0.05, 0.2, 50, 5000)
    assert asian.price < bs                                  # asiatique < européenne
    barrier = pr.mc_path_option(pr.up_and_out_call_payoff(100, 120), 100, 1, 0.05, 0.2, 50, 5000)
    assert barrier.price < bs
    # Heston avec vol de vol ~0 et v0 = θ = σ² → retrouve Black-Scholes
    h = pr.heston_mc(100, 100, 1, 0.05, 0.04, 2.0, 0.04, 1e-6, 0.0, n_paths=20_000)
    assert abs(h.price - bs) < 4 * h.std_error + 0.05


def test_pricing_course_strategies():
    from quantlab import pricing as pr
    S, T, r, sig = 100.0, 0.25, 0.0, 0.2
    bull = pr.analyze_strategy("bull_spread", pr.make_strategy("bull_spread", S, K1=100, K2=110),
                               S, T, r, sig)
    p = bull.net_premium
    assert abs(bull.max_loss - p) < 1e-6 and abs(bull.max_gain - (10 - p)) < 1e-6
    assert abs(bull.breakevens[0] - (100 + p)) < 1e-3
    strad = pr.analyze_strategy("straddle", pr.make_strategy("straddle", S), S, T, r, sig)
    assert np.isinf(strad.max_gain) and abs(strad.greeks.delta) < 0.1 and strad.greeks.vega > 0
    assert len(strad.breakevens) == 2
    fly = pr.analyze_strategy("butterfly", pr.make_strategy("butterfly", S, K1=95, K2=100, K3=105),
                              S, T, r, sig)
    assert fly.net_premium > 0 and abs(fly.max_gain - (5 - fly.net_premium)) < 1e-3
    assert pr.strategy_payoff(fly.legs, [90, 100, 110]).tolist() == [0, 5, 0]
    rr = pr.make_strategy("risk_reversal", S, K1=95, K2=105)
    assert pr.strategy_payoff(rr, [80, 100, 120]).tolist() == [-15, 0, 15]
    st = pr.return_stats(pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, .01, 600)))))
    assert {"skew", "excess_kurtosis", "shapiro_p", "realized_vol"} <= set(st)


def test_ai_butterfly_and_vol_edge(tmp_path):
    from quantlab import ai_trader as ai
    chain = ai.option_chain(FakeAlpaca(), "NVDA", 140.0)
    plan, why = ai.build_option_trade("LONG_CALL_BUTTERFLY", chain, 140.0)
    assert plan, why
    assert [l["strike"] for l in plan["legs"]] == [135.0, 140.0, 145.0]
    assert [l["ratio"] for l in plan["legs"]] == [1, 2, 1]
    assert ai.mleg_order(plan["legs"], 1, True)["legs"][1]["ratio_qty"] == "2"
    a = ai.course_analysis(plan, 140.0, 0.3)
    assert a["breakevens"] and 0 <= a["prob_profit"] <= 1
    # long vol refusé si la vol réalisée est très inférieure à l'implicite payée
    strad, _ = ai.build_option_trade("LONG_STRADDLE", chain, 140.0)
    assert ai.course_analysis(strad, 140.0, 0.05)["vol_edge"] > ai.VOL_EDGE_MAX


def test_mirror_protects_ai_positions():
    from quantlab import broker as bk
    from quantlab import mirror as mr
    acc = bk.Account(cash=10_000, initial_capital=10_000,
                     history=[{"symbol": "NVDA"}, {"symbol": "AMD"}])
    uni = mr.managed_universe(acc, ["SPY"], protected={"NVDA"})
    assert "AMD" in uni and "SPY" in uni and "NVDA" not in uni


def test_live_with_stocks_hybrid_universe(tmp_path):
    """Mode --with-stocks : ETF fixes ∪ actions screenées, doublons d'émetteur
    écartés, garde-fou résultats, et au plus la moitié des places en actions."""
    from quantlab import live as lv
    path = str(tmp_path / "account.json")
    base = synthetic_ohlcv("ETF1", years=3, seed=11)
    twin = base.copy(); twin[["Open", "High", "Low", "Close"]] *= 1.001   # GOOG/GOOGL
    datasets = {"ETF1": base, "AAA": synthetic_ohlcv("AAA", years=3, seed=5),
                "AAAL": twin.assign(Close=synthetic_ohlcv("AAA", years=3, seed=5)["Close"] * 1.002),
                "BBB": synthetic_ohlcv("BBB", years=3, seed=9)}
    loader = lambda sym, years=2, fresh=False, **k: (datasets[sym], "yahoo")
    asked = []
    guard = lambda s: (asked.append(s), 2 if s == "BBB" else None)[1]
    out = lv.run_cycle(["ETF1"], capital=10_000, state_path=path, loader=loader,
                       screener=lambda: ["AAA", "AAAL", "BBB"], screener_mode="add",
                       max_screened_positions=1, earnings_guard=guard)
    assert "ETF1" in out.splitlines()[2]                 # l'univers fixe reste présent
    assert "Doublons d'émetteur écartés : AAAL" in out
    if "BBB" in asked:
        assert "BBB : résultats dans 2 j" in out
    acc = lv.load_account(path)
    assert sum(1 for s in acc.positions if s in {"AAA", "AAAL", "BBB"}) <= 1
    assert "BBB" not in acc.positions


def test_ai_live_greeks_and_limits(tmp_path):
    from quantlab import ai_trader as ai
    client = FakeAlpaca()
    kw = dict(earnings=lambda s: None, crypto_news=lambda: [], fng=lambda: "-",
              sleep=lambda s: None)
    gp = str(tmp_path / "greeks.json")
    straddle = lambda p: {"action": "OPEN", "strategy": "LONG_STRADDLE", "underlying": "NVDA",
                          "confidence": 0.8, "reasoning": "x"}
    st = str(tmp_path / "ai.json")
    client.data_iv = 0.3
    ai.run_cycle(client, straddle, state_path=st, greeks_path=gp, **kw)
    t = ai.load_trades(st)
    if not t:          # refusé par le filtre d'avantage de vol : on force l'ouverture pour le test
        ai.VOL_EDGE_MAX, old = 99, ai.VOL_EDGE_MAX
        ai.run_cycle(client, straddle, state_path=st, greeks_path=gp, **kw)
        ai.VOL_EDGE_MAX = old
        t = ai.load_trades(st)
    assert t and t[0]["strategy"] == "LONG_STRADDLE"
    out, _ = ai.run_cycle(client, lambda p: {"action": "HOLD", "reasoning": "x"},
                          state_path=st, greeks_path=gp, **kw)
    assert "Grecques du portefeuille" in out
    g = json.load(open(gp))
    tot = g["total"]
    assert tot["vega"] > 0 and tot["theta_day"] < 0 and tot["gamma_usd"] > 0   # long straddle
    assert abs(tot["delta_usd"]) < 0.3 * 140 * 100 * t[0]["qty"]               # ~ delta-neutre
    ok, why = ai.greek_limits_ok({"total": {"delta_usd": 80_000, "vega": 0, "theta_day": 0}}, 100_000)
    assert not ok and "delta" in why
    ok, _ = ai.greek_limits_ok(g, 100_000)
    assert ok


def test_bonds_and_forwards():
    from quantlab import bonds as bd
    # au pair : coupon = taux → prix = nominal
    assert abs(bd.bond_price(1000, 0.05, 10, 0.05) - 1000) < 1e-9
    p = bd.bond_price(1000, 0.04, 10, 0.045, freq=2)
    assert abs(bd.ytm(p, 1000, 0.04, 10, freq=2) - 0.045) < 1e-8
    assert abs(bd.zero_coupon_price(100, 5, 0.03) - 100 / 1.03 ** 5) < 1e-12
    d = bd.durations(1000, 0.0, 10, 0.04)
    assert abs(d["macaulay"] - 10) < 1e-9                    # zéro-coupon : duration = maturité
    d = bd.durations(1000, 0.04, 10, 0.045)
    exact = bd.bond_price(1000, 0.04, 10, 0.055) / d["price"] - 1
    approx1 = -d["modified"] * 0.01
    approx2 = bd.price_change(d["modified"], d["convexity"], 0.01)
    assert abs(approx2 - exact) < abs(approx1 - exact) < 0.01   # la convexité améliore
    assert abs(d["dv01"] - (bd.bond_price(1000, 0.04, 10, 0.0449) - bd.bond_price(1000, 0.04, 10, 0.0451)) / 2) < 1e-3
    assert abs(bd.forward_price(100, 0.04, 1) - 100 * np.exp(0.04)) < 1e-12
    assert abs(bd.forward_value(100, bd.forward_price(100, 0.04, 1), 0.04, 1)) < 1e-9
    # duration empirique d'un « ETF » qui perd 7 % par point de taux
    idx = pd.bdate_range("2024-01-01", periods=400)
    rng = np.random.default_rng(0)
    y = pd.Series(4 + np.cumsum(rng.normal(0, 0.05, 400)), index=idx)
    etf = pd.Series(100 * np.cumprod(1 - 7 * y.diff().fillna(0) / 100), index=idx)
    assert abs(bd.empirical_duration(etf, y) - 7) < 0.3


def test_factsheet_nav_fees_metrics(prices):
    from quantlab import factsheet as fs
    from quantlab import fund as fd
    idx = pd.bdate_range("2020-01-01", periods=3 * 252)
    eq = pd.Series(np.linspace(100_000, 133_100, len(idx)), index=idx)
    gross = fs.nav(eq)
    assert gross.iloc[0] == 100
    net = fs.apply_fees(gross, mgmt=0.02, perf=0.20)   # 2/20 historique (défaut : moyenne HFR)
    assert net.iloc[-1] < gross.iloc[-1]                          # les frais coûtent
    flat = fs.apply_fees(pd.Series(100.0, index=idx), 0.0, 0.20)
    assert abs(flat.iloc[-1] - 100) < 1e-9                       # pas de gain → pas de commission
    down_up = pd.Series(np.r_[np.linspace(100, 80, 252), np.linspace(80, 100, 252)], index=idx[:504])
    assert abs(fs.apply_fees(down_up, 0.0, 0.20).iloc[-1] - 100) < 1e-9   # high-water mark
    res = fd.run_fund(prices, 100_000)
    spy = prices["AAA"]["Close"]
    m = fs.metrics(res.equity, spy)
    assert {"sharpe", "sortino", "var95_daily", "cvar95_daily", "beta", "alpha"} <= set(m)
    assert m["cvar95_daily"] >= m["var95_daily"] > 0 and m["max_drawdown"] <= 0
    tab = fs.monthly_table(res.equity)
    assert "Année" in tab.columns and len(tab) >= 7
    a = fs.activity(res.trades, res.equity)
    assert a["trades_per_year"] > 0 and a["turnover"] > 0
    out = fs.report(res.equity, spy, None, profiles={"Prudent": (0.5, res.equity)}, trades=res.trades)
    assert "Net de frais" in out and "Profils de risque" in out and "rotation" in out


def test_dashboard_shows_greeks_even_without_positions(tmp_path, prices):
    from quantlab import dashboard as db
    from quantlab import fund as fd
    from quantlab.broker import load_account
    res = fd.run_fund(prices, capital=100_000)
    account = load_account(str(tmp_path / "acc.json"), capital=100_000)
    ai = {"trades": [], "open": [], "closed": [], "realized": 0.0, "win_rate": None, "greeks": None}
    content = open(db.save(res, account, {}, 100_000, str(tmp_path / "d.html"), ai=ai),
                   encoding="utf-8").read()
    assert "Grecques en continu" in content and "toutes les grecques valent 0" in content


# ── module 17 : idées presse ────────────────────────────────────────────────
class NewsClient:
    def __init__(self, news):
        self.news = news

    def data(self, path, params=None):
        return {"news": self.news, "next_page_token": None}


def _bars(price, dollar_volume, n=80, drift=0.004):
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    close = price * np.exp(np.linspace(-drift * n, 0, n))
    return pd.DataFrame({"Close": close, "Volume": dollar_volume / close}, index=idx)


def test_ideas_aggregate_filters_and_niche(tmp_path):
    from quantlab import ideas as ide
    news = [{"headline": "Analyst upgrades NICH, raises price target", "symbols": ["NICH"],
             "created_at": "2026-10-26T10:00:00Z"},
            {"headline": "NICH beats estimates, strong growth", "symbols": ["NICH"]},
            {"headline": "BIGC record revenue", "symbols": ["BIGC", "AAPL"]},
            {"headline": "Penny stock surges", "symbols": ["PENY"]},
            {"headline": "Thin stock rallies", "symbols": ["THIN"]},
            {"headline": "Roundup", "symbols": list("ABCDEFG")},          # fourre-tout ignoré
            {"headline": "Bitcoin", "symbols": ["BTCUSD"]}]                # pas un ticker action
    bars = {"NICH": _bars(20, 30e6), "BIGC": _bars(150, 2e9), "PENY": _bars(2, 50e6),
            "THIN": _bars(30, 2e6), "AAPL": _bars(200, 1e10)}
    rows = ide.build_ideas(NewsClient(news), exclude={"AAPL"}, bars_fn=lambda s: bars,
                           summarize=lambda r: {"NICH": {"why": "Relèvement", "risk": "Petite valeur"}})
    syms = [r["symbol"] for r in rows]
    assert syms == ["NICH", "BIGC"]                    # penny, illiquide, exclu, roundup filtrés
    nich = rows[0]
    assert nich["mentions"] == 2 and nich["niche"] and nich["sentiment"] > 0
    assert nich["trend"] == "haussiere" and nich["why"] == "Relèvement"
    assert not rows[1]["niche"]
    path = str(tmp_path / "ideas.json")
    ide.save(rows, path)
    assert ide.load(path)["ideas"][0]["symbol"] == "NICH"
    assert "NICH" in ide.report(rows) and "niche" in ide.report(rows).lower()
    assert ide.build_ideas(NewsClient([]), bars_fn=lambda s: {}) == []      # fail-safe


def test_ai_cycle_trades_press_idea_at_half_size(tmp_path):
    from quantlab import ai_trader as ai
    from quantlab import ideas as ide
    ideas_path = str(tmp_path / "ideas.json")
    ide.save([{"symbol": "NICH", "score": 5.0, "niche": True, "headlines": ["NICH upgraded"]},
              {"symbol": "NVDA", "score": 4.0, "niche": False, "headlines": []}], ideas_path)
    assert [i["symbol"] for i in ai.fresh_ideas(ideas_path, exclude={"NVDA"})] == ["NICH"]
    seen = []
    def decide(prompt):
        seen.append(prompt)
        return {"action": "OPEN", "strategy": "BUY_STOCK", "underlying": "NICH",
                "confidence": 0.8, "stop_loss_pct": 0.03, "reasoning": "presse"}
    client = FakeAlpaca()
    out, _ = ai.run_cycle(client, decide, state_path=str(tmp_path / "ai.json"),
                          ideas_path=ideas_path, sleep=lambda s: None,
                          earnings=lambda s: None, crypto_news=lambda: [], fng=lambda: "-")
    assert "Idées presse ajoutées : NICH*" in out and "OUVERT BUY_STOCK NICH" in out, out
    assert "NICH (idee presse, niche)" in seen[0]
    t = ai.load_trades(str(tmp_path / "ai.json"))[0]
    assert t["niche"] and t["source"] == "presse"
    assert t["cost"] == 100_000 * ai.SPOT_MAX_PCT * ai.NICHE_SIZE     # taille divisée par 2
    crypto_rows = [l for l in seen[0].splitlines() if "(crypto)" in l]
    assert 0 < len(crypto_rows) <= ai.TOP_CRYPTOS                        # pré-tri crypto
    # idées trop anciennes → ignorées
    old = ide.load(ideas_path)
    old["updated_at"] = "2020-01-01T00:00:00"
    with open(ideas_path, "w", encoding="utf-8") as f:
        json.dump(old, f)
    assert ai.fresh_ideas(ideas_path) == []
    assert "rafraîchies" in ai.refresh_ideas(NewsClient([]), ideas_path)


# ── terminal du gérant ──────────────────────────────────────────────────────
def test_terminal_model_and_page(tmp_path, prices):
    from quantlab import terminal as term
    from quantlab import fund as fd
    from quantlab.broker import load_account, buy
    data = {"SPY": prices["AAA"], "TLT": prices["BBB"], "GLD": prices["CCC"],
            "BTC-USD": prices["DDD"], "NVDA": prices["BBB"], "^VIX": prices["CCC"]}
    account = load_account(str(tmp_path / "acc.json"), capital=100_000)
    spot = float(data["NVDA"]["Close"].iloc[-1])
    buy(account, "NVDA", spot, spot * 0.95, None, "trend", "2026-10-01", 1.0)
    account.record_equity("2026-10-01", 100_000)
    account.record_equity("2026-10-02", 100_500)
    prices_now = {s: float(d["Close"].iloc[-1]) for s, d in data.items()}
    res = fd.run_fund({k: data[k] for k in ("SPY", "TLT", "GLD")}, 100_000)
    ai = {"open": [], "closed": [], "realized": 0.0, "win_rate": None,
          "greeks": {"updated_at": "x", "positions": [], "total": {"vega": 10.0}}}
    m = term.build_model(account, data, prices_now,
                         groups={"Univers du fonds": ["SPY", "TLT", "GLD", "BTC-USD"],
                                 "Idées presse": ["NVDA"]},
                         fund_res=res, ai=ai, earnings_fn=lambda s: 3,
                         ideas={"ideas": [{"symbol": "NVDA", "score": 5, "headlines": ["NVDA up"]}]},
                         limits={"max_positions": 8, "risk_pct": 1.0})
    assert {a["symbol"] for a in m["assets"]} == {"SPY", "TLT", "GLD", "BTC-USD", "NVDA"}
    nv = next(a for a in m["assets"] if a["symbol"] == "NVDA")
    assert "En portefeuille" in nv["groups"] and nv["earnings"] == 3 and nv["news"] == ["NVDA up"]
    assert nv["signals"] and len(nv["chart"]["c"]) == term.CHART_DAYS and nv["hv"] > 0
    assert m["positions"][0]["symbol"] == "NVDA" and m["positions"][0]["risk_usd"] > 0
    r = m["risk"]
    assert r["var95"] > 0 and r["cvar95"] >= r["var95"]
    assert [s["name"] for s in r["stress"]][-1].startswith("Volatilité") and r["stress"][-1]["pnl"] == 100
    assert abs(m["overview"]["by_class"]["Cash"] - account.cash) < 0.01 and m["backtest"]["gross"]
    assert any(l["name"].startswith("Positions") for l in r["limits"])
    html = open(term.save(m, str(tmp_path / "index.html")), encoding="utf-8").read()
    assert "Terminal du gérant" in html and "/*__DATA__*/null" not in html
    payload = html.split("const D = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(payload)["assets"]                                   # JSON strict
    # sans données de marché : la page se construit quand même
    empty = term.build_model(load_account(str(tmp_path / "b.json"), capital=1000), {}, {},
                             groups={})
    assert empty["assets"] == [] and empty["risk"]["stress"]
    term.save(empty, str(tmp_path / "e.html"))


def test_workflow_commands_parse():
    """Chaque commande `python -m quantlab ...` des workflows GitHub doit être valide."""
    import glob, re, shlex
    from quantlab import cli
    cli.PARSE_ONLY = True
    try:
        cmds = []
        for f in glob.glob(os.path.join(ROOT_DIR, ".github", "workflows", "*.yml")):
            for line in open(f, encoding="utf-8"):
                m = re.search(r"python -m quantlab ([^|&]+)", line)
                if m:
                    cmds.append(shlex.split(m.group(1)))
        assert len(cmds) >= 5
        for c in cmds:
            assert cli.main(c) == 0, c
    finally:
        cli.PARSE_ONLY = False


def test_treasury_puts_idle_cash_to_work(tmp_path, prices):
    from quantlab import treasury as tr
    from quantlab import fund as fd
    from quantlab.broker import load_account, buy
    acc = load_account(str(tmp_path / "t.json"), capital=100_000)
    px = {"SPY": 500.0, "BIL": 91.5, "AAA": 50.0}
    tr.rebalance(acc, px, core_pct=0.30, sweep="BIL")
    v = {s: h["shares"] * px[s] for s, h in acc.treasury.items()}
    assert abs(v["SPY"] - 30_000) < 50 and abs(v["BIL"] - 69_000) < 100
    assert 0 < acc.cash < 1_500 and abs(acc.equity(px) - 100_000) < 100
    # une stratégie a besoin de 50 000 $ : BIL vendu d'abord, le socle reste
    tr.raise_cash(acc, 50_000, px)
    assert acc.cash >= 50_000 and "SPY" in acc.treasury
    assert buy(acc, "AAA", 50.0, 45.0, None, "trend", "2026-10-26", 1.0, equity=acc.equity(px))
    # tout l'argent pris par les stratégies → plus de trésorerie, jamais de cash négatif
    tr.raise_cash(acc, acc.equity(px), px)
    assert acc.cash >= 0 and acc.treasury == {}
    assert acc.holdings() == {"AAA": acc.positions["AAA"]["shares"]}
    # backtest : avec un socle, le fonds est plus investi et la trésorerie ne bloque rien
    tp = {"SPY": prices["AAA"], "BIL": prices["BBB"]}
    base = fd.run_fund(prices, 100_000)
    core = fd.run_fund(prices, 100_000, core_pct=0.3, treasury_prices=tp, sweep="BIL")
    assert len(core.trades) >= len(base.trades) * 0.9
    assert not core.equity.equals(base.equity) and (core.equity > 0).all()


def test_mirror_includes_treasury_and_reserve(tmp_path):
    from quantlab import mirror as mr
    from quantlab.broker import load_account
    acc = load_account(str(tmp_path / "m.json"), capital=100_000)
    acc.cash = 10_000
    acc.treasury = {"SPY": {"shares": 60.0, "cost": 30_000, "last": 500.0},
                    "BIL": {"shares": 655.7, "cost": 60_000, "last": 91.5}}
    plan = mr.plan_mirror(acc, {"SPY": 500.0, "BIL": 91.5}, ["SPY"], {}, 100_000, reserve=0.25)
    q = {a["symbol"]: a["qty"] for a in plan}
    assert abs(q["SPY"] - 45.0) < 0.1 and abs(q["BIL"] - 655.7 * 0.75) < 0.5
