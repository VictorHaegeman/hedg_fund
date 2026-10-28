"""Terminal du gérant — tableau de bord interactif (docs/index.html, GitHub Pages).

Tout ce qu'un gérant de fonds regarde chaque matin, sur une seule page :
  • Vue d'ensemble : VL, P&L jour / mois / année vs S&P 500, cash, expositions brute
    et nette, drawdown en cours, répartition par classe d'actifs et par secteur
  • Positions : fonds principal + poche IA, P&L, distance au stop, risque engagé
  • Risque : VaR / CVaR 95 % du portefeuille actuel (simulation historique 1 an),
    volatilité, bêta, scénarios de stress (actions, taux, bitcoin, or, volatilité),
    utilisation des limites, matrice de corrélation
  • Grecques : poche options (Δ Γ ν Θ ρ) + delta $ des positions au comptant
  • Marchés : ~100 actifs (ETF, actions screenées, idées presse, cryptos). Un clic →
    graphique 6 mois, indicateurs, signaux des 3 stratégies du fonds, résultats,
    news, et calculateur d'options Black-Scholes (prix + grecques call/put, straddle)
    avec la vol implicite Alpaca quand elle existe, sinon la vol réalisée
  • Idées presse (module 17), Activité, Fiche du fonds, Contexte macro

La page est autonome (HTML + JS sans dépendance externe). Les données sont calculées
ici en Python et intégrées en JSON ; les grecques du calculateur sont recalculées dans
le navigateur avec les mêmes formules que quantlab.pricing.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import indicators as ind

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "docs", "index.html")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terminal_template.html")
RISK_FREE = 0.04
CHART_DAYS = 130            # ~6 mois de bourse dans le graphique du détail

ETF_INFO = {
    "SPY": ("ETF actions", "Indice S&P 500"), "QQQ": ("ETF actions", "Nasdaq 100"),
    "IWM": ("ETF actions", "Petites capitalisations"), "DIA": ("ETF actions", "Dow Jones"),
    "XLK": ("ETF actions", "Technologie"), "XLF": ("ETF actions", "Finance"),
    "XLE": ("ETF actions", "Énergie"), "XLV": ("ETF actions", "Santé"),
    "XLY": ("ETF actions", "Consommation discrétionnaire"), "XLP": ("ETF actions", "Consommation de base"),
    "XLI": ("ETF actions", "Industrie"), "XLU": ("ETF actions", "Services publics"),
    "XLB": ("ETF actions", "Matériaux"), "GLD": ("Or / matières", "Or"),
    "USO": ("Or / matières", "Pétrole"), "TLT": ("Obligations", "Trésor US 20+ ans"),
    "IEF": ("Obligations", "Trésor US 7-10 ans"), "SHY": ("Obligations", "Trésor US 1-3 ans"),
    "TIP": ("Obligations", "Trésor indexé inflation"), "LQD": ("Obligations", "Entreprises IG"),
    "HYG": ("Obligations", "Haut rendement"),
}
STOCK_SECTORS = {
    **{s: "Technologie" for s in ("AAPL", "MSFT", "NVDA", "AMD", "AVGO", "CRM", "ORCL", "ADBE",
                                   "INTC", "QCOM", "MU", "PLTR", "SHOP")},
    **{s: "Communication" for s in ("GOOGL", "META", "NFLX", "DIS")},
    **{s: "Consommation discrétionnaire" for s in ("AMZN", "TSLA", "MCD", "NKE", "SBUX", "UBER")},
    **{s: "Finance" for s in ("JPM", "BAC", "GS", "V", "MA", "PYPL", "COIN")},
    **{s: "Santé" for s in ("UNH", "LLY", "JNJ", "PFE", "MRK")},
    **{s: "Consommation de base" for s in ("WMT", "COST", "KO", "PEP")},
    **{s: "Énergie" for s in ("XOM", "CVX")},
    **{s: "Industrie" for s in ("BA", "CAT")},
}
SECTOR_FR = {"Technology": "Technologie", "Communication Services": "Communication",
             "Consumer Cyclical": "Consommation discrétionnaire",
             "Consumer Defensive": "Consommation de base", "Financial Services": "Finance",
             "Healthcare": "Santé", "Energy": "Énergie", "Industrials": "Industrie",
             "Basic Materials": "Matériaux", "Utilities": "Services publics",
             "Real Estate": "Immobilier"}
STRESS = [("Actions −10 %", "SPY", -0.10), ("Krach actions −20 %", "SPY", -0.20),
          ("Taux +1 % (TLT ≈ −16 %)", "TLT", -0.165), ("Bitcoin −30 %", "BTC-USD", -0.30),
          ("Or −10 %", "GLD", -0.10)]
MACRO = {"^VIX": "VIX (peur)", "^TNX": "Taux US 10 ans", "DX-Y.NYB": "Dollar (DXY)",
         "CL=F": "Pétrole WTI", "GC=F": "Or (once)", "BTC-USD": "Bitcoin"}


# ── utilitaires ─────────────────────────────────────────────────────────────
def _f(x, d: int = 2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, d) if math.isfinite(x) else None


def is_crypto(sym: str) -> bool:
    return sym.endswith("-USD")


def asset_class(sym: str) -> str:
    if is_crypto(sym):
        return "Crypto"
    return ETF_INFO.get(sym, ("Actions", ""))[0]


def sector_of(sym: str, lookup=None) -> str:
    if is_crypto(sym):
        return "Crypto"
    if sym in ETF_INFO:
        return ETF_INFO[sym][1]
    if sym in STOCK_SECTORS:
        return STOCK_SECTORS[sym]
    if lookup:
        try:
            s = lookup(sym)
            if s:
                return SECTOR_FR.get(s, s)
        except Exception:
            pass
    return "Autre"


def yf_sector(sym: str) -> str | None:
    import yfinance as yf
    return (yf.Ticker(sym).info or {}).get("sector")


YF_ALIAS = {"UNI-USD": "UNI7083-USD"}     # tickers Yahoo différents du symbole Alpaca


def download(symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Historique journalier en un seul appel yfinance (fail-safe)."""
    symbols = sorted(set(symbols))
    if not symbols:
        return {}
    yf_syms = [YF_ALIAS.get(s, s) for s in symbols]
    try:
        import yfinance as yf
        raw = yf.download(yf_syms, period=period, interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
    except Exception:
        return {}
    out = {}
    for s in symbols:
        try:
            y = YF_ALIAS.get(s, s)
            df = raw[y] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=["Close"])
            if len(df) >= 30:
                out[s] = df
        except Exception:
            continue
    return out


# ── statistiques par actif ──────────────────────────────────────────────────
def strategy_signals(df: pd.DataFrame) -> list[dict]:
    """Ce que disent les 3 stratégies du fonds aujourd'hui sur cet actif."""
    from .strategies import ALL_STRATEGIES, get_strategy
    names = {"trend": "Suivi de tendance", "meanrev": "Retour à la moyenne",
             "breakout": "Cassure"}
    out = []
    for key in ALL_STRATEGIES:
        try:
            s = get_strategy(key).generate_signals(df)
            last = s.iloc[-1]
            recent = bool(s["entry"].tail(5).any())
            state = ("ACHAT" if bool(last["entry"]) else "SORTIE" if bool(last["exit"])
                     else "achat récent" if recent else "—")
            out.append({"key": key, "name": names.get(key, key), "state": state,
                        "stop": _f(last.get("stop"))})
        except Exception:
            continue
    return out


def asset_stats(sym: str, df: pd.DataFrame) -> dict:
    c = df["Close"].astype(float)
    price = float(c.iloc[-1])
    rets = np.log(c / c.shift(1)).dropna()
    e20, e50 = ind.ema(c, 20), ind.ema(c, 50)
    trend = ("haussière" if price > e20.iloc[-1] > e50.iloc[-1] else
             "baissière" if price < e20.iloc[-1] < e50.iloc[-1] else "neutre")
    per = 365 if is_crypto(sym) else 252

    def chg(n):
        return _f((price / float(c.iloc[-n - 1]) - 1) * 100, 1) if len(c) > n else None

    tail = c.tail(CHART_DAYS)
    hv = float(rets.tail(20).std() * math.sqrt(per))
    adv = float((c * df["Volume"]).tail(20).mean()) if "Volume" in df else None
    return {
        "symbol": sym, "price": _f(price, 4 if price < 1 else 2),
        "chg_1d": chg(1), "chg_5d": chg(5), "chg_1m": chg(21), "chg_3m": chg(63),
        "chg_1y": chg(len(c) - 1) if len(c) > 200 else None,
        "rsi": _f(ind.rsi(c, 14).iloc[-1], 1), "trend": trend,
        "hv": _f(hv, 4), "hv60": _f(rets.tail(60).std() * math.sqrt(per), 4),
        "atr_pct": _f(ind.atr(df, 14).iloc[-1] / price * 100, 2) if {"High", "Low"} <= set(df) else None,
        "from_high": _f((price / float(c.tail(252).max()) - 1) * 100, 1),
        "adv_musd": _f(adv / 1e6, 1) if adv else None,
        "periods": per,
        "chart": {"d": [d.strftime("%Y-%m-%d") for d in tail.index],
                  "c": [_f(v, 4) for v in tail],
                  "e20": [_f(v, 4) for v in e20.tail(CHART_DAYS)],
                  "e50": [_f(v, 4) for v in e50.tail(CHART_DAYS)]},
        "signals": strategy_signals(df) if len(df) >= 210 else [],
    }


# ── risque du portefeuille ──────────────────────────────────────────────────
def _returns(data: dict[str, pd.DataFrame], syms: list[str], days: int = 252) -> pd.DataFrame:
    frames = {s: data[s]["Close"] for s in syms if s in data}
    if not frames:
        return pd.DataFrame()
    px = pd.DataFrame(frames)
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px.groupby(level=0).last().ffill()
    return px.pct_change().dropna(how="all").tail(days)


def beta(r: pd.Series, f: pd.Series) -> float:
    both = pd.concat([r, f], axis=1).dropna()
    if len(both) < 30 or both.iloc[:, 1].var() == 0:
        return 0.0
    return float(both.cov().iloc[0, 1] / both.iloc[:, 1].var())


def portfolio_risk(weights: dict[str, float], data: dict[str, pd.DataFrame], nav: float) -> dict:
    """VaR/CVaR historiques 95 % (1 j), vol, bêta et stress du portefeuille actuel."""
    syms = [s for s, w in weights.items() if abs(w) > 0]
    factors = sorted({f for _, f, _ in STRESS})
    rets = _returns(data, sorted(set(syms) | set(factors)))
    out = {"var95": None, "cvar95": None, "vol": None, "beta": None, "stress": [],
           "betas": {}}
    if rets.empty or not syms:
        out["stress"] = [{"name": n, "pnl": 0.0, "pct": 0.0} for n, _, _ in STRESS]
        return out
    held = [s for s in syms if s in rets]
    port = (rets[held].fillna(0.0) * pd.Series({s: weights[s] for s in held})).sum(axis=1)
    if len(port) > 30:
        var = float(-np.percentile(port, 5))
        tail = port[port <= -var]
        out.update(var95=_f(var * nav), cvar95=_f(-tail.mean() * nav if len(tail) else var * nav),
                   vol=_f(port.std() * math.sqrt(252), 4),
                   beta=_f(beta(port, rets["SPY"]) if "SPY" in rets else None, 2))
    for name, fac, shock in STRESS:
        pnl = 0.0
        if fac in rets:
            for s in held:
                b = 1.0 if s == fac else beta(rets[s], rets[fac])
                out["betas"].setdefault(s, {})[fac] = _f(b, 2)
                pnl += weights[s] * nav * b * shock
        out["stress"].append({"name": name, "pnl": _f(pnl), "pct": _f(pnl / nav * 100, 2)})
    return out


def correlation(data: dict[str, pd.DataFrame], syms: list[str]) -> dict:
    rets = _returns(data, syms, 126)
    syms = [s for s in syms if s in rets and rets[s].notna().sum() > 60]
    if len(syms) < 2:
        return {"symbols": [], "matrix": []}
    m = rets[syms].corr().round(2)
    return {"symbols": syms, "matrix": [[_f(v, 2) for v in row] for row in m.to_numpy()]}


# ── modèle complet ──────────────────────────────────────────────────────────
def _curve_stats(curve: list, spy: pd.Series | None) -> dict:
    if not curve or len(curve) < 2:
        return {}
    s = pd.Series([v for _, v in curve], index=pd.to_datetime([d for d, _ in curve]))
    last = s.index[-1]

    def since(start):
        base = s[s.index < start]
        b = float(base.iloc[-1]) if len(base) else float(s.iloc[0])
        return _f((float(s.iloc[-1]) / b - 1) * 100, 2)
    out = {"day": _f((s.iloc[-1] / s.iloc[-2] - 1) * 100, 2),
           "mtd": since(pd.Timestamp(last.year, last.month, 1)),
           "ytd": since(pd.Timestamp(last.year, 1, 1)),
           "dd": _f((s.iloc[-1] / s.cummax().iloc[-1] - 1) * 100, 2),
           "peak_date": s.idxmax().strftime("%Y-%m-%d")}
    if spy is not None and len(spy):
        sp = spy.copy()
        sp.index = pd.to_datetime(sp.index).tz_localize(None).normalize()
        sp = sp[sp.index >= s.index[0]]
        if len(sp) > 1:
            out["spy_total"] = _f((sp.iloc[-1] / sp.iloc[0] - 1) * 100, 2)
            sp_ytd = sp[sp.index >= pd.Timestamp(last.year, 1, 1)]
            out["spy_curve"] = [[d.strftime("%Y-%m-%d"), _f(v / sp.iloc[0] * curve[0][1])]
                                for d, v in sp.items()]
            if len(sp_ytd) > 1:
                out["spy_ytd"] = _f((sp_ytd.iloc[-1] / sp_ytd.iloc[0] - 1) * 100, 2)
    return out


def fund_positions(account, data, prices_now, nav, sector_lookup) -> list[dict]:
    rows = []
    today = date.today()
    for sym, p in account.positions.items():
        px = prices_now.get(sym) or p["entry"]
        val = px * p["shares"]
        try:
            held = (today - date.fromisoformat(str(p["entry_date"])[:10])).days
        except ValueError:
            held = None
        rows.append({"symbol": sym, "book": "Fonds", "strategy": p.get("strategy"),
                     "shares": _f(p["shares"], 4), "entry": _f(p["entry"]), "price": _f(px),
                     "value": _f(val), "weight": _f(val / nav * 100, 2),
                     "pnl": _f((px - p["entry"]) * p["shares"]),
                     "pnl_pct": _f((px / p["entry"] - 1) * 100, 2),
                     "stop": _f(p.get("stop")),
                     "to_stop_pct": _f((p["stop"] / px - 1) * 100, 2) if p.get("stop") else None,
                     "risk_usd": _f(max(px - p["stop"], 0) * p["shares"]) if p.get("stop") else None,
                     "entry_date": str(p["entry_date"])[:10], "days": held,
                     "asset_class": asset_class(sym), "sector": sector_of(sym, sector_lookup)})
    return sorted(rows, key=lambda r: -(r["value"] or 0))


AI_KIND_FR = {
    "BUY_STOCK": "Achat d'actions", "SHORT_STOCK": "Vente à découvert (pari à la baisse)",
    "BUY_CRYPTO": "Achat de crypto",
    "BULL_CALL_SPREAD": "Option : bull call spread (pari à la hausse)",
    "BEAR_PUT_SPREAD": "Option : bear put spread (pari à la baisse)",
    "LONG_STRADDLE": "Option : straddle (gros mouvement, sens inconnu)",
    "LONG_STRANGLE": "Option : strangle (gros mouvement, sens inconnu)",
    "LONG_CALL_BUTTERFLY": "Option : butterfly (le prix reste stable)",
    "BULL_PUT_SPREAD": "Vente d'options : bull put spread (prime encaissée, vue haussière)",
    "BEAR_CALL_SPREAD": "Vente d'options : bear call spread (prime encaissée, vue baissière)",
    "IRON_CONDOR": "Vente d'options : iron condor (le prix reste dans une fourchette)",
    "COVERED_CALL": "Vente d'options : covered call (revenu sur des actions détenues)",
}


def describe_ai_trade(t: dict) -> tuple[str, str]:
    """(type en clair, détail) : jambes call/put pour les options, stop/objectif sinon."""
    kind = AI_KIND_FR.get(t.get("strategy"), t.get("strategy") or "")
    if t.get("legs"):
        legs = []
        for l in t["legs"]:
            n = l.get("ratio", 1)
            legs.append(f"{'achat' if l['side'] == 'buy' else 'vente'} {n if n > 1 else ''}"
                        f"{'×' if n > 1 else ''}{'call' if l['type'] == 'call' else 'put'} "
                        f"{l['strike']:g}")
        qty = t.get("qty")
        out = " + ".join(legs) + (f" · {qty} contrat{'s' if qty and qty > 1 else ''}" if qty else "")
        a = t.get("analysis") or {}
        if a.get("breakevens"):
            out += " · point mort " + " / ".join(f"{b:,.2f}" for b in a["breakevens"])
        if a.get("prob_profit") is not None:
            out += f" · proba. de gain {a['prob_profit'] * 100:.0f} %"
        return kind, out
    sl, tp = t.get("sl_pct"), t.get("tp_pct")
    px = t.get("entry_price")
    if sl and tp:
        d = f"stop {'+' if t.get('strategy') == 'SHORT_STOCK' else '−'}{sl * 100:.0f} % · objectif " \
            f"{'−' if t.get('strategy') == 'SHORT_STOCK' else '+'}{tp * 100:.0f} %"
        return kind, (f"à {px:,.2f} $ · " if px else "") + d
    return kind, ""


def ai_positions(ai: dict | None, prices: dict | None = None) -> list[dict]:
    if not ai:
        return []
    greeks = {g["id"]: g for g in ((ai.get("greeks") or {}).get("positions") or [])}
    rows = []
    for t in ai.get("open", []):
        g = greeks.get(t["id"], {})
        und = t["underlying"].replace("/USD", "-USD")
        what, detail = describe_ai_trade(t)
        cur = (prices or {}).get(und)
        entry = t.get("entry_price")
        short = t.get("kind") == "short"
        live = {}
        if t.get("kind") != "options" and entry and t.get("sl_pct") and t.get("tp_pct"):
            live = {"entry_price": entry,
                    "stop_price": round(entry * (1 + t["sl_pct"]) if short else entry * (1 - t["sl_pct"]), 4),
                    "target_price": round(entry * (1 - t["tp_pct"]) if short else entry * (1 + t["tp_pct"]), 4)}
            if cur:
                ret = (entry / cur - 1) if short else (cur / entry - 1)
                live.update(price=cur, pnl=round(t["cost"] * ret, 2), pnl_pct=round(ret * 100, 2))
        elif t.get("kind") == "options" and cur:
            live = {"price": cur, "spot_at_entry": t.get("spot_at_entry")}
        if not g and t.get("kind") in ("stock", "crypto", "short") and t.get("cost"):
            g = {"delta_usd": -t["cost"] if t["kind"] == "short" else t["cost"]}   # action : Δ = 1
        spot = t.get("kind") != "options"
        max_loss = t.get("max_loss") if not spot else (
            round(t["cost"] * t["sl_pct"], 2) if t.get("cost") and t.get("sl_pct") else None)
        max_gain = t.get("max_gain") if not spot else (
            round(t["cost"] * t["tp_pct"], 2) if t.get("cost") and t.get("tp_pct") else None)
        rows.append({"symbol": und, "book": "Poche IA", "strategy": t["strategy"],
                     "what": what, "detail": detail, **live,
                     "confidence": t.get("confidence"), "qty": t.get("qty"),
                     "credit": t.get("credit"), "delta_hedged": t.get("delta_hedged"),
                     "hedge_shares": t.get("hedge_shares") or 0,
                     "hedge_trades": len(t.get("hedge_log") or []),
                     "kind": t["kind"], "cost": t.get("cost"), "max_loss": max_loss,
                     "max_gain": max_gain, "gain_is_target": spot, "expiry": t.get("expiry"),
                     "opened": str(t.get("opened_at", ""))[:10], "niche": t.get("niche"),
                     "source": t.get("source"), "reasoning": t.get("reasoning"),
                     "delta_usd": g.get("delta_usd"), "gamma_usd": g.get("gamma_usd"),
                     "vega": g.get("vega"), "theta_day": g.get("theta_day"), "rho": g.get("rho"),
                     "legs": [{"type": l["type"], "side": l["side"], "strike": l["strike"],
                               "ratio": l.get("ratio", 1)} for l in t.get("legs", [])]})
    return rows


def iv_lookup(client, symbols: list[str], prices: dict, budget_s: float = 75.0) -> dict:
    """Vol implicite ATM ~30 j par sous-jacent (Alpaca options), dans un budget de temps."""
    out = {}
    if client is None:
        return out
    from . import ai_trader as ai
    t0 = time.time()
    for s in symbols:
        if time.time() - t0 > budget_s or is_crypto(s) or not prices.get(s):
            continue
        try:
            iv = ai.atm_iv(ai.option_chain(client, s, prices[s]), prices[s])
            if iv and 0.02 < iv < 5:
                out[s] = round(float(iv), 4)
        except Exception:
            continue
    return out


def build_model(account, data: dict[str, pd.DataFrame], prices_now: dict[str, float], *,
                groups: dict[str, list[str]], fund_res=None, factsheet: dict | None = None,
                ideas: dict | None = None, ai: dict | None = None,
                alpaca_lines: list[str] | None = None, gauge: dict | None = None,
                client=None, earnings_fn=None, news: dict | None = None,
                headlines: list[str] | None = None, sector_lookup=None,
                limits: dict | None = None, ai_limits: dict | None = None,
                budget_s: float = 75.0) -> dict:
    """Assemble toutes les données du terminal (fail-safe à chaque étape)."""
    nav = account.equity(prices_now)
    spy = data["SPY"]["Close"] if "SPY" in data else None
    pos = fund_positions(account, data, prices_now, nav, sector_lookup)
    aip = ai_positions(ai, {s_: float(d_["Close"].iloc[-1]) for s_, d_ in data.items()})
    invested = sum(r["value"] or 0 for r in pos)
    weights = {r["symbol"]: (r["value"] or 0) / nav for r in pos}
    tre_rows = []
    for sym, h in (getattr(account, "treasury", None) or {}).items():
        px_ = prices_now.get(sym) or h.get("last") or 0.0
        val = h["shares"] * px_
        tre_rows.append({"symbol": sym, "label": "Socle S&P 500" if sym == "SPY" else "Bons du Trésor",
                         "shares": _f(h["shares"], 4), "price": _f(px_), "value": _f(val),
                         "weight": _f(val / nav * 100, 2) if nav else None,
                         "pnl": _f(val - h.get("cost", val))})
        weights[sym] = weights.get(sym, 0.0) + (val / nav if nav else 0.0)
    tre_value = sum(r["value"] or 0 for r in tre_rows)

    # univers « Marchés » : chaque actif une seule fois, avec ses groupes
    member: dict[str, list[str]] = {}
    for g, syms in groups.items():
        for s in syms:
            member.setdefault(s, [])
            if g not in member[s]:
                member[s].append(g)
    for r in pos:
        member.setdefault(r["symbol"], []).append("En portefeuille")
    for r in aip:
        member.setdefault(r["symbol"], []).append("Poche IA (position)")

    t0 = time.time()
    idea_map = {i["symbol"]: i for i in (ideas or {}).get("ideas", [])}
    priority = ([r["symbol"] for r in pos] + [r["symbol"] for r in aip]
                + groups.get("Actions screenées", []) + list(idea_map) + list(member))
    seen, order = set(), []
    for s in priority:
        if s not in seen:
            seen.add(s)
            order.append(s)
    ivs = iv_lookup(client, order, {s: float(data[s]["Close"].iloc[-1]) for s in order if s in data},
                    budget_s)
    assets = []
    for s in order:
        df = data.get(s)
        if df is None:
            continue
        try:
            a = asset_stats(s, df)
        except Exception:
            continue
        a["groups"] = member.get(s, [])
        a["asset_class"] = asset_class(s)
        a["sector"] = sector_of(s, sector_lookup if time.time() - t0 < budget_s else None)
        a["name"] = ETF_INFO.get(s, ("", ""))[1] if s in ETF_INFO else ""
        a["iv"] = ivs.get(s)
        a["vol_source"] = "implicite (Alpaca)" if s in ivs else "réalisée 20 j"
        a["earnings"] = None
        if earnings_fn and not is_crypto(s) and s not in ETF_INFO and time.time() - t0 < budget_s * 1.5:
            try:
                a["earnings"] = earnings_fn(s)
            except Exception:
                pass
        idea = idea_map.get(s)
        a["news"] = ((news or {}).get(s) or (idea or {}).get("headlines") or [])[:4]
        if idea:
            a["idea"] = {k: idea.get(k) for k in ("score", "mentions", "sentiment", "why",
                                                     "risk", "niche")}
        assets.append(a)

    # répartition
    by_class: dict[str, float] = {}
    by_sector: dict[str, float] = {}
    for r in pos:
        by_class[r["asset_class"]] = by_class.get(r["asset_class"], 0) + (r["value"] or 0)
        by_sector[r["sector"]] = by_sector.get(r["sector"], 0) + (r["value"] or 0)
    for r in tre_rows:
        by_class[r["label"]] = by_class.get(r["label"], 0) + (r["value"] or 0)
    by_class["Cash"] = account.cash

    risk = portfolio_risk(weights, data, nav)
    risk["used_risk_usd"] = _f(sum(r["risk_usd"] or 0 for r in pos))
    corr_syms = [r["symbol"] for r in pos] + [s for s in groups.get("Univers du fonds", [])]
    risk["corr"] = correlation(data, list(dict.fromkeys(corr_syms))[:16])
    g = (ai or {}).get("greeks") or {}
    tot = g.get("total") or {}
    risk["stress"].append({"name": "Volatilité +10 pts (poche options)",
                           "pnl": _f((tot.get("vega") or 0) * 10),
                           "pct": _f((tot.get("vega") or 0) * 10 / nav * 100, 2)})

    lim = limits or {}
    lim_rows = [
        {"name": "Positions ouvertes (fonds)", "used": len(pos), "max": lim.get("max_positions", 8),
         "unit": ""},
        {"name": "Exposition des stratégies", "used": _f(invested / nav * 100, 1), "max": 100, "unit": "%"},
        {"name": "Risque engagé jusqu'aux stops", "used": _f(risk["used_risk_usd"] / nav * 100, 2),
         "max": _f(lim.get("risk_pct", 1.0) * lim.get("max_positions", 8), 2), "unit": "%"},
    ]
    al = ai_limits or {}
    if al:
        eq = al.get("equity") or nav
        lim_rows += [
            {"name": "Trades IA ouverts", "used": len(aip), "max": al.get("max_trades", 5), "unit": ""},
            {"name": "|Δ $| poche IA / capital", "used": _f(abs(tot.get("delta_usd") or 0) / eq * 100, 2),
             "max": _f(al.get("max_delta_pct", 0.5) * 100, 1), "unit": "%"},
            {"name": "|Vega| poche IA (par pt de vol)", "used": _f(abs(tot.get("vega") or 0) / eq * 100, 3),
             "max": _f(al.get("max_vega_pct", 0.01) * 100, 2), "unit": "%"},
            {"name": "Θ poche IA par jour", "used": _f(max(-(tot.get("theta_day") or 0), 0) / eq * 100, 3),
             "max": _f(al.get("max_theta_pct", 0.002) * 100, 2), "unit": "%"},
        ]
    risk["limits"] = lim_rows

    # activité
    hist = list(account.history)
    closed = [h for h in hist if h.get("pnl") is not None]
    wins = [h["pnl"] for h in closed if h["pnl"] > 0]
    losses = [h["pnl"] for h in closed if h["pnl"] <= 0]
    try:
        first = date.fromisoformat(str(account.created)[:10])
        months = max((date.today() - first).days / 30.44, 1)
    except ValueError:
        months = 1
    activity = {
        "trades": len(closed), "per_month": _f(len(closed) / months, 1),
        "win_rate": _f(len(wins) / len(closed) * 100, 0) if closed else None,
        "avg_win": _f(np.mean(wins)) if wins else None,
        "avg_loss": _f(np.mean(losses)) if losses else None,
        "profit_factor": _f(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None,
        "realized": _f(sum(h["pnl"] for h in closed)),
        "history": [{k: (_f(v) if isinstance(v, float) else v) for k, v in h.items()
                     if k in ("symbol", "strategy", "entry_date", "exit_date", "entry", "exit",
                              "pnl", "pnl_pct", "reason")} for h in reversed(hist[-60:])],
        "ai_closed": [{k: t.get(k) for k in ("underlying", "strategy", "opened_at", "closed_at",
                                              "pnl", "close_reason", "reasoning")}
                      for t in reversed((ai or {}).get("closed", [])[-40:])],
        "ai_realized": (ai or {}).get("realized"), "ai_win_rate": (ai or {}).get("win_rate"),
    }

    curve = [[d, _f(v)] for d, v in account.equity_curve]
    cs = _curve_stats(account.equity_curve, spy)

    backtest = None
    if fund_res is not None:
        try:
            from . import factsheet as fsh
            gross = fsh.nav(fund_res.equity)
            net = fsh.apply_fees(gross, (factsheet or {}).get("mgmt", fsh.MGMT_FEE),
                                 (factsheet or {}).get("perf", fsh.PERF_FEE))
            step = max(len(gross) // 400, 1)
            sp = None
            if spy is not None:
                s2 = spy.copy()
                s2.index = pd.to_datetime(s2.index).tz_localize(None).normalize()
                gi = pd.to_datetime(gross.index).tz_localize(None).normalize()
                s2 = s2.reindex(gi, method="ffill")
                sp = (s2 / s2.dropna().iloc[0] * 100).to_numpy()
            backtest = {"d": [d.strftime("%Y-%m-%d") for d in gross.index[::step]],
                        "gross": [_f(v) for v in gross.to_numpy()[::step]],
                        "net": [_f(v) for v in net.to_numpy()[::step]],
                        "spy": [_f(v) for v in sp[::step]] if sp is not None else None}
        except Exception:
            backtest = None

    macro = []
    for sym, label in MACRO.items():
        df = data.get(sym)
        if df is None:
            continue
        c = df["Close"].dropna()
        macro.append({"symbol": sym, "label": label, "last": _f(c.iloc[-1]),
                      "chg_1d": _f((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2) if len(c) > 1 else None,
                      "chg_1m": _f((c.iloc[-1] / c.iloc[-22] - 1) * 100, 2) if len(c) > 22 else None,
                      "spark": [_f(v, 3) for v in c.tail(60)]})

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "risk_free": RISK_FREE,
        "overview": {"nav": _f(nav), "initial": account.initial_capital,
                     "pnl": _f(nav - account.initial_capital),
                     "pnl_pct": _f((nav / account.initial_capital - 1) * 100, 2),
                     "cash": _f(account.cash), "cash_pct": _f(account.cash / nav * 100, 1),
                     "gross": _f(invested / nav * 100, 1), "net": _f(invested / nav * 100, 1),
                     "treasury": _f(tre_value), "treasury_pct": _f(tre_value / nav * 100, 1) if nav else None,
                     "n_pos": len(pos), "n_ai": len(aip), "created": str(account.created)[:10],
                     "last_cycle": str(getattr(account, "last_cycle", ""))[:16],
                     "by_class": {k: _f(v) for k, v in by_class.items()},
                     "by_sector": {k: _f(v) for k, v in sorted(by_sector.items(), key=lambda x: -x[1])},
                     "alpaca": alpaca_lines or [], **cs},
        "curve": curve,
        "positions": pos, "ai_positions": aip, "treasury": tre_rows,
        "ai_greeks": g, "risk": risk, "assets": assets,
        "ideas": (ideas or {}), "activity": activity,
        "factsheet": {k: v for k, v in (factsheet or {}).items() if k != "monthly_html"},
        "monthly_html": (factsheet or {}).get("monthly_html", ""),
        "backtest": backtest, "gauge": gauge or {}, "macro": macro,
        "headlines": headlines or [],
    }


def _clean(o):
    """NaN/inf → None, types numpy → Python (JSON strict)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def render(model: dict, template_path: str = TEMPLATE) -> str:
    with open(template_path, encoding="utf-8") as f:
        tpl = f.read()
    payload = json.dumps(_clean(model), ensure_ascii=False, allow_nan=False) \
        .replace("</", "<\\/")
    return tpl.replace("/*__DATA__*/null", payload)


def save(model: dict, path: str = OUT_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(model))
    return path
