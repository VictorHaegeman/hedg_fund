"""Couche « fonds d'investissement » — valeur de part, frais, indicateurs de risque, factsheet.

Ce qu'un investisseur (ou un régulateur) attend d'un fonds, calculé sur une courbe
d'équité quotidienne :

  • Valeur liquidative (VL) par part, base 100 à la création
  • Frais : gestion (ex. 2 %/an, prélevés chaque jour) et performance (ex. 20 % de
    la surperformance au-dessus du plus haut historique net — « high-water mark » —
    cristallisée chaque fin d'année) → performance NETTE de frais
  • Rendement annualisé, volatilité, Sharpe, Sortino, Calmar, max drawdown
  • VaR et CVaR historiques à 95 % (perte journalière à ne dépasser que 5 % du temps,
    et perte moyenne au-delà)
  • Bêta, alpha de Jensen (CAPM) et corrélation vs un indice de référence (SPY)
  • Tableau des rendements mensuels, meilleur/pire mois, % de mois positifs
  • Profils de risque (prudent / équilibré / dynamique) = même stratégie, risque
    par trade différent

⚠ Une performance simulée ou en paper trading n'est pas un historique réel. Commercialiser
un fonds auprès de tiers en France exige un agrément AMF (société de gestion).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_PROFILES = {"Prudent": 0.5, "Équilibré": 1.0, "Dynamique": 1.5}


def nav(equity: pd.Series, base: float = 100.0) -> pd.Series:
    """Valeur liquidative par part, base 100 au premier jour."""
    return equity / float(equity.iloc[0]) * base


def apply_fees(gross_nav: pd.Series, mgmt: float = 0.02, perf: float = 0.20) -> pd.Series:
    """VL nette de frais de gestion (quotidiens) et de performance (high-water mark,
    cristallisée au dernier jour de chaque année civile)."""
    rets = gross_nav.pct_change().fillna(0.0).to_numpy()
    idx = gross_nav.index
    daily_mgmt = mgmt / TRADING_DAYS
    out = np.empty(len(rets))
    v = float(gross_nav.iloc[0])
    hwm = v
    for i, r in enumerate(rets):
        if i > 0:
            v *= (1 + r) * (1 - daily_mgmt)
        year_end = i == len(rets) - 1 or idx[i].year != idx[i + 1].year
        if year_end and v > hwm:
            v -= perf * (v - hwm)
            hwm = v
        out[i] = v
    return pd.Series(out, index=idx, name="nav_net")


def _ann_return(s: pd.Series) -> float:
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (float(s.iloc[-1]) / float(s.iloc[0])) ** (1 / yrs) - 1 if yrs > 0 else 0.0


def metrics(equity: pd.Series, bench: pd.Series | None = None, rf: float = 0.0) -> dict:
    """Indicateurs standard d'un fonds sur une courbe d'équité/VL quotidienne."""
    eq = equity.dropna()
    r = eq.pct_change().dropna()
    if len(r) < 2:
        return {}
    ann = _ann_return(eq)
    vol = float(r.std() * math.sqrt(TRADING_DAYS))
    ex = r - rf / TRADING_DAYS
    downside = float(np.sqrt((np.minimum(ex, 0) ** 2).mean()) * math.sqrt(TRADING_DAYS))
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    var95 = float(-np.percentile(r, 5))
    cvar95 = float(-r[r <= -var95].mean()) if (r <= -var95).any() else var95
    monthly = eq.resample("ME").last().pct_change().dropna()
    out = {
        "ann_return": ann, "volatility": vol,
        "sharpe": float(ex.mean() / r.std() * math.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0,
        "sortino": float(ex.mean() * TRADING_DAYS / downside) if downside > 0 else 0.0,
        "max_drawdown": mdd, "calmar": ann / abs(mdd) if mdd < 0 else float("nan"),
        "var95_daily": var95, "cvar95_daily": cvar95,
        "best_month": float(monthly.max()) if len(monthly) else float("nan"),
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
        "pct_positive_months": float((monthly > 0).mean()) if len(monthly) else float("nan"),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
    }
    if bench is not None:
        b = bench.reindex(eq.index).ffill().pct_change()
        both = pd.concat([r, b], axis=1, keys=["f", "b"]).dropna()
        if len(both) > 20 and both["b"].var() > 0:
            beta = float(both.cov().iloc[0, 1] / both["b"].var())
            alpha = float((both["f"].mean() - rf / TRADING_DAYS
                           - beta * (both["b"].mean() - rf / TRADING_DAYS)) * TRADING_DAYS)
            out.update(beta=beta, alpha=alpha, correlation=float(both.corr().iloc[0, 1]),
                       bench_ann_return=_ann_return(bench.reindex(eq.index).ffill().dropna()))
    return out


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    """Rendements mensuels (%) : une ligne par année, colonnes Jan…Déc + Année."""
    m = equity.resample("ME").last()
    rets = m.pct_change()
    if len(m):   # premier mois : depuis le premier point de la courbe
        rets.iloc[0] = m.iloc[0] / equity.iloc[0] - 1
    df = pd.DataFrame({"y": rets.index.year, "m": rets.index.month, "r": rets.to_numpy() * 100})
    tab = df.pivot(index="y", columns="m", values="r")
    names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    tab.columns = [names[c - 1] for c in tab.columns]
    yearly = equity.resample("YE").last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / equity.iloc[0] - 1
    tab["Année"] = (yr.to_numpy() * 100)[: len(tab)]
    tab.index.name = None
    return tab.round(1)


def activity(trades: pd.DataFrame | list, equity: pd.Series) -> dict:
    """Activité de trading : trades par an/mois, durée de détention, rotation annuelle.

    Rotation (turnover) = valeur échangée (achats + ventes) / 2 / équité moyenne, par an
    — la mesure utilisée par l'industrie pour comparer l'activité de fonds différents."""
    df = pd.DataFrame(trades)
    if df.empty or equity is None or len(equity) < 2:
        return {}
    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    traded = (df["entry"] * df["shares"]).sum() + (df["exit"] * df["shares"]).sum()
    hold = df["bars_held"] if "bars_held" in df else (
        pd.to_datetime(df["exit_date"]) - pd.to_datetime(df["entry_date"])).dt.days
    return {"trades": int(len(df)), "trades_per_year": len(df) / yrs,
            "trades_per_month": len(df) / yrs / 12,
            "avg_holding_days": float(hold.mean()), "median_holding_days": float(hold.median()),
            "turnover": float(traded / 2 / equity.mean() / yrs)}


def activity_line(a: dict) -> str:
    if not a:
        return "  Pas encore de trades clôturés."
    return (f"  {a['trades']} trades | {a['trades_per_year']:.0f}/an ({a['trades_per_month']:.1f}/mois) | "
            f"détention moyenne {a['avg_holding_days']:.0f} j (médiane {a['median_holding_days']:.0f}) | "
            f"rotation {a['turnover'] * 100:.0f} %/an")


def _pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:+.{d}f} %"


def summary_rows(m: dict) -> list[tuple[str, str]]:
    rows = [("Rendement annualisé", _pct(m.get("ann_return"))),
            ("Volatilité annualisée", _pct(m.get("volatility")).replace("+", "")),
            ("Ratio de Sharpe", f"{m.get('sharpe', float('nan')):.2f}"),
            ("Ratio de Sortino", f"{m.get('sortino', float('nan')):.2f}"),
            ("Perte maximale (max drawdown)", _pct(m.get("max_drawdown"))),
            ("Ratio de Calmar", f"{m.get('calmar', float('nan')):.2f}"),
            ("VaR 95 % (1 jour)", _pct(-m.get("var95_daily", float("nan")), 2)),
            ("CVaR 95 % (1 jour)", _pct(-m.get("cvar95_daily", float("nan")), 2)),
            ("Meilleur / pire mois", f"{_pct(m.get('best_month'))} / {_pct(m.get('worst_month'))}"),
            ("Mois positifs", f"{m.get('pct_positive_months', float('nan')) * 100:.0f} %")]
    if "beta" in m:
        rows += [("Bêta vs S&P 500", f"{m['beta']:.2f}"),
                 ("Alpha annuel (Jensen)", _pct(m["alpha"])),
                 ("Corrélation vs S&P 500", f"{m['correlation']:.2f}")]
    return rows


def report(fund_equity: pd.Series, bench: pd.Series | None, live_curve: list | None = None,
           mgmt: float = 0.02, perf: float = 0.20, profiles: dict | None = None,
           trades=None) -> str:
    gross = nav(fund_equity)
    net = apply_fees(gross, mgmt, perf)
    mg, mn = metrics(gross, bench), metrics(net, bench)
    lines = ["=" * 78, "FICHE DU FONDS (factsheet) — simulation sur données réelles", "=" * 78,
             f"Période : {gross.index[0]:%Y-%m-%d} → {gross.index[-1]:%Y-%m-%d} | VL base 100 → "
             f"brute {gross.iloc[-1]:,.1f} | nette {net.iloc[-1]:,.1f} "
             f"(frais {mgmt:.0%} gestion + {perf:.0%} performance, high-water mark)", "",
             f"  {'Indicateur':<32}{'Brut':>14}{'Net de frais':>16}"]
    for (k, vg), (_, vn) in zip(summary_rows(mg), summary_rows(mn)):
        lines.append(f"  {k:<32}{vg:>14}{vn:>16}")
    if "bench_ann_return" in mg:
        lines.append(f"  {'Indice de référence (SPY)':<32}{_pct(mg['bench_ann_return']):>14}")
    if trades is not None:
        lines += ["", "— Activité de trading (simulation) —",
                  activity_line(activity(trades, fund_equity))]
    lines += ["", "— Rendements mensuels nets (%) —", monthly_table(net).to_string()]
    if live_curve and len(live_curve) >= 2:
        s = pd.Series([v for _, v in live_curve], index=pd.to_datetime([d for d, _ in live_curve]))
        ml = metrics(s, bench)
        lines += ["", f"— Historique RÉEL du compte papier ({s.index[0]:%Y-%m-%d} → "
                      f"{s.index[-1]:%Y-%m-%d}) —"]
        lines += [f"  {k:<32}{v:>14}" for k, v in summary_rows(ml)[:6]]
    if profiles:
        lines += ["", "— Profils de risque (même stratégie, risque par trade différent) —",
                  f"  {'Profil':<12}{'Risque/trade':>13}{'Rendement/an':>14}{'Volatilité':>12}"
                  f"{'Sharpe':>8}{'Pire baisse':>13}"]
        for name, (risk, eq) in profiles.items():
            m = metrics(eq)
            lines.append(f"  {name:<12}{risk:>12.2f}%{_pct(m['ann_return']):>14}"
                         f"{_pct(m['volatility']).replace('+', ''):>12}{m['sharpe']:>8.2f}"
                         f"{_pct(m['max_drawdown']):>13}")
    lines += ["", "⚠ Performance simulée (backtest) et paper trading : pas un historique réel.",
              "  Commercialiser un fonds en France nécessite un agrément AMF."]
    return "\n".join(lines)
