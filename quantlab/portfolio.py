"""Module 7 — Construction de portefeuille diversifié.

Allocation par inverse de volatilité, ajustée à la tolérance au risque,
avec rendement attendu, volatilité, max drawdown, corrélations et justification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import max_drawdown

RISK_PROFILES = {
    # part actifs risqués max, vol cible annualisée
    "low": {"max_risky": 0.50, "target_vol": 0.08},
    "medium": {"max_risky": 0.80, "target_vol": 0.12},
    "high": {"max_risky": 1.00, "target_vol": 0.18},
}

ASSET_NOTES = {
    "SPY": "cœur actions US large-cap — moteur de rendement long terme",
    "QQQ": "tech/croissance US — rendement élevé, drawdowns plus profonds",
    "EFA": "actions développées hors US — diversification géographique",
    "TLT": "obligations longues US — amortisseur en récession (corrélation souvent négative)",
    "IEF": "obligations 7-10 ans — ballast défensif moins volatil que TLT",
    "GLD": "or — couverture inflation/crises, faible corrélation aux actions",
    "BTC-USD": "crypto — convexité forte, petite dose suffit (vol extrême)",
    "ETH-USD": "crypto — bêta élevé sur l'adoption blockchain, très volatil",
}


def asset_stats(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    rets = {}
    for sym, df in prices.items():
        c = df["Close"]
        r = c.pct_change().dropna()
        years = len(c) / 252
        rows[sym] = {
            "CAGR_%": round(((c.iloc[-1] / c.iloc[0]) ** (1 / years) - 1) * 100, 1),
            "Vol_%": round(r.std() * np.sqrt(252) * 100, 1),
            "MaxDD_%": round(max_drawdown(c) * 100, 1),
        }
        rets[sym] = r
    stats = pd.DataFrame(rows).T
    ret_df = pd.DataFrame(rets).dropna()
    return stats, ret_df


def build(prices: dict[str, pd.DataFrame], risk_tolerance: str = "medium") -> dict:
    profile = RISK_PROFILES[risk_tolerance]
    stats, ret_df = asset_stats(prices)

    # Pondération inverse-volatilité
    inv_vol = 1.0 / stats["Vol_%"]
    w = inv_vol / inv_vol.sum()

    # Scaling vers la vol cible via la matrice de covariance réelle
    cov = ret_df.cov() * 252
    port_vol = float(np.sqrt(w.values @ cov.values @ w.values))
    scale = min(profile["target_vol"] / port_vol, 1.0) if port_vol > 0 else 1.0
    w_scaled = w * scale  # le reste en cash
    cash = max(0.0, 1.0 - w_scaled.sum())

    exp_ret = float((w_scaled * stats["CAGR_%"] / 100).sum() + cash * 0.04)  # cash ~4%
    final_vol = port_vol * scale

    # Drawdown du portefeuille simulé (poids constants, rebalancement implicite quotidien)
    port_curve = (1 + (ret_df * w_scaled).sum(axis=1)).cumprod()
    port_mdd = max_drawdown(port_curve)

    alloc = pd.DataFrame({
        "allocation_%": (w_scaled * 100).round(1),
        "CAGR_hist_%": stats["CAGR_%"],
        "Vol_%": stats["Vol_%"],
        "MaxDD_%": stats["MaxDD_%"],
    }).sort_values("allocation_%", ascending=False)

    return {
        "alloc": alloc, "cash_pct": round(cash * 100, 1),
        "expected_return": exp_ret, "expected_vol": final_vol,
        "expected_mdd": port_mdd, "corr": ret_df.corr().round(2),
        "profile": risk_tolerance,
    }


def report(prices: dict[str, pd.DataFrame], risk_tolerance: str = "medium",
           horizon_years: int = 3, capital: float = 10_000.0) -> str:
    p = build(prices, risk_tolerance)
    alloc = p["alloc"].copy()
    alloc["montant_$"] = (alloc["allocation_%"] / 100 * capital).round(0)
    lines = [
        "=" * 78,
        "MODULE 7 — CONSTRUCTION DE PORTEFEUILLE",
        f"Tolérance au risque : {risk_tolerance} | Horizon : {horizon_years} ans | "
        f"Capital : ${capital:,.0f}",
        f"Méthode : inverse de volatilité + scaling vers vol cible "
        f"({RISK_PROFILES[risk_tolerance]['target_vol'] * 100:.0f}% annualisée)",
        "=" * 78,
        "",
        "ALLOCATION :",
        alloc.to_string(),
        f"Cash / monétaire : {p['cash_pct']:.1f}% "
        f"(${p['cash_pct'] / 100 * capital:,.0f})",
        "",
        f"Rendement attendu : {p['expected_return'] * 100:.1f}%/an "
        "(CAGR historiques pondérés — pas une garantie)",
        f"Risque attendu    : volatilité {p['expected_vol'] * 100:.1f}%/an | "
        f"max drawdown simulé {p['expected_mdd'] * 100:.1f}%",
        "",
        "Matrice de corrélation (rendements quotidiens) :",
        p["corr"].to_string(),
        "",
        "Pourquoi chaque actif est inclus :",
    ]
    for sym in alloc.index:
        note = ASSET_NOTES.get(sym, "diversification — profil rendement/risque complémentaire")
        lines.append(f"  • {sym:<8}: {note}")
    lines += [
        "",
        "Rebalancement recommandé : trimestriel, ou si une ligne dévie de ±20 % de sa cible.",
    ]
    return "\n".join(lines)
