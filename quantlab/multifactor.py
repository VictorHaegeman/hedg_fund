"""Module 5 — Stratégie multi-facteurs : momentum, value, volatilité, trend.

Scoring cross-sectionnel en z-scores sur un univers d'actifs, pondérations
explicites, rebalancement mensuel, portefeuille exemple sur les données chargées.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

WEIGHTS = {"momentum": 0.40, "value": 0.20, "volatility": 0.20, "trend": 0.20}

FORMULAS = {
    "momentum": "rendement 12 mois hors dernier mois : Close[-21] / Close[-252] − 1 "
                "(le dernier mois est exclu pour éviter le retour à la moyenne court terme)",
    "value": "proxy de cherté : −(Close / moyenne 3 ans − 1) — plus l'actif est sous sa "
             "moyenne longue, plus le score value est élevé (mean reversion long terme)",
    "volatility": "−volatilité réalisée 60 j annualisée — les actifs calmes sont surpondérés "
                  "(anomalie low-vol)",
    "trend": "Close / SMA200 − 1 — prime aux actifs au-dessus de leur tendance long terme",
}


def factor_scores(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calcule les 4 facteurs bruts par actif à la dernière date disponible."""
    rows = {}
    for sym, df in prices.items():
        c = df["Close"]
        if len(c) < 260:
            continue
        mom = float(c.iloc[-21] / c.iloc[-252] - 1)
        lt_mean = float(c.tail(756).mean()) if len(c) >= 504 else float(c.mean())
        value = -float(c.iloc[-1] / lt_mean - 1)
        vol = -float(ind.realized_vol(c, 60).iloc[-1])
        trend = float(c.iloc[-1] / ind.sma(c, 200).iloc[-1] - 1)
        rows[sym] = {"momentum": mom, "value": value, "volatility": vol, "trend": trend}
    return pd.DataFrame(rows).T


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0).replace(0, np.nan)


def composite(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = factor_scores(prices)
    z = zscore(raw).clip(-3, 3)
    z["score"] = sum(z[f] * w for f, w in WEIGHTS.items())
    return pd.concat([raw.round(3), z[["score"]].round(2)], axis=1).sort_values(
        "score", ascending=False)


def example_portfolio(scores: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    """Top-N pondéré par score décalé en positif (score-min +.1), normalisé à 100 %."""
    top = scores.head(top_n).copy()
    shifted = top["score"] - top["score"].min() + 0.1
    top["allocation_%"] = (shifted / shifted.sum() * 100).round(1)
    return top[["score", "allocation_%"]]


def report(prices: dict[str, pd.DataFrame], sources: dict[str, str]) -> str:
    scores = composite(prices)
    port = example_portfolio(scores)
    lines = [
        "=" * 78,
        "MODULE 5 — STRATÉGIE MULTI-FACTEURS",
        f"Univers : {', '.join(prices)} "
        f"(sources : {', '.join(sorted(set(sources.values())))})",
        "=" * 78,
        "",
        "Formule exacte de chaque facteur :",
        *[f"  • {f.upper():<10} (poids {w * 100:.0f}%) : {FORMULAS[f]}"
          for f, w in WEIGHTS.items()],
        "",
        "Score composite = Σ poids × z-score(facteur), z-scores bornés à ±3.",
        "Rebalancement : MENSUEL (1er jour de bourse du mois) — assez fréquent pour",
        "suivre le momentum, assez lent pour limiter les frais (~12 rotations/an).",
        "",
        "Scores actuels (facteurs bruts + score composite) :",
        scores.to_string(),
        "",
        f"Portefeuille exemple (top {len(port)} par score, pondéré par score) :",
        port.to_string(),
        "",
        "Règles d'exécution : à chaque rebalancement, vendre les actifs sortis du top,",
        "acheter les entrants, ramener chaque ligne à son allocation cible si l'écart",
        "dépasse 2 points de pourcentage (bande de tolérance anti-frais).",
    ]
    return "\n".join(lines)
