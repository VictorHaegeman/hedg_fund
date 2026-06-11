"""Module 10 — Analyse des drawdowns : profondeur, durée, récupération, recommandations."""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def drawdown_episodes(equity: pd.Series, min_depth: float = 0.01) -> pd.DataFrame:
    """Liste les épisodes de drawdown (pic → creux → récupération)."""
    dd = drawdown_series(equity)
    episodes = []
    in_dd = False
    peak_date = trough_date = None
    trough = 0.0
    for date, val in dd.items():
        if not in_dd and val < 0:
            in_dd = True
            peak_date, trough_date, trough = date, date, val
        elif in_dd:
            if val < trough:
                trough, trough_date = val, date
            if val == 0:
                episodes.append({
                    "pic": peak_date, "creux": trough_date, "récupéré": date,
                    "profondeur_%": round(trough * 100, 1),
                    "jours_chute": (trough_date - peak_date).days,
                    "jours_récup": (date - trough_date).days,
                    "durée_totale_j": (date - peak_date).days,
                })
                in_dd = False
    if in_dd:  # drawdown en cours, non récupéré
        episodes.append({
            "pic": peak_date, "creux": trough_date, "récupéré": pd.NaT,
            "profondeur_%": round(trough * 100, 1),
            "jours_chute": (trough_date - peak_date).days,
            "jours_récup": None, "durée_totale_j": None,
        })
    df = pd.DataFrame(episodes)
    if len(df):
        df = df[df["profondeur_%"] <= -min_depth * 100]
    return df


def report(res: BacktestResult) -> str:
    eq = res.equity
    eps = drawdown_episodes(eq)
    dd = drawdown_series(eq)
    lines = [
        "=" * 78,
        "MODULE 10 — ANALYSE DES DRAWDOWNS",
        f"Stratégie : {res.strategy_name} | Actif : {res.symbol}",
        "=" * 78,
        "",
        f"Max drawdown          : {dd.min() * 100:.1f}%",
        f"Drawdown actuel       : {dd.iloc[-1] * 100:.1f}%",
        f"Temps passé en DD >5% : {(dd < -0.05).mean() * 100:.0f}% des jours",
    ]
    if len(eps):
        recovered = eps.dropna(subset=["jours_récup"])
        lines += [
            f"Épisodes de DD > 1%   : {len(eps)}",
            f"Profondeur moyenne    : {eps['profondeur_%'].mean():.1f}%",
        ]
        if len(recovered):
            lines.append(
                f"Récupération moyenne  : {recovered['jours_récup'].mean():.0f} jours "
                f"(max {recovered['jours_récup'].max():.0f} j)"
            )
        top = eps.nsmallest(5, "profondeur_%")
        lines += ["", "5 pires drawdowns :", top.to_string(index=False)]
    lines += [
        "",
        "3 façons de réduire les drawdowns :",
        "  1. Filtre de régime : couper l'exposition quand le prix passe sous la SMA 200",
        "     ou quand la volatilité réalisée dépasse son 80e percentile.",
        "  2. Stop d'équité : réduire la taille des positions de moitié après un DD de 10 %",
        "     sur la courbe d'équité, reprendre la taille normale après un nouveau plus haut.",
        "  3. Diversification : répartir le risque sur plusieurs actifs/stratégies peu corrélés",
        "     au lieu d'un seul instrument (voir module 7).",
        "",
        "Améliorations du position sizing :",
        "  • Volatility targeting : taille ∝ vol cible / ATR — positions réduites quand le",
        "    marché devient nerveux, ce qui écrête mécaniquement les pires drawdowns.",
        "  • Risque fractionnel sur l'équité courante (déjà implémenté) plutôt que sur le",
        "    capital initial : la taille diminue automatiquement pendant les pertes.",
        "  • Plafonner le risque total ouvert (somme des distances aux stops) à ~5 % de l'équité.",
    ]
    return "\n".join(lines)
