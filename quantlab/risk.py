"""Module 3 — Analyse risque/rendement d'une stratégie backtestée."""

from __future__ import annotations

import numpy as np

from .backtest import BacktestResult
from .drawdown import drawdown_episodes, drawdown_series


def report(res: BacktestResult, risk_pct: float = 1.0) -> str:
    t = res.trades
    m = res.metrics
    lines = [
        "=" * 78,
        "MODULE 3 — ANALYSE RISQUE / RENDEMENT",
        f"Stratégie : {res.strategy_name} | Actif : {res.symbol}",
        "=" * 78,
        "",
        "— Risque par trade —",
        f"  Risque défini      : {risk_pct:.1f}% de l'équité courante (distance au stop)",
    ]
    if len(t):
        worst = t["pnl_pct"].min()
        r = t["r_multiple"].dropna()
        lines += [
            f"  Pire trade réalisé : {worst:.1f}% du prix d'entrée "
            f"({t['pnl'].min():,.0f} $)",
            f"  R-multiple moyen   : {r.mean():.2f}R | médian : {r.median():.2f}R"
            if len(r) else "",
            "",
            "— Ratio rendement/risque —",
            f"  Gain moyen / |perte moyenne| : "
            f"{abs(m['avg_win'] / m['avg_loss']):.2f}:1" if m["avg_loss"] != 0 else
            "  Aucune perte enregistrée",
            f"  Win rate : {m['win_rate'] * 100:.1f}% → espérance par trade : "
            f"{(m['win_rate'] * m['avg_win'] + (1 - m['win_rate']) * m['avg_loss']):,.0f} $",
            f"  Profit factor : {m['profit_factor']:.2f} "
            f"({'sain (>1.5)' if m['profit_factor'] > 1.5 else 'acceptable (>1.2)' if m['profit_factor'] > 1.2 else 'fragile (<1.2)'})",
        ]
    dd = drawdown_series(res.equity)
    eps = drawdown_episodes(res.equity)
    lines += [
        "",
        "— Patterns de drawdown —",
        f"  Max drawdown : {dd.min() * 100:.1f}% | épisodes >1% : {len(eps)}",
    ]
    if len(eps) >= 3:
        depths = eps["profondeur_%"]
        lines.append(
            f"  Distribution des profondeurs : médiane {depths.median():.1f}%, "
            f"pire {depths.min():.1f}% — "
            + ("queue épaisse : quelques épisodes dominent le risque total"
               if abs(depths.min()) > 3 * abs(depths.median())
               else "profil régulier, pas de queue extrême")
        )
    # Ratio rendement annuel / drawdown
    lines += [
        f"  CAGR / |MaxDD| (Calmar) : {m['calmar']:.2f} "
        f"({'bon (>0.5)' if m['calmar'] > 0.5 else 'à améliorer (<0.5)'})",
        "",
        "— 3 améliorations pour réduire le risque —",
        "  1. Filtre de volatilité : ne pas ouvrir de nouveau trade quand la volatilité",
        "     réalisée 20 j dépasse son 80e percentile historique — les pires pertes",
        "     surviennent presque toutes en régime de haute volatilité (voir module 4).",
        "  2. Stop suiveur en plus du stop initial : remonter le stop à breakeven après",
        "     +1R, puis suivre à 2×ATR — convertit des trades gagnants-puis-perdants",
        "     en trades neutres ou gagnants.",
        f"  3. Réduire le risque par trade de {risk_pct:.1f}% à {max(risk_pct / 2, 0.5):.1f}% pendant un drawdown",
        "     d'équité supérieur à 10 % (anti-martingale) — la profondeur du DD max baisse",
        "     quasi proportionnellement.",
        "",
        "— 2 façons d'augmenter le rendement SANS augmenter le risque —",
        "  1. Diversifier sur 3–5 actifs peu corrélés avec le même budget de risque",
        "     total : plus d'opportunités de trades pour le même risque par position",
        "     (le Sharpe du portefeuille monte avec la racine du nombre de paris",
        "     indépendants).",
        "  2. Réinvestir la composante cash : la stratégie est souvent hors marché ;",
        "     placer le cash inutilisé en monétaire/T-bills ajoute ~rendement sans",
        "     risque de marché ajouté (le stop sizing reste inchangé).",
    ]
    return "\n".join([l for l in lines if l is not None])
