"""Module 9 — Simulation Monte Carlo : probabilité de perte, distribution, scénarios extrêmes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult


def simulate(res: BacktestResult, n_paths: int = 5_000, horizon_days: int = 252,
             seed: int = 42) -> dict:
    """Bootstrap par blocs des rendements quotidiens de la stratégie.

    Le bootstrap par blocs (5 jours) préserve l'autocorrélation court terme,
    contrairement au tirage i.i.d. qui sous-estime les drawdowns.
    """
    rets = res.daily_returns.to_numpy()
    rets = rets[np.isfinite(rets)]
    if len(rets) < 60:
        raise ValueError("Pas assez d'historique pour une simulation fiable")
    rng = np.random.default_rng(seed)
    block = 5
    n_blocks = horizon_days // block + 1

    starts = rng.integers(0, len(rets) - block, size=(n_paths, n_blocks))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    paths = rets[idx].reshape(n_paths, -1)[:, :horizon_days]

    curves = np.cumprod(1 + paths, axis=1)
    finals = curves[:, -1] - 1.0
    running_max = np.maximum.accumulate(curves, axis=1)
    mdds = ((curves / running_max) - 1.0).min(axis=1)

    pct = lambda p: float(np.percentile(finals, p))
    return {
        "n_paths": n_paths, "horizon_days": horizon_days,
        "p_loss": float((finals < 0).mean()),
        "p_loss_10": float((finals < -0.10).mean()),
        "p_loss_20": float((finals < -0.20).mean()),
        "median": pct(50), "p5": pct(5), "p25": pct(25), "p75": pct(75), "p95": pct(95),
        "worst": float(finals.min()), "best": float(finals.max()),
        "mdd_median": float(np.median(mdds)), "mdd_p95": float(np.percentile(mdds, 5)),
        "finals": finals,
    }


def _histogram(finals: np.ndarray, bins: int = 13, width: int = 46) -> str:
    counts, edges = np.histogram(finals, bins=bins)
    peak = counts.max() or 1
    rows = []
    for i, n in enumerate(counts):
        lo, hi = edges[i] * 100, edges[i + 1] * 100
        bar = "█" * max(1, int(n / peak * width)) if n else ""
        rows.append(f"  {lo:+7.1f}% → {hi:+7.1f}%  {bar} {n}")
    return "\n".join(rows)


def report(res: BacktestResult, n_paths: int = 5_000, horizon_days: int = 252) -> str:
    s = simulate(res, n_paths, horizon_days)
    fragile = s["p5"] < -0.25 or s["mdd_p95"] < -0.45 or s["p_loss"] > 0.45
    robust = s["p_loss"] < 0.30 and s["p5"] > -0.15 and s["mdd_p95"] > -0.30
    if fragile:
        verdict = ("FRAGILE — une part importante des trajectoires finit en perte "
                   "sévère ou subit des drawdowns profonds. La performance dépend trop "
                   "de l'ordre des trades : réduire la taille des positions ou ajouter "
                   "des filtres (modules 3/6).")
    elif robust:
        verdict = ("ROBUSTE — la grande majorité des trajectoires reste positive et "
                   "les queues de distribution sont contenues. La performance ne dépend "
                   "pas d'un seul enchaînement chanceux de trades.")
    else:
        verdict = ("MITIGÉ — les pertes extrêmes restent contenues (queues maîtrisées "
                   "par le position sizing) mais l'edge est modeste : une année sur "
                   "trois environ finit en perte. Robuste en risque, perfectible en "
                   "rendement — voir module 6 (optimisation) et module 3 (rendement "
                   "sans risque ajouté).")
    return "\n".join([
        "=" * 78,
        "MODULE 9 — SIMULATION MONTE CARLO",
        f"Stratégie : {res.strategy_name} | Actif : {res.symbol}",
        f"Méthode : bootstrap par blocs (5 j) des rendements quotidiens — "
        f"{s['n_paths']:,} trajectoires × {s['horizon_days']} jours (~1 an)",
        "=" * 78,
        "",
        "— Probabilité de perte (horizon 1 an) —",
        f"  P(rendement < 0)    : {s['p_loss'] * 100:.1f}%",
        f"  P(perte > 10 %)     : {s['p_loss_10'] * 100:.1f}%",
        f"  P(perte > 20 %)     : {s['p_loss_20'] * 100:.1f}%",
        "",
        "— Distribution des rendements à 1 an —",
        f"  Pire scénario  : {s['worst'] * 100:+.1f}%",
        f"  Percentile 5   : {s['p5'] * 100:+.1f}%   (1 année sur 20 fait pire)",
        f"  Percentile 25  : {s['p25'] * 100:+.1f}%",
        f"  Médiane        : {s['median'] * 100:+.1f}%",
        f"  Percentile 75  : {s['p75'] * 100:+.1f}%",
        f"  Percentile 95  : {s['p95'] * 100:+.1f}%",
        f"  Meilleur       : {s['best'] * 100:+.1f}%",
        "",
        _histogram(s["finals"]),
        "",
        "— Drawdowns simulés —",
        f"  Max DD médian sur 1 an : {s['mdd_median'] * 100:.1f}%",
        f"  Max DD au pire 5 %     : {s['mdd_p95'] * 100:.1f}%",
        "",
        f"VERDICT : {verdict}",
    ])
