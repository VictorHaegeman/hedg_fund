"""Module 12 — Détection d'alpha / d'edge : opportunités sous-exploitées.

Quatre anomalies connues sont MESURÉES sur les données réelles de l'actif
(pas d'affirmations en l'air : moyenne, t-stat, significativité) :

  Inefficiences comportementales :
    • Réaction excessive aux chutes (reversal post -2 %) — peur court terme
    • Effet tournant du mois (turn-of-month) — flux mécaniques salaires/fonds

  Failles de structure de marché :
    • Écart overnight vs intraday — prime de détention nocturne
    • Anomalie basse volatilité — le calme paie plus que l'agitation

Le rapport classe les edges par |t-stat| et détaille les 2 plus forts :
pourquoi la plupart des traders les ratent + exécution pas à pas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def edge_overnight(df: pd.DataFrame) -> dict:
    on = (df["Open"] / df["Close"].shift(1) - 1).dropna()
    intra = (df["Close"] / df["Open"] - 1).dropna()
    return {
        "key": "overnight", "type": "structure de marché",
        "name": "Écart overnight vs intraday",
        "stat": f"rendement annualisé overnight {on.mean() * 252 * 100:+.1f}% "
                f"vs intraday {intra.mean() * 252 * 100:+.1f}%",
        "tstat": _tstat(on - intra.reindex(on.index)),
        "edge_daily": float(on.mean() - intra.mean()),
        "why_missed": "Invisible sur un graphique en bougies quotidiennes : la prime se "
                      "forme quand le marché est fermé. Les day-traders, qui ferment tout "
                      "le soir, se privent exactement de la partie rémunérée du risque.",
        "steps": [
            "1. Acheter l'actif dans les 10 dernières minutes de la séance (ordre MOC)",
            "2. Revendre à l'ouverture du lendemain (ordre MOO)",
            "3. Taille constante, pas de levier ; coûts critiques → réserver aux ETF",
            "   très liquides (spread < 2 bps) ou via futures",
            "4. Couper la stratégie si la moyenne mobile 1 an de l'edge devient négative",
        ],
    }


def edge_reversal(df: pd.DataFrame, drop: float = -0.02) -> dict:
    ret = df["Close"].pct_change()
    next_ret = ret.shift(-1)
    after_drop = next_ret[ret < drop].dropna()
    base = float(next_ret.mean())
    return {
        "key": "reversal", "type": "comportementale",
        "name": f"Sur-réaction aux chutes (lendemain d'une baisse > {abs(drop) * 100:.0f}%)",
        "stat": f"rendement moyen le lendemain : {after_drop.mean() * 100:+.2f}% "
                f"(vs {base * 100:+.2f}% un jour normal, {len(after_drop)} occurrences)",
        "tstat": _tstat(after_drop - base),
        "edge_daily": float(after_drop.mean() - base),
        "why_missed": "Le biais d'aversion aux pertes pousse la foule à VENDRE après une "
                      "grosse baisse (douleur, appels de marge, stops), précisément quand "
                      "l'espérance du lendemain est la meilleure. Acheter ce moment est "
                      "émotionnellement très coûteux — c'est pour ça que l'edge persiste.",
        "steps": [
            f"1. Filtrer : clôture du jour < {abs(drop) * 100:.0f}% vs veille ET prix > SMA 200",
            "   (on n'achète pas les chutes d'un marché baissier)",
            "2. Acheter à la clôture du jour de la chute (ou à l'open du lendemain)",
            "3. Stop : 3×ATR(14) sous l'entrée ; sortie : clôture du lendemain ou RSI(2)>70",
            "4. Risque max 1 % de l'équité ; jamais 2 unités en même temps sur le même actif",
        ],
    }


def edge_turn_of_month(df: pd.DataFrame, window: int = 3) -> dict:
    ret = df["Close"].pct_change().dropna()
    month = ret.index.to_period("M")
    rank_in_month = pd.Series(ret.groupby(month).cumcount().to_numpy(), index=ret.index)
    days_in_month = rank_in_month.groupby(month).transform("max")
    tom = (rank_in_month < window) | (rank_in_month == days_in_month)  # 3 premiers + dernier
    tom_ret, other_ret = ret[tom], ret[~tom]
    return {
        "key": "turn_of_month", "type": "comportementale",
        "name": "Effet tournant du mois (dernier jour + 3 premiers jours)",
        "stat": f"moyenne jours TOM {tom_ret.mean() * 100:+.3f}%/j "
                f"vs autres jours {other_ret.mean() * 100:+.3f}%/j",
        "tstat": _tstat(tom_ret - other_ret.mean()),
        "edge_daily": float(tom_ret.mean() - other_ret.mean()),
        "why_missed": "Flux mécaniques (salaires investis, rebalancements de fonds, "
                      "fenêtres comptables) trop lents et trop 'ennuyeux' pour les traders "
                      "actifs : l'edge est petit par jour mais récurrent 12 fois par an, et "
                      "personne ne se vante d'un trade de 4 jours calendaires.",
        "steps": [
            "1. Acheter à la clôture de l'avant-dernier jour de bourse du mois",
            "2. Vendre à la clôture du 3e jour de bourse du mois suivant",
            "3. Le reste du temps : cash rémunéré (l'exposition ne dure que ~20 % des jours)",
            "4. Améliorations : sauter le trade si prix < SMA 200 ; combiner avec le module 4",
        ],
    }


def edge_low_vol(df: pd.DataFrame, horizon: int = 20) -> dict:
    c = df["Close"]
    vol = ind.realized_vol(c, 20)
    fwd = c.shift(-horizon) / c - 1
    med = vol.median()
    calm, stress = fwd[vol < med].dropna(), fwd[vol >= med].dropna()
    return {
        "key": "low_vol", "type": "structure de marché",
        "name": "Anomalie basse volatilité (rendement à 20 j selon le calme du marché)",
        "stat": f"après vol basse : {calm.mean() * 100:+.2f}%/20j | "
                f"après vol haute : {stress.mean() * 100:+.2f}%/20j",
        "tstat": _tstat(calm - stress.mean()) / np.sqrt(horizon),  # corrige le chevauchement
        "edge_daily": float((calm.mean() - stress.mean()) / horizon),
        "why_missed": "Le cerveau assimile volatilité à opportunité : les traders "
                      "augmentent la taille quand ça bouge (et perdent), la réduisent quand "
                      "c'est calme (et ratent la dérive haussière). L'edge est l'inverse "
                      "exact de l'instinct.",
        "steps": [
            "1. Calculer chaque soir la volatilité réalisée 20 j et sa médiane historique",
            "2. Exposition 100 % quand vol < médiane, 30 % quand vol ≥ médiane",
            "   (ou cible de vol : taille = vol_cible / vol_courante, plafonnée à 1)",
            "3. Appliquer en overlay sur n'importe quelle stratégie des modules 1/5/11",
            "4. Rebalancer seulement si le changement de taille dépasse 10 points de %",
        ],
    }


def detect_edges(df: pd.DataFrame) -> list[dict]:
    edges = [edge_overnight(df), edge_reversal(df), edge_turn_of_month(df), edge_low_vol(df)]
    return sorted(edges, key=lambda e: abs(e["tstat"]), reverse=True)


def _signif(t: float) -> str:
    a = abs(t)
    return ("très significatif (|t|>3)" if a > 3 else
            "significatif (|t|>2)" if a > 2 else
            "faible (|t|>1)" if a > 1 else "non significatif — à ignorer")


def report(df: pd.DataFrame, symbol: str, source: str = "") -> str:
    edges = detect_edges(df)
    lines = [
        "=" * 78,
        "MODULE 12 — DÉTECTION D'ALPHA / D'EDGE",
        f"Marché analysé : {symbol} | Source : {source} | "
        f"{len(df)} séances ({df.index[0].date()} → {df.index[-1].date()})",
        "=" * 78,
        "",
        "Anomalies mesurées sur CES données (classées par force statistique) :",
    ]
    for e in edges:
        lines.append(f"  [{e['tstat']:+.1f}t] {e['name']} ({e['type']}) — {e['stat']}"
                     f" → {_signif(e['tstat'])}")
    # Stratégies proposées : les 2 meilleurs t SIGNÉS (l'effet doit jouer dans
    # le sens exploitable ; un t négatif = anomalie inversée sur cet actif)
    top2 = sorted(edges, key=lambda e: e["tstat"], reverse=True)[:2]
    for i, e in enumerate(top2, 1):
        lines += [
            "",
            f"STRATÉGIE UNIQUE {i} : {e['name']}",
            "-" * 78,
            f"  Inefficience exploitée ({e['type']}) : {e['stat']}",
            f"  Force statistique : t = {e['tstat']:+.2f} ({_signif(e['tstat'])})",
            *(["  ⚠ Effet INVERSÉ sur cet actif (t < 0) : ne pas trader cette anomalie "
               "ici telle quelle."] if e["tstat"] < 0 else []),
            f"  Pourquoi la plupart des traders la ratent :",
            f"    {e['why_missed']}",
            "  Exécution pas à pas :",
            *[f"    {s}" for s in e["steps"]],
        ]
    lines += [
        "",
        "Garde-fous : un edge mesuré sur le passé peut s'éroder (arbitrage, frais).",
        "Toujours retester sur la moitié récente des données avant d'engager du capital,",
        "et dimensionner comme si l'edge réel valait la moitié de l'edge mesuré.",
    ]
    return "\n".join(lines)
