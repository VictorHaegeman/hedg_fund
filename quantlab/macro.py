"""Module 11 — Stratégie macro : taux d'intérêt, inflation, croissance économique.

Les trois facteurs sont mesurés avec des proxys de marché observables
quotidiennement (pas de données retardées de plusieurs mois) :
  • Taux      : ^TNX (rendement du Treasury US 10 ans)
  • Inflation : ratio TIP/IEF (obligations indexées vs nominales = anticipations
                d'inflation, "breakeven" implicite)
  • Croissance: ratio XLY/XLP (consommation cyclique vs consommation de base —
                monte quand le marché price une accélération économique)

Le couple (croissance, inflation) définit 4 quadrants macro ; la direction des
taux module le choix obligataire. Signaux : chaque proxy comparé à sa moyenne
mobile 126 jours (≈ 6 mois).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

PROXIES = ["^TNX", "TIP", "IEF", "XLY", "XLP"]

FACTOR_EFFECTS = [
    ("TAUX D'INTÉRÊT (^TNX)",
     "Des taux qui MONTENT compriment les valorisations (surtout tech/growth, "
     "duration longue) et pénalisent les obligations existantes ; des taux qui "
     "BAISSENT soutiennent actions de croissance, immobilier, or et obligations "
     "longues (TLT). Effet direct sur le coût du levier des positions."),
    ("INFLATION (ratio TIP/IEF)",
     "Une inflation anticipée en HAUSSE favorise actifs réels (or, matières "
     "premières, énergie) et détruit les obligations nominales ; en BAISSE, elle "
     "redonne de la valeur aux obligations et aux actions de duration longue. "
     "C'est le facteur qui décide si actions et obligations sont corrélées (+) "
     "ou décorrélées (−)."),
    ("CROISSANCE (ratio XLY/XLP)",
     "Une croissance qui ACCÉLÈRE favorise cycliques, small caps et crédit "
     "risqué ; une croissance qui RALENTIT favorise défensives, qualité et "
     "obligations d'État. Le ratio cyclique/défensif réagit des mois avant "
     "les statistiques officielles (PIB, PMI)."),
]

# (croissance, inflation) -> quadrant
QUADRANTS = {
    ("up", "down"): ("GOLDILOCKS (croissance ↑, inflation ↓)",
                     ["QQQ", "SPY"], ["matières premières", "cash excessif"],
                     "le meilleur régime pour les actions, surtout growth/tech"),
    ("up", "up"): ("REFLATION (croissance ↑, inflation ↑)",
                   ["SPY", "GLD"], ["TLT", "obligations longues"],
                   "cycliques et actifs réels mènent ; les obligations souffrent"),
    ("down", "up"): ("STAGFLATION (croissance ↓, inflation ↑)",
                     ["GLD"], ["SPY", "QQQ", "TLT"],
                     "le pire régime : seuls or/actifs réels et cash protègent"),
    ("down", "down"): ("RALENTISSEMENT DÉSINFLATIONNISTE (croissance ↓, inflation ↓)",
                       ["TLT", "IEF"], ["cycliques", "small caps", "crypto"],
                       "les obligations longues redeviennent la meilleure classe"),
}


def _direction(series: pd.Series, lookback: int = 126) -> tuple[str, float]:
    """'up'/'down' selon la position vs SMA(lookback), + variation 6 mois en %."""
    s = series.dropna()
    sma = ind.sma(s, lookback).iloc[-1]
    chg = float(s.iloc[-1] / s.iloc[-min(lookback, len(s) - 1)] - 1)
    return ("up" if float(s.iloc[-1]) > float(sma) else "down"), chg


def detect_macro(prices: dict[str, pd.DataFrame]) -> dict:
    tnx = prices["^TNX"]["Close"]
    infl_ratio = (prices["TIP"]["Close"] / prices["IEF"]["Close"]).dropna()
    growth_ratio = (prices["XLY"]["Close"] / prices["XLP"]["Close"]).dropna()

    rates_dir, rates_chg = _direction(tnx)
    infl_dir, infl_chg = _direction(infl_ratio)
    growth_dir, growth_chg = _direction(growth_ratio)

    name, favor, avoid, why = QUADRANTS[(growth_dir, infl_dir)]
    # Les taux modulent la poche obligataire
    if rates_dir == "up":
        favor = [a for a in favor if a not in ("TLT", "IEF")] or favor
        if "TLT" not in avoid and "obligations longues" not in avoid:
            avoid = avoid + ["TLT (taux en hausse)"]
    return {
        "rates": (rates_dir, rates_chg, float(tnx.iloc[-1])),
        "inflation": (infl_dir, infl_chg, float(infl_ratio.iloc[-1])),
        "growth": (growth_dir, growth_chg, float(growth_ratio.iloc[-1])),
        "quadrant": name, "favor": favor, "avoid": avoid, "why": why,
    }


def example_trades(m: dict, prices: dict[str, pd.DataFrame],
                   capital: float = 10_000.0, risk_pct: float = 1.0) -> list[str]:
    lines = []
    for sym in m["favor"]:
        if sym not in prices:
            continue
        df = prices[sym]
        c = float(df["Close"].iloc[-1])
        atr_v = float(ind.atr(df).iloc[-1])
        stop, target = c - 2 * atr_v, c + 4 * atr_v
        size = capital * risk_pct / 100 / (c - stop)
        lines += [
            f"  LONG {sym} @ {c:,.2f} | stop {stop:,.2f} (2×ATR) | "
            f"target {target:,.2f} (4×ATR) | R/R 2:1 | taille {size:,.4f} unités",
            f"    Sortie macro : clôturer si le quadrant change (proxy directeur "
            f"repasse sa SMA 126 j) même si le stop n'est pas touché.",
        ]
    return lines


def report(prices: dict[str, pd.DataFrame], sources: dict[str, str],
           capital: float = 10_000.0, risk_pct: float = 1.0) -> str:
    m = detect_macro(prices)
    fr = {"up": "EN HAUSSE", "down": "EN BAISSE"}
    lines = [
        "=" * 78,
        "MODULE 11 — STRATÉGIE MACRO",
        f"Proxys : {', '.join(PROXIES)} "
        f"(sources : {', '.join(sorted(set(sources.values())))})",
        "=" * 78,
        "",
        "— Lecture actuelle des 3 facteurs (vs moyenne mobile 126 j ≈ 6 mois) —",
        f"  Taux 10 ans  : {fr[m['rates'][0]]}  "
        f"(niveau {m['rates'][2]:.2f}%, {m['rates'][1] * 100:+.1f}% sur 6 mois)",
        f"  Inflation    : {fr[m['inflation'][0]]}  "
        f"(TIP/IEF = {m['inflation'][2]:.3f}, {m['inflation'][1] * 100:+.1f}% sur 6 mois)",
        f"  Croissance   : {fr[m['growth'][0]]}  "
        f"(XLY/XLP = {m['growth'][2]:.3f}, {m['growth'][1] * 100:+.1f}% sur 6 mois)",
        "",
        f"QUADRANT MACRO : {m['quadrant']}",
        f"  → {m['why']}",
        f"  Classes favorisées : {', '.join(m['favor'])}",
        f"  À éviter           : {', '.join(m['avoid'])}",
        "",
        "— Comment chaque facteur affecte les trades —",
    ]
    for title, text in FACTOR_EFFECTS:
        lines += [f"  • {title}", f"    {text}"]
    lines += [
        "",
        "— Signaux d'entrée / sortie (règles mécaniques) —",
        "  1. Recalculer les 3 directions chaque fin de semaine (proxy vs SMA 126 j).",
        "  2. Entrer LONG sur les classes favorisées du quadrant courant quand le",
        "     proxy directeur vient de croiser sa SMA 126 j (signal frais < 20 jours).",
        "  3. Sortir quand (a) le quadrant change, ou (b) stop 2×ATR touché.",
        "  4. Taille : risque fixe par position (1–2 % de l'équité), max 3 positions.",
        "",
        "— Exemples de trades sur les prix actuels —",
        *(example_trades(m, prices, capital, risk_pct) or
          ["  (aucun ETF favorisé dans l'univers chargé)"]),
        "",
        "Limite honnête : les proxys de marché anticipent la macro mais génèrent des",
        "faux signaux autour des retournements ; ce module se combine avec le module 4",
        "(régime technique) — n'agir que quand les deux pointent dans le même sens.",
    ]
    return "\n".join(lines)
