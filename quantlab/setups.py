"""Module 8 — Génération de trade setups haute probabilité sur les données actuelles.

Scanne un univers d'actifs avec les 3 stratégies : signaux déclenchés aujourd'hui
ou proches du déclenchement, avec entrée / stop / take-profit / R:R / raisonnement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .regime import detect
from .strategies import ALL_STRATEGIES, get_strategy


def scan_asset(symbol: str, df: pd.DataFrame) -> list[dict]:
    """Renvoie les setups actifs (signal sur la dernière clôture) + setups proches."""
    setups = []
    regime = detect(df)
    c = float(df["Close"].iloc[-1])
    atr_v = float(ind.atr(df).iloc[-1])

    for key in ALL_STRATEGIES:
        strat = get_strategy(key)
        sig = strat.generate_signals(df)
        last = sig.iloc[-1]
        triggered = bool(last["entry"])

        # Setup "proche" : conditions presque réunies
        near, near_note = _near_trigger(key, df)
        if not triggered and not near:
            continue

        stop = float(last["stop"])
        target = float(last["target"]) if np.isfinite(last["target"]) else c + 2 * (c - stop)
        risk = c - stop
        rr = (target - c) / risk if risk > 0 else np.nan
        setups.append({
            "symbol": symbol, "strategy": strat.spec.name, "status":
                "DÉCLENCHÉ (entrer à l'open prochain)" if triggered else f"EN APPROCHE — {near_note}",
            "entry": round(c, 2), "stop": round(stop, 2), "target": round(target, 2),
            "rr": round(rr, 2),
            "reasoning": _reasoning(key, df, regime),
            "triggered": triggered,
        })
    return setups


def _near_trigger(key: str, df: pd.DataFrame) -> tuple[bool, str]:
    c = df["Close"]
    last = float(c.iloc[-1])
    if key == "trend":
        ema_f, ema_s = ind.ema(c, 20), ind.ema(c, 50)
        gap = float(ema_f.iloc[-1] / ema_s.iloc[-1] - 1)
        above_trend = last > float(ind.sma(c, 200).iloc[-1])
        if above_trend and -0.01 < gap < 0:
            return True, f"EMA20 à {gap * 100:.1f}% de l'EMA50 (croisement imminent)"
    elif key == "meanrev":
        rsi_v = float(ind.rsi(c, 2).iloc[-1])
        above_trend = last > float(ind.sma(c, 200).iloc[-1])
        if above_trend and rsi_v < 25:
            return True, f"RSI(2) = {rsi_v:.0f}, achat sous 10"
    elif key == "breakout":
        upper, _ = ind.donchian(df, 55)
        dist = float(last / upper.shift(1).iloc[-1] - 1)
        if -0.02 < dist < 0:
            return True, f"à {abs(dist) * 100:.1f}% du plus haut 55 jours"
    return False, ""


def _reasoning(key: str, df: pd.DataFrame, regime) -> str:
    tech = {
        "trend": "alignement momentum (EMA20>EMA50) au-dessus de la SMA200 avec ADX confirmant",
        "meanrev": "survente extrême court terme (RSI2) à l'intérieur d'une tendance haussière",
        "breakout": "cassure du plus haut 55 j confirmée par le volume — déséquilibre acheteur",
    }[key]
    macro = (f"régime {regime.trend}/{regime.volatility} vol — "
             + ("environnement favorable à cette stratégie" if _fits(key, regime)
                else "régime peu favorable : taille réduite conseillée"))
    return f"Technique : {tech}. Macro : {macro}."


def _fits(key: str, regime) -> bool:
    # Cohérent avec les recommandations du module 4 (regime.RECOMMENDATIONS)
    if regime.trend == "bull":
        if regime.volatility == "high":
            return key == "meanrev"
        if regime.volatility == "normal":
            return True  # trend en taille normale + meanrev sur les replis
        return key in ("trend", "breakout")
    if regime.trend == "sideways":
        return key == "meanrev" and regime.volatility != "high"
    return False


def report(prices: dict[str, pd.DataFrame], capital: float = 10_000.0,
           risk_pct: float = 1.0, top: int = 3) -> str:
    all_setups = []
    for sym, df in prices.items():
        all_setups.extend(scan_asset(sym, df))
    # Priorité : déclenchés d'abord, puis meilleur R:R
    all_setups.sort(key=lambda s: (not s["triggered"], -(s["rr"] if np.isfinite(s["rr"]) else 0)))
    chosen = all_setups[:top]

    lines = [
        "=" * 78,
        "MODULE 8 — TRADE SETUPS HAUTE PROBABILITÉ",
        f"Univers scanné : {', '.join(prices)} | Capital : ${capital:,.0f} | "
        f"Risque/trade : {risk_pct:.1f}%",
        "=" * 78,
    ]
    if not chosen:
        lines += ["", "Aucun setup déclenché ou en approche aujourd'hui.",
                  "C'est une information : ne pas forcer de trade quand le système est muet."]
        return "\n".join(lines)
    for i, s in enumerate(chosen, 1):
        risk_amount = capital * risk_pct / 100
        size = risk_amount / (s["entry"] - s["stop"]) if s["entry"] > s["stop"] else 0
        lines += [
            "",
            f"TRADE {i} — {s['symbol']} | {s['strategy']}",
            f"  Statut      : {s['status']}",
            f"  Entrée      : {s['entry']:,.2f}",
            f"  Stop-loss   : {s['stop']:,.2f}  (risque {s['entry'] - s['stop']:,.2f}/unité)",
            f"  Take-profit : {s['target']:,.2f}",
            f"  Ratio R/R   : {s['rr']:.2f}:1",
            f"  Taille      : {size:,.4f} unités (risque ${risk_amount:,.0f})",
            f"  Raisonnement: {s['reasoning']}",
        ]
    lines += ["", "Rappel : exécution à l'ouverture de la séance suivante ; annuler le setup",
              "si le prix ouvre au-delà de 1×ATR du niveau d'entrée (gap défavorable)."]
    return "\n".join(lines)
