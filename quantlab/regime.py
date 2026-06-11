"""Module 4 — Détection du régime de marché : tendance, volatilité, volume, recommandations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class Regime:
    trend: str          # bull / bear / sideways
    volatility: str     # low / normal / high
    volume: str         # rising / falling / stable
    detail: dict


def detect(df: pd.DataFrame) -> Regime:
    c, v = df["Close"], df["Volume"]
    sma50, sma200 = ind.sma(c, 50), ind.sma(c, 200)
    last = float(c.iloc[-1])
    s50, s200 = float(sma50.iloc[-1]), float(sma200.iloc[-1])
    slope50 = float(sma50.iloc[-1] / sma50.iloc[-21] - 1) if len(sma50.dropna()) > 21 else 0.0
    adx_v = float(ind.adx(df).iloc[-1])

    if last > s200 and s50 > s200 and slope50 > 0.005:
        trend = "bull"
    elif last < s200 and s50 < s200 and slope50 < -0.005:
        trend = "bear"
    else:
        trend = "sideways"

    rv = ind.realized_vol(c, 20).dropna()
    cur_vol = float(rv.iloc[-1])
    pct = float((rv < cur_vol).mean())  # percentile historique
    volatility = "high" if pct > 0.8 else "low" if pct < 0.3 else "normal"

    v20, v100 = float(v.rolling(20).mean().iloc[-1]), float(v.rolling(100).mean().iloc[-1])
    ratio = v20 / v100 if v100 > 0 else 1.0
    volume = "rising" if ratio > 1.15 else "falling" if ratio < 0.85 else "stable"

    return Regime(trend, volatility, volume, {
        "close": last, "sma50": s50, "sma200": s200, "slope50_1m": slope50,
        "adx": adx_v, "realized_vol_20d": cur_vol, "vol_percentile": pct,
        "volume_ratio_20_100": ratio,
    })


RECOMMENDATIONS = {
    ("bull", "low"): ("Trend following (clé 'trend') ou breakout : les tendances calmes sont "
                      "l'environnement le plus rentable pour laisser courir les gains.",
                      "Éviter : shorter les rallyes, prendre ses profits trop tôt."),
    ("bull", "normal"): ("Trend following en taille normale + mean reversion sur les replis "
                         "(clé 'meanrev').",
                         "Éviter : sur-trader les micro-cassures intraday."),
    ("bull", "high"): ("Mean reversion en taille réduite : les replis violents en tendance "
                       "haussière rebondissent, mais réduire le risque par trade de moitié.",
                       "Éviter : breakouts (taux de faux signaux élevé en haute volatilité)."),
    ("bear", "low"): ("Rester majoritairement cash ; les stratégies long-only n'ont pas d'edge. "
                      "Attendre un croisement SMA50>SMA200 pour réengager.",
                      "Éviter : acheter les creux 'pas chers' sans signal de retournement."),
    ("bear", "normal"): ("Cash ou exposition minimale ; préparer une watchlist pour le "
                         "retournement.", "Éviter : moyennes à la baisse, levier."),
    ("bear", "high"): ("Cash. Les krachs offrent les pires conditions long-only : corrélations "
                       "à 1, gaps, stops sautés.",
                       "Éviter : 'attraper le couteau qui tombe', tout levier."),
    ("sideways", "low"): ("Mean reversion : acheter le bas du range (RSI(2)<10), vendre le haut. "
                          "Compression de volatilité = breakout possible, garder une alerte "
                          "Donchian 55j.",
                          "Éviter : trend following (whipsaw garanti)."),
    ("sideways", "normal"): ("Mean reversion en taille normale ; ignorer les signaux de "
                             "tendance tant que SMA50 et SMA200 sont plates.",
                             "Éviter : courir après les cassures non confirmées par le volume."),
    ("sideways", "high"): ("Réduire fortement l'activité : range + haute volatilité = faux "
                           "signaux des deux côtés. Taille minimale ou cash.",
                           "Éviter : augmenter la taille pour 'se refaire'."),
}


def report(df: pd.DataFrame, symbol: str, source: str = "") -> str:
    r = detect(df)
    d = r.detail
    reco, avoid = RECOMMENDATIONS[(r.trend, r.volatility)]
    trend_fr = {"bull": "HAUSSIER", "bear": "BAISSIER", "sideways": "SANS TENDANCE (range)"}
    vol_fr = {"low": "BASSE", "normal": "NORMALE", "high": "ÉLEVÉE"}
    volu_fr = {"rising": "en hausse", "falling": "en baisse", "stable": "stable"}
    return "\n".join([
        "=" * 78,
        "MODULE 4 — DÉTECTION DU RÉGIME DE MARCHÉ",
        f"Actif : {symbol} | Source : {source} | Dernière clôture : {d['close']:,.2f} "
        f"({df.index[-1].date()})",
        "=" * 78,
        "",
        f"Tendance    : {trend_fr[r.trend]}",
        f"  Close vs SMA200 : {(d['close'] / d['sma200'] - 1) * 100:+.1f}% | "
        f"SMA50 vs SMA200 : {(d['sma50'] / d['sma200'] - 1) * 100:+.1f}% | "
        f"pente SMA50 (1 mois) : {d['slope50_1m'] * 100:+.1f}% | ADX : {d['adx']:.0f}",
        f"Volatilité  : {vol_fr[r.volatility]}",
        f"  Vol réalisée 20 j : {d['realized_vol_20d'] * 100:.1f}% annualisée "
        f"({d['vol_percentile'] * 100:.0f}e percentile historique)",
        f"Volume      : {volu_fr[r.volume]} "
        f"(moyenne 20 j / 100 j = {d['volume_ratio_20_100']:.2f})",
        "",
        f"Stratégie recommandée : {reco}",
        f"À éviter maintenant  : {avoid}",
    ])
