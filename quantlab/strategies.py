"""Module 1 — Génération de stratégies.

Trois stratégies rentables, définies avec règles exactes (indicateurs + réglages,
entrée/sortie pas à pas, stop-loss / take-profit, conditions de marché, edge).
Chaque stratégie produit des signaux exploitables directement par le backtester.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class StrategySpec:
    """Description complète d'une stratégie (format demandé : 5 points)."""

    key: str
    name: str
    timeframe: str
    indicators_used: list[str]
    entry_rules: list[str]
    exit_rules: list[str]
    stop_take_logic: str
    best_conditions: str
    edge: str
    params: dict = field(default_factory=dict)


class Strategy:
    """Interface : generate_signals(df) -> DataFrame avec colonnes
    entry (bool), exit (bool), stop (prix), target (prix, NaN = pas de TP fixe)."""

    spec: StrategySpec

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def with_params(self, **overrides) -> "Strategy":
        params = {**self.spec.params, **overrides}
        return type(self)(**params)


class TrendFollowing(Strategy):
    """EMA 20/50 + filtre SMA 200 + ADX. Suiveur de tendance long-only."""

    def __init__(self, fast: int = 20, slow: int = 50, trend: int = 200,
                 adx_min: float = 20.0, atr_n: int = 14, stop_mult: float = 2.0,
                 target_mult: float = 4.0):
        self.fast, self.slow, self.trend = fast, slow, trend
        self.adx_min, self.atr_n = adx_min, atr_n
        self.stop_mult, self.target_mult = stop_mult, target_mult
        self.spec = StrategySpec(
            key="trend",
            name="Trend Following (croisement EMA + filtre de tendance)",
            timeframe="1D",
            indicators_used=[
                f"EMA({fast}) et EMA({slow}) sur le Close",
                f"SMA({trend}) — filtre de tendance long terme",
                f"ADX({atr_n}) — force de tendance, seuil {adx_min}",
                f"ATR({atr_n}) — dimensionnement du stop",
            ],
            entry_rules=[
                f"1. Le Close est au-dessus de la SMA({trend}) (marché haussier)",
                f"2. L'EMA({fast}) croise au-dessus de l'EMA({slow})",
                f"3. ADX({atr_n}) > {adx_min} (la tendance a de la force)",
                "4. Entrée à l'ouverture de la bougie suivante",
            ],
            exit_rules=[
                f"1. L'EMA({fast}) recroise sous l'EMA({slow}) → sortie à l'open suivant",
                "2. OU stop-loss / take-profit touché en intraday",
            ],
            stop_take_logic=(
                f"Stop-loss initial = entrée − {stop_mult}×ATR({atr_n}). "
                f"Take-profit = entrée + {target_mult}×ATR({atr_n}) "
                f"(ratio R/R {target_mult / stop_mult:.1f}:1)."
            ),
            best_conditions="Marchés en tendance haussière soutenue, volatilité basse à moyenne "
                            "(indices, grandes capitalisations, crypto en bull market).",
            edge="Capture systématique des tendances longues : les pertes sont coupées vite "
                 "(2×ATR) et les gains courent (4×ATR + sortie sur croisement), ce qui produit "
                 "une espérance positive même avec un win rate < 50 %.",
            params=dict(fast=fast, slow=slow, trend=trend, adx_min=adx_min,
                        atr_n=atr_n, stop_mult=stop_mult, target_mult=target_mult),
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        ema_f, ema_s = ind.ema(c, self.fast), ind.ema(c, self.slow)
        sma_t = ind.sma(c, self.trend)
        adx_v = ind.adx(df, self.atr_n)
        atr_v = ind.atr(df, self.atr_n)

        cross_up = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
        cross_dn = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))
        entry = cross_up & (c > sma_t) & (adx_v > self.adx_min)

        return pd.DataFrame({
            "entry": entry.fillna(False),
            "exit": cross_dn.fillna(False),
            "stop": c - self.stop_mult * atr_v,
            "target": c + self.target_mult * atr_v,
        }, index=df.index)


class MeanReversion(Strategy):
    """RSI(2) survendu dans une tendance haussière — retour à la moyenne court terme."""

    def __init__(self, rsi_n: int = 2, rsi_buy: float = 10.0, rsi_exit: float = 70.0,
                 trend: int = 200, atr_n: int = 14, stop_mult: float = 3.0):
        self.rsi_n, self.rsi_buy, self.rsi_exit = rsi_n, rsi_buy, rsi_exit
        self.trend, self.atr_n, self.stop_mult = trend, atr_n, stop_mult
        self.spec = StrategySpec(
            key="meanrev",
            name="Mean Reversion (RSI court terme en tendance haussière)",
            timeframe="1D",
            indicators_used=[
                f"RSI({rsi_n}) — survente extrême court terme, seuil d'achat {rsi_buy}",
                f"SMA({trend}) — n'acheter les replis que dans un marché haussier",
                f"ATR({atr_n}) — stop de sécurité large ({stop_mult}×)",
            ],
            entry_rules=[
                f"1. Le Close est au-dessus de la SMA({trend})",
                f"2. RSI({rsi_n}) clôture sous {rsi_buy} (repli excessif)",
                "3. Entrée à l'ouverture de la bougie suivante",
            ],
            exit_rules=[
                f"1. RSI({rsi_n}) clôture au-dessus de {rsi_exit} (rebond consommé) → sortie à l'open suivant",
                "2. OU stop-loss touché",
            ],
            stop_take_logic=(
                f"Stop-loss = entrée − {stop_mult}×ATR({atr_n}) (large : on laisse respirer le rebond). "
                "Pas de take-profit fixe : la sortie RSI sert de prise de profit adaptative."
            ),
            best_conditions="Marchés haussiers avec des replis réguliers (indices actions type "
                            "S&P 500/Nasdaq). Évite les marchés baissiers grâce au filtre SMA 200.",
            edge="Les replis brutaux dans une tendance haussière sur-réagissent statistiquement : "
                 "acheter la peur court terme et revendre le rebond donne un win rate élevé "
                 "(souvent 60–75 %) avec une durée de détention de quelques jours.",
            params=dict(rsi_n=rsi_n, rsi_buy=rsi_buy, rsi_exit=rsi_exit,
                        trend=trend, atr_n=atr_n, stop_mult=stop_mult),
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        rsi_v = ind.rsi(c, self.rsi_n)
        sma_t = ind.sma(c, self.trend)
        atr_v = ind.atr(df, self.atr_n)
        entry = (rsi_v < self.rsi_buy) & (c > sma_t)
        exit_ = rsi_v > self.rsi_exit
        return pd.DataFrame({
            "entry": entry.fillna(False),
            "exit": exit_.fillna(False),
            "stop": c - self.stop_mult * atr_v,
            "target": np.nan,
        }, index=df.index)


class Breakout(Strategy):
    """Cassure de canal Donchian (style Turtle) avec sortie sur canal opposé court."""

    def __init__(self, entry_n: int = 55, exit_n: int = 20, atr_n: int = 14,
                 stop_mult: float = 2.0, vol_filter: int = 50):
        self.entry_n, self.exit_n, self.atr_n = entry_n, exit_n, atr_n
        self.stop_mult, self.vol_filter = stop_mult, vol_filter
        self.spec = StrategySpec(
            key="breakout",
            name=f"Breakout Donchian {entry_n}/{exit_n} (style Turtle)",
            timeframe="1D",
            indicators_used=[
                f"Canal de Donchian({entry_n}) — plus haut de {entry_n} jours pour l'entrée",
                f"Canal de Donchian({exit_n}) — plus bas de {exit_n} jours pour la sortie",
                f"ATR({atr_n}) — stop initial {stop_mult}×ATR",
                f"Volume > moyenne {vol_filter} jours — confirmation de la cassure",
            ],
            entry_rules=[
                f"1. Le Close dépasse le plus haut des {entry_n} jours précédents",
                f"2. Volume du jour > moyenne mobile {vol_filter} jours du volume",
                "3. Entrée à l'ouverture de la bougie suivante",
            ],
            exit_rules=[
                f"1. Le Close casse le plus bas des {exit_n} jours précédents → sortie à l'open suivant",
                "2. OU stop-loss initial touché",
            ],
            stop_take_logic=(
                f"Stop-loss initial = entrée − {stop_mult}×ATR({atr_n}). Pas de take-profit fixe : "
                f"le canal bas {exit_n} jours agit comme stop suiveur et laisse courir les grands mouvements."
            ),
            best_conditions="Marchés capables de grands mouvements directionnels : crypto, "
                            "matières premières, actions momentum. Mauvais en range serré.",
            edge="Les nouveaux plus hauts de 55 jours signalent un déséquilibre offre/demande "
                 "persistant ; quelques très gros gagnants paient les nombreuses fausses cassures "
                 "(distribution des gains à queue épaisse).",
            params=dict(entry_n=entry_n, exit_n=exit_n, atr_n=atr_n,
                        stop_mult=stop_mult, vol_filter=vol_filter),
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c, v = df["Close"], df["Volume"]
        upper, _ = ind.donchian(df, self.entry_n)
        _, lower_exit = ind.donchian(df, self.exit_n)
        atr_v = ind.atr(df, self.atr_n)
        vol_ok = v > v.rolling(self.vol_filter).mean()
        entry = (c > upper.shift(1)) & vol_ok
        exit_ = c < lower_exit.shift(1)
        return pd.DataFrame({
            "entry": entry.fillna(False),
            "exit": exit_.fillna(False),
            "stop": c - self.stop_mult * atr_v,
            "target": np.nan,
        }, index=df.index)


ALL_STRATEGIES: dict[str, type[Strategy]] = {
    "trend": TrendFollowing,
    "meanrev": MeanReversion,
    "breakout": Breakout,
}


def get_strategy(key: str, **params) -> Strategy:
    if key not in ALL_STRATEGIES:
        raise KeyError(f"Stratégie inconnue : {key} (choix : {list(ALL_STRATEGIES)})")
    return ALL_STRATEGIES[key](**params)


def describe_all(capital: float = 10_000.0, risk_pct: float = 1.0) -> str:
    lines = [
        "=" * 78,
        "MODULE 1 — GÉNÉRATION DE STRATÉGIES",
        f"Capital : ${capital:,.0f} | Risque par trade : {risk_pct:.1f}% "
        f"(${capital * risk_pct / 100:,.0f}) | Timeframe : 1D",
        "=" * 78,
    ]
    for i, cls in enumerate(ALL_STRATEGIES.values(), 1):
        s = cls().spec
        lines += [
            "",
            f"STRATÉGIE {i} : {s.name}   [clé CLI : {s.key}]",
            "-" * 78,
            "  Indicateurs (réglages exacts) :",
            *[f"    • {x}" for x in s.indicators_used],
            "  Règles d'entrée :",
            *[f"    {x}" for x in s.entry_rules],
            "  Règles de sortie :",
            *[f"    {x}" for x in s.exit_rules],
            f"  Stop-loss / Take-profit : {s.stop_take_logic}",
            f"  Conditions de marché idéales : {s.best_conditions}",
            f"  Edge : {s.edge}",
        ]
    lines += [
        "",
        "Position sizing commun : taille = (capital × risque%) / (entrée − stop).",
        "Backtester une stratégie :  python -m quantlab backtest --strategy <clé> --symbol SPY",
    ]
    return "\n".join(lines)
