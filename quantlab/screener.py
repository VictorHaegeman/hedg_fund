"""Screener TradingView — génération d'univers dynamique (source de données FORWARD).

Hedg_fund travaille sur un univers figé en dur (DEFAULT_UNIVERSE, 14 symboles).
Ce module interroge le screener TradingView (via la lib `tradingview-screener`,
sans clé API) pour proposer un univers de candidats *liquides* à chaque cycle.

⚠ GARDE-FOU D'INTÉGRITÉ — À LIRE AVANT TOUT USAGE
Ce screener ne doit alimenter QUE le cycle LIVE (génération de candidats à venir).
Il ne doit JAMAIS alimenter un backtest (backtest / fund / validate / all) : choisir
les actifs qui sortent du screener *aujourd'hui* puis les backtester dans le passé
injecterait un biais de sélection / look-ahead massif — exactement ce que l'univers
fixe (ETF sectoriels, voir cli.py) est conçu pour éviter.

Tout est fail-safe, comme le reste du projet (data.py → synthétique, news.py →
neutre) : en cas d'échec réseau, de lib absente ou de réponse vide, on renvoie une
liste vide et l'appelant retombe sur l'univers fixe. Aucune exception propagée.
"""

from __future__ import annotations

import re

# Quotes crypto courantes sur les paires TradingView (BTCUSDT, ETHUSD, ...).
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")
# Ticker action plausible (lettres/chiffres, classes d'actions en tiret : BRK-B).
_TICKER_RE = re.compile(r"[A-Z][A-Z0-9-]{0,9}")


def to_yf_symbol(tv_symbol: str, crypto: bool = False) -> str | None:
    """Convertit un symbole TradingView ('NASDAQ:AAPL', 'BINANCE:BTCUSDT') au
    format yfinance attendu par data.load ('AAPL', 'BTC-USD').

    Conservateur : renvoie None pour tout ce qui n'est pas mappable proprement
    (mieux vaut écarter un symbole douteux que charger une mauvaise série)."""
    if not tv_symbol:
        return None
    s = str(tv_symbol).split(":")[-1].strip().upper()
    if not s:
        return None
    if crypto:
        for q in _CRYPTO_QUOTES:
            if s.endswith(q) and len(s) > len(q):
                return f"{s[:-len(q)]}-USD"
        return None  # paire crypto non reconnue → écartée
    s = s.replace(".", "-")  # classes d'actions : BRK.B → BRK-B (convention yfinance)
    return s if _TICKER_RE.fullmatch(s) else None


def _fetch(market: str, limit: int, min_mcap: float, min_volume: float,
           crypto: bool) -> list[str]:
    """Interroge le screener TradingView et renvoie les symboles bruts ('name').

    Entièrement fail-safe : renvoie [] sur toute erreur (lib absente, réseau,
    réponse vide). C'est le seul point qui touche le réseau."""
    try:
        from tradingview_screener import Query, col
    except Exception:
        return []
    try:
        if crypto:
            q = (Query().select("name", "close", "volume")
                 .set_markets("crypto")
                 .order_by("volume", ascending=False)
                 .limit(int(limit)))
        else:
            q = (Query().select("name", "close", "volume", "market_cap_basic")
                 .set_markets(market)
                 .where(col("market_cap_basic") > min_mcap,
                        col("volume") > min_volume)
                 .order_by("market_cap_basic", ascending=False)
                 .limit(int(limit)))
        _, df = q.get_scanner_data()
    except Exception:
        return []
    if df is None or len(df) == 0:
        return []
    source = df["name"] if "name" in df.columns else df.index
    return [str(x) for x in source]


def candidates(market: str = "america", limit: int = 15, min_mcap: float = 2e9,
               min_volume: float = 5e5, crypto: bool = False, fetch=None) -> list[str]:
    """Liste de candidats liquides, déjà mappés au format yfinance.

    `fetch` est injectable (tests hors-ligne) ; par défaut on interroge le réseau
    via `_fetch`. Renvoie [] si rien d'exploitable → l'appelant retombe alors sur
    l'univers fixe."""
    raw = (fetch or _fetch)(market=market, limit=int(limit) * 2, min_mcap=min_mcap,
                            min_volume=min_volume, crypto=crypto)
    out: list[str] = []
    for tv in raw or []:
        sym = to_yf_symbol(tv, crypto=crypto)
        if sym and sym not in out:
            out.append(sym)
        if len(out) >= limit:
            break
    return out


def report(market: str = "america", limit: int = 15, crypto: bool = False,
           fetch=None) -> str:
    """Rapport lisible pour la sous-commande `screen` (inspection, aucun ordre)."""
    syms = candidates(market=market, limit=limit, crypto=crypto, fetch=fetch)
    lines = [
        "=" * 78,
        "SCREENER TRADINGVIEW — univers de candidats (FORWARD / cycle live uniquement)",
        f"Marché : {'crypto' if crypto else market} | limite : {limit}",
        "=" * 78,
    ]
    if not syms:
        lines.append("  Aucun candidat (réseau/lib indisponible ou réponse vide).")
        lines.append("  → Le cycle live retombe alors sur l'univers fixe DEFAULT_UNIVERSE.")
    else:
        lines.append(f"  {len(syms)} candidats (mappés yfinance) :")
        lines.append(f"    {', '.join(syms)}")
        lines.append("")
        lines.append("  ⚠ Source FORWARD : génère des candidats live. Ne JAMAIS l'utiliser")
        lines.append("    pour backtester (biais de sélection / look-ahead).")
    return "\n".join(lines)
