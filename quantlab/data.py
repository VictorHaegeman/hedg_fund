"""Chargement des données : yfinance avec cache CSV local, fallback synthétique hors-ligne."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(symbol: str, years: int) -> str:
    safe = symbol.replace("^", "_").replace("/", "_").replace("=", "_").replace("-", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{years}y.csv")


def _from_cache(symbol: str, years: int, max_age_days: int = 3) -> pd.DataFrame | None:
    path = _cache_path(symbol, years)
    if not os.path.exists(path):
        return None
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    if age > timedelta(days=max_age_days):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if not df.empty else None


def _to_cache(symbol: str, years: int, df: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(_cache_path(symbol, years))


def _download(symbol: str, years: int) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        df = yf.download(
            symbol,
            period=f"{years}y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None
        # yfinance peut renvoyer des colonnes MultiIndex (ticker en 2e niveau)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[[c for c in REQUIRED_COLS if c in df.columns]].copy()
        df = df.dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        return df if len(df) > 100 else None
    except Exception:
        return None


def synthetic_ohlcv(symbol: str, years: int = 10, seed: int | None = None) -> pd.DataFrame:
    """Série OHLCV synthétique réaliste (GBM avec changements de régime) — fallback hors-ligne."""
    if seed is None:
        seed = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    n = int(years * 252)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)

    # Régimes alternés bull/bear/range pour produire tendances et drawdowns crédibles
    mu, sigma = np.zeros(n), np.zeros(n)
    i = 0
    while i < n:
        length = int(rng.integers(60, 400))
        regime = rng.choice(["bull", "bear", "range"], p=[0.5, 0.2, 0.3])
        if regime == "bull":
            m, s = 0.12, 0.14
        elif regime == "bear":
            m, s = -0.18, 0.30
        else:
            m, s = 0.02, 0.18
        mu[i : i + length] = m / 252
        sigma[i : i + length] = s / np.sqrt(252)
        i += length

    rets = rng.normal(mu, sigma)
    close = 100.0 * np.exp(np.cumsum(rets))
    intraday = np.abs(rng.normal(0, sigma)) + 0.002
    open_ = close * (1 + rng.normal(0, sigma * 0.3))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    volume = (rng.lognormal(13, 0.4, n) * (1 + 2 * np.abs(rets) / (sigma + 1e-9))).astype(int)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def load(symbol: str, years: int = 10, allow_synthetic: bool = True) -> tuple[pd.DataFrame, str]:
    """Renvoie (df OHLCV journalier, source) — source ∈ {'cache', 'yahoo', 'synthetic'}."""
    cached = _from_cache(symbol, years)
    if cached is not None:
        return cached, "cache"
    df = _download(symbol, years)
    if df is not None:
        _to_cache(symbol, years, df)
        return df, "yahoo"
    if allow_synthetic:
        return synthetic_ohlcv(symbol, years), "synthetic"
    raise RuntimeError(f"Impossible de charger les données pour {symbol}")
